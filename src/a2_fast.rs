//! The fast path for narrow A2-shaped WaveNets: the reference's
//! `a2_fast.cpp` for three channels, in the arithmetic of this crate's
//! generic path.
//!
//! An A2 Lite model is three channels wide and twenty-three layers deep. The
//! generic path runs it as the batched kernels do everything else, with
//! lanes across output channels, and at three channels that is no lanes at
//! all:
//! every step is scalar, and the time goes on the steps around the
//! arithmetic rather than in it. Here the lanes are *frames*. Each layer's
//! history is planar, one row per channel, so four consecutive frames of a
//! channel are one vector load; a tap is then nine broadcast weights against
//! frame vectors, with the accumulator for every four frames independent of
//! every other, which is what keeps the core's multiply-add units full. The
//! mixin, the activation, the head sum and the 1x1 residual follow in the
//! same frame vectors. Eight channels are the generic path's own shape,
//! where lanes across channels fill a vector, so that model keeps it.
//!
//! # Bit-exactness
//!
//! For every output channel and frame, each number is computed in the order
//! the generic path computes it: a tap's contribution is a fused chain over
//! the inputs from zero, added to the running sum tap by tap, the bias last;
//! the mixin is one product; the 1x1 the same chain then its bias; the head
//! the same over its taps. Which frames advance together is the only
//! difference, and frames are independent. The tests below hold the two
//! paths to each other bit for bit on the A2 container's narrow
//! child, and the recorded cases hold the result to the C++ reference.

use crate::buffer::Buf;
use crate::engine::{Engine, prewarm_with_silence};
use crate::kernels::macc_row;
use crate::wavenet::{A2Parts, WaveNet};

/// Frames per vector.
const L: usize = 4;
/// The most channels this path takes: a weight per channel is a scalar
/// here, and the tails hold `C * C` of them in registers.
const MAX_C: usize = 4;
/// Blocks of history kept between rewinds, in units of the block size.
const SLACK: usize = 16;

/// A planar history: one row per channel, frames contiguous, linear with an
/// occasional rewind so every tap's read for a block is one contiguous run.
#[derive(Debug, Clone)]
struct Planes {
    rows: [Vec<f32>; MAX_C],
    lookback: usize,
    cap: usize,
    /// Where the next block's first frame goes; at least `lookback`.
    pos: usize,
}

impl Planes {
    fn new(lookback: usize, max_block: usize) -> Self {
        let cap = lookback + SLACK * max_block + L;
        Self {
            rows: std::array::from_fn(|_| vec![0.0; cap]),
            lookback,
            cap,
            pos: lookback,
        }
    }

    /// Make room for `n` more frames: rewind when the end is near, keeping
    /// the last `lookback` frames.
    #[inline(always)]
    fn begin(&mut self, n: usize) {
        if self.pos + n > self.cap {
            for row in &mut self.rows {
                row.copy_within(self.pos - self.lookback..self.pos, 0);
            }
            self.pos = self.lookback;
        }
    }

    /// The frames `lookback` before the block's first, `n` of them.
    #[inline(always)]
    fn read(&self, c: usize, lookback: usize, n: usize) -> &[f32] {
        let start = self.pos - lookback;
        &self.rows[c][start..start + n]
    }

    fn clear(&mut self) {
        for row in &mut self.rows {
            row.iter_mut().for_each(|v| *v = 0.0);
        }
        self.pos = self.lookback;
    }
}

#[derive(Debug, Clone)]
struct Layer {
    kernel_size: usize,
    dilation: usize,
    /// `w[k][o * MAX_C + i]`: tap `k`, from input `i` to output `o`.
    conv_w: Vec<[f32; MAX_C * MAX_C]>,
    conv_b: [f32; MAX_C],
    mixin_w: [f32; MAX_C],
    /// `[o * MAX_C + i]`.
    l1x1_w: [f32; MAX_C * MAX_C],
    l1x1_b: [f32; MAX_C],
    slope: f32,
    hist: Planes,
}

/// The engine.
#[derive(Debug, Clone)]
pub(crate) struct A2Fast {
    c: usize,
    rechannel: [f32; MAX_C],
    layers: Vec<Layer>,
    /// Per tap, a weight per channel.
    head_w: Vec<[f32; MAX_C]>,
    head_b: f32,
    head_scale: f32,
    head_hist: Planes,
    prewarm_samples: usize,
    max_buffer: usize,
    /// The block size rounded up to whole vectors; the work buffers' length.
    padded: usize,
    prewarm_on_reset: bool,
    /// The residual stream through the layers, planar.
    stream: [Vec<f32>; MAX_C],
    /// A layer's activated output, planar.
    z: [Vec<f32>; MAX_C],
    /// The activations summed over the layers, planar.
    head_sum: [Vec<f32>; MAX_C],
    cond: Vec<f32>,
    out: Vec<f32>,
}

/// Column-major `(rows, rows)` into `[o * MAX_C + i]`.
fn square(m: &[f32], rows: usize) -> [f32; MAX_C * MAX_C] {
    let mut out = [0.0; MAX_C * MAX_C];
    for i in 0..rows {
        for o in 0..rows {
            out[o * MAX_C + i] = m[i * rows + o];
        }
    }
    out
}

fn pad(v: &[f32]) -> [f32; MAX_C] {
    let mut out = [0.0; MAX_C];
    out[..v.len()].copy_from_slice(v);
    out
}

impl A2Fast {
    /// Whether this path takes a model of that many channels.
    fn takes(channels: usize) -> bool {
        (1..=MAX_C).contains(&channels)
    }

    /// Build from the parts of a model with at most [`MAX_C`] channels.
    fn from_parts(parts: &A2Parts) -> Self {
        let c = parts.channels;
        assert!(Self::takes(c), "{c} channels is not this path's shape");
        let layers = parts
            .layers
            .iter()
            .map(|l| Layer {
                kernel_size: l.kernel_size,
                dilation: l.dilation,
                conv_w: l.conv_w.iter().map(|tap| square(tap, c)).collect(),
                conv_b: pad(&l.conv_b),
                mixin_w: pad(&l.mixin_w),
                l1x1_w: square(&l.l1x1_w, c),
                l1x1_b: pad(&l.l1x1_b),
                slope: l.slope,
                hist: Planes::new((l.kernel_size - 1) * l.dilation, 0),
            })
            .collect();
        Self {
            c,
            rechannel: pad(&parts.rechannel),
            layers,
            head_w: parts.head_w.iter().map(|tap| pad(tap)).collect(),
            head_b: parts.head_b,
            head_scale: parts.head_scale,
            head_hist: Planes::new(parts.head_w.len().saturating_sub(1), 0),
            prewarm_samples: parts.prewarm_samples,
            max_buffer: 0,
            padded: 0,
            prewarm_on_reset: true,
            stream: std::array::from_fn(|_| Vec::new()),
            z: std::array::from_fn(|_| Vec::new()),
            head_sum: std::array::from_fn(|_| Vec::new()),
            cond: Vec::new(),
            out: Vec::new(),
        }
    }

    /// One block, `n` frames from `self.cond` into `self.out`, for a model of
    /// `C` channels. Everything is done in whole vectors of frames; the
    /// frames past `n` are computed on stale data and never kept.
    #[inline(always)]
    fn block<const C: usize, const FUSE: bool>(&mut self, n: usize) {
        let groups = n.div_ceil(L);
        let m = groups * L;
        let cond = &self.cond[..m];
        // The rechannel: `fma(w, cond, 0)` is the product rounded once.
        for o in 0..C {
            let w = self.rechannel[o];
            for (x, cd) in self.stream[o][..m].iter_mut().zip(cond) {
                *x = if FUSE { w.mul_add(*cd, 0.0) } else { w * *cd };
            }
            self.head_sum[o][..m].iter_mut().for_each(|v| *v = 0.0);
        }
        for layer in &mut self.layers {
            layer.hist.begin(m);
            let pos = layer.hist.pos;
            for o in 0..C {
                layer.hist.rows[o][pos..pos + n].copy_from_slice(&self.stream[o][..n]);
            }
            let mut z: [&mut [[f32; L]]; MAX_C] =
                self.z.each_mut().map(|row| row[..m].as_chunks_mut::<L>().0);
            for zo in z.iter_mut().take(C) {
                zo.iter_mut().for_each(|g| *g = [0.0; L]);
            }
            let last = layer.kernel_size - 1;
            for (k, w) in layer.conv_w.iter().enumerate() {
                let lookback = (last - k) * layer.dilation;
                let x: [&[[f32; L]]; MAX_C] = std::array::from_fn(|i| {
                    if i < C {
                        layer.hist.read(i, lookback, m).as_chunks::<L>().0
                    } else {
                        &[]
                    }
                });
                for g in 0..groups {
                    // This group's inputs, loaded once for every output.
                    let xg: [[f32; L]; MAX_C] =
                        std::array::from_fn(|i| if i < C { x[i][g] } else { [0.0; L] });
                    for (o, zo) in z.iter_mut().take(C).enumerate() {
                        let mut partial = [0.0f32; L];
                        for (i, xi) in xg.iter().take(C).enumerate() {
                            macc_row::<L, FUSE>(&mut partial, xi, w[o * MAX_C + i]);
                        }
                        for (zl, p) in zo[g].iter_mut().zip(&partial) {
                            *zl += p;
                        }
                    }
                }
            }
            // Bias, mixin, activation, and the head sum.
            for o in 0..C {
                let (b, wm, slope) = (layer.conv_b[o], layer.mixin_w[o], layer.slope);
                for ((z, cd), hs) in self.z[o][..m]
                    .iter_mut()
                    .zip(cond)
                    .zip(&mut self.head_sum[o][..m])
                {
                    let mix = if FUSE { wm.mul_add(*cd, 0.0) } else { wm * *cd };
                    let v = (*z + b) + mix;
                    *z = if v > 0.0 { v } else { slope * v };
                    *hs += *z;
                }
            }
            // The 1x1 and the residual, into the stream for the next layer.
            let zc: [&[[f32; L]]; MAX_C] = std::array::from_fn(|i| {
                if i < C {
                    self.z[i][..m].as_chunks::<L>().0
                } else {
                    &[]
                }
            });
            let mut sc: [&mut [[f32; L]]; MAX_C] = self
                .stream
                .each_mut()
                .map(|row| row[..m].as_chunks_mut::<L>().0);
            for g in 0..groups {
                let zg: [[f32; L]; MAX_C] =
                    std::array::from_fn(|i| if i < C { zc[i][g] } else { [0.0; L] });
                for (o, so) in sc.iter_mut().take(C).enumerate() {
                    let mut q = [0.0f32; L];
                    for (i, zi) in zg.iter().take(C).enumerate() {
                        macc_row::<L, FUSE>(&mut q, zi, layer.l1x1_w[o * MAX_C + i]);
                    }
                    let b = layer.l1x1_b[o];
                    for (s, q) in so[g].iter_mut().zip(&q) {
                        *s += q + b;
                    }
                }
            }
            layer.hist.pos += n;
        }
        // The head: a convolution over the summed activations, one output.
        self.head_hist.begin(m);
        let pos = self.head_hist.pos;
        for o in 0..C {
            self.head_hist.rows[o][pos..pos + n].copy_from_slice(&self.head_sum[o][..n]);
        }
        let taps = self.head_w.len();
        let (yc, _) = self.out[..m].as_chunks_mut::<L>();
        yc.iter_mut().for_each(|g| *g = [0.0; L]);
        for (k, w) in self.head_w.iter().enumerate() {
            let lookback = taps - 1 - k;
            let x: [&[[f32; L]]; MAX_C] = std::array::from_fn(|i| {
                if i < C {
                    self.head_hist.read(i, lookback, m).as_chunks::<L>().0
                } else {
                    &[]
                }
            });
            for (g, yg) in yc.iter_mut().enumerate() {
                let mut partial = [0.0f32; L];
                for (xi, wi) in x.iter().zip(w).take(C) {
                    macc_row::<L, FUSE>(&mut partial, &xi[g], *wi);
                }
                for (yl, p) in yg.iter_mut().zip(&partial) {
                    *yl += p;
                }
            }
        }
        let (b, scale) = (self.head_b, self.head_scale);
        for y in &mut self.out[..m] {
            *y = scale * (*y + b);
        }
        self.head_hist.pos += n;
    }

    /// The block for this model's channel count, fused or not as this
    /// machine does it: see [`crate::kernels::fused`].
    fn block_here(&mut self, n: usize) {
        #[cfg(target_arch = "x86_64")]
        if crate::kernels::fused() {
            // SAFETY: `fused()` reported the CPU has FMA and AVX.
            return unsafe { self.block_fma(n) };
        }
        self.block_c::<{ !cfg!(target_arch = "x86_64") }>(n)
    }

    #[inline(always)]
    fn block_c<const FUSE: bool>(&mut self, n: usize) {
        match self.c {
            1 => self.block::<1, FUSE>(n),
            2 => self.block::<2, FUSE>(n),
            3 => self.block::<3, FUSE>(n),
            _ => self.block::<4, FUSE>(n),
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx,fma")]
    unsafe fn block_fma(&mut self, n: usize) {
        self.block_c::<true>(n)
    }
}

impl Engine for A2Fast {
    fn in_channels(&self) -> usize {
        1
    }

    fn out_channels(&self) -> usize {
        1
    }

    fn prewarm_samples(&self) -> usize {
        self.prewarm_samples
    }

    fn max_buffer_size(&self) -> usize {
        self.max_buffer
    }

    /// Allocating: the histories are rebuilt for the block, and cleared, as
    /// the generic path's arena is.
    fn set_max_buffer_size(&mut self, max_buffer: usize) {
        self.max_buffer = max_buffer;
        self.padded = max_buffer.div_ceil(L) * L;
        for l in &mut self.layers {
            l.hist = Planes::new((l.kernel_size - 1) * l.dilation, self.padded);
        }
        self.head_hist = Planes::new(self.head_w.len().saturating_sub(1), self.padded);
        for row in self
            .stream
            .iter_mut()
            .chain(&mut self.z)
            .chain(&mut self.head_sum)
        {
            *row = vec![0.0; self.padded];
        }
        self.cond = vec![0.0; self.padded];
        self.out = vec![0.0; self.padded];
    }

    fn prewarm(&mut self) {
        for l in &mut self.layers {
            l.hist.clear();
        }
        self.head_hist.clear();
        let max_buffer = self.max_buffer;
        prewarm_with_silence(self, max_buffer);
    }

    fn process(&mut self, input: &[&[f32]], output: &mut [&mut [f32]], n: usize) {
        debug_assert!(!input.is_empty() && !output.is_empty());
        debug_assert!(
            n <= self.max_buffer,
            "block of {n} frames, but sized for {}",
            self.max_buffer
        );
        self.cond[..n].copy_from_slice(&input[0][..n]);
        self.block_here(n);
        output[0][..n].copy_from_slice(&self.out[..n]);
    }

    fn process_buf(&mut self, input: &Buf, output: &mut Buf, n: usize) {
        debug_assert_eq!(input.rows(), 1);
        debug_assert_eq!(output.rows(), 1);
        self.cond[..n].copy_from_slice(input.left(n));
        self.block_here(n);
        output.left_mut(n).copy_from_slice(&self.out[..n]);
    }

    fn set_prewarm_on_reset(&mut self, on: bool) {
        self.prewarm_on_reset = on;
    }

    fn prewarm_on_reset(&self) -> bool {
        self.prewarm_on_reset
    }
}

/// The engine for a WaveNet: the fast path when the model has the narrow A2
/// shape, the model itself otherwise.
pub(crate) fn select(net: WaveNet) -> Box<dyn Engine> {
    match net.a2_parts() {
        Some(parts) if A2Fast::takes(parts.channels) => Box::new(A2Fast::from_parts(&parts)),
        _ => Box::new(net),
    }
}

#[cfg(test)]
mod tests {
    //! The fast path against the generic path: bit for bit, on both children
    //! of the A2 container, over a random signal in ragged blocks,
    //! with and without a prewarm.

    use super::*;
    use crate::format::{ArchConfig, parse_value};
    use std::path::Path;

    fn children() -> Vec<(String, serde_json::Value)> {
        let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/models/a2.nam");
        let v: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap();
        v["config"]["submodels"]
            .as_array()
            .unwrap()
            .iter()
            .map(|sm| {
                let channels = sm["model"]["config"]["layers"][0]["channels"]
                    .as_u64()
                    .unwrap();
                (format!("{channels} channels"), sm["model"].clone())
            })
            .collect()
    }

    fn generic(v: &serde_json::Value) -> WaveNet {
        let file = parse_value(v).unwrap();
        match file.config {
            ArchConfig::WaveNet(cfg) => WaveNet::new(cfg, &file.weights).unwrap(),
            _ => panic!("not a WaveNet"),
        }
    }

    fn signal(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5
            })
            .collect()
    }

    fn run(
        engine: &mut dyn Engine,
        input: &[f32],
        blocks: &[usize],
        max_buffer: usize,
        prewarm: bool,
    ) -> Vec<f32> {
        engine.set_max_buffer_size(max_buffer);
        if prewarm {
            engine.prewarm();
        }
        let mut out = vec![0.0f32; input.len()];
        let mut pos = 0;
        let mut i = 0;
        while pos < input.len() {
            let n = blocks[i % blocks.len()].min(input.len() - pos);
            let src = &input[pos..pos + n];
            let dst = &mut out[pos..pos + n];
            engine.process(&[src], &mut [dst], n);
            pos += n;
            i += 1;
        }
        out
    }

    #[test]
    fn the_narrow_child_takes_the_fast_path_and_the_wide_one_keeps_the_generic() {
        let mut seen = Vec::new();
        for (name, v) in children() {
            let parts = generic(&v)
                .a2_parts()
                .expect("both children have the A2 shape");
            let fast_path = A2Fast::takes(parts.channels);
            assert_eq!(fast_path, name.starts_with("3 "), "{name}");
            seen.push(fast_path);
        }
        assert!(
            seen.contains(&true) && seen.contains(&false),
            "both shapes were exercised"
        );
    }

    #[test]
    fn the_fast_path_is_the_generic_path_bit_for_bit() {
        let input = signal(48_000 * 2, 7);
        for (name, v) in children() {
            for (blocks, max_buffer, prewarm) in [
                (vec![64usize], 64, true),
                (vec![1usize, 7, 64, 3, 128, 33], 128, true),
                (vec![512usize], 512, false),
                (vec![31usize, 1, 1, 100], 100, false),
            ] {
                let mut g = generic(&v);
                let want = run(&mut g, &input, &blocks, max_buffer, prewarm);
                let mut f = select(generic(&v));
                let got = run(f.as_mut(), &input, &blocks, max_buffer, prewarm);
                let first = want
                    .iter()
                    .zip(&got)
                    .position(|(a, b)| a.to_bits() != b.to_bits());
                assert!(
                    first.is_none(),
                    "{name}, blocks {blocks:?}, prewarm {prewarm}: first difference at frame {} ({} vs {})",
                    first.unwrap(),
                    want[first.unwrap()],
                    got[first.unwrap()]
                );
                assert!(
                    want.iter().any(|v| v.abs() > 1e-3),
                    "{name}: the model made no sound"
                );
            }
        }
    }

    #[test]
    fn the_fast_path_reports_what_the_model_does() {
        for (name, v) in children() {
            let g = generic(&v);
            let f = select(generic(&v));
            assert_eq!(f.in_channels(), g.in_channels(), "{name}");
            assert_eq!(f.out_channels(), g.out_channels(), "{name}");
            assert_eq!(f.prewarm_samples(), g.prewarm_samples(), "{name}");
        }
    }
}
