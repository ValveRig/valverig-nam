//! Convolution kernels.
//!
//! # Shape specialisation
//!
//! The C++ reference carries hand-written specialisations for a handful of
//! channel counts and kernel sizes, and anything outside that set falls into
//! a generic path. The model shape is known at load time and never changes,
//! so it is a code-generation problem: [`matmul`] is written once and
//! monomorphised for every `(OUT, IN)` pair in [`for_each_shape`], and
//! [`select_matmul`] hands a layer the function pointer for its shape when it
//! is built. Shapes outside the table run [`matmul_dyn`], the same arithmetic
//! with runtime bounds, written to be good rather than to be a cliff.
//!
//! # Weight reuse
//!
//! These models are bound by weight bandwidth, not arithmetic: a 16-channel
//! layer streams its whole weight matrix per frame for a few hundred
//! multiply-adds. The batched kernels load a weight column once and apply it
//! to a tile of frames before touching the next one, cutting weight traffic
//! by the tile's size; [`tile_for`] picks the tile per shape.
//!
//! # Vectorisation
//!
//! The accumulator is indexed by *output channel*, and distinct output
//! channels are independent sums, so widening that loop to a SIMD register
//! cannot reorder any addition. The row step, one input's contribution to
//! every output channel, is written out as four NEON or SSE lanes in
//! [`lane4`], because rustc's auto-vectoriser does not take that loop at 16
//! channels and the scalar form it emits spills. Every other target gets the
//! scalar loop, which is the same arithmetic in the same order.
//!
//! Two bounds checks the compiler cannot prove away are hoisted: `out` is
//! narrowed to `&mut [f32; OUT]` once so the write-back loop vectorises, and
//! a tile's input columns are taken as `&[f32; IN]` once per tile rather
//! than indexed through a runtime stride inside the inner loop.
//!
//! # Bit-exactness
//!
//! Every kernel in this module accumulates in exactly the same order:
//! `acc[o] += w[o, i] * x[i]` for `i` ascending, starting from `0.0`, with
//! the bias added last and separately by the caller. The specialised,
//! dynamic and batched paths differ only in loop bounds and in which frames
//! advance together, so they are bit-identical to *each other* by
//! construction. The tests below check it anyway, and `tests/reference.rs`
//! checks the whole-model consequence against output from the C++ reference.
//!
//! # Fused multiply-add
//!
//! The multiply-accumulate step fuses where the machine has the instruction:
//! one rounding, where `a * b + c` is two. The C++ reference makes the same
//! choice at its own release flags, where Eigen's `pmadd` lowers to a fused
//! multiply-add. Rust never contracts on its own, so every fusion here is a
//! written one, through [`macc`].
//!
//! What it costs is bit-identical output against one pinned build of the
//! reference; what stays is agreement to within that reference's own
//! build-to-build spread, recorded per case in `assets/expectations.txt`.
//! What it buys is up to a quarter of the throughput on the large captures.
//!
//! Only the *vector-lane* accumulations fuse. Where the accumulator is a
//! single scalar carried across a loop, in the `out_ch > DYN_ACC` branch of
//! [`matvec_dyn`] and in the LSTM gate products, fusing is slower, because
//! the whole loop is one serial dependency chain and a fused multiply-add is
//! a longer link in it than an add. Those sites are written unfused.
//!
//! On x86-64 the fused instructions are not in the baseline, so [`fused`]
//! asks the CPU once and every kernel follows its answer: each has a twin
//! compiled for AVX and FMA, entered only after that check, and the unfused
//! form otherwise. On aarch64 NEON is baseline and always fuses.
//!
//! # Layout
//!
//! Weight matrices are column-major `(out_channels, in_channels)`, matching
//! `Eigen::MatrixXf`: `w[o, i]` lives at `w[o + i * OUT]`. Frames are
//! columns, so one frame of activations is contiguous.

/// One multiply-accumulate step, fused: one instruction and one rounding,
/// where `acc + a * b` would be two of each.
#[inline(always)]
pub(crate) fn macc(acc: f32, a: f32, b: f32) -> f32 {
    a.mul_add(b, acc)
}

/// One multiply-accumulate step, fused or not by the const.
#[inline(always)]
fn step<const FUSE: bool>(acc: f32, a: f32, b: f32) -> f32 {
    if FUSE { macc(acc, a, b) } else { acc + a * b }
}

/// Whether this machine fuses: yes wherever the fused instruction is in the
/// target's baseline, and on x86-64 exactly when the CPU has FMA and AVX.
///
/// One answer per process, and every kernel here follows it, so they stay
/// bit-identical to each other on every machine. Without the instruction,
/// fusing would mean a library call per multiply-accumulate; the reference
/// does not fuse on such a machine either.
#[inline(always)]
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
pub(crate) fn fused() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::is_x86_feature_detected!("fma") && std::arch::is_x86_feature_detected!("avx")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        true
    }
}

/// Four lanes of `acc[k] = acc[k] + w[k] * x`, fused by `FUSE`, as one
/// vector operation where the target has one.
///
/// A lane is one output channel, so the vector form adds nothing in a
/// different order: it is the scalar loop bit for bit.
#[inline(always)]
fn lane4<const FUSE: bool>(acc: &mut [f32; 4], w: &[f32; 4], x: f32) {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        use core::arch::aarch64::{
            vaddq_f32, vdupq_n_f32, vfmaq_f32, vld1q_f32, vmulq_f32, vst1q_f32,
        };
        // SAFETY: the cfg above holds only where NEON is a baseline feature
        // of the target, and both pointers are to a whole `[f32; 4]`.
        unsafe {
            let (av, wv, xv) = (
                vld1q_f32(acc.as_ptr()),
                vld1q_f32(w.as_ptr()),
                vdupq_n_f32(x),
            );
            let r = if FUSE {
                vfmaq_f32(av, wv, xv)
            } else {
                vaddq_f32(av, vmulq_f32(wv, xv))
            };
            vst1q_f32(acc.as_mut_ptr(), r);
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        use core::arch::x86_64::{
            _mm_add_ps, _mm_fmadd_ps, _mm_loadu_ps, _mm_mul_ps, _mm_set1_ps, _mm_storeu_ps,
        };
        // SAFETY: SSE2 is baseline on x86-64, and both pointers are to a
        // whole `[f32; 4]`. The fused form needs FMA, which is not: it is
        // only ever instantiated inside a twin compiled with
        // `target_feature(enable = "avx,fma")` that `fused_dispatch!` enters
        // after `fused()` reported the CPU has it. Nothing in this crate
        // instantiates `lane4::<true>` anywhere else, and nothing outside
        // the crate can.
        unsafe {
            let (av, wv, xv) = (
                _mm_loadu_ps(acc.as_ptr()),
                _mm_loadu_ps(w.as_ptr()),
                _mm_set1_ps(x),
            );
            let r = if FUSE {
                _mm_fmadd_ps(wv, xv, av)
            } else {
                _mm_add_ps(av, _mm_mul_ps(wv, xv))
            };
            _mm_storeu_ps(acc.as_mut_ptr(), r);
        }
    }
    #[cfg(not(any(
        all(target_arch = "aarch64", target_feature = "neon"),
        target_arch = "x86_64"
    )))]
    {
        for (a, w) in acc.iter_mut().zip(w) {
            *a = step::<FUSE>(*a, *w, x);
        }
    }
}

/// One input's contribution to every output channel of one frame:
/// `acc[o] = acc[o] + w[o] * x` for each `o`, fused by `FUSE`, four lanes at
/// a time with the odd lanes one at a time.
#[inline(always)]
pub(crate) fn macc_row<const OUT: usize, const FUSE: bool>(
    acc: &mut [f32; OUT],
    w: &[f32; OUT],
    x: f32,
) {
    let (a4, ar) = acc.as_chunks_mut::<4>();
    let (w4, wr) = w.as_chunks::<4>();
    for (a, w) in a4.iter_mut().zip(w4) {
        lane4::<FUSE>(a, w, x);
    }
    for (a, w) in ar.iter_mut().zip(wr) {
        *a = step::<FUSE>(*a, *w, x);
    }
}

/// [`macc_row`] for a row whose length is only known at run time, unfused:
/// `acc[o] = acc[o] + w[o] * x`, two roundings, for every `o`.
///
/// This is the LSTM's gate product, which the reference computes unfused
/// and whose reference vectors are pinned that way.
#[inline(always)]
pub(crate) fn macc_row_unfused(acc: &mut [f32], w: &[f32], x: f32) {
    debug_assert_eq!(acc.len(), w.len());
    let (a4, ar) = acc.as_chunks_mut::<4>();
    let (w4, wr) = w.as_chunks::<4>();
    for (a, w) in a4.iter_mut().zip(w4) {
        lane4::<false>(a, w, x);
    }
    for (a, w) in ar.iter_mut().zip(wr) {
        *a += *w * x;
    }
}

/// Defines `$name` as the machine-appropriate face of `$body::<…, FUSE>`.
///
/// On x86-64 it also defines `$fma`, the body compiled for AVX and FMA, and
/// enters it when [`fused`] reports the CPU has them; otherwise it runs the
/// unfused body. On every other target it runs the body fused, because the
/// fused instruction is in the baseline there.
macro_rules! fused_dispatch {
    (
        $(#[$meta:meta])*
        $vis:vis fn $name:ident / $fma:ident $(<$(const $g:ident: usize),+>)?
            ($($arg:ident: $ty:ty),* $(,)?) => $body:ident
    ) => {
        $(#[$meta])*
        $vis fn $name$(<$(const $g: usize),+>)?($($arg: $ty),*) {
            #[cfg(target_arch = "x86_64")]
            if fused() {
                // SAFETY: `fused()` reported the CPU has FMA and AVX.
                return unsafe { $fma$(::<$($g),+>)?($($arg),*) };
            }
            $body::<$($($g,)+)? { !cfg!(target_arch = "x86_64") }>($($arg),*)
        }

        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "avx,fma")]
        unsafe fn $fma$(<$(const $g: usize),+>)?($($arg: $ty),*) {
            $body::<$($($g,)+)? true>($($arg),*)
        }
    };
}

/// The `(OUT, IN)` pairs that get a monomorphised kernel.
///
/// Chosen to cover the shapes real captures instantiate: NAM's standard,
/// lite, feather and nano WaveNets use 16 and 8 channels, A2 uses 3/8/16,
/// and 1 appears at every model's input and output and, since a
/// non-parametric capture's conditioning signal is the input itself, as the
/// `IN` of every layer's input mixin. Anything else runs [`matmul_dyn`].
/// `the_bundled_captures_stay_on_the_table` checks the table against the
/// bundled models, so a common shape cannot quietly fall back.
macro_rules! for_each_shape {
    ($mac:ident) => {
        $mac!(1, 1);
        $mac!(1, 2);
        $mac!(1, 3);
        $mac!(1, 4);
        $mac!(1, 6);
        $mac!(1, 8);
        $mac!(1, 16);
        $mac!(2, 1);
        $mac!(2, 2);
        $mac!(3, 1);
        $mac!(3, 3);
        $mac!(3, 6);
        $mac!(4, 1);
        $mac!(4, 4);
        $mac!(6, 3);
        $mac!(6, 6);
        $mac!(8, 1);
        $mac!(8, 4);
        $mac!(8, 8);
        $mac!(8, 16);
        $mac!(16, 1);
        $mac!(16, 8);
        $mac!(16, 16);
        $mac!(16, 32);
        $mac!(32, 16);
    };
}

#[inline(always)]
fn matvec_body<const OUT: usize, const IN: usize, const FUSE: bool>(
    w: &[f32],
    x: &[f32],
    out: &mut [f32],
) {
    debug_assert!(w.len() >= OUT * IN && x.len() >= IN && out.len() >= OUT);
    // Narrowing to a fixed-length view up front. Without it the write-back
    // loop keeps a bounds check per element and stays scalar.
    let out: &mut [f32; OUT] = (&mut out[..OUT]).try_into().expect("checked above");
    let mut acc = [0.0f32; OUT];
    let (cols, _) = w.as_chunks::<OUT>();
    for (i, wcol) in cols.iter().take(IN).enumerate() {
        macc_row::<OUT, FUSE>(&mut acc, wcol, x[i]);
    }
    for o in 0..OUT {
        out[o] += acc[o];
    }
}

/// Largest output-channel count the dynamic kernel accumulates on the stack.
///
/// Comfortably above anything a real capture uses: NAM's widest standard
/// WaveNet is 16 channels, doubled to 32 by gating.
const DYN_ACC: usize = 64;

#[inline(always)]
fn matvec_dyn_body<const FUSE: bool>(
    w: &[f32],
    x: &[f32],
    out: &mut [f32],
    out_ch: usize,
    in_ch: usize,
) {
    debug_assert!(w.len() >= out_ch * in_ch && x.len() >= in_ch && out.len() >= out_ch);
    if out_ch <= DYN_ACC {
        let mut acc = [0.0f32; DYN_ACC];
        let acc = &mut acc[..out_ch];
        for i in 0..in_ch {
            let xi = x[i];
            let wcol = &w[i * out_ch..i * out_ch + out_ch];
            for o in 0..out_ch {
                acc[o] = step::<FUSE>(acc[o], wcol[o], xi);
            }
        }
        for o in 0..out_ch {
            out[o] += acc[o];
        }
    } else {
        for o in 0..out_ch {
            let mut acc = 0.0f32;
            for i in 0..in_ch {
                acc += w[o + i * out_ch] * x[i];
            }
            out[o] += acc;
        }
    }
}

/// Frames a batched kernel accumulates before moving to the next weight
/// column, chosen by the shape's output width.
///
/// The tile is the accumulator, `[[f32; OUT]; TILE]`, which has to stay in
/// registers: NEON has 32 of them, four lanes each. Measured on an Apple M3
/// core, ns per frame at block 64:
///
/// | shape | 2 | 4 | 8 |
/// |---|---|---|---|
/// | 16 × 16 | 5.6 | 5.2 | 6.7 |
/// | 8 × 16 | 3.6 | 3.0 | 3.8 |
/// | 8 × 8 | 1.9 | 1.7 | 1.6 |
/// | 32 × 16 | 10.1 | 13.6 | 18.8 |
///
/// Four is faster than two on every shape up to 16 channels, and eight adds
/// nothing past four except on the narrowest shapes, within noise. At 32
/// channels four frames are the whole register file and the accumulator
/// spills, so that shape keeps two.
pub(crate) const fn tile_for(out_ch: usize) -> usize {
    if out_ch >= 32 { 2 } else { 4 }
}

/// The dynamic kernel's tile: its accumulator is `DYN_ACC` wide whatever the
/// shape, so it keeps the conservative choice.
const DYN_TILE: usize = 2;

fused_dispatch! {
    /// Fixed-shape matrix-matrix product over `n` frames, accumulating into
    /// `out`: `out[o + f * OUT] += sum_i w[o + i * OUT] * x[i + f * x_stride]`.
    ///
    /// `x_stride` is the input's column stride, which is not always `IN` -
    /// the reference regularly hands a `Conv1x1` a `topRows()` block whose
    /// stride is the source's row count. `out` is always tightly packed at
    /// `OUT`.
    #[inline(always)]
    pub(crate) fn matmul / matmul_fma<const OUT: usize, const IN: usize, const TILE: usize>
        (w: &[f32], x: &[f32], x_stride: usize, out: &mut [f32], n: usize) => matmul_body
}

/// The safe face of `matmul_fma`, for the dispatch table.
#[cfg(target_arch = "x86_64")]
fn matmul_x86_fma<const OUT: usize, const IN: usize, const TILE: usize>(
    w: &[f32],
    x: &[f32],
    x_stride: usize,
    out: &mut [f32],
    n: usize,
) {
    // SAFETY: `select_matmul` returns this only after `fused()` reported the
    // CPU has FMA and AVX, and nothing else names it.
    unsafe { matmul_fma::<OUT, IN, TILE>(w, x, x_stride, out, n) }
}

#[inline(always)]
fn matmul_body<const OUT: usize, const IN: usize, const TILE: usize, const FUSE: bool>(
    w: &[f32],
    x: &[f32],
    x_stride: usize,
    out: &mut [f32],
    n: usize,
) {
    debug_assert!(w.len() >= OUT * IN);
    debug_assert!(n == 0 || x.len() >= (n - 1) * x_stride + IN);
    debug_assert!(out.len() >= n * OUT);

    let (cols, _) = w.as_chunks::<OUT>();
    let mut f = 0;
    while f + TILE <= n {
        // One accumulator per frame in the tile, each indexed by output
        // channel, the dimension `matvec` vectorises over, for the same
        // reason: distinct output channels are independent sums.
        let mut acc = [[0.0f32; OUT]; TILE];
        // The tile's input columns as fixed-length views, checked once here.
        // Indexing `x[(f + t) * x_stride + i]` in the loop below costs a
        // bounds check per element that the compiler cannot elide, because
        // `x_stride` is a runtime value.
        let xt: [&[f32; IN]; TILE] = std::array::from_fn(|t| {
            let start = (f + t) * x_stride;
            (&x[start..start + IN]).try_into().expect("checked above")
        });
        for (i, wcol) in cols.iter().take(IN).enumerate() {
            // Loaded once, then used TILE times. This is the whole point.
            for (t, a) in acc.iter_mut().enumerate() {
                macc_row::<OUT, FUSE>(a, wcol, xt[t][i]);
            }
        }
        for (t, a) in acc.iter().enumerate() {
            let o: &mut [f32; OUT] = (&mut out[(f + t) * OUT..(f + t + 1) * OUT])
                .try_into()
                .expect("checked above");
            for k in 0..OUT {
                o[k] += a[k];
            }
        }
        f += TILE;
    }
    // Tail: fewer than TILE frames left, so fall back to the per-frame kernel
    // rather than carry a partial-tile branch through the hot loop.
    while f < n {
        matvec_body::<OUT, IN, FUSE>(w, &x[f * x_stride..], &mut out[f * OUT..]);
        f += 1;
    }
}

fused_dispatch! {
    /// Runtime-shape counterpart to [`matmul`], for shapes outside the table.
    pub(crate) fn matmul_dyn / matmul_dyn_fma
        (w: &[f32], x: &[f32], x_stride: usize, out: &mut [f32], out_ch: usize, in_ch: usize, n: usize)
        => matmul_dyn_body
}

#[inline(always)]
fn matmul_dyn_body<const FUSE: bool>(
    w: &[f32],
    x: &[f32],
    x_stride: usize,
    out: &mut [f32],
    out_ch: usize,
    in_ch: usize,
    n: usize,
) {
    debug_assert!(w.len() >= out_ch * in_ch);
    debug_assert!(out.len() >= n * out_ch);

    if out_ch > DYN_ACC {
        // Too wide to tile without spilling; the per-frame kernel already has
        // a sensible shape for this case.
        for f in 0..n {
            matvec_dyn_body::<FUSE>(w, &x[f * x_stride..], &mut out[f * out_ch..], out_ch, in_ch);
        }
        return;
    }

    let mut f = 0;
    while f + DYN_TILE <= n {
        let mut acc = [[0.0f32; DYN_ACC]; DYN_TILE];
        for i in 0..in_ch {
            let wcol = &w[i * out_ch..i * out_ch + out_ch];
            for (t, a) in acc.iter_mut().enumerate() {
                let xi = x[(f + t) * x_stride + i];
                for o in 0..out_ch {
                    a[o] = step::<FUSE>(a[o], wcol[o], xi);
                }
            }
        }
        for (t, a) in acc.iter().enumerate() {
            let o = &mut out[(f + t) * out_ch..(f + t + 1) * out_ch];
            for k in 0..out_ch {
                o[k] += a[k];
            }
        }
        f += DYN_TILE;
    }
    while f < n {
        matvec_dyn_body::<FUSE>(w, &x[f * x_stride..], &mut out[f * out_ch..], out_ch, in_ch);
        f += 1;
    }
}

/// A monomorphised [`matmul`], selected once at load time.
pub(crate) type MatMulFn = fn(&[f32], &[f32], usize, &mut [f32], usize);

/// Pick the specialised batched kernel for `(out_ch, in_ch)`, if there is one.
///
/// Resolved once at load time and stored in the layer, so `process()` does no
/// shape dispatch at all.
pub(crate) fn select_matmul(out_ch: usize, in_ch: usize) -> Option<MatMulFn> {
    #[cfg(target_arch = "x86_64")]
    let fma = fused();
    macro_rules! arm {
        ($o:expr, $i:expr) => {
            if out_ch == $o && in_ch == $i {
                #[cfg(target_arch = "x86_64")]
                if fma {
                    return Some(matmul_x86_fma::<$o, $i, { tile_for($o) }>);
                }
                return Some(matmul::<$o, $i, { tile_for($o) }>);
            }
        };
    }
    for_each_shape!(arm);
    None
}

/// The arguments of one batched product, so that a convolution can hand its
/// shape to [`matmul_auto`] without a positional list eight long.
pub(crate) struct Product<'a> {
    /// The column-major `(out_ch, in_ch)` weight matrix.
    pub w: &'a [f32],
    /// The input, `n` columns of `x_stride` floats, the first `in_ch` of each
    /// meaningful.
    pub x: &'a [f32],
    /// Distance in floats between consecutive input columns.
    pub x_stride: usize,
    /// Output channels.
    pub out_ch: usize,
    /// Input channels.
    pub in_ch: usize,
    /// Frames.
    pub n: usize,
}

/// Run the batched kernel chosen at load time, or the dynamic one,
/// accumulating into `out`, which is tightly packed at `out_ch`.
#[inline(always)]
pub(crate) fn matmul_auto(kernel: Option<MatMulFn>, p: Product<'_>, out: &mut [f32]) {
    match kernel {
        Some(f) => f(p.w, p.x, p.x_stride, out, p.n),
        None => matmul_dyn(p.w, p.x, p.x_stride, out, p.out_ch, p.in_ch, p.n),
    }
}

fused_dispatch! {
    /// Batched depthwise accumulate: `out[c + f * ch] += w[c] * x[c + f * x_stride]`.
    ///
    /// The weight vector is only `channels` long, so there is nothing to
    /// tile: the whole of `w` stays resident across the frame loop on its own.
    pub(crate) fn depthwise_accum / depthwise_accum_fma
        (w: &[f32], x: &[f32], x_stride: usize, out: &mut [f32], channels: usize, n: usize)
        => depthwise_accum_body
}

#[inline(always)]
fn depthwise_accum_body<const FUSE: bool>(
    w: &[f32],
    x: &[f32],
    x_stride: usize,
    out: &mut [f32],
    channels: usize,
    n: usize,
) {
    for f in 0..n {
        let x = &x[f * x_stride..f * x_stride + channels];
        let o = &mut out[f * channels..(f + 1) * channels];
        for c in 0..channels {
            o[c] = step::<FUSE>(o[c], w[c], x[c]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::splitmix_stream as rng_stream;

    // The per-frame kernels: the bodies the batched kernels finish their
    // tails with, exposed whole as oracles for the tests below.
    fused_dispatch! {
        /// Fixed-shape matrix-vector product, accumulating into `out`:
        /// `out[o] += sum_i w[o + i * OUT] * x[i]`.
        #[inline(always)]
        fn matvec / matvec_fma<const OUT: usize, const IN: usize>
            (w: &[f32], x: &[f32], out: &mut [f32]) => matvec_body
    }

    fused_dispatch! {
        /// Runtime-shape matrix-vector product, accumulating into `out`.
        ///
        /// Identical arithmetic to [`matvec`], same order and same
        /// association, with runtime bounds. The accumulator block is what makes the loop
        /// order possible, and the loop order is what makes it fast: `i`-outer
        /// walks one contiguous column of `w` per input channel, where `o`-outer
        /// would read `w` with stride `out_ch`, several times slower at 16×16.
        /// The `o`-outer form is kept only for shapes too wide for the stack
        /// block; it cannot accumulate straight into `out`, which already holds
        /// earlier kernel taps, without re-associating the sum.
        #[inline]
        fn matvec_dyn / matvec_dyn_fma
            (w: &[f32], x: &[f32], out: &mut [f32], out_ch: usize, in_ch: usize) => matvec_dyn_body
    }

    /// One step the way this machine does it, for the scalar oracles below.
    fn step_here(acc: f32, a: f32, b: f32) -> f32 {
        if fused() {
            macc(acc, a, b)
        } else {
            acc + a * b
        }
    }

    /// The batched kernels exist only to change *when* weights are loaded,
    /// not what is computed. Tiling over frames cannot reorder an addition,
    /// since distinct frames are independent output columns, so batched and
    /// per-frame must agree bit for bit, not merely to within a tolerance.
    ///
    /// Exercised at frame counts either side of every tile so the tiled
    /// body, the scalar tail and the empty case are all covered, and with a
    /// column stride wider than `in_ch`, which is what a `topRows()`-style
    /// view hands a `Conv1x1`. The per-frame oracle is [`matvec_dyn`], which
    /// `specialised_and_dynamic_kernels_agree_bit_for_bit` ties to
    /// [`matvec`] and to the definition.
    #[test]
    fn batched_kernels_are_bit_identical_to_the_per_frame_ones() {
        let shapes = [
            (1usize, 1usize),
            (1, 8),
            (3, 3),
            (8, 8),
            (8, 16),
            (16, 1),
            (16, 16),
            (16, 8),
        ];
        for (out_ch, in_ch) in shapes {
            for n in [0usize, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 33, 64] {
                for extra_stride in [0usize, 5] {
                    let stride = in_ch + extra_stride;
                    let tag = (out_ch * 131 + in_ch * 17 + n * 7 + stride) as u64;
                    let w = rng_stream(0xBA7C4E5 ^ tag, out_ch * in_ch);
                    let x = rng_stream(0xC0FFEE ^ tag, n.max(1) * stride + in_ch);
                    // Seed both outputs identically and non-zero, so an
                    // accumulate that silently became an assign would show.
                    let seed = rng_stream(0xD00D ^ tag, n * out_ch);

                    let mut want = seed.clone();
                    for f in 0..n {
                        matvec_dyn(&w, &x[f * stride..], &mut want[f * out_ch..], out_ch, in_ch);
                    }

                    let mut got = seed.clone();
                    let p = Product {
                        w: &w,
                        x: &x,
                        x_stride: stride,
                        out_ch,
                        in_ch,
                        n,
                    };
                    matmul_auto(select_matmul(out_ch, in_ch), p, &mut got);
                    for (i, (a, b)) in want.iter().zip(&got).enumerate() {
                        assert_eq!(
                            a.to_bits(),
                            b.to_bits(),
                            "{out_ch}x{in_ch}, n={n}, stride={stride}, element {i}"
                        );
                    }

                    // And the dynamic path must match the specialised one too,
                    // for the shapes that have both.
                    let mut dynamic = seed.clone();
                    matmul_dyn(&w, &x, stride, &mut dynamic, out_ch, in_ch, n);
                    for (i, (a, b)) in want.iter().zip(&dynamic).enumerate() {
                        assert_eq!(
                            a.to_bits(),
                            b.to_bits(),
                            "dyn {out_ch}x{in_ch}, n={n}, stride={stride}, element {i}"
                        );
                    }
                }
            }
        }
    }

    /// The definition, as a scalar loop, with this machine's fusion policy:
    /// what is pinned is the accumulation *order*.
    fn naive(w: &[f32], x: &[f32], out_ch: usize, in_ch: usize) -> Vec<f32> {
        (0..out_ch)
            .map(|o| {
                let mut acc = 0.0f32;
                for i in 0..in_ch {
                    acc = step_here(acc, w[o + i * out_ch], x[i]);
                }
                acc
            })
            .collect()
    }

    /// Every monomorphised kernel in the table, against the dynamic one and
    /// against the definition.
    #[test]
    fn specialised_and_dynamic_kernels_agree_bit_for_bit() {
        macro_rules! check {
            ($o:expr, $i:expr) => {{
                let (o, i): (usize, usize) = ($o, $i);
                let w = rng_stream(0xC0FFEE + o as u64 * 131 + i as u64, o * i);
                let x = rng_stream(0xBEEF + o as u64 * 17 + i as u64, i);
                let expected = naive(&w, &x, o, i);

                let mut a = vec![0.0f32; o];
                matvec::<$o, $i>(&w, &x, &mut a);
                let mut b = vec![0.0f32; o];
                matvec_dyn(&w, &x, &mut b, o, i);

                for k in 0..o {
                    assert_eq!(a[k].to_bits(), b[k].to_bits(), "shape {o}x{i} lane {k}");
                    assert_eq!(
                        a[k].to_bits(),
                        expected[k].to_bits(),
                        "shape {o}x{i} lane {k}"
                    );
                }
            }};
        }
        for_each_shape!(check);
    }

    #[test]
    fn depthwise_matches_the_definition() {
        let (channels, stride, n) = (5usize, 7usize, 6usize);
        let w = rng_stream(1, channels);
        let x = rng_stream(2, n * stride);
        let seed = rng_stream(3, n * channels);
        let mut got = seed.clone();
        depthwise_accum(&w, &x, stride, &mut got, channels, n);
        for f in 0..n {
            for c in 0..channels {
                let want = step_here(seed[f * channels + c], w[c], x[f * stride + c]);
                assert_eq!(
                    got[f * channels + c].to_bits(),
                    want.to_bits(),
                    "f={f} c={c}"
                );
            }
        }
    }

    #[test]
    fn every_tabulated_shape_resolves() {
        macro_rules! check {
            ($o:expr, $i:expr) => {
                assert!(
                    select_matmul($o, $i).is_some(),
                    "missing kernel {}x{}",
                    $o,
                    $i
                );
            };
        }
        for_each_shape!(check);
        assert!(select_matmul(7, 13).is_none());
    }

    /// The table is maintained by hand, so this holds it to the shapes the
    /// bundled captures instantiate; a shape that falls off the table here
    /// is a shape real captures pay for.
    #[test]
    fn the_bundled_captures_stay_on_the_table() {
        for (o, i) in [
            (16usize, 1usize),
            (16, 16),
            (8, 1),
            (8, 8),
            (16, 8),
            (8, 16),
            (3, 1),
            (3, 3),
        ] {
            assert!(
                select_matmul(o, i).is_some(),
                "{o}x{i} runs on the dynamic kernel"
            );
        }
    }

    /// Nanoseconds per frame for each tabulated shape at a block of 64,
    /// through the function pointer a loaded model holds, so a compiler
    /// change can be pinned to a shape. Prints; asserts nothing.
    ///
    /// It lives here rather than in `tests/benchmark.rs`, as the other
    /// crates' measurements do, because it measures `select_matmul` and the
    /// kernels behind it, which are internal: moving it would mean widening
    /// the crate's API to relocate a test.
    ///
    /// ```text
    /// cargo test --release --lib kernel_bench -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "measures rather than checks; run with --release --ignored"]
    fn kernel_bench() {
        use std::hint::black_box;
        use std::time::Instant;
        let shapes = [
            (16usize, 16usize),
            (16, 8),
            (8, 16),
            (16, 1),
            (8, 8),
            (8, 1),
            (3, 3),
            (4, 4),
            (6, 6),
            (32, 16),
        ];
        let n = 64;
        for (o, i) in shapes {
            let f = black_box(select_matmul(o, i).expect("tabulated"));
            let w: Vec<f32> = (0..o * i).map(|k| (k as f32 * 0.37).sin()).collect();
            let x: Vec<f32> = (0..n * i).map(|k| (k as f32 * 0.11).cos()).collect();
            let mut out = vec![0.0f32; n * o];
            for _ in 0..2000 {
                f(&w, &x, i, &mut out, n);
            }
            let reps = 200_000;
            let t = Instant::now();
            for _ in 0..reps {
                f(&w, black_box(&x), i, &mut out, n);
            }
            let ns = t.elapsed().as_nanos() as f64 / (reps as f64 * n as f64);
            println!(
                "({o:2},{i:2}) {ns:7.2} ns/frame  {:5.1} fma/ns  checksum {:.3}",
                (o * i) as f64 / ns,
                out.iter().sum::<f32>()
            );
        }
    }
}
