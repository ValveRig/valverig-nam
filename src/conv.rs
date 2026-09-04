//! `Conv1D` (dilated, grouped) and `Conv1x1` (pointwise, grouped).
//!
//! Both mirror `NAM/conv1d.cpp` and the `Conv1x1` section of `NAM/dsp.cpp`,
//! including the exact order in which floats are drawn from the `.nam` weight
//! array and the exact order in which products are accumulated.
//!
//! # Accumulation order
//!
//! The reference computes a dilated convolution as one Eigen matrix product
//! per kernel tap, accumulated into a zeroed output:
//!
//! ```text
//! out.leftCols(n).setZero();
//! for (k = 0; k < K; k++)
//!     out.leftCols(n).noalias() += weight[k] * ring.Read(n, dilation * (K - 1 - k));
//! out.leftCols(n).colwise() += bias;
//! ```
//!
//! With `+=`, Eigen evaluates each tap's contribution into a temporary and
//! then adds it, so the per-element order is: sum over input channels
//! *within* a tap, then sum across taps, then the bias. That association is
//! reproduced here exactly, because collapsing it into a single running
//! accumulator over `(k, i)` would give different (equally valid, but
//! different) results.
//!
//! # Weights
//!
//! A convolution reads its weights when it is built, in the order the
//! reference's constructor and `set_weights` visit them, and before it
//! allocates storage for them it checks that the file has that many left.
//! So a file whose shapes describe more weights than it carries fails with
//! [`Error::WeightCount`] at the first convolution that cannot be filled,
//! rather than sizing an allocation for a shape that was never real.

use crate::buffer::Buf;
use crate::error::{Error, Result};
use crate::history::{Arena, History};
use crate::kernels::{MatMulFn, Product, depthwise_accum, matmul_auto, select_matmul};
use crate::weights::WeightReader;

/// A borrowed column-major view whose columns may be strided.
///
/// The reference regularly passes `_z.topRows(bottleneck)` into a `Conv1x1`:
/// a block whose column stride is the *source* row count, not its own. This
/// carries that distinction rather than forcing a copy.
#[derive(Debug, Clone, Copy)]
pub(crate) struct View<'a> {
    data: &'a [f32],
    rows: usize,
    stride: usize,
}

impl<'a> View<'a> {
    /// A view over a whole buffer.
    #[inline]
    pub(crate) fn full(b: &'a Buf) -> Self {
        Self {
            data: b.data(),
            rows: b.rows(),
            stride: b.rows(),
        }
    }

    /// A view over the top `rows` channels of a buffer.
    #[inline]
    pub(crate) fn top_rows(b: &'a Buf, rows: usize) -> Self {
        debug_assert!(rows <= b.rows());
        Self {
            data: b.data(),
            rows,
            stride: b.rows(),
        }
    }

    /// Channels in the view.
    #[inline]
    pub(crate) fn rows(&self) -> usize {
        self.rows
    }

    /// One column.
    #[inline]
    pub(crate) fn col(&self, c: usize) -> &'a [f32] {
        let s = c * self.stride;
        &self.data[s..s + self.rows]
    }
}

/// How the weights of a grouped convolution are stored: one weight per
/// channel per tap when every group is one channel wide, a full matrix per
/// tap otherwise.
///
/// The reference stores a grouped convolution whose groups equal both
/// channel counts as one weight per channel per tap.
fn is_depthwise(in_ch: usize, out_ch: usize, groups: usize) -> bool {
    groups == in_ch && in_ch == out_ch
}

fn check_groups(in_ch: usize, out_ch: usize, groups: usize) -> Result<()> {
    if groups == 0 {
        return Err(Error::Config("groups must be positive".into()));
    }
    if !in_ch.is_multiple_of(groups) {
        return Err(Error::Config(format!(
            "in_channels ({in_ch}) must be divisible by numGroups ({groups})"
        )));
    }
    if !out_ch.is_multiple_of(groups) {
        return Err(Error::Config(format!(
            "out_channels ({out_ch}) must be divisible by numGroups ({groups})"
        )));
    }
    Ok(())
}

/// Floats a convolution of this shape consumes from the weight array, or an
/// error when the product does not fit in `usize`.
fn weight_count(
    in_ch: usize,
    out_ch: usize,
    groups: usize,
    kernel_size: usize,
    bias: bool,
) -> Result<usize> {
    let per_tap = if is_depthwise(in_ch, out_ch, groups) {
        Some(in_ch)
    } else {
        (out_ch / groups)
            .checked_mul(in_ch / groups)
            .and_then(|n| n.checked_mul(groups))
    };
    per_tap
        .and_then(|n| n.checked_mul(kernel_size))
        .and_then(|n| n.checked_add(if bias { out_ch } else { 0 }))
        .ok_or_else(|| {
            Error::Config(format!(
                "a {out_ch}x{in_ch} convolution with kernel size {kernel_size} has more weights than can be counted"
            ))
        })
}

/// The `(row, col)` positions of a grouped weight matrix in the order the
/// reference fills them, *"Crazy ordering because that's how it gets
/// flattened."* For each group, for each output channel in the group, for
/// each input channel in the group.
fn grouped_positions(
    out_ch: usize,
    in_ch: usize,
    groups: usize,
) -> impl Iterator<Item = (usize, usize)> {
    let opg = out_ch / groups;
    let ipg = in_ch / groups;
    (0..groups).flat_map(move |g| {
        (0..opg).flat_map(move |i| (0..ipg).map(move |j| (g * opg + i, g * ipg + j)))
    })
}

/// `out.leftCols(n).colwise() += bias`, on a tightly packed `(out_ch, n)`
/// block.
#[inline]
fn add_bias(out: &mut [f32], bias: &[f32]) {
    if bias.is_empty() {
        return;
    }
    for frame in out.chunks_exact_mut(bias.len()) {
        for (v, b) in frame.iter_mut().zip(bias) {
            *v += *b;
        }
    }
}

/// A 1D dilated, optionally grouped convolution with its own input history.
#[derive(Debug, Clone)]
pub(crate) struct Conv1D {
    in_ch: usize,
    out_ch: usize,
    kernel_size: usize,
    dilation: usize,
    groups: usize,
    /// One weight per channel per tap rather than a matrix; see
    /// [`is_depthwise`].
    depthwise: bool,
    /// `weight[k]`, column-major `(out_ch, in_ch)`. Empty when depthwise.
    weight: Vec<Vec<f32>>,
    /// `depthwise_weight[k]`, one float per channel. Empty when not depthwise.
    depthwise_weight: Vec<Vec<f32>>,
    /// One per output channel; empty when the convolution has no bias.
    bias: Vec<f32>,
    /// The batched kernel for this shape, resolved once at load time so
    /// `process()` does no shape dispatch. `None` falls back to `matmul_dyn`.
    kernel: Option<MatMulFn>,
    history: Option<History>,
    cached_prewarm_state: Vec<f32>,
    has_cached_prewarm_state: bool,
    /// `(out_ch, max_buffer)`: the convolution's result for the current block.
    pub(crate) output: Buf,
}

impl Conv1D {
    /// Build with the given shape and read the weights from `r`: the taps
    /// in the reference's grouped order, then the bias if `bias` is set.
    ///
    /// Fails with [`Error::Config`] when the groups do not divide the
    /// channel counts and [`Error::WeightCount`] when `r` cannot supply the
    /// shape.
    pub(crate) fn new(
        in_ch: usize,
        out_ch: usize,
        kernel_size: usize,
        bias: bool,
        dilation: usize,
        groups: usize,
        r: &mut WeightReader<'_>,
    ) -> Result<Self> {
        check_groups(in_ch, out_ch, groups)?;
        r.check(weight_count(in_ch, out_ch, groups, kernel_size, bias)?)?;
        let depthwise = is_depthwise(in_ch, out_ch, groups);
        let mut weight = Vec::new();
        let mut depthwise_weight = Vec::new();
        // In both layouts the tap index varies fastest.
        if depthwise {
            depthwise_weight = vec![vec![0.0; in_ch]; kernel_size];
            for c in 0..in_ch {
                for tap in depthwise_weight.iter_mut() {
                    tap[c] = r.next()?;
                }
            }
        } else {
            weight = vec![vec![0.0; out_ch * in_ch]; kernel_size];
            for (row, col) in grouped_positions(out_ch, in_ch, groups) {
                for tap in weight.iter_mut() {
                    tap[row + col * out_ch] = r.next()?;
                }
            }
        }
        let mut b = vec![0.0; if bias { out_ch } else { 0 }];
        r.fill(&mut b)?;
        Ok(Self {
            in_ch,
            out_ch,
            kernel_size,
            dilation,
            groups,
            depthwise,
            weight,
            depthwise_weight,
            bias: b,
            kernel: if depthwise {
                None
            } else {
                select_matmul(out_ch, in_ch)
            },
            history: None,
            cached_prewarm_state: vec![0.0; in_ch],
            has_cached_prewarm_state: false,
            output: Buf::new(),
        })
    }

    /// Input channels.
    pub(crate) fn in_channels(&self) -> usize {
        self.in_ch
    }

    /// Output channels.
    pub(crate) fn out_channels(&self) -> usize {
        self.out_ch
    }

    /// Kernel size.
    pub(crate) fn kernel_size(&self) -> usize {
        self.kernel_size
    }

    /// Dilation factor.
    pub(crate) fn dilation(&self) -> usize {
        self.dilation
    }

    /// The weights per tap, column-major `(out_ch, in_ch)`; empty when
    /// depthwise.
    pub(crate) fn weight_taps(&self) -> &[Vec<f32>] {
        &self.weight
    }

    /// The bias, one per output channel; empty when the layer has none.
    pub(crate) fn bias(&self) -> &[f32] {
        &self.bias
    }

    /// Whether the convolution is plain: one group and a full matrix, which
    /// a 1-channel convolution is not, since the reference stores that as
    /// depthwise.
    pub(crate) fn is_plain(&self) -> bool {
        self.groups == 1 && !self.depthwise
    }

    /// How many frames of history this convolution reaches back:
    /// `(kernel_size - 1) * dilation`, or an error when that does not fit.
    pub(crate) fn receptive_field(&self) -> Result<usize> {
        self.kernel_size
            .saturating_sub(1)
            .checked_mul(self.dilation)
            .ok_or_else(|| {
                Error::Config(format!(
                    "a kernel of {} at dilation {} reaches further back than can be counted",
                    self.kernel_size, self.dilation
                ))
            })
    }

    /// Floats of input history this convolution keeps:
    /// `in_channels × receptive_field`, or an error when that does not fit.
    pub(crate) fn history_floats(&self) -> Result<usize> {
        self.in_ch
            .checked_mul(self.receptive_field()?)
            .ok_or_else(|| {
                Error::Config(format!(
                    "{} channels of {}-frame history is more than can be counted",
                    self.in_ch, self.kernel_size
                ))
            })
    }

    /// Reserve history and output storage for blocks of up to `max_buffer`
    /// frames. Allocating; never call it from `process`.
    ///
    /// The receptive field has been checked by [`Conv1D::receptive_field`]
    /// before this is reached.
    pub(crate) fn set_max_buffer_size(&mut self, arena: &mut Arena, max_buffer: usize) {
        let lookback = self.kernel_size.saturating_sub(1) * self.dilation;
        self.history = Some(History::reserve(arena, self.in_ch, lookback, max_buffer));
        self.output.resize(self.out_ch, max_buffer);
        // The cached prewarm column is one frame wide and independent of the
        // block size, so resizing does not invalidate it. The reference keeps
        // it too, which is what lets a second Reset() skip the prewarm run.
    }

    /// True when a steady-state history has been cached by
    /// [`Conv1D::cache_prewarm_state`].
    pub(crate) fn has_cached_prewarm_state(&self) -> bool {
        self.has_cached_prewarm_state
    }

    /// Snapshot the last written input column as the steady prewarm state.
    pub(crate) fn cache_prewarm_state(&mut self, arena: &Arena) {
        if let Some(h) = &self.history {
            h.cache_last_written(arena, &mut self.cached_prewarm_state);
            self.has_cached_prewarm_state = true;
        }
    }

    /// Refill the history from the cached steady state, skipping a prewarm run.
    pub(crate) fn prewarm_from_cache(&mut self, arena: &mut Arena) {
        debug_assert!(self.has_cached_prewarm_state);
        if let Some(h) = &mut self.history {
            h.fill_with_sample(arena, &self.cached_prewarm_state);
        }
    }

    /// Process `n` frames of `input` into [`Conv1D::output`].
    ///
    /// Panics if [`Conv1D::set_max_buffer_size`] has not run or `n` exceeds
    /// the size it was given.
    pub(crate) fn process(&mut self, arena: &mut Arena, input: &Buf, n: usize) {
        debug_assert_eq!(input.rows(), self.in_ch);
        let hist = self
            .history
            .as_mut()
            .expect("Conv1D::process before set_max_buffer_size");
        hist.write(arena, input.data(), n);

        let out = self.output.left_mut(n);
        out.fill(0.0);

        let k_last = self.kernel_size.saturating_sub(1);
        for k in 0..self.kernel_size {
            let lookback = self.dilation * (k_last - k);
            let runs = hist.read_runs(n, lookback);
            let mut frame = 0usize;
            for (col, count) in runs {
                if count == 0 {
                    continue;
                }
                let src = hist.run(arena, col, count);
                // The run is contiguous and column-major, so its stride is
                // just `in_ch`. Batching over it is what keeps each weight
                // column resident across the tile's frames instead of being
                // re-streamed per frame; see `kernels::matmul`.
                let o = &mut out[frame * self.out_ch..(frame + count) * self.out_ch];
                if self.depthwise {
                    depthwise_accum(
                        &self.depthwise_weight[k],
                        src,
                        self.in_ch,
                        o,
                        self.in_ch,
                        count,
                    );
                } else {
                    let p = Product {
                        w: &self.weight[k],
                        x: src,
                        x_stride: self.in_ch,
                        out_ch: self.out_ch,
                        in_ch: self.in_ch,
                        n: count,
                    };
                    matmul_auto(self.kernel, p, o);
                }
                frame += count;
            }
        }

        add_bias(out, &self.bias);
        hist.advance(n);
    }
}

/// A pointwise (1x1) convolution: a per-frame linear layer.
#[derive(Debug, Clone)]
pub(crate) struct Conv1x1 {
    in_ch: usize,
    out_ch: usize,
    groups: usize,
    /// One weight per channel rather than a matrix; see [`is_depthwise`].
    depthwise: bool,
    /// Column-major `(out_ch, in_ch)`. Empty when depthwise.
    weight: Vec<f32>,
    /// One float per channel. Empty when not depthwise.
    depthwise_weight: Vec<f32>,
    /// One per output channel; empty when the convolution has no bias.
    bias: Vec<f32>,
    /// The batched kernel for this shape, resolved once at load time so
    /// `process()` does no shape dispatch. `None` falls back to `matmul_dyn`.
    kernel: Option<MatMulFn>,
    /// `(out_ch, max_buffer)`.
    pub(crate) output: Buf,
}

impl Conv1x1 {
    /// Build with the given shape and read the weights from `r`: the matrix
    /// in the reference's grouped order, then the bias if `bias` is set.
    ///
    /// Fails as [`Conv1D::new`] does.
    pub(crate) fn new(
        in_ch: usize,
        out_ch: usize,
        bias: bool,
        groups: usize,
        r: &mut WeightReader<'_>,
    ) -> Result<Self> {
        check_groups(in_ch, out_ch, groups)?;
        r.check(weight_count(in_ch, out_ch, groups, 1, bias)?)?;
        let depthwise = is_depthwise(in_ch, out_ch, groups);
        let mut weight = Vec::new();
        let mut depthwise_weight = Vec::new();
        if depthwise {
            depthwise_weight = vec![0.0; in_ch];
            r.fill(&mut depthwise_weight)?;
        } else {
            weight = vec![0.0; out_ch * in_ch];
            for (row, col) in grouped_positions(out_ch, in_ch, groups) {
                weight[row + col * out_ch] = r.next()?;
            }
        }
        let mut b = vec![0.0; if bias { out_ch } else { 0 }];
        r.fill(&mut b)?;
        Ok(Self {
            in_ch,
            out_ch,
            groups,
            depthwise,
            weight,
            depthwise_weight,
            bias: b,
            kernel: if depthwise {
                None
            } else {
                select_matmul(out_ch, in_ch)
            },
            output: Buf::new(),
        })
    }

    /// Input channels.
    pub(crate) fn in_channels(&self) -> usize {
        self.in_ch
    }

    /// Output channels.
    pub(crate) fn out_channels(&self) -> usize {
        self.out_ch
    }

    /// The weights, column-major `(out_ch, in_ch)`; empty when depthwise.
    pub(crate) fn weight(&self) -> &[f32] {
        &self.weight
    }

    /// The bias, one per output channel; empty when the layer has none.
    pub(crate) fn bias(&self) -> &[f32] {
        &self.bias
    }

    /// Whether the convolution is plain: one group and a full matrix, which
    /// a 1-channel convolution is not, since the reference stores that as
    /// depthwise.
    pub(crate) fn is_plain(&self) -> bool {
        self.groups == 1 && !self.depthwise
    }

    /// Reserve output storage for blocks of up to `max_buffer` frames.
    /// Allocating.
    pub(crate) fn set_max_buffer_size(&mut self, max_buffer: usize) {
        self.output.resize(self.out_ch, max_buffer);
    }

    /// Process `n` frames of a whole buffer into [`Conv1x1::output`].
    pub(crate) fn process(&mut self, input: &Buf, n: usize) {
        self.process_view(View::full(input), n);
    }

    /// Process `n` frames of a possibly strided view into
    /// [`Conv1x1::output`].
    pub(crate) fn process_view(&mut self, input: View<'_>, n: usize) {
        debug_assert_eq!(input.rows(), self.in_ch);
        let out = self.output.left_mut(n);
        if self.depthwise {
            for f in 0..n {
                let x = input.col(f);
                let o = &mut out[f * self.out_ch..(f + 1) * self.out_ch];
                for c in 0..self.in_ch {
                    o[c] = self.depthwise_weight[c] * x[c];
                }
            }
        } else {
            // The reference assigns (`=`) rather than accumulating; zeroing
            // first and accumulating is bit-identical because `0.0 + v == v`.
            out.fill(0.0);
            // `input` may be a `topRows()`-style block, so its column stride
            // is the source's row count rather than `in_ch`.
            let p = Product {
                w: &self.weight,
                x: input.data,
                x_stride: input.stride,
                out_ch: self.out_ch,
                in_ch: self.in_ch,
                n,
            };
            matmul_auto(self.kernel, p, out);
        }
        add_bias(out, &self.bias);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seq_weights(n: usize) -> Vec<f32> {
        (0..n).map(|i| ((i % 17) as f32 - 8.0) * 0.125).collect()
    }

    #[test]
    fn conv1d_matches_direct_definition() {
        // out[o, t] = bias[o] + sum_k sum_i w[k][o, i] * x[i, t - (K-1-k)*dilation]
        let (in_ch, out_ch, k, dil) = (3usize, 2usize, 3usize, 2usize);
        let w = seq_weights(weight_count(in_ch, out_ch, 1, k, true).unwrap());
        let mut r = WeightReader::new(&w);
        let mut c = Conv1D::new(in_ch, out_ch, k, true, dil, 1, &mut r).unwrap();
        r.finish().unwrap();

        let mut arena = Arena::new();
        c.set_max_buffer_size(&mut arena, 8);

        // Feed a single block; history before it is zero.
        let n = 8;
        let mut input = Buf::zeros(in_ch, n);
        for t in 0..n {
            for i in 0..in_ch {
                input.set(i, t, (t * in_ch + i) as f32 * 0.25 - 1.0);
            }
        }
        c.process(&mut arena, &input, n);

        for t in 0..n {
            for o in 0..out_ch {
                let mut acc = 0.0f32;
                for tap in 0..k {
                    let lag = (k - 1 - tap) * dil;
                    let mut inner = 0.0f32;
                    for i in 0..in_ch {
                        let x = if t >= lag { input.at(i, t - lag) } else { 0.0 };
                        inner += c.weight[tap][o + i * out_ch] * x;
                    }
                    acc += inner;
                }
                acc += c.bias[o];
                assert_eq!(
                    c.output.at(o, t).to_bits(),
                    acc.to_bits(),
                    "mismatch at o={o} t={t}"
                );
            }
        }
    }

    #[test]
    fn depthwise_is_selected_when_groups_equal_channels() {
        assert_eq!(weight_count(8, 8, 8, 3, false).unwrap(), 3 * 8);
        assert_eq!(weight_count(8, 8, 4, 3, false).unwrap(), 3 * 2 * 2 * 4);
        let w = seq_weights(24);
        let c = Conv1D::new(8, 8, 3, false, 1, 8, &mut WeightReader::new(&w)).unwrap();
        assert!(c.depthwise && c.weight.is_empty() && c.depthwise_weight.len() == 3);
        let w = seq_weights(48);
        let c = Conv1D::new(8, 8, 3, false, 1, 4, &mut WeightReader::new(&w)).unwrap();
        assert!(!c.depthwise && !c.weight.is_empty() && c.depthwise_weight.is_empty());
        // One channel in, one out, one group: depthwise by the reference's
        // rule, so not "plain".
        let w = seq_weights(3);
        let c = Conv1D::new(1, 1, 3, false, 1, 1, &mut WeightReader::new(&w)).unwrap();
        assert!(c.depthwise && !c.is_plain());
    }

    #[test]
    fn grouped_weights_land_on_the_block_diagonal() {
        let w: Vec<f32> = (1..=8).map(|i| i as f32).collect();
        let c = Conv1x1::new(4, 4, false, 2, &mut WeightReader::new(&w)).unwrap();
        // groups=2 => two 2x2 blocks; off-diagonal blocks stay zero.
        let at = |o: usize, i: usize| c.weight[o + i * 4];
        assert_eq!(at(0, 0), 1.0);
        assert_eq!(at(0, 1), 2.0);
        assert_eq!(at(1, 0), 3.0);
        assert_eq!(at(1, 1), 4.0);
        assert_eq!(at(2, 2), 5.0);
        assert_eq!(at(3, 3), 8.0);
        assert_eq!(at(0, 2), 0.0);
        assert_eq!(at(2, 0), 0.0);
    }

    #[test]
    fn a_shape_the_file_cannot_fill_is_refused_before_allocating() {
        // 16 x 16 x 3 taps = 768 weights; the file has 10.
        let w = seq_weights(10);
        match Conv1D::new(16, 16, 3, false, 1, 1, &mut WeightReader::new(&w)) {
            Err(Error::WeightCount { expected, found }) => assert_eq!((expected, found), (768, 10)),
            other => panic!("{other:?}"),
        }
        // A product that does not fit in usize is a config error, not a wrap.
        assert!(matches!(
            weight_count(usize::MAX / 2, 4, 1, 3, false),
            Err(Error::Config(_))
        ));
    }

    #[test]
    fn groups_must_divide_the_channel_counts() {
        let w = seq_weights(64);
        assert!(matches!(
            Conv1x1::new(4, 4, false, 0, &mut WeightReader::new(&w)),
            Err(Error::Config(_))
        ));
        assert!(matches!(
            Conv1x1::new(4, 4, false, 3, &mut WeightReader::new(&w)),
            Err(Error::Config(_))
        ));
        assert!(matches!(
            Conv1x1::new(6, 4, false, 3, &mut WeightReader::new(&w)),
            Err(Error::Config(_))
        ));
    }
}
