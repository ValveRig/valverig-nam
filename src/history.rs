//! Input history for dilated convolutions.
//!
//! Every `Conv1D` needs the last `(kernel_size - 1) * dilation` frames of
//! its input in addition to the current block. The C++ reference keeps that
//! in a `RingBuffer` that, when the write pointer runs out of room,
//! bulk-copies `max_lookback` columns back to the start.
//!
//! This does not copy. A dilated convolution never needs the whole lookback
//! contiguous. It needs `num_frames` contiguous at each of `kernel_size`
//! fixed offsets, so the read pointers wrap instead, and each read splits
//! into at most two runs. That is bit-exact against the reference by
//! construction, because a read is a read: the values returned are the same
//! ones, reached by different arithmetic on the index. `tests/reference.rs`
//! holds the whole-model consequence against the reference's own output.
//!
//! The storage is `max_lookback + max_buffer` columns per convolution, 40%
//! less than the reference's `2 * max_lookback + max_buffer`, and no block
//! is more expensive than any other, since none triggers a copy. The copy
//! is not where a model's time goes, since removing it measures within
//! noise, so the design is about storage and worst case, not speed.
//!
//! Per-layer buffers would scatter the working set, so every history in the
//! model is backed by one [`Arena`]: a single contiguous allocation, laid
//! out in execution order.

/// One contiguous allocation holding every convolution's input history.
#[derive(Debug, Clone, Default)]
pub(crate) struct Arena {
    data: Vec<f32>,
}

impl Arena {
    /// An empty arena.
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Drop all reservations. Call before re-reserving for a new buffer size.
    pub(crate) fn clear(&mut self) {
        self.data.clear();
    }

    /// Reserve `len` zeroed floats and return the offset of the reservation.
    pub(crate) fn alloc(&mut self, len: usize) -> usize {
        let off = self.data.len();
        self.data.resize(off + len, 0.0);
        off
    }

    /// Total size in floats.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.data.len()
    }

    #[inline]
    fn slice(&self, off: usize, len: usize) -> &[f32] {
        &self.data[off..off + len]
    }

    #[inline]
    fn slice_mut(&mut self, off: usize, len: usize) -> &mut [f32] {
        &mut self.data[off..off + len]
    }
}

/// A single convolution's input history, backed by a slice of an [`Arena`].
#[derive(Debug, Clone)]
pub(crate) struct History {
    rows: usize,
    /// Number of columns of storage.
    cols: usize,
    /// `(kernel_size - 1) * dilation`.
    max_lookback: usize,
    /// Largest block this history was sized for.
    max_buffer: usize,
    /// Column index of the next write.
    write_pos: usize,
    /// Offset into the arena, in floats.
    offset: usize,
}

impl History {
    /// Reserve storage for `rows` channels, `max_lookback` frames of history
    /// and blocks of up to `max_buffer` frames.
    ///
    /// `max_lookback + max_buffer` columns is enough because nothing is ever
    /// copied: after writing `n` columns there are still
    /// `cols - n >= max_lookback` columns of intact history behind the write
    /// pointer. The reference reserves `2 * max_lookback + max_buffer`, the
    /// doubled lookback being what keeps its rewind copy non-overlapping.
    pub(crate) fn reserve(
        arena: &mut Arena,
        rows: usize,
        max_lookback: usize,
        max_buffer: usize,
    ) -> Self {
        let cols = max_lookback + max_buffer;
        let offset = arena.alloc(rows * cols);
        Self {
            rows,
            cols,
            max_lookback,
            max_buffer,
            write_pos: 0,
            offset,
        }
    }

    /// Write `n` frames from `src` (a column-major `(rows, >= n)` buffer) at
    /// the write pointer, wrapping around the end if needed.
    pub(crate) fn write(&mut self, arena: &mut Arena, src: &[f32], n: usize) {
        debug_assert!(
            n <= self.max_buffer,
            "block of {n} exceeds max buffer {}",
            self.max_buffer
        );
        debug_assert!(src.len() >= self.rows * n);
        let first = (self.cols - self.write_pos).min(n);
        let start = self.offset + self.write_pos * self.rows;
        arena
            .slice_mut(start, self.rows * first)
            .copy_from_slice(&src[..self.rows * first]);
        if first < n {
            let rest = n - first;
            arena
                .slice_mut(self.offset, self.rows * rest)
                .copy_from_slice(&src[self.rows * first..self.rows * n]);
        }
    }

    /// Advance the write pointer by `n`, after the block has been read.
    #[inline]
    pub(crate) fn advance(&mut self, n: usize) {
        self.write_pos += n;
        self.write_pos %= self.cols;
    }

    /// The runs of storage covering `n` frames ending `lookback` frames before
    /// the write pointer.
    ///
    /// Returns `[(first_col, count), (0, count)]`; the second run has
    /// `count == 0` unless the window wraps the end of the storage.
    #[inline]
    pub(crate) fn read_runs(&self, n: usize, lookback: usize) -> [(usize, usize); 2] {
        debug_assert!(lookback <= self.max_lookback);
        debug_assert!(n <= self.max_buffer);
        let start = (self.write_pos + self.cols - lookback) % self.cols;
        let first = (self.cols - start).min(n);
        [(start, first), (0, n - first)]
    }

    /// Borrow `count` columns starting at storage column `col`.
    #[inline]
    pub(crate) fn run<'a>(&self, arena: &'a Arena, col: usize, count: usize) -> &'a [f32] {
        arena.slice(self.offset + col * self.rows, self.rows * count)
    }

    /// Copy the most recently written column into `dst`.
    ///
    /// Used by the prewarm cache: after prewarming on silence the model is in
    /// a steady state, and every convolution's history is a constant column.
    pub(crate) fn cache_last_written(&self, arena: &Arena, dst: &mut [f32]) {
        debug_assert_eq!(dst.len(), self.rows);
        let col = (self.write_pos + self.cols - 1) % self.cols;
        dst.copy_from_slice(self.run(arena, col, 1));
    }

    /// Fill every column with `sample` and restore the initial write position.
    ///
    /// The reference's `RingBuffer::FillWithSample`. Since every column ends up
    /// identical, the write position only has to be somewhere legal, which
    /// is why this can restart at 0 where the reference restarts at
    /// `max_lookback` and still agree with it.
    pub(crate) fn fill_with_sample(&mut self, arena: &mut Arena, sample: &[f32]) {
        debug_assert_eq!(sample.len(), self.rows);
        let rows = self.rows;
        let base = self.offset;
        for c in 0..self.cols {
            arena
                .slice_mut(base + c * rows, rows)
                .copy_from_slice(sample);
        }
        self.write_pos = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The ring is checked against a naive history that simply keeps every
    /// frame ever written, over ragged block schedules and every lookback.
    ///
    /// The oracle is the definition, not a second ring buffer, because two
    /// rings can agree with each other and both be wrong: the window ending `lookback`
    /// frames before the write pointer is the last `n` frames of everything
    /// written, offset by `lookback`.
    #[test]
    fn reads_match_a_naive_full_history() {
        const ROWS: usize = 3;
        for &(lookback, max_buf) in &[
            (0usize, 8usize),
            (1, 8),
            (7, 8),
            (16, 64),
            (239, 64),
            (5, 1),
            (512, 128),
        ] {
            let mut arena = Arena::new();
            let mut h = History::reserve(&mut arena, ROWS, lookback, max_buf);
            // Every frame ever written, column-major, starting from the zeroed
            // history the ring begins with.
            let mut all = vec![0.0f32; ROWS * (lookback + max_buf)];

            let schedule = [1usize, 7, 3, max_buf, 2, max_buf, 5, 1, max_buf.min(4)];
            let mut counter = 0.0f32;
            for round in 0..40 {
                let n = schedule[round % schedule.len()].clamp(1, max_buf);
                let mut src = vec![0.0f32; ROWS * n];
                for v in src.iter_mut() {
                    counter += 1.0;
                    *v = counter;
                }
                h.write(&mut arena, &src, n);
                all.extend_from_slice(&src);

                for lb in [0, lookback / 2, lookback] {
                    let mut got = Vec::new();
                    for (c, k) in h.read_runs(n, lb) {
                        if k > 0 {
                            got.extend_from_slice(h.run(&arena, c, k));
                        }
                    }
                    // The same window, taken straight from the full history.
                    let end = all.len() / ROWS - lb;
                    let want = &all[(end - n) * ROWS..end * ROWS];
                    assert_eq!(
                        got, want,
                        "lookback={lookback} max_buf={max_buf} round={round} lb={lb}"
                    );
                }
                h.advance(n);
            }
        }
    }

    /// Storage is `max_lookback + max_buffer` columns, not the reference's
    /// `2 * max_lookback + max_buffer`.
    #[test]
    fn storage_is_lookback_plus_buffer() {
        let mut arena = Arena::new();
        let _ = History::reserve(&mut arena, 8, 239, 64);
        assert_eq!(arena.len(), 8 * (239 + 64));
    }

    #[test]
    fn fill_with_sample_is_seen_at_every_lookback() {
        let mut arena = Arena::new();
        let mut h = History::reserve(&mut arena, 2, 10, 4);
        h.fill_with_sample(&mut arena, &[0.25, -0.5]);
        for lb in 0..=10 {
            let runs = h.read_runs(4, lb);
            for (c, k) in runs {
                for col in 0..k {
                    assert_eq!(h.run(&arena, c + col, 1), &[0.25, -0.5]);
                }
            }
        }
    }
}
