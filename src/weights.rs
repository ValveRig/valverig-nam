//! Cursor over the flat `weights` array of a `.nam` file.
//!
//! A `.nam` file stores every parameter of the model as one flat list of
//! floats. Which parameter each float belongs to is defined entirely by the
//! order in which the model's constructors consume them: there are no names,
//! shapes or offsets in the file. Getting that order wrong produces a model
//! that loads cleanly and sounds like noise, so the consumption order is the
//! single most load-bearing detail of the format.
//!
//! [`WeightReader`] makes running off the end an error rather than a panic,
//! [`WeightReader::check`] lets a constructor refuse a shape before it
//! allocates storage the file cannot fill, and [`WeightReader::finish`]
//! enforces the reference's rule that a file must contain *exactly* as many
//! weights as the architecture consumes.

use crate::error::{Error, Result};

/// Sequential reader over a `.nam` weight array.
pub(crate) struct WeightReader<'a> {
    data: &'a [f32],
    pos: usize,
}

impl<'a> WeightReader<'a> {
    /// Start at the beginning of `data`.
    pub(crate) fn new(data: &'a [f32]) -> Self {
        Self { data, pos: 0 }
    }

    /// Consume one float.
    ///
    /// Deliberately named `next` despite the resemblance to `Iterator::next`:
    /// the whole module exists to mirror the reference's `*(weights++)`
    /// idiom, and reading the two side by side is the point.
    #[allow(clippy::should_implement_trait)]
    #[inline]
    pub(crate) fn next(&mut self) -> Result<f32> {
        let v = self.data.get(self.pos).copied().ok_or(Error::WeightCount {
            expected: self.pos + 1,
            found: self.data.len(),
        })?;
        self.pos += 1;
        Ok(v)
    }

    /// Consume `n` floats into `dst`, in order.
    #[inline]
    pub(crate) fn fill(&mut self, dst: &mut [f32]) -> Result<()> {
        for v in dst.iter_mut() {
            *v = self.next()?;
        }
        Ok(())
    }

    /// How many floats remain.
    pub(crate) fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }

    /// Fail unless at least `needed` floats remain.
    ///
    /// A constructor calls this before allocating storage for the weights it
    /// is about to read, so that a file which lies about its shape is
    /// refused with [`Error::WeightCount`] rather than sizing an allocation
    /// the process cannot survive. `expected` in the error is the count the
    /// architecture would have reached, not the count it did.
    pub(crate) fn check(&self, needed: usize) -> Result<()> {
        if needed > self.remaining() {
            return Err(Error::WeightCount {
                expected: self.pos.saturating_add(needed),
                found: self.data.len(),
            });
        }
        Ok(())
    }

    /// Assert the array is exactly exhausted.
    ///
    /// The reference raises "Weight mismatch" in both directions, too few and
    /// too many, and so does this.
    pub(crate) fn finish(self) -> Result<()> {
        if self.pos != self.data.len() {
            return Err(Error::WeightCount {
                expected: self.pos,
                found: self.data.len(),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reports_shortfall_and_surplus() {
        let d = [1.0f32, 2.0];
        let mut r = WeightReader::new(&d);
        assert_eq!(r.next().unwrap(), 1.0);
        assert!(r.finish().is_err(), "one weight left over must fail");

        let mut r = WeightReader::new(&d);
        let mut buf = [0.0f32; 3];
        assert!(r.fill(&mut buf).is_err(), "reading past the end must fail");
    }

    #[test]
    fn check_refuses_a_shape_the_file_cannot_fill() {
        let d = [1.0f32, 2.0, 3.0];
        let mut r = WeightReader::new(&d);
        r.next().unwrap();
        assert!(r.check(2).is_ok());
        match r.check(3) {
            Err(Error::WeightCount { expected, found }) => assert_eq!((expected, found), (4, 3)),
            other => panic!("{other:?}"),
        }
        assert!(r.check(usize::MAX).is_err(), "must not overflow");
    }
}
