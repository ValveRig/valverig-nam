//! Deterministic generators and comparisons shared by the unit tests.
//!
//! The `film` and `gating` tests pin their output against values produced by
//! a C++ program linked directly against the reference headers, built at the
//! flags `tools/regen-assets.sh` pins. That program seeded itself from the
//! generator defined here, so the two languages walk the same sequence; a
//! test module that drifted to a different generator would silently stop
//! meaning anything. The program itself is not part of this repository, so
//! those pins are evidence, not something to regenerate.

use crate::buffer::Buf;

/// The oracle's generator: a Lehmer/Knuth LCG, integer arithmetic only, so
/// Rust and C++ produce the same sequence bit for bit.
pub(crate) struct Lcg(u32);

impl Lcg {
    /// Seed it.
    pub(crate) fn new(seed: u32) -> Self {
        Self(seed)
    }

    /// Next sample, roughly in [-3.33, 3.33].
    pub(crate) fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let v = ((self.0 >> 8) as i32) % 20_001 - 10_000;
        v as f32 / 3000.0_f32
    }
}

/// Fill a `(rows, cols)` buffer from the generator, in storage order.
pub(crate) fn fill(g: &mut Lcg, rows: usize, cols: usize) -> Buf {
    let mut b = Buf::zeros(rows, cols);
    for v in b.data_mut() {
        *v = g.next();
    }
    b
}

/// Compare against pinned `f32` bit patterns.
///
/// Bits, not `approx_eq`: for a value the arithmetic reproduces exactly, an
/// activation or a reduction whose order is fixed, a one-ULP drift is a real
/// change and a tolerance would hide it. Where the multiply-accumulate is
/// fused and the pinned value was not, use [`assert_close`].
pub(crate) fn assert_bits(got: &[f32], want: &[u32]) {
    assert_eq!(got.len(), want.len(), "length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert_eq!(
            g.to_bits(),
            *w,
            "element {i}: {g} vs {}",
            f32::from_bits(*w)
        );
    }
}

/// Relative tolerance for a value the reference produced with a different
/// rounding schedule than ours.
///
/// `kernels::macc` fuses, so anything downstream of a convolution rounds once
/// per multiply-accumulate where the pinned oracle rounded twice. The gap is a
/// few ULP per accumulation step and grows with the length of the reduction;
/// 1e-5 relative sits about two orders of magnitude above what these small
/// cases actually show, and still fails a wrong *formula*, the thing these
/// tests exist to catch, by orders of magnitude more than that.
pub(crate) const REF_TOL: f32 = 1e-5;

/// Compare against pinned `f32` bit patterns, to within [`REF_TOL`].
///
/// Same oracle values as [`assert_bits`], read as numbers rather than as bits.
/// The scale is per-element, falling back to absolute near zero so that a
/// cancelled sum does not demand impossible precision.
pub(crate) fn assert_close(got: &[f32], want: &[u32]) {
    assert_eq!(got.len(), want.len(), "length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let w = f32::from_bits(*w);
        let scale = g.abs().max(w.abs()).max(1.0);
        assert!(
            (g - w).abs() <= REF_TOL * scale,
            "element {i}: {g} vs {w} (rel {:.2e} > {REF_TOL:.0e})",
            (g - w).abs() / scale
        );
    }
}

/// Compare against pinned `f32` bit patterns to within `max` units in the
/// last place, per element.
///
/// For a value that went through this crate's own `tanh`: the oracle ran
/// libm's, and the two are within a unit of each other, so a pinned bit
/// pattern is right to the last bit or the one beside it. A wrong formula
/// still fails such a test by orders of magnitude, and it is far tighter
/// than [`assert_close`].
pub(crate) fn assert_ulps(got: &[f32], want: &[u32], max: u32) {
    assert_eq!(got.len(), want.len(), "length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let w = f32::from_bits(*w);
        let d = if *g == w {
            0
        } else {
            assert!(g.signum() == w.signum(), "element {i}: {g} vs {w}");
            g.to_bits().abs_diff(w.to_bits())
        };
        assert!(d <= max, "element {i}: {g} vs {w} ({d} ulps > {max})");
    }
}

/// splitmix64, matching `tools/gen_vectors.cpp`, as `f32` in [-1, 1).
///
/// The 24-bit mantissa keeps every value exactly representable, so a test
/// signal survives an `f32`/`f64` round trip unchanged.
pub(crate) fn splitmix_stream(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            ((z >> 40) as u32) as f32 / 8_388_608.0 - 1.0
        })
        .collect()
}
