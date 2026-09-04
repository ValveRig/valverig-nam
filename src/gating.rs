//! Gated and blended activations for WaveNet layers.
//!
//! A port of `NAM/gating_activations.h`. Both take a `2 * channels`-row
//! block: the top half is the signal, the bottom half controls what happens
//! to it. Gating multiplies the two activated halves; blending uses the
//! bottom half as a per-element mix between the activated and *un*-activated
//! signal.
//!
//! The reference applies each activation to a `(channels, 1)` scratch column
//! rather than to the whole block. That matters for [`Activation::PReLU`],
//! whose flat-buffer path indexes its slope table by position modulo slope
//! count: on a single column, position *is* channel. Reproducing the
//! column-at-a-time shape keeps that indexing correct.
//!
//! The scratch is also observable, not just an implementation detail.
//! Because the reference activates *copies* of the two halves, the bottom
//! half of the caller's block is left exactly as it was, and an
//! implementation that activated the bottom half in place would agree on the
//! result but not on the block. `wavenet::Layer` only reads the top half afterwards, so
//! nothing downstream depends on it; the tests pin it anyway.
//!
//! The reference has one class per mode. They share every field and differ
//! in one line of arithmetic, so that line is the only thing the two gated
//! variants of [`Nonlinearity`] do not share.

use crate::activations::Activation;
use crate::buffer::Buf;
use crate::format::GatingMode;

/// A layer's activation stage: a plain activation over the block, or one of
/// the two gated forms over a doubled block.
#[derive(Debug, Clone)]
pub(crate) enum Nonlinearity {
    /// One activation over the whole block, in place.
    Plain(Activation),
    /// `output = primary(top) * secondary(bottom)`.
    Gated(Gate),
    /// `output = alpha * primary(top) + (1 - alpha) * top`, where
    /// `alpha = secondary(bottom)`.
    Blended(Gate),
}

impl Nonlinearity {
    /// Build for `mode`. `channels` is the layer's bottleneck: the width of
    /// the result, and half the block for the gated modes.
    pub(crate) fn new(
        mode: GatingMode,
        primary: Activation,
        secondary: Activation,
        channels: usize,
    ) -> Self {
        match mode {
            GatingMode::None => Nonlinearity::Plain(primary),
            GatingMode::Gated => Nonlinearity::Gated(Gate::new(primary, secondary, channels)),
            GatingMode::Blended => Nonlinearity::Blended(Gate::new(primary, secondary, channels)),
        }
    }

    /// Apply to the first `n` columns of `z`, in place. For the gated modes
    /// `z` has `2 * channels` rows and on return its top `channels` rows hold
    /// the result.
    ///
    /// The combining expressions are transcribed from the reference rather
    /// than rearranged. Blending is two products, each rounded, then one
    /// sum: `alpha * a + pre - alpha * pre` and friends are algebraically
    /// equal but not bit-equal.
    pub(crate) fn apply(&mut self, z: &mut Buf, n: usize) {
        match self {
            Nonlinearity::Plain(a) => a.apply(z.left_mut(n)),
            Nonlinearity::Gated(g) => g.run(z, n, |a, b, _pre| a * b),
            Nonlinearity::Blended(g) => {
                g.run(z, n, |a, alpha, pre| alpha * a + (1.0 - alpha) * pre)
            }
        }
    }
}

/// The shared half of the two gated modes: the two activations and the
/// per-column scratch they are applied on.
///
/// The reference keeps the scratch as members of the activation object too.
/// It is sized once at construction, since the bottleneck never changes, so
/// nothing here allocates after that.
#[derive(Debug, Clone)]
pub(crate) struct Gate {
    primary: Activation,
    secondary: Activation,
    channels: usize,
    a: Vec<f32>,
    b: Vec<f32>,
}

impl Gate {
    fn new(primary: Activation, secondary: Activation, channels: usize) -> Self {
        Self {
            primary,
            secondary,
            channels,
            a: vec![0.0; channels],
            b: vec![0.0; channels],
        }
    }

    /// For each column: activate copies of both halves, then write
    /// `combine(primary(top), secondary(bottom), top)` over the top half.
    ///
    /// The reference writes into a block that aliases the top of its input,
    /// which is safe because both halves of a column are read into scratch
    /// before anything is written back. This does the same; the un-activated
    /// `top` that blending needs is read from the block just before it is
    /// overwritten.
    #[inline]
    fn run(&mut self, z: &mut Buf, n: usize, combine: impl Fn(f32, f32, f32) -> f32) {
        let c = self.channels;
        debug_assert_eq!(z.rows(), 2 * c);
        for f in 0..n {
            let (top, bottom) = z.col_mut(f).split_at_mut(c);
            self.a.copy_from_slice(top);
            self.b.copy_from_slice(bottom);
            self.primary.apply(&mut self.a);
            self.secondary.apply(&mut self.b);
            for ((dst, &a), &b) in top.iter_mut().zip(&self.a).zip(&self.b) {
                *dst = combine(a, b, *dst);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{Lcg, assert_bits, assert_ulps, fill};
    use std::hint::black_box;

    fn mode(blended: bool) -> GatingMode {
        if blended {
            GatingMode::Blended
        } else {
            GatingMode::Gated
        }
    }

    fn filled(rows: usize, n: usize) -> Buf {
        let mut b = Buf::zeros(rows, n);
        for t in 0..n {
            for r in 0..rows {
                b.set(r, t, (r as f32) * 0.3 - (t as f32) * 0.45 + 0.1);
            }
        }
        b
    }

    /// The oracle's `gating_case`: seed 777, a `(2 * channels, max_cols)`
    /// block, activation over the first `n` columns, then the *whole* block
    /// compared, so the untouched bottom half and the untouched tail are
    /// checked too.
    fn gating_case(
        primary: Activation,
        secondary: Activation,
        c: usize,
        n: usize,
        max_cols: usize,
        blended: bool,
    ) -> Vec<f32> {
        let mut g = Lcg::new(777);
        let mut z = fill(&mut g, 2 * c, max_cols);
        Nonlinearity::new(mode(blended), primary, secondary, c).apply(&mut z, n);
        z.data().to_vec()
    }

    /// The oracle's `gating_prelu_case`: seed 31337, per-channel slopes.
    fn prelu_case(c: usize, n: usize, max_cols: usize, blended: bool) -> Vec<f32> {
        let mut g = Lcg::new(31337);
        let mut z = fill(&mut g, 2 * c, max_cols);
        let primary = Activation::PReLU {
            negative_slopes: (0..c).map(|i| 0.1_f32 * (i + 1) as f32).collect(),
        };
        let secondary = Activation::PReLU {
            negative_slopes: (0..c).map(|i| -0.05_f32 * (i + 1) as f32).collect(),
        };
        Nonlinearity::new(mode(blended), primary, secondary, c).apply(&mut z, n);
        z.data().to_vec()
    }

    // Expected values below come from running `NAM/gating_activations.h`
    // itself, built against the vendored Eigen at the flags
    // `tools/regen-assets.sh` pins. Gating agrees across every build variant;
    // blending's `alpha * a + (1 - alpha) * pre` is FMA-sensitive in the same
    // way FiLM's combine step is, and these are the non-contracted values.
    // The two cases through `Tanh` read the pins to within two ULP: the
    // reference ran libm's `tanhf`, this crate runs its own.

    #[test]
    fn gating_matches_the_reference() {
        // Tanh over the top half, Sigmoid over the bottom, 4 channels,
        // 5 of 7 frames.
        assert_ulps(
            &gating_case(Activation::Tanh, Activation::Sigmoid, 4, 5, 7, false),
            &[
                0x3ebc_0fbc,
                0x3ed5_dc6b,
                0x3f3c_e965,
                0xbf2a_f6a2,
                0x3eb2_dbd2,
                0x3e9e_8ca1,
                0x3f94_0da7,
                0x3fe2_4dd3,
                0x3f37_cfb8,
                0x3e4e_b3f4,
                0x3bb8_5064,
                0x3f2a_e74f,
                0x3f87_a328,
                0xbfae_f9db,
                0xbedf_be77,
                0x3f5c_c1e1,
                0xbf39_3340,
                0x3d4c_c116,
                0xbf0d_7801,
                0x3f54_78c8,
                0x403c_4f30,
                0xc018_8312,
                0x3e66_6666,
                0x3ff4_02bb,
                0xbf1e_acdd,
                0x3e9e_7360,
                0xbe86_f9f5,
                0xbdda_11ee,
                0x3f6d_65b8,
                0x3f46_508e,
                0xbf80_0aec,
                0xc006_8ca1,
                0xbf07_30d3,
                0xbf2d_ea7d,
                0x3eff_3e36,
                0xbf1a_5a9f,
                0x3e74_0da7,
                0x3f4d_0e56,
                0x3dbe_76c9,
                0x403e_5604,
                0x3fdb_7a33,
                0x4034_f87e,
                0xbf93_f7cf,
                0xc00d_4fdf,
                0xc040_a94d,
                0x400d_08e0,
                0xbf82_a535,
                0x3fa9_999a,
                0x4016_2a53,
                0x4017_983c,
                0xc02e_0419,
                0x4022_0c4a,
                0xbfcd_1942,
                0xbeef_7201,
                0xbe83_69d0,
                0x3fd6_a7f0,
            ],
            2,
        );
    }

    #[test]
    fn gating_with_other_activations_matches_the_reference() {
        // ReLU over the top half, Softsign over the bottom, 2 channels,
        // 3 of 3 frames.
        assert_bits(
            &gating_case(Activation::ReLU, Activation::Softsign, 2, 3, 3, false),
            &[
                0x3efe_c467,
                0xbef0_4d86,
                0x4005_f3b6,
                0xbf86_6666,
                0x3e3f_d9fb,
                0x3e4a_8a37,
                0x3f94_0da7,
                0x3fe2_4dd3,
                0x3cec_579a,
                0x3fef_28b7,
                0x3c6a_d65b,
                0x3fe9_ba5e,
            ],
        );
    }

    #[test]
    fn blending_matches_the_reference() {
        // Tanh activated, Sigmoid blend weight, 3 channels, 4 of 6 frames.
        assert_ulps(
            &gating_case(Activation::Tanh, Activation::Sigmoid, 3, 4, 6, true),
            &[
                0x3f35_02c6,
                0x3f4d_bac4,
                0x3fb8_feb6,
                0xbf86_6666,
                0x3eb2_dbd2,
                0x3e9e_8ca1,
                0x3f56_7048,
                0x3fad_2697,
                0x3f8e_d76b,
                0x4039_1111,
                0x3c6a_d65b,
                0x3fe9_ba5e,
                0x3f5d_edf9,
                0xbf9e_2485,
                0xbed6_f5bb,
                0x3f5c_c1e1,
                0xbf80_0000,
                0x3f2e_2a53,
                0xc02e_e75c,
                0x3fad_f320,
                0x3f9f_8d24,
                0xc018_8312,
                0x3e66_6666,
                0x3ff4_02bb,
                0xbfa8_1062,
                0x3ef9_83c1,
                0xc013_d194,
                0xc010_0576,
                0x3f6d_65b8,
                0x3f46_508e,
                0xbf80_0aec,
                0xc006_8ca1,
                0xbfe3_3e1f,
                0xc01a_e148,
                0x3fee_353f,
                0xbf3f_6715,
            ],
            2,
        );
    }

    #[test]
    fn blending_with_other_activations_matches_the_reference() {
        // SiLU activated, Hardswish blend weight (so alpha leaves [0, 1] -
        // the reference does not clamp it), 5 channels, 5 of 5 frames.
        assert_bits(
            &gating_case(Activation::SiLU, Activation::Hardswish, 5, 5, 5, true),
            &[
                0x3f31_d382,
                0x3f34_e05a,
                0x3fe2_955e,
                0x3e91_9bca,
                0xbd79_50b0,
                0x3e9e_8ca1,
                0x3f94_0da7,
                0x3fe2_4dd3,
                0x4002_aaab,
                0x4039_1111,
                0x3c2a_1ffe,
                0x3ff4_8904,
                0x3f72_237e,
                0xbfb6_25f4,
                0xbd0f_0738,
                0x3f5c_c1e1,
                0xbf80_0000,
                0x3f2e_2a53,
                0xc039_2c60,
                0x3fee_f9db,
                0x4039_a316,
                0xc03d_9de4,
                0x3e81_94b5,
                0x3fe0_d88e,
                0xbf4f_0a97,
                0x3ef9_83c1,
                0xc013_d194,
                0xc010_0576,
                0x3f6d_65b8,
                0x3f46_508e,
                0xbf9a_5135,
                0xbfee_4086,
                0xbf80_a36a,
                0xc014_0fa8,
                0x3f8f_46a2,
                0xbf3f_6715,
                0x3e74_0da7,
                0x3f4d_0e56,
                0x3dbe_76c9,
                0x403e_5604,
                0x3f9b_7bf9,
                0x4038_5fa9,
                0xbea1_e6fe,
                0x3ff9_be6a,
                0x4044_7ec2,
                0x400d_08e0,
                0xbf82_a535,
                0x3fa9_999a,
                0x4016_2a53,
                0x4017_983c,
            ],
        );
    }

    #[test]
    fn gating_prelu_indexes_each_half_from_channel_zero() {
        // Both halves get their own slope table, each indexed from 0: the
        // secondary's slopes are *not* offset by `channels`. 3 channels,
        // 4 of 5 frames.
        assert_bits(
            &prelu_case(3, 4, 5, false),
            &[
                0xbc0f_8a68,
                0x3feb_3c79,
                0xbd1f_c6da,
                0xc054_85cd,
                0x3fc3_8a95,
                0x3e7b_38a9,
                0xbd29_1811,
                0xbe1e_c7e0,
                0x3dac_db11,
                0xc053_fd45,
                0x3ebe_4b18,
                0xbfcb_4396,
                0x3de4_b6f8,
                0x40a2_b915,
                0xbed4_afff,
                0x3f59_af72,
                0x4043_8a95,
                0xc03a_2d0e,
                0x3b89_1fcf,
                0x3d27_2f52,
                0x4063_eb08,
                0xbe63_53f8,
                0xbe76_19f1,
                0x401c_8057,
                0x4047_983c,
                0xc001_cac1,
                0x3e37_a328,
                0xc04e_f465,
                0x3fe7_a328,
                0x3f3d_2f1b,
            ],
        );
    }

    #[test]
    fn blending_prelu_matches_the_reference() {
        assert_bits(
            &prelu_case(3, 4, 5, true),
            &[
                0xbee5_cb67,
                0x3f99_fbe8,
                0xbee0_c229,
                0xc054_85cd,
                0x3fc3_8a95,
                0x3e7b_38a9,
                0xc007_c077,
                0xbfbb_9e1c,
                0x3eb5_6b2e,
                0xc053_fd45,
                0x3ebe_4b18,
                0xbfcb_4396,
                0x3e06_7c3f,
                0x3fd5_08de,
                0xc00d_0f3b,
                0x3f59_af72,
                0x4043_8a95,
                0xc03a_2d0e,
                0x3ec1_0625,
                0x3fd9_62fe,
                0x3fba_6920,
                0xbe63_53f8,
                0xbe76_19f1,
                0x401c_8057,
                0x4047_983c,
                0xc001_cac1,
                0x3e37_a328,
                0xc04e_f465,
                0x3fe7_a328,
                0x3f3d_2f1b,
            ],
        );
    }

    #[test]
    fn the_bottom_half_and_the_tail_survive() {
        let (c, n, max_cols) = (3usize, 4usize, 6usize);
        let mut g = Lcg::new(777);
        let src = fill(&mut g, 2 * c, max_cols);
        let mut z = src.clone();
        Nonlinearity::new(GatingMode::Gated, Activation::Tanh, Activation::Sigmoid, c)
            .apply(&mut z, n);
        for t in 0..max_cols {
            for r in 0..2 * c {
                if t < n && r < c {
                    continue;
                }
                assert_eq!(z.at(r, t).to_bits(), src.at(r, t).to_bits(), "r={r} t={t}");
            }
        }
    }

    // Bit-exact, and not by luck: gating multiplies two activated values, so
    // there is no accumulation here to fuse.
    //
    // `black_box` on the inputs of the expected value: `filled` is arithmetic
    // on loop indices, which the optimiser sees through, and LLVM then folds
    // `expf` on the resulting constants through its double-precision folder.
    // That lands one ULP away from the runtime libm often enough to fail
    // under `--release` while passing in debug.
    #[test]
    fn gating_multiplies_the_two_activated_halves() {
        let c = 3usize;
        let mut g = Nonlinearity::new(GatingMode::Gated, Activation::Tanh, Activation::Sigmoid, c);
        let src = filled(2 * c, 5);
        let mut z = src.clone();
        g.apply(&mut z, 5);
        for t in 0..5 {
            for i in 0..c {
                let want = crate::activations::tanh(black_box(src.at(i, t)))
                    * crate::activations::sigmoid(black_box(src.at(i + c, t)));
                assert_eq!(z.at(i, t).to_bits(), want.to_bits());
            }
        }
    }

    #[test]
    fn blending_interpolates_towards_the_unactivated_signal() {
        let c = 2usize;
        let mut b = Nonlinearity::new(
            GatingMode::Blended,
            Activation::ReLU,
            Activation::Sigmoid,
            c,
        );
        let src = filled(2 * c, 4);
        let mut z = src.clone();
        b.apply(&mut z, 4);
        for t in 0..4 {
            for i in 0..c {
                let pre = black_box(src.at(i, t));
                let alpha = crate::activations::sigmoid(black_box(src.at(i + c, t)));
                let want = alpha * crate::activations::relu(pre) + (1.0 - alpha) * pre;
                assert_eq!(z.at(i, t).to_bits(), want.to_bits());
            }
        }
    }

    /// alpha == 1 must reproduce the activation and alpha == 0 the raw input,
    /// which is the property the blend is there to provide.
    #[test]
    fn blending_endpoints_are_the_activation_and_the_identity() {
        let c = 2usize;
        let mut b = Nonlinearity::new(
            GatingMode::Blended,
            Activation::ReLU,
            Activation::Identity,
            c,
        );
        let mut z = Buf::zeros(2 * c, 2);
        z.set(0, 0, -3.0);
        z.set(1, 0, 2.5);
        z.set(2, 0, 1.0); // alpha = 1 -> pure ReLU
        z.set(3, 0, 1.0);
        z.set(0, 1, -3.0);
        z.set(1, 1, 2.5);
        z.set(2, 1, 0.0); // alpha = 0 -> pass through
        z.set(3, 1, 0.0);
        b.apply(&mut z, 2);
        assert_eq!(z.at(0, 0), 0.0);
        assert_eq!(z.at(1, 0), 2.5);
        assert_eq!(z.at(0, 1), -3.0);
        assert_eq!(z.at(1, 1), 2.5);
    }

    #[test]
    fn prelu_slopes_follow_the_channel_not_the_frame() {
        let c = 2usize;
        let mut g = Nonlinearity::new(
            GatingMode::Gated,
            Activation::PReLU {
                negative_slopes: vec![0.5, 0.25],
            },
            Activation::Identity,
            c,
        );
        let mut z = Buf::zeros(2 * c, 2);
        for t in 0..2 {
            z.set(0, t, -1.0);
            z.set(1, t, -1.0);
            z.set(2, t, 1.0);
            z.set(3, t, 1.0);
        }
        g.apply(&mut z, 2);
        for t in 0..2 {
            assert_eq!(z.at(0, t), -0.5);
            assert_eq!(z.at(1, t), -0.25);
        }
    }

    /// The ungated mode is the plain activation over the first `n` columns,
    /// and nothing past them.
    #[test]
    fn plain_touches_only_the_first_n_columns() {
        let src = filled(3, 4);
        let mut z = src.clone();
        Nonlinearity::new(GatingMode::None, Activation::ReLU, Activation::Tanh, 3).apply(&mut z, 2);
        for t in 0..4 {
            for r in 0..3 {
                let want = if t < 2 {
                    crate::activations::relu(src.at(r, t))
                } else {
                    src.at(r, t)
                };
                assert_eq!(z.at(r, t).to_bits(), want.to_bits(), "r={r} t={t}");
            }
        }
    }
}
