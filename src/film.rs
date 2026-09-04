//! Feature-wise Linear Modulation.
//!
//! A port of `NAM/film.h`. A `Conv1x1` maps the conditioning signal to a
//! per-channel scale (and, optionally, a shift), which then modulate the
//! signal passing through:
//!
//! ```text
//! scale, shift = Conv1x1(condition)     // split top/bottom half
//! output       = input * scale + shift
//! ```
//!
//! Every learned parameter belongs to that `Conv1x1`, which `film.h` always
//! builds with a bias, so weight consumption is exactly the convolution's.
//!
//! This is the mechanism behind parametric captures: knob positions
//! conditioning the model.
//!
//! # Why the combine step must not fuse
//!
//! `film.h` has two implementations of `input * scale + shift`: an Eigen
//! array expression and, under `NAM_USE_INLINE_GEMM`, a hand-written loop.
//! Compiled with clang's default `-ffp-contract=on` they are *not*
//! bit-identical. The hand-written statement contracts into an FMA, which
//! rounds once, while the Eigen expression rounds the multiply and the add
//! separately, and the two disagree by 1 ULP on a large fraction of outputs.
//! Under the reference build this crate pins (`-ffp-contract=off
//! -DEIGEN_DONT_VECTORIZE`, see `tools/regen-assets.sh`) both paths agree,
//! and that is what this port reproduces: two roundings, never `mul_add`.

use crate::buffer::Buf;
use crate::conv::{Conv1x1, View};
use crate::error::Result;
use crate::weights::WeightReader;

/// Per-channel scale-and-shift modulation driven by a conditioning signal.
#[derive(Debug, Clone)]
pub(crate) struct Film {
    cond_to_scale_shift: Conv1x1,
    do_shift: bool,
    input_dim: usize,
    /// `(input_dim, max_buffer)`: the modulated signal.
    pub(crate) output: Buf,
}

impl Film {
    /// Build a FiLM site modulating `input_dim` channels from `condition_dim`
    /// conditioning channels, reading its weights from `r`.
    ///
    /// The generating convolution has `2 * input_dim` outputs when `shift` is
    /// set (scale on top, shift below) and `input_dim` otherwise, and always
    /// carries a bias. Fails as [`Conv1x1::new`] does.
    pub(crate) fn new(
        condition_dim: usize,
        input_dim: usize,
        shift: bool,
        groups: usize,
        r: &mut WeightReader<'_>,
    ) -> Result<Self> {
        let out = if shift { 2 * input_dim } else { input_dim };
        Ok(Self {
            cond_to_scale_shift: Conv1x1::new(condition_dim, out, true, groups, r)?,
            do_shift: shift,
            input_dim,
            output: Buf::new(),
        })
    }

    /// Reserve buffers for blocks of up to `max_buffer` frames. Allocating.
    pub(crate) fn set_max_buffer_size(&mut self, max_buffer: usize) {
        self.cond_to_scale_shift.set_max_buffer_size(max_buffer);
        self.output.resize(self.input_dim, max_buffer);
    }

    /// Modulate `input`, leaving it untouched; the result lands in
    /// [`Film::output`]. The reference's `Process`.
    ///
    /// Takes a [`View`] rather than a `&Buf` because `wavenet/model.cpp`
    /// feeds a gated layer's post-activation FiLM with `_z.topRows(bottleneck)`,
    /// a block whose column stride is the source row count.
    pub(crate) fn process(&mut self, input: View<'_>, condition: &Buf, n: usize) {
        debug_assert_eq!(input.rows(), self.input_dim);
        debug_assert_eq!(condition.rows(), self.cond_to_scale_shift.in_channels());
        debug_assert!(n <= self.output.cols());

        self.cond_to_scale_shift.process(condition, n);
        let ss = &self.cond_to_scale_shift.output;
        let ss_rows = ss.rows();
        let d = self.input_dim;
        let out = self.output.left_mut(n);
        // The branch is hoisted out of the frame loop, as it is in the
        // reference, so the inner loop is straight-line.
        if self.do_shift {
            for f in 0..n {
                let x = input.col(f);
                let sscol = &ss.data()[f * ss_rows..(f + 1) * ss_rows];
                let o = &mut out[f * d..(f + 1) * d];
                for c in 0..d {
                    o[c] = x[c] * sscol[c] + sscol[c + d];
                }
            }
        } else {
            for f in 0..n {
                let x = input.col(f);
                let sscol = &ss.data()[f * ss_rows..(f + 1) * ss_rows];
                let o = &mut out[f * d..(f + 1) * d];
                for c in 0..d {
                    o[c] = x[c] * sscol[c];
                }
            }
        }
    }

    /// Modulate `target` in place. The reference's `Process_`, which is
    /// `Process` followed by copying the output back.
    ///
    /// Only the first `n` columns are touched; the tail of `target` keeps
    /// whatever it held, as it does in the reference.
    pub(crate) fn process_in_place(&mut self, target: &mut Buf, condition: &Buf, n: usize) {
        debug_assert_eq!(target.rows(), self.input_dim);
        self.process(View::full(target), condition, n);
        target.left_mut(n).copy_from_slice(self.output.left(n));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{Lcg, assert_bits, assert_close, fill};

    /// Weights a site of this shape consumes: the generator's
    /// `(out / groups) * (cond / groups) * groups` matrix plus its bias.
    fn weight_count(cond: usize, dim: usize, shift: bool, groups: usize) -> usize {
        let out = if shift { 2 * dim } else { dim };
        (out / groups) * (cond / groups) * groups + out
    }

    /// A site with weights drawn from `g`, in the oracle's order.
    fn load(cond: usize, dim: usize, shift: bool, groups: usize, g: &mut Lcg) -> Film {
        let w: Vec<f32> = (0..weight_count(cond, dim, shift, groups))
            .map(|_| g.next())
            .collect();
        let mut r = WeightReader::new(&w);
        let f = Film::new(cond, dim, shift, groups, &mut r).unwrap();
        r.finish().unwrap();
        f
    }

    /// The oracle's `film_case`: seed 12345, then weights, input and condition
    /// drawn from that one stream in that order.
    fn film_case(cond: usize, dim: usize, shift: bool, groups: usize, n: usize) -> Vec<f32> {
        let mut g = Lcg::new(12345);
        let mut f = load(cond, dim, shift, groups, &mut g);
        f.set_max_buffer_size(n);
        let input = fill(&mut g, dim, n);
        let condition = fill(&mut g, cond, n);
        f.process(View::full(&input), &condition, n);
        f.output.left(n).to_vec()
    }

    // Expected values below come from running `NAM/film.h` itself, built the
    // way `tools/regen-assets.sh` builds the reference:
    //
    //   c++ -std=c++20 -O2 -ffp-contract=off \
    //       -DEIGEN_DONT_VECTORIZE -DEIGEN_MAX_ALIGN_BYTES=0
    //
    // Both of that header's code paths (Eigen and `NAM_USE_INLINE_GEMM`) agree
    // under those flags. They do *not* agree when contraction is left on, and
    // the generator's values shift when Eigen is allowed to vectorize its
    // reduction, so these numbers are only meaningful against that build.
    //
    // The cases through the generator are compared to within `REF_TOL`
    // rather than bit for bit: the generator's multiply-accumulate fuses here
    // and did not in the oracle, so the two round differently. What each case
    // still pins is the formula. The depthwise case has no accumulation and
    // stays bit-exact.

    #[test]
    fn scale_and_shift_match_the_reference() {
        // FiLM(condition_dim=3, input_dim=4, shift, groups=1), 5 frames.
        assert_close(
            &film_case(3, 4, true, 1, 5),
            &[
                0xc1c4_8f81,
                0xc015_db80,
                0xc0b3_1fc9,
                0x403d_8bc1,
                0xc1a0_9c2b,
                0xc020_56da,
                0x4067_4949,
                0x40bc_3be0,
                0xc10b_f687,
                0x4055_e0cc,
                0xc1c9_0b87,
                0xc1c5_54da,
                0x41e9_db77,
                0x416e_f48c,
                0x4091_6e24,
                0x410d_82de,
                0xc054_9676,
                0xc125_8ea8,
                0xc175_c0c6,
                0xc069_4b94,
            ],
        );
    }

    #[test]
    fn scale_only_and_grouped_matches_the_reference() {
        // FiLM(4, 4, no shift, groups=2), 3 frames.
        assert_close(
            &film_case(4, 4, false, 2, 3),
            &[
                0x4233_66ae,
                0xc011_b8b0,
                0x404e_e018,
                0x40d9_b4a1,
                0xc061_441e,
                0xc0f4_7a4b,
                0xc036_cc3f,
                0xc189_6d8d,
                0x3f34_b093,
                0xc0ae_93c0,
                0x3fcf_f7d5,
                0x3fa7_26af,
            ],
        );
    }

    #[test]
    fn grouped_with_shift_matches_the_reference() {
        // FiLM(6, 3, shift, groups=3): the generator is a grouped 6 -> 6 map,
        // so the group structure straddles the scale/shift split.
        assert_close(
            &film_case(6, 3, true, 3, 4),
            &[
                0x3f46_3fc0,
                0xc02a_fa29,
                0xc062_29de,
                0xc1f9_29ba,
                0x413a_be9b,
                0x4086_25bc,
                0xc224_8360,
                0xc069_512e,
                0x40df_362f,
                0xc0ae_114b,
                0x405c_b6d1,
                0x408b_53cf,
            ],
        );
    }

    #[test]
    fn depthwise_generator_matches_the_reference() {
        // groups == in == out sends the generator down Conv1x1's depthwise path.
        assert_bits(
            &film_case(4, 4, false, 4, 3),
            &[
                0x4082_d413,
                0xc0bc_47a7,
                0xbff9_590b,
                0x40e8_5979,
                0x41b1_bfe7,
                0xc182_25d6,
                0x40d6_c2bb,
                0xc0ca_d515,
                0xbe33_1a68,
                0xc14f_e009,
                0x4023_ed03,
                0xc10f_84dc,
            ],
        );
    }

    #[test]
    fn in_place_matches_the_reference_and_leaves_the_tail_alone() {
        // Oracle `film_inplace`: seed 999, FiLM(2, 3, shift), 4 of 6 frames.
        let (cond, dim, n, max_cols) = (2usize, 3usize, 4usize, 6usize);
        let mut g = Lcg::new(999);
        let mut f = load(cond, dim, true, 1, &mut g);
        f.set_max_buffer_size(max_cols);
        let mut target = fill(&mut g, dim, max_cols);
        let condition = fill(&mut g, cond, max_cols);
        let tail = target.data()[dim * n..].to_vec();

        f.process_in_place(&mut target, &condition, n);

        assert_close(
            target.data(),
            &[
                0xc148_59d4,
                0x4113_09b2,
                0x405a_39ae,
                0x3fd9_c74e,
                0x4029_072e,
                0xc056_f50d,
                0x4090_b398,
                0x4051_648c,
                0xc037_b805,
                0x4005_e45f,
                0xbf93_f6c0,
                0xc046_a80c,
                0xbf7c_ed91,
                0x4020_bf26,
                0x402c_ac08,
                0x3f81_3cc2,
                0x4009_9423,
                0x4002_f1aa,
            ],
        );
        assert_eq!(
            &target.data()[dim * n..],
            &tail[..],
            "columns past n were touched"
        );
    }

    #[test]
    fn strided_top_rows_input_matches_the_reference() {
        // Oracle `film_toprows`: seed 4242, FiLM(5, 4) over the top 4 of 8 rows
        // of `z` for 3 of 5 frames, result copied back over those same rows -
        // what `Layer::Process` does for a gated layer's post-activation FiLM.
        let (cond, bn, n, max_cols) = (5usize, 4usize, 3usize, 5usize);
        let mut g = Lcg::new(4242);
        let mut f = load(cond, bn, true, 1, &mut g);
        f.set_max_buffer_size(max_cols);
        let mut z = fill(&mut g, 2 * bn, max_cols);
        let condition = fill(&mut g, cond, max_cols);

        f.process(View::top_rows(&z, bn), &condition, n);
        let rows = z.rows();
        for c in 0..n {
            let src = f.output.col(c).to_vec();
            z.data_mut()[c * rows..c * rows + bn].copy_from_slice(&src);
        }

        assert_close(
            z.data(),
            &[
                0xc021_fd20,
                0xc0df_7207,
                0x4175_6c5a,
                0x409c_fc78,
                0x3d5e_8ca1,
                0x3f68_3127,
                0xbee0_15d8,
                0xc024_02bb,
                0xc124_6a7b,
                0x3fb9_fa5d,
                0x410e_64c5,
                0xc003_0580,
                0x4037_04c7,
                0xbff6_b2dc,
                0x3dd2_f1aa,
                0xbe4f_3078,
                0x4106_dbc4,
                0x4095_d75e,
                0xc095_913b,
                0x4161_d324,
                0x3f84_dd2f,
                0xc02f_6c8b,
                0x4009_1111,
                0x3d79_db23,
                0x3ed0_e560,
                0x3ef5_6b2e,
                0x3f25_e354,
                0x3d83_126f,
                0x3f66_0f05,
                0xbfd7_2b02,
                0x4054_b6f4,
                0x3fba_6921,
                0xbfdc_ed91,
                0x4013_900b,
                0x3dd2_f1aa,
                0xbf6a_2798,
                0x3fcf_7201,
                0x3f06_3ab6,
                0xc04b_7a33,
                0xbf4d_fea2,
            ],
        );
    }

    #[test]
    fn scale_and_shift_split_across_the_generator_output() {
        let (cond, dim) = (2usize, 3usize);
        let w: Vec<f32> = (0..weight_count(cond, dim, true, 1))
            .map(|i| (i as f32) * 0.1 - 0.5)
            .collect();
        let mut f = Film::new(cond, dim, true, 1, &mut WeightReader::new(&w)).unwrap();
        f.set_max_buffer_size(4);

        let mut condition = Buf::zeros(cond, 4);
        let mut input = Buf::zeros(dim, 4);
        for t in 0..4 {
            for c in 0..cond {
                condition.set(c, t, (t as f32) * 0.3 - (c as f32) * 0.7);
            }
            for c in 0..dim {
                input.set(c, t, (c as f32) - (t as f32) * 0.25);
            }
        }
        f.process(View::full(&input), &condition, 4);

        // Independently: run the generator, then apply scale/shift by hand.
        let mut generator =
            Conv1x1::new(cond, 2 * dim, true, 1, &mut WeightReader::new(&w)).unwrap();
        generator.set_max_buffer_size(4);
        generator.process(&condition, 4);
        for t in 0..4 {
            for c in 0..dim {
                let want =
                    input.at(c, t) * generator.output.at(c, t) + generator.output.at(c + dim, t);
                assert_eq!(f.output.at(c, t).to_bits(), want.to_bits(), "c={c} t={t}");
            }
        }
    }

    /// With a zeroed generator matrix the scale and shift reduce to its bias,
    /// which pins the combine step down independently of `Conv1x1`.
    #[test]
    fn combine_step_is_input_times_scale_plus_shift() {
        let (cond, dim, n) = (3usize, 5usize, 4usize);
        let mut w = vec![0.0f32; 2 * dim * cond];
        let scale: Vec<f32> = (0..dim).map(|i| 0.5 + i as f32 * 0.25).collect();
        let shift: Vec<f32> = (0..dim).map(|i| -1.0 + i as f32 * 0.125).collect();
        w.extend_from_slice(&scale);
        w.extend_from_slice(&shift);
        assert_eq!(w.len(), weight_count(cond, dim, true, 1));
        let mut f = Film::new(cond, dim, true, 1, &mut WeightReader::new(&w)).unwrap();
        f.set_max_buffer_size(n);

        let mut g = Lcg::new(7);
        let input = fill(&mut g, dim, n);
        let condition = fill(&mut g, cond, n);
        f.process(View::full(&input), &condition, n);

        for t in 0..n {
            for c in 0..dim {
                let want = input.at(c, t) * scale[c] + shift[c];
                assert_eq!(f.output.at(c, t).to_bits(), want.to_bits(), "c={c} t={t}");
            }
        }
    }

    #[test]
    fn weight_count_follows_the_generator_shape() {
        // (out/g) * (in/g) * g weights, then one bias per output channel; the
        // site refuses one weight fewer and leaves none over with exactly that.
        for (cond, dim, shift, groups, want) in [
            (3usize, 4usize, false, 1usize, 4 * 3 + 4),
            (3, 4, true, 1, 8 * 3 + 8),
            (4, 4, false, 2, 2 * 2 * 2 + 4),
            // Depthwise: one weight per channel plus the bias.
            (4, 4, false, 4, 4 + 4),
        ] {
            assert_eq!(weight_count(cond, dim, shift, groups), want);
            let w = vec![0.5f32; want];
            let mut r = WeightReader::new(&w);
            Film::new(cond, dim, shift, groups, &mut r).unwrap();
            r.finish().unwrap();
            assert!(Film::new(cond, dim, shift, groups, &mut WeightReader::new(&w[1..])).is_err());
        }
    }

    #[test]
    fn output_is_as_wide_as_the_modulated_signal_not_the_generator() {
        let w = vec![0.0f32; weight_count(3, 4, true, 1)];
        let mut f = Film::new(3, 4, true, 1, &mut WeightReader::new(&w)).unwrap();
        f.set_max_buffer_size(16);
        assert_eq!(f.output.rows(), 4);
        assert_eq!(f.output.cols(), 16);
    }
}
