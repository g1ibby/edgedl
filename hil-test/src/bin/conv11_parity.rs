//! HIL: Conv2D 1x1 (11cn) SIMD vs scalar parity on ESP32-S3
//% CHIPS: esp32s3
//% FEATURES: defmt trace

#![no_std]
#![no_main]

use hil_test as _;

// Keep modules outside of src/bin so xtask can scan src/bin as files only
#[path = "../conv11_parity/blocked.rs"]
mod blocked;
#[path = "../conv11_parity/ic_tail.rs"]
mod ic_tail;
#[path = "../conv11_parity/oc_tail.rs"]
mod oc_tail;
#[path = "../conv11_parity/ohwi.rs"]
mod ohwi;
#[path = "../conv11_parity/padding.rs"]
mod padding;
#[path = "../conv11_parity/util.rs"]
mod util;

#[embedded_test::tests(default_timeout = 60, executor = hil_test::Executor::new())]
mod tests {
    use super::*;

    pub struct Ctx;

    #[init]
    fn init() -> Ctx {
        let _p = esp_hal::init(
            esp_hal::Config::default().with_cpu_clock(esp_hal::clock::CpuClock::max()),
        );
        Ctx
    }

    #[test]
    fn conv11_ohwi_parity(_ctx: Ctx) {
        ohwi::run_group();
    }

    #[test]
    fn conv11_blocked_parity(_ctx: Ctx) {
        blocked::run_group();
    }

    #[test]
    fn conv11_oc_tail_parity(_ctx: Ctx) {
        oc_tail::run_group();
    }

    #[test]
    fn conv11_ic_tail_parity(_ctx: Ctx) {
        ic_tail::run_group();
    }

    #[test]
    fn conv11_padding_1111_linear_pt(_ctx: Ctx) {
        padding::test_padding_1111_linear_pt();
    }

    #[test]
    fn conv11_padding_1111_relu_bias_pt(_ctx: Ctx) {
        padding::test_padding_1111_relu_bias_pt();
    }

    #[test]
    fn conv11_padding_1111_linear_pc(_ctx: Ctx) {
        padding::test_padding_1111_linear_pc();
    }

    #[test]
    fn conv11_padding_1111_relu_bias_pc(_ctx: Ctx) {
        padding::test_padding_1111_relu_bias_pc();
    }

    #[test]
    fn conv11_padding_blocked_relu_bias_pt(_ctx: Ctx) {
        padding::test_padding_blocked_relu_bias_pt();
    }

    #[test]
    fn conv11_ohwi_neg_shift_one_pixel_sanity(_ctx: Ctx) {
        // Sanity: per-tensor + bias with negative mac_shift (out_exp=2) should
        // route to the per-channel vector epilogue. Compare only one pixel and
        // log max|Δ| across 16 lanes to isolate the epilogue behavior.
        use edgedl::{
            arena::Arena,
            kernels::{
                conv2d::conv2d_scalar,
                simd::conv2d::{ExecOutcome, try_conv2d},
            },
            model::{Activation, NodeOp},
            rounding::DEFAULT_ROUNDING,
        };

        use super::util::*;

        let case = Case {
            layout: Layout::OHWI,
            w_exps: WeightExps::PerTensor0,
            has_bias: true,
            activation: Activation::Linear,
            out_exp: 2, // ensure negative mac_shift -> per-channel vector epilogue
        };
        let spec = build_spec(&case);
        let conv = match spec.plan[0].op {
            NodeOp::Conv2d(ref c) => c,
            _ => unreachable!(),
        };

        // Arena
        const ARENA_TOTAL: usize = SIZE_BYTES + SCRATCH_BYTES;
        let mut arena_buf = Aligned::<ARENA_TOTAL>([0; ARENA_TOTAL]);
        let mut arena = Arena::from_buf(&mut arena_buf.0);

        // Input pattern (same as util)
        let mut input = [0i8; IN_ELEMS];
        for (i, v) in input.iter_mut().enumerate() {
            let t = (i % 7) as i8;
            *v = match t {
                0 => -3,
                1 => -1,
                2 => 0,
                3 => 1,
                4 => 2,
                5 => -2,
                _ => 3,
            };
        }
        let in_mut = arena.value_slice_mut(PLAN.offset_of(IN_ID), IN_ELEMS);
        in_mut.copy_from_slice(&input);
        let out_mut = arena.value_slice_mut(PLAN.offset_of(OUT_ID), OUT_ELEMS);
        out_mut.fill(0);

        // Scalar
        conv2d_scalar(&PLAN, &mut arena, &spec, conv, DEFAULT_ROUNDING);
        let golden = {
            let s = arena.value_slice(PLAN.offset_of(OUT_ID), OUT_ELEMS);
            let mut tmp = [0i8; OUT_ELEMS];
            tmp.copy_from_slice(s);
            tmp
        };

        // SIMD
        let out_mut2 = arena.value_slice_mut(PLAN.offset_of(OUT_ID), OUT_ELEMS);
        out_mut2.fill(0);
        let outcome = try_conv2d(&PLAN, &mut arena, &spec, conv, DEFAULT_ROUNDING);
        assert!(
            matches!(outcome, ExecOutcome::Executed),
            "SIMD path fell back for neg-shift sanity"
        );
        let got = arena.value_slice(PLAN.offset_of(OUT_ID), OUT_ELEMS);

        // One pixel (0,0) across OC lanes
        let base = 0usize; // ((0*W)+0)*OC
        let mut max_abs = 0i16;
        for i in 0..OC {
            // lanes
            let a = golden[base + i] as i16;
            let b = got[base + i] as i16;
            let d = if a > b { a - b } else { b - a };
            if d > max_abs {
                max_abs = d;
            }
        }
        defmt::info!(
            "neg_shift_one_pixel_sanity: max_abs_diff={} (expected ≤1)",
            max_abs
        );
    }
}
