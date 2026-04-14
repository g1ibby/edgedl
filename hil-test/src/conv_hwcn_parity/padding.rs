use edgedl::{
    arena::Arena,
    kernels::{
        conv2d::conv2d_scalar,
        simd::conv2d::{ExecOutcome, try_conv2d},
    },
    model::{Activation, NodeOp, NodeSpec},
    rounding::DEFAULT_ROUNDING,
};

use super::util::{Case, IN_ELEMS, IN_ID, OUT_ELEMS, OUT_ID, PLAN, WeightExps, build_spec};

macro_rules! run_padding_test {
    ($case:expr, $pads:expr, $strides:expr, $dilations:expr $(,)?) => {{
        // Build base spec
        let spec = build_spec($case);

        // Patch the plan to include padding, stride, dilation
        let mut node = spec.plan[0];
        if let NodeOp::Conv2d(ref mut c) = node.op {
            c.pads_hw = $pads;
            c.strides_hw = $strides;
            c.dilations_hw = $dilations;
        }

        // This expands to a unique static cell per call site
        let plan_slice = hil_test::mk_static!([NodeSpec; 1], [node]);

        let spec_with_padding = edgedl::model::ModelSpec {
            alignment: spec.alignment,
            inputs: spec.inputs,
            outputs: spec.outputs,
            values: spec.values,
            initializers: spec.initializers,
            plan: plan_slice,
        };

        // Aligned arena buffer
        const ARENA_TOTAL: usize = 2048;
        let mut arena_buf = super::util::Aligned::<ARENA_TOTAL>([0; ARENA_TOTAL]);
        let mut arena = Arena::from_buf(&mut arena_buf.0);

        // Prepare input pattern
        let mut input = [0i8; IN_ELEMS];
        for (i, v) in input.iter_mut().enumerate() {
            let t = (i % 11) as i8;
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

        // Copy input into arena
        let in_mut = arena.value_slice_mut(PLAN.offset_of(IN_ID), IN_ELEMS);
        in_mut.copy_from_slice(&input);
        let out_mut = arena.value_slice_mut(PLAN.offset_of(OUT_ID), OUT_ELEMS);
        out_mut.fill(0);

        // Run scalar
        conv2d_scalar(
            &PLAN,
            &mut arena,
            &spec_with_padding,
            match spec_with_padding.plan[0].op {
                NodeOp::Conv2d(ref c) => c,
                _ => panic!(),
            },
            DEFAULT_ROUNDING,
        );
        let mut golden = [0i8; OUT_ELEMS];
        golden.copy_from_slice(arena.value_slice(PLAN.offset_of(OUT_ID), OUT_ELEMS));

        // Clear output and run SIMD
        let out_mut2 = arena.value_slice_mut(PLAN.offset_of(OUT_ID), OUT_ELEMS);
        out_mut2.fill(0);
        let outcome = try_conv2d(
            &PLAN,
            &mut arena,
            &spec_with_padding,
            match spec_with_padding.plan[0].op {
                NodeOp::Conv2d(ref c) => c,
                _ => panic!(),
            },
            DEFAULT_ROUNDING,
        );

        if !matches!(outcome, ExecOutcome::Executed) {
            edgedl::ne_error!("conv_hwcn_padding: SIMD path fell back");
            assert!(
                matches!(outcome, ExecOutcome::Executed),
                "SIMD path fell back"
            );
        }

        // Compare
        let got = arena.value_slice(PLAN.offset_of(OUT_ID), OUT_ELEMS);
        for i in 0..OUT_ELEMS {
            let a = golden[i] as i16;
            let b = got[i] as i16;
            let diff = if a > b { a - b } else { b - a };
            if diff > 1 {
                edgedl::ne_error!(
                    "conv_hwcn_padding: mismatch at idx={} scalar={} simd={}",
                    i,
                    golden[i],
                    got[i]
                );
                assert!(
                    diff <= 1,
                    "mismatch at {}: scalar={} simd={}",
                    i,
                    golden[i],
                    got[i]
                );
            }
        }
    }};
}

pub fn test_padding_55_linear_pt() {
    run_padding_test!(
        &Case {
            layout: super::util::Layout::OHWI,
            w_exps: WeightExps::PerTensor0,
            has_bias: false,
            activation: Activation::Linear,
        },
        [1, 1, 1, 1], // pads
        [1, 1],       // strides
        [1, 1],       // dilations
    );
}

pub fn test_padding_55_relu_bias_pt() {
    run_padding_test!(
        &Case {
            layout: super::util::Layout::OHWI,
            w_exps: WeightExps::PerTensor0,
            has_bias: true,
            activation: Activation::ReLU,
        },
        [2, 2, 2, 2],
        [1, 1],
        [1, 1],
    );
}

pub fn test_padding_stride2() {
    run_padding_test!(
        &Case {
            layout: super::util::Layout::OHWI,
            w_exps: WeightExps::PerTensor0,
            has_bias: true,
            activation: Activation::Linear,
        },
        [1, 1, 1, 1],
        [2, 2],
        [1, 1],
    );
}

pub fn test_padding_dilation2() {
    run_padding_test!(
        &Case {
            layout: super::util::Layout::OHWI,
            w_exps: WeightExps::PerTensor0,
            has_bias: true,
            activation: Activation::Linear,
        },
        [1, 1, 1, 1],
        [1, 1],
        [2, 2],
    );
}

/// Blocked layout case inspired by noise_model first conv:
/// 5x5 kernel, stride 2, SAME padding [2,2,2,2], ReLU, with bias,
/// and weights in BlockedN16HWC16 layout.
pub fn test_padding_blocked_stride2_relu_bias_pt() {
    run_padding_test!(
        &Case {
            layout: super::util::Layout::BlockedN16HWC16,
            w_exps: WeightExps::PerTensor0,
            has_bias: true,
            activation: Activation::ReLU,
        },
        [2, 2, 2, 2],
        [2, 2],
        [1, 1],
    );
}

pub fn run_group() {
    test_padding_55_linear_pt();
    test_padding_55_relu_bias_pt();
    test_padding_stride2();
    test_padding_dilation2();
    test_padding_blocked_stride2_relu_bias_pt();
}
