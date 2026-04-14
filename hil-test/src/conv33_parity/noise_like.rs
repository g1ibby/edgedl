use edgedl::{
    arena::Arena,
    kernels::{
        conv2d::conv2d_scalar,
        simd::conv2d::{ExecOutcome, try_conv2d},
    },
    model::{Initializer, ModelSpec, NodeOp, ParamExponents, ParamLayout, Shape4dNHWC, ValueMeta},
    rounding::DEFAULT_ROUNDING,
};

use super::util::{
    Aligned,
    BIAS_I32_BYTES,
    IN_ELEMS,
    IN_ID,
    INPUTS,
    KH,
    KW,
    OC,
    OUT_ELEMS,
    OUT_ID,
    OUTPUTS,
    PLAN,
    PLAN_WITH_BIAS_LINEAR,
    SCRATCH_BYTES,
    SIZE_BYTES,
    W_ID,
    W_OHWI_ONES,
};

// Noise-model-like exponent patterns for 3x3 Conv2D:
// - Case A (matches Conv id=5):  in_e = -4, w_e = -8, out_e = -4
// - Case B (matches Conv id=6):  in_e = -4, w_e = -9, out_e = -4
// - Case C (matches Conv id=2):  in_e = -4, w_e = -10, out_e = -5

// Value tables: reuse the same shapes as conv33_parity util, but override exponents.
pub static VALUES_NOISE_INM4_OUTM4: [ValueMeta; 4] = [
    // Input
    ValueMeta {
        shape: Shape4dNHWC::new(
            1,
            super::util::H as u16,
            super::util::W as u16,
            super::util::IC as u16,
        ),
        exp: -4,
    },
    // Weights (exp unused for Conv2D; exponents come from Initializer)
    ValueMeta {
        shape: Shape4dNHWC::new(OC as u16, KH as u16, KW as u16, super::util::IC as u16),
        exp: 0,
    },
    // Bias
    ValueMeta {
        shape: Shape4dNHWC::new(OC as u16, 1, 1, 1),
        exp: 0,
    },
    // Output
    ValueMeta {
        shape: Shape4dNHWC::new(1, super::util::OH as u16, super::util::OW as u16, OC as u16),
        exp: -4,
    },
];

pub static VALUES_NOISE_INM4_OUTM5: [ValueMeta; 4] = [
    // Input
    ValueMeta {
        shape: Shape4dNHWC::new(
            1,
            super::util::H as u16,
            super::util::W as u16,
            super::util::IC as u16,
        ),
        exp: -4,
    },
    // Weights
    ValueMeta {
        shape: Shape4dNHWC::new(OC as u16, KH as u16, KW as u16, super::util::IC as u16),
        exp: 0,
    },
    // Bias
    ValueMeta {
        shape: Shape4dNHWC::new(OC as u16, 1, 1, 1),
        exp: 0,
    },
    // Output
    ValueMeta {
        shape: Shape4dNHWC::new(1, super::util::OH as u16, super::util::OW as u16, OC as u16),
        exp: -5,
    },
];

// BlockedN16HWC16 weights with strong negative exponents and bias.
pub static INITS_BLOCKED_NEG8_BIAS: [Initializer; 2] = [
    Initializer {
        id: W_ID,
        layout: ParamLayout::WeightsI8BlockedN16HWC16 {
            oc: OC as u16,
            kh: KH as u16,
            kw: KW as u16,
            ic: super::util::IC as u16,
        },
        data: &W_OHWI_ONES,
        exponents: ParamExponents::PerTensor(-8),
    },
    Initializer {
        id: super::util::B_ID,
        layout: ParamLayout::BiasI32 { oc: OC as u16 },
        data: &BIAS_I32_BYTES,
        exponents: ParamExponents::PerTensor(-8),
    },
];

pub static INITS_BLOCKED_NEG9_BIAS: [Initializer; 2] = [
    Initializer {
        id: W_ID,
        layout: ParamLayout::WeightsI8BlockedN16HWC16 {
            oc: OC as u16,
            kh: KH as u16,
            kw: KW as u16,
            ic: super::util::IC as u16,
        },
        data: &W_OHWI_ONES,
        exponents: ParamExponents::PerTensor(-9),
    },
    Initializer {
        id: super::util::B_ID,
        layout: ParamLayout::BiasI32 { oc: OC as u16 },
        data: &BIAS_I32_BYTES,
        exponents: ParamExponents::PerTensor(-9),
    },
];

pub static INITS_BLOCKED_NEG10_BIAS: [Initializer; 2] = [
    Initializer {
        id: W_ID,
        layout: ParamLayout::WeightsI8BlockedN16HWC16 {
            oc: OC as u16,
            kh: KH as u16,
            kw: KW as u16,
            ic: super::util::IC as u16,
        },
        data: &W_OHWI_ONES,
        exponents: ParamExponents::PerTensor(-10),
    },
    Initializer {
        id: super::util::B_ID,
        layout: ParamLayout::BiasI32 { oc: OC as u16 },
        data: &BIAS_I32_BYTES,
        exponents: ParamExponents::PerTensor(-10),
    },
];

pub static INITS_BLOCKED_NEG8_NO_BIAS: [Initializer; 1] = [Initializer {
    id: W_ID,
    layout: ParamLayout::WeightsI8BlockedN16HWC16 {
        oc: OC as u16,
        kh: KH as u16,
        kw: KW as u16,
        ic: super::util::IC as u16,
    },
    data: &W_OHWI_ONES,
    exponents: ParamExponents::PerTensor(-8),
}];

pub static SPEC_K33_NEG8_OUTM4: ModelSpec = ModelSpec {
    alignment: 16,
    inputs: &INPUTS,
    outputs: &OUTPUTS,
    values: &VALUES_NOISE_INM4_OUTM4,
    initializers: &INITS_BLOCKED_NEG8_BIAS,
    plan: &PLAN_WITH_BIAS_LINEAR,
};

pub static SPEC_K33_NEG9_OUTM4: ModelSpec = ModelSpec {
    alignment: 16,
    inputs: &INPUTS,
    outputs: &OUTPUTS,
    values: &VALUES_NOISE_INM4_OUTM4,
    initializers: &INITS_BLOCKED_NEG9_BIAS,
    plan: &PLAN_WITH_BIAS_LINEAR,
};

pub static SPEC_K33_NEG10_OUTM5: ModelSpec = ModelSpec {
    alignment: 16,
    inputs: &INPUTS,
    outputs: &OUTPUTS,
    values: &VALUES_NOISE_INM4_OUTM5,
    initializers: &INITS_BLOCKED_NEG10_BIAS,
    plan: &PLAN_WITH_BIAS_LINEAR,
};

pub static SPEC_K33_NEG8_OUTM4_NO_BIAS: ModelSpec = ModelSpec {
    alignment: 16,
    inputs: &INPUTS,
    outputs: &OUTPUTS,
    values: &VALUES_NOISE_INM4_OUTM4,
    initializers: &INITS_BLOCKED_NEG8_NO_BIAS,
    plan: &super::util::PLAN_NO_BIAS_LINEAR,
};

fn run_parity_for_spec(spec: &ModelSpec, label: &str) {
    let conv = match spec.plan[0].op {
        NodeOp::Conv2d(ref c) => c,
        _ => panic!("unexpected plan kind"),
    };

    const ARENA_TOTAL: usize = SIZE_BYTES + SCRATCH_BYTES;
    let mut arena_buf = Aligned::<ARENA_TOTAL>([0; ARENA_TOTAL]);
    let mut arena = Arena::from_buf(&mut arena_buf.0);

    // Deterministic input pattern, same as util::run_parity
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

    // Scalar reference
    conv2d_scalar(&PLAN, &mut arena, spec, conv, DEFAULT_ROUNDING);
    let mut golden = [0i8; OUT_ELEMS];
    golden.copy_from_slice(arena.value_slice(PLAN.offset_of(OUT_ID), OUT_ELEMS));

    // SIMD path
    let out_mut2 = arena.value_slice_mut(PLAN.offset_of(OUT_ID), OUT_ELEMS);
    out_mut2.fill(0);
    let outcome = try_conv2d(&PLAN, &mut arena, spec, conv, DEFAULT_ROUNDING);
    if !matches!(outcome, ExecOutcome::Executed) {
        edgedl::ne_error!(
            "conv33_noise_like_parity({}): SIMD path fell back; scalar result only (assert will fail)",
            label
        );
        assert!(
            matches!(outcome, ExecOutcome::Executed),
            "SIMD path fell back for noise-like case {}",
            label
        );
    }

    let got = arena.value_slice(PLAN.offset_of(OUT_ID), OUT_ELEMS);
    for i in 0..OUT_ELEMS {
        let a = golden[i] as i16;
        let b = got[i] as i16;
        let diff = if a > b { a - b } else { b - a };
        if diff > 1 {
            edgedl::ne_error!(
                "conv33_noise_like_parity({}): mismatch at idx={} scalar={} simd={} (assert will fail)",
                label,
                i,
                golden[i],
                got[i]
            );
            assert!(
                diff <= 1,
                "noise-like mismatch {} at {}: scalar={} simd={}",
                label,
                i,
                golden[i],
                got[i]
            );
        }
    }
}

pub fn run_group() {
    // These three specs mirror the exponent relations of the noise model's 3x3 convs.
    run_parity_for_spec(&SPEC_K33_NEG8_OUTM4, "k33_neg8_outm4");
    run_parity_for_spec(&SPEC_K33_NEG9_OUTM4, "k33_neg9_outm4");
    run_parity_for_spec(&SPEC_K33_NEG10_OUTM5, "k33_neg10_outm5");
    // Bias-free variant to isolate epilogue bias handling.
    run_parity_for_spec(&SPEC_K33_NEG8_OUTM4_NO_BIAS, "k33_neg8_outm4_no_bias");

    // Real noise-model Conv2D id=5: N=1, H=32, W=51, IC=16, OC=16, KH=3, KW=3,
    // BlockedN16HWC16 weights, bias, padding [1,1,1,1], stride [1,1], ReLU,
    // exponents: in_e=-4, w_e=-8, out_e=-4, bias_e=-8.
    run_k33_real_shape_parity();
}

// Real-shape K33 parity (noise model Conv id=5)
const REAL_N: usize = 1;
const REAL_H: usize = 32;
const REAL_W: usize = 51;
const REAL_IC: usize = 16;
const REAL_OC: usize = 16;
const REAL_KH: usize = 3;
const REAL_KW: usize = 3;
const REAL_OH: usize = 32;
const REAL_OW: usize = 51;

const REAL_IN_ID: edgedl::model::ValueId = 0;
const REAL_W_ID: edgedl::model::ValueId = 1;
const REAL_B_ID: edgedl::model::ValueId = 2;
const REAL_OUT_ID: edgedl::model::ValueId = 3;

const REAL_IN_ELEMS: usize = REAL_N * REAL_H * REAL_W * REAL_IC;
const REAL_OUT_ELEMS: usize = REAL_N * REAL_OH * REAL_OW * REAL_OC;

const REAL_OFFSETS: [usize; 4] = [0, 0, 0, REAL_IN_ELEMS];

const REAL_SIZE_BYTES: usize = REAL_IN_ELEMS + REAL_OUT_ELEMS;
const REAL_SCRATCH_BYTES: usize = 256;

static REAL_PLAN: edgedl::arena::PlannedArena = edgedl::arena::PlannedArena {
    size_bytes: REAL_SIZE_BYTES,
    offsets: &REAL_OFFSETS,
    scratch_bytes: REAL_SCRATCH_BYTES,
};

static REAL_VALUES: [ValueMeta; 4] = [
    ValueMeta {
        shape: Shape4dNHWC::new(REAL_N as u16, REAL_H as u16, REAL_W as u16, REAL_IC as u16),
        exp: -4,
    },
    ValueMeta {
        shape: Shape4dNHWC::new(
            REAL_OC as u16,
            REAL_KH as u16,
            REAL_KW as u16,
            REAL_IC as u16,
        ),
        exp: 0,
    },
    ValueMeta {
        shape: Shape4dNHWC::new(REAL_OC as u16, 1, 1, 1),
        exp: -8,
    },
    ValueMeta {
        shape: Shape4dNHWC::new(
            REAL_N as u16,
            REAL_OH as u16,
            REAL_OW as u16,
            REAL_OC as u16,
        ),
        exp: -4,
    },
];

static REAL_INITS: [Initializer; 2] = [
    Initializer {
        id: REAL_W_ID,
        layout: ParamLayout::WeightsI8BlockedN16HWC16 {
            oc: REAL_OC as u16,
            kh: REAL_KH as u16,
            kw: REAL_KW as u16,
            ic: REAL_IC as u16,
        },
        data: &W_OHWI_ONES,
        exponents: ParamExponents::PerTensor(-8),
    },
    Initializer {
        id: REAL_B_ID,
        layout: ParamLayout::BiasI32 { oc: REAL_OC as u16 },
        data: &BIAS_I32_BYTES,
        exponents: ParamExponents::PerTensor(-8),
    },
];

static REAL_INPUTS: [edgedl::model::ValueId; 1] = [REAL_IN_ID];
static REAL_OUTPUTS: [edgedl::model::ValueId; 1] = [REAL_OUT_ID];

static REAL_PLAN_NODES: [edgedl::model::NodeSpec; 1] = [edgedl::model::NodeSpec {
    op: edgedl::model::NodeOp::Conv2d(edgedl::model::Conv2dSpec {
        input: REAL_IN_ID,
        weights: REAL_W_ID,
        bias: Some(REAL_B_ID),
        output: REAL_OUT_ID,
        strides_hw: [1, 1],
        dilations_hw: [1, 1],
        pads_hw: [1, 1, 1, 1],
        activation: edgedl::model::Activation::ReLU,
        groups: 1,
    }),
}];

static REAL_SPEC: ModelSpec = ModelSpec {
    alignment: 16,
    inputs: &REAL_INPUTS,
    outputs: &REAL_OUTPUTS,
    values: &REAL_VALUES,
    initializers: &REAL_INITS,
    plan: &REAL_PLAN_NODES,
};

fn run_k33_real_shape_parity() {
    let conv = match REAL_SPEC.plan[0].op {
        NodeOp::Conv2d(ref c) => c,
        _ => panic!("unexpected plan kind"),
    };

    const ARENA_TOTAL: usize = REAL_SIZE_BYTES + REAL_SCRATCH_BYTES;
    let mut arena_buf = Aligned::<ARENA_TOTAL>([0; ARENA_TOTAL]);
    let mut arena = Arena::from_buf(&mut arena_buf.0);

    let mut input = [0i8; REAL_IN_ELEMS];
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

    let in_mut = arena.value_slice_mut(REAL_PLAN.offset_of(REAL_IN_ID), REAL_IN_ELEMS);
    in_mut.copy_from_slice(&input);
    let out_mut = arena.value_slice_mut(REAL_PLAN.offset_of(REAL_OUT_ID), REAL_OUT_ELEMS);
    out_mut.fill(0);

    conv2d_scalar(&REAL_PLAN, &mut arena, &REAL_SPEC, conv, DEFAULT_ROUNDING);
    let mut golden = [0i8; REAL_OUT_ELEMS];
    golden.copy_from_slice(arena.value_slice(REAL_PLAN.offset_of(REAL_OUT_ID), REAL_OUT_ELEMS));

    let out_mut2 = arena.value_slice_mut(REAL_PLAN.offset_of(REAL_OUT_ID), REAL_OUT_ELEMS);
    out_mut2.fill(0);
    let outcome = try_conv2d(&REAL_PLAN, &mut arena, &REAL_SPEC, conv, DEFAULT_ROUNDING);
    if !matches!(outcome, ExecOutcome::Executed) {
        edgedl::ne_error!(
            "conv33_noise_like_real: SIMD path fell back; scalar result only (assert will fail)",
        );
        assert!(matches!(outcome, ExecOutcome::Executed));
    }

    let got = arena.value_slice(REAL_PLAN.offset_of(REAL_OUT_ID), REAL_OUT_ELEMS);
    for i in 0..REAL_OUT_ELEMS {
        let a = golden[i] as i16;
        let b = got[i] as i16;
        let diff = if a > b { a - b } else { b - a };
        if diff > 1 {
            edgedl::ne_error!(
                "conv33_noise_like_real: mismatch at idx={} scalar={} simd={} (assert will fail)",
                i,
                golden[i],
                got[i]
            );
            assert!(
                diff <= 1,
                "noise-like real-shape mismatch at {}: scalar={} simd={}",
                i,
                golden[i],
                got[i]
            );
        }
    }
}
