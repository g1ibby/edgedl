use edgedl::{
    arena::{Arena, PlannedArena},
    kernels::{
        conv2d::conv2d_scalar,
        simd::conv2d::{ExecOutcome, try_conv2d},
    },
    model::{
        Activation,
        Conv2dSpec,
        Initializer,
        ModelSpec,
        NodeOp,
        NodeSpec,
        ParamExponents,
        ParamLayout,
        Shape4dNHWC,
        ValueId,
        ValueMeta,
    },
    rounding::DEFAULT_ROUNDING,
};

// Fixed shapes for this suite
pub const N: usize = 1;
pub const H: usize = 5;
pub const W: usize = 5;
pub const IC: usize = 16;
pub const OC: usize = 16;
pub const KH: usize = 3;
pub const KW: usize = 3;

// Output size for valid padding: OH = H - KH + 1 = 5 - 3 + 1 = 3
pub const OH: usize = 3;
pub const OW: usize = 3;

pub const IN_ELEMS: usize = N * H * W * IC;
pub const OUT_ELEMS: usize = N * OH * OW * OC;

// Value IDs
pub const IN_ID: ValueId = 0;
pub const W_ID: ValueId = 1;
pub const B_ID: ValueId = 2;
pub const OUT_ID: ValueId = 3;

// Planned arena
const OFFSETS: [usize; 4] = [
    0,        // input
    0,        // weights (not in arena)
    0,        // bias    (not in arena)
    IN_ELEMS, // output
];

pub const SIZE_BYTES: usize = IN_ELEMS + OUT_ELEMS;
pub const SCRATCH_BYTES: usize = 64; // 16*2 + 16*2 (scales + bias16)

pub static PLAN: PlannedArena = PlannedArena {
    size_bytes: SIZE_BYTES,
    offsets: &OFFSETS,
    scratch_bytes: SCRATCH_BYTES,
};

// Values tables
pub static VALUES_OUT_EXP_0: [ValueMeta; 4] = [
    ValueMeta {
        shape: Shape4dNHWC::new(1, H as u16, W as u16, IC as u16),
        exp: 0,
    },
    ValueMeta {
        shape: Shape4dNHWC::new(OC as u16, 3, 3, IC as u16),
        exp: 0,
    },
    ValueMeta {
        shape: Shape4dNHWC::new(OC as u16, 1, 1, 1),
        exp: 0,
    },
    ValueMeta {
        shape: Shape4dNHWC::new(1, OH as u16, OW as u16, OC as u16),
        exp: 0,
    },
];

pub static VALUES_OUT_EXP_2: [ValueMeta; 4] = [
    ValueMeta {
        shape: Shape4dNHWC::new(1, H as u16, W as u16, IC as u16),
        exp: 0,
    },
    ValueMeta {
        shape: Shape4dNHWC::new(OC as u16, 3, 3, IC as u16),
        exp: 0,
    },
    ValueMeta {
        shape: Shape4dNHWC::new(OC as u16, 1, 1, 1),
        exp: 0,
    },
    ValueMeta {
        shape: Shape4dNHWC::new(1, OH as u16, OW as u16, OC as u16),
        exp: 2,
    },
];

pub static VALUES_STRIDE_2: [ValueMeta; 4] = [
    ValueMeta {
        shape: Shape4dNHWC::new(1, H as u16, W as u16, IC as u16),
        exp: 0,
    },
    ValueMeta {
        shape: Shape4dNHWC::new(OC as u16, 3, 3, IC as u16),
        exp: 0,
    },
    ValueMeta {
        shape: Shape4dNHWC::new(OC as u16, 1, 1, 1),
        exp: 0,
    },
    ValueMeta {
        shape: Shape4dNHWC::new(1, 2, 2, OC as u16), // OH=2, OW=2 for stride 2
        exp: 0,
    },
];

pub static VALUES_DILATION_2: [ValueMeta; 4] = [
    ValueMeta {
        shape: Shape4dNHWC::new(1, H as u16, W as u16, IC as u16),
        exp: 0,
    },
    ValueMeta {
        shape: Shape4dNHWC::new(OC as u16, 3, 3, IC as u16),
        exp: 0,
    },
    ValueMeta {
        shape: Shape4dNHWC::new(OC as u16, 1, 1, 1),
        exp: 0,
    },
    ValueMeta {
        shape: Shape4dNHWC::new(1, 1, 1, OC as u16), // OH=1, OW=1 for dilation 2
        exp: 0,
    },
];

pub static INPUTS: [ValueId; 1] = [IN_ID];
pub static OUTPUTS: [ValueId; 1] = [OUT_ID];

// Static parameters
// Weights (OHWI): use all-ones for simplicity
pub static W_OHWI_ONES: [u8; OC * KH * KW * IC] = [1u8; OC * KH * KW * IC];

// Bias (I32)
pub static BIAS_I32_BYTES: [u8; OC * 4] = {
    let mut bytes = [0u8; OC * 4];
    let mut i = 0usize;
    while i < OC {
        let v: i32 = ((i as i32) + 1) * 10;
        let le = v.to_le_bytes();
        bytes[i * 4 + 0] = le[0];
        bytes[i * 4 + 1] = le[1];
        bytes[i * 4 + 2] = le[2];
        bytes[i * 4 + 3] = le[3];
        i += 1;
    }
    bytes
};

// Per-channel weight exponent table
pub static W_EXP_PC: [i8; OC] = [0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0];

// NodeSpec variants
const CONV_BASE_NO_BIAS_LINEAR: NodeSpec = NodeSpec {
    op: NodeOp::Conv2d(Conv2dSpec {
        input: IN_ID,
        weights: W_ID,
        bias: None,
        output: OUT_ID,
        strides_hw: [1, 1],
        dilations_hw: [1, 1],
        pads_hw: [0, 0, 0, 0],
        activation: Activation::Linear,
        groups: 1,
    }),
};
const CONV_BASE_NO_BIAS_RELU: NodeSpec = NodeSpec {
    op: NodeOp::Conv2d(Conv2dSpec {
        input: IN_ID,
        weights: W_ID,
        bias: None,
        output: OUT_ID,
        strides_hw: [1, 1],
        dilations_hw: [1, 1],
        pads_hw: [0, 0, 0, 0],
        activation: Activation::ReLU,
        groups: 1,
    }),
};
const CONV_BASE_WITH_BIAS_LINEAR: NodeSpec = NodeSpec {
    op: NodeOp::Conv2d(Conv2dSpec {
        input: IN_ID,
        weights: W_ID,
        bias: Some(B_ID),
        output: OUT_ID,
        strides_hw: [1, 1],
        dilations_hw: [1, 1],
        pads_hw: [0, 0, 0, 0],
        activation: Activation::Linear,
        groups: 1,
    }),
};
const CONV_BASE_WITH_BIAS_RELU: NodeSpec = NodeSpec {
    op: NodeOp::Conv2d(Conv2dSpec {
        input: IN_ID,
        weights: W_ID,
        bias: Some(B_ID),
        output: OUT_ID,
        strides_hw: [1, 1],
        dilations_hw: [1, 1],
        pads_hw: [0, 0, 0, 0],
        activation: Activation::ReLU,
        groups: 1,
    }),
};

pub static PLAN_NO_BIAS_LINEAR: [NodeSpec; 1] = [CONV_BASE_NO_BIAS_LINEAR];
pub static PLAN_NO_BIAS_RELU: [NodeSpec; 1] = [CONV_BASE_NO_BIAS_RELU];
pub static PLAN_WITH_BIAS_LINEAR: [NodeSpec; 1] = [CONV_BASE_WITH_BIAS_LINEAR];
pub static PLAN_WITH_BIAS_RELU: [NodeSpec; 1] = [CONV_BASE_WITH_BIAS_RELU];

// Stride 2 variants
const CONV_S2_NO_BIAS_LINEAR: NodeSpec = NodeSpec {
    op: NodeOp::Conv2d(Conv2dSpec {
        input: IN_ID,
        weights: W_ID,
        bias: None,
        output: OUT_ID,
        strides_hw: [2, 2],
        dilations_hw: [1, 1],
        pads_hw: [0, 0, 0, 0],
        activation: Activation::Linear,
        groups: 1,
    }),
};
const CONV_S2_WITH_BIAS_LINEAR: NodeSpec = NodeSpec {
    op: NodeOp::Conv2d(Conv2dSpec {
        input: IN_ID,
        weights: W_ID,
        bias: Some(B_ID),
        output: OUT_ID,
        strides_hw: [2, 2],
        dilations_hw: [1, 1],
        pads_hw: [0, 0, 0, 0],
        activation: Activation::Linear,
        groups: 1,
    }),
};
pub static PLAN_S2_NO_BIAS_LINEAR: [NodeSpec; 1] = [CONV_S2_NO_BIAS_LINEAR];
pub static PLAN_S2_WITH_BIAS_LINEAR: [NodeSpec; 1] = [CONV_S2_WITH_BIAS_LINEAR];

// Dilation 2 variants
const CONV_D2_NO_BIAS_LINEAR: NodeSpec = NodeSpec {
    op: NodeOp::Conv2d(Conv2dSpec {
        input: IN_ID,
        weights: W_ID,
        bias: None,
        output: OUT_ID,
        strides_hw: [1, 1],
        dilations_hw: [2, 2],
        pads_hw: [0, 0, 0, 0],
        activation: Activation::Linear,
        groups: 1,
    }),
};
const CONV_D2_WITH_BIAS_LINEAR: NodeSpec = NodeSpec {
    op: NodeOp::Conv2d(Conv2dSpec {
        input: IN_ID,
        weights: W_ID,
        bias: Some(B_ID),
        output: OUT_ID,
        strides_hw: [1, 1],
        dilations_hw: [2, 2],
        pads_hw: [0, 0, 0, 0],
        activation: Activation::Linear,
        groups: 1,
    }),
};
pub static PLAN_D2_NO_BIAS_LINEAR: [NodeSpec; 1] = [CONV_D2_NO_BIAS_LINEAR];
pub static PLAN_D2_WITH_BIAS_LINEAR: [NodeSpec; 1] = [CONV_D2_WITH_BIAS_LINEAR];

#[derive(Copy, Clone)]
pub enum Layout {
    OHWI,
    BlockedN16HWC16,
}
#[derive(Copy, Clone)]
pub enum WeightExps {
    PerTensor0,
    PerChannel,
}

pub static INITS_OHWI_PT: [Initializer; 1] = [Initializer {
    id: W_ID,
    layout: ParamLayout::WeightsI8OHWI {
        oc: OC as u16,
        kh: KH as u16,
        kw: KW as u16,
        ic: IC as u16,
    },
    data: &W_OHWI_ONES,
    exponents: ParamExponents::PerTensor(0),
}];
pub static INITS_OHWI_PT_BIAS: [Initializer; 2] = [
    INITS_OHWI_PT[0],
    Initializer {
        id: B_ID,
        layout: ParamLayout::BiasI32 { oc: OC as u16 },
        data: &BIAS_I32_BYTES,
        exponents: ParamExponents::PerTensor(0),
    },
];
pub static INITS_OHWI_PC: [Initializer; 1] = [Initializer {
    id: W_ID,
    layout: ParamLayout::WeightsI8OHWI {
        oc: OC as u16,
        kh: KH as u16,
        kw: KW as u16,
        ic: IC as u16,
    },
    data: &W_OHWI_ONES,
    exponents: ParamExponents::PerChannel(&W_EXP_PC),
}];
pub static INITS_OHWI_PC_BIAS: [Initializer; 2] = [
    INITS_OHWI_PC[0],
    Initializer {
        id: B_ID,
        layout: ParamLayout::BiasI32 { oc: OC as u16 },
        data: &BIAS_I32_BYTES,
        exponents: ParamExponents::PerTensor(0),
    },
];

pub static INITS_BLOCKED_PT: [Initializer; 1] = [Initializer {
    id: W_ID,
    layout: ParamLayout::WeightsI8BlockedN16HWC16 {
        oc: OC as u16,
        kh: KH as u16,
        kw: KW as u16,
        ic: IC as u16,
    },
    data: &W_OHWI_ONES,
    exponents: ParamExponents::PerTensor(0),
}];
pub static INITS_BLOCKED_PT_BIAS: [Initializer; 2] = [
    INITS_BLOCKED_PT[0],
    Initializer {
        id: B_ID,
        layout: ParamLayout::BiasI32 { oc: OC as u16 },
        data: &BIAS_I32_BYTES,
        exponents: ParamExponents::PerTensor(0),
    },
];

pub struct Case {
    pub layout: Layout,
    pub w_exps: WeightExps,
    pub has_bias: bool,
    pub activation: Activation,
    pub out_exp: i8,
    pub stride: usize,
    pub dilation: usize,
}

pub fn build_spec(case: &Case) -> ModelSpec {
    let values: &'static [ValueMeta] = if case.stride == 2 {
        &VALUES_STRIDE_2
    } else if case.dilation == 2 {
        &VALUES_DILATION_2
    } else if case.out_exp == 2 {
        &VALUES_OUT_EXP_2
    } else {
        &VALUES_OUT_EXP_0
    };

    // Note: The static PLAN_* specs have fixed strides/dilations [1,1].
    // If we want to support variable strides/dilations, we can't use the static PLAN_* arrays
    // directly if they are hardcoded.
    // However, `NodeSpec` contains `Conv2dSpec` which has `strides_hw`.
    // We need to construct the `NodeSpec` dynamically or have variants.
    // Since `ModelSpec` requires a `'static` reference to `plan`, we can't easily create it on the
    // stack unless we leak it or use a static cell.
    // But wait, `ModelSpec` fields are references.

    // Let's define a few more static plans for common stride/dilation combos?
    // Or use `mk_static!` macro if available (it is in lib.rs).

    // For simplicity, let's add specific static plans for the test cases we added.
    // Stride=2, Dilation=1
    // Stride=1, Dilation=2

    let plan_slice: &'static [NodeSpec] =
        match (case.has_bias, case.activation, case.stride, case.dilation) {
            (false, Activation::Linear, 1, 1) => &PLAN_NO_BIAS_LINEAR,
            (false, Activation::ReLU, 1, 1) => &PLAN_NO_BIAS_RELU,
            (true, Activation::Linear, 1, 1) => &PLAN_WITH_BIAS_LINEAR,
            (true, Activation::ReLU, 1, 1) => &PLAN_WITH_BIAS_RELU,

            (false, Activation::Linear, 2, 1) => &PLAN_S2_NO_BIAS_LINEAR,
            (true, Activation::Linear, 2, 1) => &PLAN_S2_WITH_BIAS_LINEAR,

            (false, Activation::Linear, 1, 2) => &PLAN_D2_NO_BIAS_LINEAR,
            (true, Activation::Linear, 1, 2) => &PLAN_D2_WITH_BIAS_LINEAR,

            _ => panic!("Unsupported case combination"),
        };
    let inits: &'static [Initializer] = match (case.layout, case.w_exps, case.has_bias) {
        (Layout::OHWI, WeightExps::PerTensor0, false) => &INITS_OHWI_PT,
        (Layout::OHWI, WeightExps::PerTensor0, true) => &INITS_OHWI_PT_BIAS,
        (Layout::OHWI, WeightExps::PerChannel, false) => &INITS_OHWI_PC,
        (Layout::OHWI, WeightExps::PerChannel, true) => &INITS_OHWI_PC_BIAS,
        (Layout::BlockedN16HWC16, WeightExps::PerTensor0, false) => &INITS_BLOCKED_PT,
        (Layout::BlockedN16HWC16, WeightExps::PerTensor0, true) => &INITS_BLOCKED_PT_BIAS,
        (Layout::BlockedN16HWC16, WeightExps::PerChannel, _) => {
            panic!("Blocked per-channel not impl in util")
        }
    };
    ModelSpec {
        alignment: 16,
        inputs: &INPUTS,
        outputs: &OUTPUTS,
        values,
        initializers: inits,
        plan: plan_slice,
    }
}

#[repr(align(16))]
pub struct Aligned<const N: usize>(pub [i8; N]);

pub fn run_parity(case: &Case) {
    let spec = build_spec(case);
    let conv = match spec.plan[0].op {
        NodeOp::Conv2d(ref c) => c,
        _ => panic!("unexpected plan kind"),
    };

    const ARENA_TOTAL: usize = SIZE_BYTES + SCRATCH_BYTES;
    let mut arena_buf = Aligned::<ARENA_TOTAL>([0; ARENA_TOTAL]);
    let mut arena = Arena::from_buf(&mut arena_buf.0);

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

    conv2d_scalar(&PLAN, &mut arena, &spec, conv, DEFAULT_ROUNDING);
    let mut golden = [0i8; OUT_ELEMS];
    golden.copy_from_slice(arena.value_slice(PLAN.offset_of(OUT_ID), OUT_ELEMS));

    let out_mut2 = arena.value_slice_mut(PLAN.offset_of(OUT_ID), OUT_ELEMS);
    out_mut2.fill(0);
    let outcome = try_conv2d(&PLAN, &mut arena, &spec, conv, DEFAULT_ROUNDING);
    if !matches!(outcome, ExecOutcome::Executed) {
        edgedl::ne_error!(
            "conv33_parity: SIMD path fell back; scalar result only (assert will fail)"
        );
        assert!(
            matches!(outcome, ExecOutcome::Executed),
            "SIMD path fell back for case"
        );
    }

    let got = arena.value_slice(PLAN.offset_of(OUT_ID), OUT_ELEMS);
    for i in 0..OUT_ELEMS {
        let a = golden[i] as i16;
        let b = got[i] as i16;
        let diff = if a > b { a - b } else { b - a };
        if diff > 1 {
            edgedl::ne_error!(
                "conv33_parity: scalar/SIMD mismatch at idx={} scalar={} simd={} (assert will fail)",
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
}
