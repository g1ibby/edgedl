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

use super::util::Aligned;

// Common constants for unaligned tests
// DEBUG: Using KH=2, KW=1 to get Khw family with only 2 taps
const N: usize = 1;
const H: usize = 7;
const W: usize = 7;
const KH: usize = 2; // Was 5, now 2 for debugging (forces Khw, not K11)
const KW: usize = 1; // Was 5, now 1 for debugging
const OH: usize = 6; // H - KH + 1 = 7 - 2 + 1 = 6
const OW: usize = 7; // W - KW + 1 = 7 - 1 + 1 = 7

const IN_ID: ValueId = 0;
const W_ID: ValueId = 1;
const OUT_ID: ValueId = 2;

// Helper to ensure 16-byte alignment for SIMD
const fn ceil16(x: usize) -> usize {
    (x + 15) & !15
}

// IC Tail Case - testing with IC=17 (unaligned) and KH=1,KW=1 (single tap)
mod ic_tail {
    use super::*;

    const IC: usize = 17; // Unaligned IC to test remainder path
    const OC: usize = 16;
    const IN_ELEMS: usize = N * H * W * IC;
    const OUT_ELEMS: usize = N * OH * OW * OC;

    // Offsets must be 16-byte aligned for TIE728 SIMD
    const IN_ELEMS_ALIGNED: usize = ceil16(IN_ELEMS);
    const OFFSETS: [usize; 3] = [0, 0, IN_ELEMS_ALIGNED];
    const SIZE: usize = IN_ELEMS_ALIGNED + OUT_ELEMS;
    const SCRATCH_BYTES: usize = 128;

    static PLAN: PlannedArena = PlannedArena {
        size_bytes: SIZE,
        offsets: &OFFSETS,
        scratch_bytes: SCRATCH_BYTES,
    };

    static VALUES: [ValueMeta; 3] = [
        ValueMeta {
            shape: Shape4dNHWC::new(N as u16, H as u16, W as u16, IC as u16),
            exp: 0,
        },
        ValueMeta {
            shape: Shape4dNHWC::new(1, 1, 1, 1),
            exp: 0,
        },
        ValueMeta {
            shape: Shape4dNHWC::new(N as u16, OH as u16, OW as u16, OC as u16),
            exp: 0,
        },
    ];

    fn build_spec() -> ModelSpec {
        static W_ONES: [u8; IC * OC * KH * KW] = [1u8; IC * OC * KH * KW];
        static INITS: [Initializer; 1] = [Initializer {
            id: W_ID,
            layout: ParamLayout::WeightsI8OHWI {
                oc: OC as u16,
                kh: KH as u16,
                kw: KW as u16,
                ic: IC as u16,
            },
            data: &W_ONES,
            exponents: ParamExponents::PerTensor(0),
        }];
        static INPUTS: [ValueId; 1] = [IN_ID];
        static OUTPUTS: [ValueId; 1] = [OUT_ID];
        static PLAN_NODES: [NodeSpec; 1] = [NodeSpec {
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
        }];

        ModelSpec {
            alignment: 16,
            inputs: &INPUTS,
            outputs: &OUTPUTS,
            values: &VALUES,
            initializers: &INITS,
            plan: &PLAN_NODES,
        }
    }

    pub fn run() {
        let spec = build_spec();
        const ARENA_TOTAL: usize = SIZE + SCRATCH_BYTES;
        // Use a static arena buffer to avoid large stack allocations on MCU.
        let arena_buf: &mut Aligned<ARENA_TOTAL> =
            hil_test::mk_static!(Aligned<ARENA_TOTAL>, Aligned([0; ARENA_TOTAL]));
        let mut arena = Arena::from_buf(&mut arena_buf.0);

        let mut input = [0i8; IN_ELEMS];
        for (i, v) in input.iter_mut().enumerate() {
            // *v = (i as i8 % 7) - 3;
            *v = ((i % 7) as i8) - 3;
        }

        let in_mut = arena.value_slice_mut(PLAN.offset_of(IN_ID), IN_ELEMS);
        in_mut.copy_from_slice(&input);
        arena
            .value_slice_mut(PLAN.offset_of(OUT_ID), OUT_ELEMS)
            .fill(0);

        let conv = match spec.plan[0].op {
            NodeOp::Conv2d(ref c) => c,
            _ => unreachable!(),
        };

        conv2d_scalar(&PLAN, &mut arena, &spec, conv, DEFAULT_ROUNDING);
        let mut golden = [0i8; OUT_ELEMS];
        golden.copy_from_slice(arena.value_slice(PLAN.offset_of(OUT_ID), OUT_ELEMS));

        arena
            .value_slice_mut(PLAN.offset_of(OUT_ID), OUT_ELEMS)
            .fill(0);
        let simd = try_conv2d(&PLAN, &mut arena, &spec, conv, DEFAULT_ROUNDING);

        if !matches!(simd, ExecOutcome::Executed) {
            edgedl::ne_error!("conv_hwcn_unaligned (ic_tail): SIMD path fell back");
            assert!(matches!(simd, ExecOutcome::Executed));
        }

        let got = arena.value_slice(PLAN.offset_of(OUT_ID), OUT_ELEMS);
        for i in 0..OUT_ELEMS {
            let a = golden[i] as i16;
            let b = got[i] as i16;
            let d = (a - b).abs();
            if d > 1 {
                edgedl::ne_error!(
                    "conv_hwcn_unaligned (ic_tail): mismatch at idx={} scalar={} simd={}",
                    i,
                    golden[i],
                    got[i]
                );
                assert!(d <= 1);
            }
        }
    }
}

// OC Tail Case
mod oc_tail {
    use super::*;

    const IC: usize = 16;
    const OC: usize = 17; // 16 + 1
    const IN_ELEMS: usize = N * H * W * IC;
    const OUT_ELEMS: usize = N * OH * OW * OC;

    // Offsets must be 16-byte aligned for TIE728 SIMD
    const IN_ELEMS_ALIGNED: usize = ceil16(IN_ELEMS);
    const OFFSETS: [usize; 3] = [0, 0, IN_ELEMS_ALIGNED];
    const SIZE: usize = IN_ELEMS_ALIGNED + OUT_ELEMS;
    const SCRATCH_BYTES: usize = 128;

    static PLAN: PlannedArena = PlannedArena {
        size_bytes: SIZE,
        offsets: &OFFSETS,
        scratch_bytes: SCRATCH_BYTES,
    };

    static VALUES: [ValueMeta; 3] = [
        ValueMeta {
            shape: Shape4dNHWC::new(N as u16, H as u16, W as u16, IC as u16),
            exp: 0,
        },
        ValueMeta {
            shape: Shape4dNHWC::new(1, 1, 1, 1),
            exp: 0,
        },
        ValueMeta {
            shape: Shape4dNHWC::new(N as u16, OH as u16, OW as u16, OC as u16),
            exp: 0,
        },
    ];

    fn build_spec() -> ModelSpec {
        static W_ONES: [u8; IC * OC * KH * KW] = [1u8; IC * OC * KH * KW];
        static INITS: [Initializer; 1] = [Initializer {
            id: W_ID,
            layout: ParamLayout::WeightsI8OHWI {
                oc: OC as u16,
                kh: KH as u16,
                kw: KW as u16,
                ic: IC as u16,
            },
            data: &W_ONES,
            exponents: ParamExponents::PerTensor(0),
        }];
        static INPUTS: [ValueId; 1] = [IN_ID];
        static OUTPUTS: [ValueId; 1] = [OUT_ID];
        static PLAN_NODES: [NodeSpec; 1] = [NodeSpec {
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
        }];

        ModelSpec {
            alignment: 16,
            inputs: &INPUTS,
            outputs: &OUTPUTS,
            values: &VALUES,
            initializers: &INITS,
            plan: &PLAN_NODES,
        }
    }

    pub fn run() {
        let spec = build_spec();
        const ARENA_TOTAL: usize = SIZE + SCRATCH_BYTES;
        let mut arena_buf = Aligned::<ARENA_TOTAL>([0; ARENA_TOTAL]);
        let mut arena = Arena::from_buf(&mut arena_buf.0);

        let mut input = [0i8; IN_ELEMS];
        for (i, v) in input.iter_mut().enumerate() {
            *v = (i as i8 % 5) - 2;
        }

        let in_mut = arena.value_slice_mut(PLAN.offset_of(IN_ID), IN_ELEMS);
        in_mut.copy_from_slice(&input);
        arena
            .value_slice_mut(PLAN.offset_of(OUT_ID), OUT_ELEMS)
            .fill(0);

        let conv = match spec.plan[0].op {
            NodeOp::Conv2d(ref c) => c,
            _ => unreachable!(),
        };

        conv2d_scalar(&PLAN, &mut arena, &spec, conv, DEFAULT_ROUNDING);
        let mut golden = [0i8; OUT_ELEMS];
        golden.copy_from_slice(arena.value_slice(PLAN.offset_of(OUT_ID), OUT_ELEMS));

        arena
            .value_slice_mut(PLAN.offset_of(OUT_ID), OUT_ELEMS)
            .fill(0);
        let simd = try_conv2d(&PLAN, &mut arena, &spec, conv, DEFAULT_ROUNDING);

        if !matches!(simd, ExecOutcome::Executed) {
            edgedl::ne_error!("conv_hwcn_unaligned (oc_tail): SIMD path fell back");
            assert!(matches!(simd, ExecOutcome::Executed));
        }

        let got = arena.value_slice(PLAN.offset_of(OUT_ID), OUT_ELEMS);
        for i in 0..OUT_ELEMS {
            let a = golden[i] as i16;
            let b = got[i] as i16;
            let d = (a - b).abs();
            if d > 1 {
                edgedl::ne_error!(
                    "conv_hwcn_unaligned (oc_tail): mismatch at idx={} scalar={} simd={}",
                    i,
                    golden[i],
                    got[i]
                );
                assert!(d <= 1);
            }
        }
    }
}

// Noise-model-like first conv (IC=1, OC=16, 5x5, stride 2, SAME padding, blocked weights)
mod noise_like_blocked_ic1 {
    use super::*;

    const N: usize = 1;
    const H: usize = 64;
    const W: usize = 101;
    const IC: usize = 1;
    const OC: usize = 16;
    const KH: usize = 5;
    const KW: usize = 5;

    // Output shape for stride 2 with SAME padding [2,2,2,2]
    // OH = floor((H + pt + pb - (KH - 1) - 1) / stride) + 1
    //    = floor((64 + 4 - 4) / 2) + 1 = 32
    // OW = floor((101 + 4 - 4) / 2) + 1 = 51
    const OH: usize = 32;
    const OW: usize = 51;

    const IN_ID: ValueId = 0;
    const W_ID: ValueId = 1;
    const B_ID: ValueId = 2;
    const OUT_ID: ValueId = 3;

    const IN_ELEMS: usize = N * H * W * IC;
    const OUT_ELEMS: usize = N * OH * OW * OC;

    // Offsets must be 16-byte aligned for TIE728 SIMD
    const IN_ELEMS_ALIGNED: usize = ceil16(IN_ELEMS);
    const OFFSETS: [usize; 4] = [
        0,                // input
        0,                // weights (not in arena)
        0,                // bias (not in arena)
        IN_ELEMS_ALIGNED, // output
    ];

    const SIZE: usize = IN_ELEMS_ALIGNED + OUT_ELEMS;
    const SCRATCH_BYTES: usize = 256;

    static PLAN: PlannedArena = PlannedArena {
        size_bytes: SIZE,
        offsets: &OFFSETS,
        scratch_bytes: SCRATCH_BYTES,
    };

    static VALUES: [ValueMeta; 4] = [
        // Input: [N, H, W, C] = [1, 64, 101, 1]
        ValueMeta {
            shape: Shape4dNHWC::new(N as u16, H as u16, W as u16, IC as u16),
            exp: 0,
        },
        // Weights meta: [OC, KH, KW, IC]
        ValueMeta {
            shape: Shape4dNHWC::new(OC as u16, KH as u16, KW as u16, IC as u16),
            exp: 0,
        },
        // Bias meta: [OC, 1, 1, 1]
        ValueMeta {
            shape: Shape4dNHWC::new(OC as u16, 1, 1, 1),
            exp: 0,
        },
        // Output: [1, OH, OW, OC] = [1, 32, 51, 16]
        ValueMeta {
            shape: Shape4dNHWC::new(N as u16, OH as u16, OW as u16, OC as u16),
            exp: 0,
        },
    ];

    static INPUTS: [ValueId; 1] = [IN_ID];
    static OUTPUTS: [ValueId; 1] = [OUT_ID];

    const TILES: usize = OC / 16;
    const W_ELEMS: usize = KH * KW * IC * 16 * TILES;

    // BlockedN16HWC16 weights buffer; contents are all ones for simplicity.
    static W_BLOCKED_ONES: [u8; W_ELEMS] = [1u8; W_ELEMS];

    // Bias (I32), simple 10, 20, ... pattern
    static BIAS_I32_BYTES: [u8; OC * 4] = {
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

    static INITS: [Initializer; 2] = [
        Initializer {
            id: W_ID,
            layout: ParamLayout::WeightsI8BlockedN16HWC16 {
                oc: OC as u16,
                kh: KH as u16,
                kw: KW as u16,
                ic: IC as u16,
            },
            data: &W_BLOCKED_ONES,
            exponents: ParamExponents::PerTensor(0),
        },
        Initializer {
            id: B_ID,
            layout: ParamLayout::BiasI32 { oc: OC as u16 },
            data: &BIAS_I32_BYTES,
            exponents: ParamExponents::PerTensor(0),
        },
    ];

    static PLAN_NODES: [NodeSpec; 1] = [NodeSpec {
        op: NodeOp::Conv2d(Conv2dSpec {
            input: IN_ID,
            weights: W_ID,
            bias: Some(B_ID),
            output: OUT_ID,
            strides_hw: [2, 2],
            dilations_hw: [1, 1],
            pads_hw: [2, 2, 2, 2],
            activation: Activation::ReLU,
            groups: 1,
        }),
    }];

    fn build_spec() -> ModelSpec {
        ModelSpec {
            alignment: 16,
            inputs: &INPUTS,
            outputs: &OUTPUTS,
            values: &VALUES,
            initializers: &INITS,
            plan: &PLAN_NODES,
        }
    }

    pub fn run() {
        let spec = build_spec();
        const ARENA_TOTAL: usize = SIZE + SCRATCH_BYTES;
        let mut arena_buf = Aligned::<ARENA_TOTAL>([0; ARENA_TOTAL]);
        let mut arena = Arena::from_buf(&mut arena_buf.0);

        // Input pattern similar to other HIL tests
        let mut input = [0i8; IN_ELEMS];
        for (i, v) in input.iter_mut().enumerate() {
            *v = ((i % 7) as i8) - 3;
        }

        let in_mut = arena.value_slice_mut(PLAN.offset_of(IN_ID), IN_ELEMS);
        in_mut.copy_from_slice(&input);
        arena
            .value_slice_mut(PLAN.offset_of(OUT_ID), OUT_ELEMS)
            .fill(0);

        let conv = match spec.plan[0].op {
            NodeOp::Conv2d(ref c) => c,
            _ => unreachable!(),
        };

        // Scalar golden
        conv2d_scalar(&PLAN, &mut arena, &spec, conv, DEFAULT_ROUNDING);
        let mut golden = [0i8; OUT_ELEMS];
        golden.copy_from_slice(arena.value_slice(PLAN.offset_of(OUT_ID), OUT_ELEMS));

        // SIMD path
        arena
            .value_slice_mut(PLAN.offset_of(OUT_ID), OUT_ELEMS)
            .fill(0);
        let simd = try_conv2d(&PLAN, &mut arena, &spec, conv, DEFAULT_ROUNDING);

        if !matches!(simd, ExecOutcome::Executed) {
            edgedl::ne_error!("conv_hwcn_unaligned (noise_like_blocked_ic1): SIMD path fell back");
            assert!(matches!(simd, ExecOutcome::Executed));
        }

        let got = arena.value_slice(PLAN.offset_of(OUT_ID), OUT_ELEMS);
        for i in 0..OUT_ELEMS {
            let a = golden[i] as i16;
            let b = got[i] as i16;
            let d = (a - b).abs();
            if d > 1 {
                edgedl::ne_error!(
                    "conv_hwcn_unaligned (noise_like_blocked_ic1): mismatch at idx={} scalar={} simd={}",
                    i,
                    golden[i],
                    got[i]
                );
                assert!(d <= 1);
            }
        }
    }
}

pub fn run_group() {
    // Run all unaligned test cases - ic_tail first for simpler debugging
    ic_tail::run();
    oc_tail::run();
    noise_like_blocked_ic1::run();
}
