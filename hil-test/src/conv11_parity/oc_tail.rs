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

const N: usize = 1;
const H: usize = 2;
const W: usize = 2;
const IC: usize = 16;
const OC: usize = 17; // OC tail = 1

const IN_ELEMS: usize = N * H * W * IC;
const OUT_ELEMS: usize = N * H * W * OC;

const IN_ID: ValueId = 0;
const W_ID: ValueId = 1;
const B_ID: ValueId = 2;
const OUT_ID: ValueId = 3;

// Aligned plan
const OFFSETS_ALIGNED: [usize; 4] = [
    0,        // input
    0,        // weights (not in arena)
    0,        // bias    (not in arena)
    IN_ELEMS, // output
];
const SIZE_ALIGNED: usize = IN_ELEMS + OUT_ELEMS;
const SCRATCH_BYTES: usize = 128;

static PLAN_ALIGNED: PlannedArena = PlannedArena {
    size_bytes: SIZE_ALIGNED,
    offsets: &OFFSETS_ALIGNED,
    scratch_bytes: SCRATCH_BYTES,
};

// Misaligned output base by +1 to exercise unaligned stores
const OFFSETS_MISALIGNED: [usize; 4] = [0, 0, 0, IN_ELEMS + 1];
const SIZE_MISALIGNED: usize = IN_ELEMS + OUT_ELEMS + 1;

static PLAN_MISALIGNED: PlannedArena = PlannedArena {
    size_bytes: SIZE_MISALIGNED,
    offsets: &OFFSETS_MISALIGNED,
    scratch_bytes: SCRATCH_BYTES,
};

// Value metadata table (fixed shapes/exp)
static VALUES: [ValueMeta; 4] = [
    ValueMeta {
        shape: Shape4dNHWC::new(N as u16, H as u16, W as u16, IC as u16),
        exp: 0,
    },
    ValueMeta {
        shape: Shape4dNHWC::new(1, 1, 1, 1),
        exp: 0,
    },
    ValueMeta {
        shape: Shape4dNHWC::new(1, 1, 1, 1),
        exp: 0,
    },
    ValueMeta {
        shape: Shape4dNHWC::new(N as u16, H as u16, W as u16, OC as u16),
        exp: 0,
    },
];

fn build_spec(aligned: bool) -> ModelSpec {
    let values: &'static [ValueMeta] = &VALUES;

    // Weights: OHWI ones
    static W_ONES: [u8; IC * OC] = [1u8; IC * OC];
    static INITS: [Initializer; 1] = [Initializer {
        id: W_ID,
        layout: ParamLayout::WeightsI8OHWI {
            oc: OC as u16,
            kh: 1,
            kw: 1,
            ic: IC as u16,
        },
        data: &W_ONES,
        exponents: ParamExponents::PerTensor(0),
    }];
    let inits: &'static [Initializer] = &INITS;

    static INPUTS: [ValueId; 1] = [IN_ID];
    static OUTPUTS: [ValueId; 1] = [OUT_ID];
    static PLAN_1: [NodeSpec; 1] = [NodeSpec {
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
        alignment: if aligned { 16 } else { 1 },
        inputs: &INPUTS,
        outputs: &OUTPUTS,
        values,
        initializers: inits,
        plan: &PLAN_1,
    }
}

pub fn run_group() {
    oc_tail_aligned_output();
    oc_tail_misaligned_output();
}

fn oc_tail_aligned_output() {
    // Build model and plan
    let spec = build_spec(true);
    const ARENA_TOTAL: usize = SIZE_ALIGNED + SCRATCH_BYTES;
    let mut arena_buf = Aligned::<ARENA_TOTAL>([0; ARENA_TOTAL]);
    let mut arena = Arena::from_buf(&mut arena_buf.0);

    // Prepare input pattern
    let mut input = [0i8; IN_ELEMS];
    for (i, v) in input.iter_mut().enumerate() {
        *v = (i as i8 % 5) - 2;
    }

    // Copy input, clear output
    let in_mut = arena.value_slice_mut(PLAN_ALIGNED.offset_of(IN_ID), IN_ELEMS);
    in_mut.copy_from_slice(&input);
    let out_mut = arena.value_slice_mut(PLAN_ALIGNED.offset_of(OUT_ID), OUT_ELEMS);
    out_mut.fill(0);

    // Scalar
    let conv = match spec.plan[0].op {
        NodeOp::Conv2d(ref c) => c,
        _ => unreachable!(),
    };
    conv2d_scalar(&PLAN_ALIGNED, &mut arena, &spec, conv, DEFAULT_ROUNDING);
    let mut golden = [0i8; OUT_ELEMS];
    golden.copy_from_slice(arena.value_slice(PLAN_ALIGNED.offset_of(OUT_ID), OUT_ELEMS));

    // Clear and SIMD
    let out_mut2 = arena.value_slice_mut(PLAN_ALIGNED.offset_of(OUT_ID), OUT_ELEMS);
    out_mut2.fill(0);
    let simd = try_conv2d(&PLAN_ALIGNED, &mut arena, &spec, conv, DEFAULT_ROUNDING);
    if !matches!(simd, ExecOutcome::Executed) {
        edgedl::ne_error!(
            "conv11_oc_tail_parity (aligned): SIMD path fell back; scalar result only (assert will fail)"
        );
        assert!(matches!(simd, ExecOutcome::Executed));
    }

    // Compare
    let got = arena.value_slice(PLAN_ALIGNED.offset_of(OUT_ID), OUT_ELEMS);
    for i in 0..OUT_ELEMS {
        let a = golden[i] as i16;
        let b = got[i] as i16;
        let d = (a - b).abs();
        if d > 1 {
            edgedl::ne_error!(
                "conv11_oc_tail_parity (aligned): scalar/SIMD mismatch at idx={} scalar={} simd={} (assert will fail)",
                i,
                golden[i],
                got[i]
            );
            assert!(d <= 1, "idx={} scalar={} simd={}", i, golden[i], got[i]);
        }
    }
}

fn oc_tail_misaligned_output() {
    let spec = build_spec(false);
    const ARENA_TOTAL: usize = SIZE_MISALIGNED + SCRATCH_BYTES;
    let mut arena_buf = Aligned::<ARENA_TOTAL>([0; ARENA_TOTAL]);
    let mut arena = Arena::from_buf(&mut arena_buf.0);

    let mut input = [0i8; IN_ELEMS];
    for (i, v) in input.iter_mut().enumerate() {
        *v = (i as i8 % 7) - 3;
    }

    let in_mut = arena.value_slice_mut(PLAN_MISALIGNED.offset_of(IN_ID), IN_ELEMS);
    in_mut.copy_from_slice(&input);
    let out_mut = arena.value_slice_mut(PLAN_MISALIGNED.offset_of(OUT_ID), OUT_ELEMS);
    out_mut.fill(0);

    let conv = match spec.plan[0].op {
        NodeOp::Conv2d(ref c) => c,
        _ => unreachable!(),
    };
    conv2d_scalar(&PLAN_MISALIGNED, &mut arena, &spec, conv, DEFAULT_ROUNDING);
    let mut golden = [0i8; OUT_ELEMS];
    golden.copy_from_slice(arena.value_slice(PLAN_MISALIGNED.offset_of(OUT_ID), OUT_ELEMS));

    let out_mut2 = arena.value_slice_mut(PLAN_MISALIGNED.offset_of(OUT_ID), OUT_ELEMS);
    out_mut2.fill(0);
    let simd = try_conv2d(&PLAN_MISALIGNED, &mut arena, &spec, conv, DEFAULT_ROUNDING);
    if !matches!(simd, ExecOutcome::Executed) {
        defmt::info!(
            "conv11_oc_tail_parity (misaligned): SIMD path fell back; scalar result only (assert will fail)"
        );
        edgedl::ne_error!(
            "conv11_oc_tail_parity (misaligned): SIMD path fell back; scalar result only (assert will fail)"
        );
        assert!(matches!(simd, ExecOutcome::Executed));
    }

    let got = arena.value_slice(PLAN_MISALIGNED.offset_of(OUT_ID), OUT_ELEMS);
    for i in 0..OUT_ELEMS {
        let a = golden[i] as i16;
        let b = got[i] as i16;
        let d = (a - b).abs();
        if d > 1 {
            edgedl::ne_error!(
                "conv11_oc_tail_parity (misaligned): scalar/SIMD mismatch at idx={} scalar={} simd={} (assert will fail)",
                i,
                golden[i],
                got[i]
            );
            assert!(d <= 1, "idx={} scalar={} simd={}", i, golden[i], got[i]);
        }
    }
}
