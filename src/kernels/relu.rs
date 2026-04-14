//! Standalone ReLU kernel: y = max(0, x) for INT8 activations.

#![allow(unused_variables)]

use crate::{
    arena::{Arena, PlannedArena},
    model::{ActivationSpec, ModelSpec},
};

/// Apply ReLU to an activation tensor: clamp negatives to zero.
pub fn relu_scalar(
    plan: &PlannedArena,
    arena: &mut Arena<'_>,
    spec: &ModelSpec,
    node: &ActivationSpec,
) {
    let in_meta = spec.values[node.input as usize];
    let out_meta = spec.values[node.output as usize];

    // Shapes must match (NHWC)
    debug_assert_eq!(in_meta.shape.n, out_meta.shape.n);
    debug_assert_eq!(in_meta.shape.h, out_meta.shape.h);
    debug_assert_eq!(in_meta.shape.w, out_meta.shape.w);
    debug_assert_eq!(in_meta.shape.c, out_meta.shape.c);

    let elems = in_meta.shape.elements();
    let in_off = plan.offset_of(node.input);
    let out_off = plan.offset_of(node.output);

    let (input, out) = arena.io_slices(in_off, elems, out_off, elems);

    for i in 0..elems {
        let v = input[i];
        out[i] = if v < 0 { 0 } else { v };
    }

    #[cfg(feature = "trace")]
    {
        let view = arena.value_slice(out_off, elems);
        crate::trace::inspect::log_value_i8("ReLU", node.output, out_meta, view);
    }
}
