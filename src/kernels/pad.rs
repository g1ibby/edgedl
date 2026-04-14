//! Scalar Pad (INT8) skeleton.
//!
//! Pad reproduces ONNX semantics with modes Constant, Edge, Reflect. The exponent
//! is unchanged.

#![allow(unused_variables)]

use crate::{
    arena::{Arena, PlannedArena},
    model::{ModelSpec, PadSpec, PaddingMode},
};

fn map_edge(idx: isize, len: usize) -> usize {
    if idx < 0 {
        0
    } else if idx as usize >= len {
        len - 1
    } else {
        idx as usize
    }
}

fn map_reflect(mut idx: isize, len: usize) -> usize {
    // Reflect about edges excluding endpoints, per ONNX reflect mode semantics.
    // Valid only when len >= 2 and pads < len; macro should ensure this.
    let hi = len as isize - 1;
    while idx < 0 || idx > hi {
        if idx < 0 {
            idx = -idx; // -1 -> 1, -2 -> 2, ...
        } else if idx > hi {
            idx = 2 * hi - idx; // hi+1 -> hi-1, hi+2 -> hi-2, ...
        }
    }
    idx as usize
}

pub fn pad_scalar(plan: &PlannedArena, arena: &mut Arena<'_>, spec: &ModelSpec, node: &PadSpec) {
    let in_meta = spec.values[node.input as usize];
    let out_meta = spec.values[node.output as usize];

    let in_off = plan.offset_of(node.input);
    let out_off = plan.offset_of(node.output);
    let in_elems = in_meta.shape.elements();
    let out_elems = out_meta.shape.elements();
    let (input, output) = arena.io_slices(in_off, in_elems, out_off, out_elems);

    // Shapes
    let n = out_meta.shape.n as usize;
    let oh = out_meta.shape.h as usize;
    let ow = out_meta.shape.w as usize;
    let c = out_meta.shape.c as usize;
    let ih = in_meta.shape.h as usize;
    let iw = in_meta.shape.w as usize;

    let n0 = node.pads_nhwc[0] as isize;
    let h0 = node.pads_nhwc[1] as isize;
    let w0 = node.pads_nhwc[2] as isize;
    let c0 = node.pads_nhwc[3] as isize;

    for bn in 0..n {
        for y in 0..oh {
            for x in 0..ow {
                for ch in 0..c {
                    let src_n = (bn as isize) - n0;
                    let src_h = (y as isize) - h0;
                    let src_w = (x as isize) - w0;
                    let src_c = (ch as isize) - c0;
                    let val = match node.mode {
                        PaddingMode::Constant => {
                            if src_n < 0
                                || src_h < 0
                                || src_w < 0
                                || src_c < 0
                                || src_h as usize >= ih
                                || src_w as usize >= iw
                                || src_c as usize >= c
                            {
                                node.const_i8
                            } else {
                                let in_idx = ((((src_n as usize) * ih + (src_h as usize)) * iw
                                    + (src_w as usize))
                                    * c
                                    + (src_c as usize))
                                    as usize;
                                input[in_idx]
                            }
                        }
                        PaddingMode::Edge => {
                            if src_n < 0 || src_c < 0 || src_c as usize >= c {
                                node.const_i8
                            } else {
                                let sh = map_edge(src_h, ih);
                                let sw = map_edge(src_w, iw);
                                let sc = map_edge(src_c, c);
                                let in_idx =
                                    ((((src_n as usize) * ih + sh) * iw + sw) * c + sc) as usize;
                                input[in_idx]
                            }
                        }
                        PaddingMode::Reflect => {
                            if ih < 2 || iw < 2 {
                                node.const_i8
                            } else {
                                let sh = map_reflect(src_h, ih);
                                let sw = map_reflect(src_w, iw);
                                let sc = map_edge(src_c, c); // clamp channels
                                let in_idx =
                                    ((((src_n as usize) * ih + sh) * iw + sw) * c + sc) as usize;
                                input[in_idx]
                            }
                        }
                    };
                    let out_idx = ((((bn) * oh + y) * ow + x) * c + ch) as usize;
                    output[out_idx] = val;
                }
            }
        }
    }

    #[cfg(feature = "trace")]
    {
        let out_view = arena.value_slice(out_off, out_meta.shape.elements());
        crate::trace::inspect::log_value_i8("Pad", node.output, out_meta, out_view);
    }
}
