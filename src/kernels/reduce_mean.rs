//! Scalar ReduceMean (INT8) skeleton.
//!
//! Sum in i32, divide by element count using exponent math (power‑of‑two scaling)
//! and configured rounding, then narrow to INT8 with saturation.

#![allow(unused_variables)]

use crate::{
    arena::{Arena, PlannedArena},
    model::{ModelSpec, ReduceMeanSpec},
    rounding::RoundingMode,
};

pub fn reduce_mean_scalar(
    plan: &PlannedArena,
    arena: &mut Arena<'_>,
    spec: &ModelSpec,
    node: &ReduceMeanSpec,
    round: RoundingMode,
) {
    let in_meta = spec.values[node.input as usize];
    let out_meta = spec.values[node.output as usize];

    let n = in_meta.shape.n as usize;
    let ih = in_meta.shape.h as usize;
    let iw = in_meta.shape.w as usize;
    let ic = in_meta.shape.c as usize;

    let on = out_meta.shape.n as usize;
    let oh = out_meta.shape.h as usize;
    let ow = out_meta.shape.w as usize;
    let oc = out_meta.shape.c as usize;

    let in_off = plan.offset_of(node.input);
    let out_off = plan.offset_of(node.output);
    let (input, output) = arena.io_slices(
        in_off,
        in_meta.shape.elements(),
        out_off,
        out_meta.shape.elements(),
    );

    #[cfg(feature = "trace")]
    {
        crate::ne_info!(
            "ReduceMean in id={} out id={} axes_bits={} keepdims={} in_shape=N{}x{}x{}x{} out_shape=N{}x{}x{}x{}",
            node.input,
            node.output,
            node.axes_bitmap_nhwc,
            node.keepdims,
            n,
            ih,
            iw,
            ic,
            out_meta.shape.n,
            out_meta.shape.h,
            out_meta.shape.w,
            out_meta.shape.c
        );
    }

    // Axes bitmap over NHWC: bit0=N, bit1=H, bit2=W, bit3=C
    let ax_n = (node.axes_bitmap_nhwc & 0b0001) != 0;
    let ax_h = (node.axes_bitmap_nhwc & 0b0010) != 0;
    let ax_w = (node.axes_bitmap_nhwc & 0b0100) != 0;
    let ax_c = (node.axes_bitmap_nhwc & 0b1000) != 0;

    // Denominator (element count reduced)
    let mut denom: i64 = 1;
    if ax_n {
        denom *= n as i64;
    }
    if ax_h {
        denom *= ih as i64;
    }
    if ax_w {
        denom *= iw as i64;
    }
    if ax_c {
        denom *= ic as i64;
    }
    if denom <= 0 {
        return;
    }

    // Iterate output positions; map to ranges in input.
    // Compute scale: y = round(sum(x) * 2^(e_in - e_out) / denom)
    let diff = (in_meta.exp as i32) - (out_meta.exp as i32);
    for bn in 0..on {
        for oy in 0..oh {
            for ox in 0..ow {
                for ch in 0..oc {
                    let mut sum: i64 = 0;

                    // Determine input ranges depending on keepdims/out dims mapping.
                    // We assume out dims equal 1 for reduced axes when keepdims==true, otherwise
                    // out dims omit the axis. The macro ensures shapes; here we infer mapping by
                    // clamping to 1.
                    let rn = if ax_n { 0..n } else { (bn)..(bn + 1) };
                    let rh = if ax_h { 0..ih } else { (oy)..(oy + 1) };
                    let rw = if ax_w { 0..iw } else { (ox)..(ox + 1) };
                    let rc = if ax_c { 0..ic } else { (ch)..(ch + 1) };

                    for n0 in rn.clone() {
                        for h0 in rh.clone() {
                            for w0 in rw.clone() {
                                for c0 in rc.clone() {
                                    let idx = ((((n0) * ih + h0) * iw + w0) * ic + c0) as usize;
                                    sum += input[idx] as i8 as i64;
                                }
                            }
                        }
                    }

                    // Scale and divide with HALF_UP rounding; avoid float.
                    let y_i32: i32 = if diff >= 0 {
                        let num = (sum as i64) << (diff as u32);
                        let den = denom;
                        let rounded = if num >= 0 {
                            (num + den / 2) / den
                        } else {
                            (num - den / 2) / den
                        };
                        if rounded > i32::MAX as i64 {
                            i32::MAX
                        } else if rounded < i32::MIN as i64 {
                            i32::MIN
                        } else {
                            rounded as i32
                        }
                    } else {
                        // Effective denominator multiplies by 2^(-diff)
                        let den = denom << ((-diff) as u32);
                        let num = sum as i64;
                        let rounded = if num >= 0 {
                            (num + den / 2) / den
                        } else {
                            (num - den / 2) / den
                        };
                        if rounded > i32::MAX as i64 {
                            i32::MAX
                        } else if rounded < i32::MIN as i64 {
                            i32::MIN
                        } else {
                            rounded as i32
                        }
                    };

                    // Saturate to i8
                    let out_val = if y_i32 > 127 {
                        127
                    } else if y_i32 < -128 {
                        -128
                    } else {
                        y_i32 as i8
                    };
                    let out_idx = ((((bn) * oh + oy) * ow + ox) * oc + ch) as usize;
                    output[out_idx] = out_val;
                }
            }
        }
    }

    #[cfg(feature = "trace")]
    {
        // Stats on output
        let out_view = arena.value_slice(out_off, out_meta.shape.elements());
        crate::trace::inspect::log_value_i8("ReduceMean", node.output, out_meta, out_view);
    }
}
