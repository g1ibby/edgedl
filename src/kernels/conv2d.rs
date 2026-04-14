//! Scalar Conv2D (INT8) skeleton.
//!
//! Contract
//! - Activations: NHWC INT8, exponent `e_in` (per‑tensor).
//! - Weights: OHWI INT8, exponent `e_w` (per‑tensor or per‑OC).
//! - Bias: I32 optional, exponent `e_b` aligned to accumulator domain.
//! - Output: NHWC INT8, exponent `e_out` (per‑tensor).
//! - Accumulator domain: I32. Requantization uses integer rounding (see rounding module).
//! - Fused activation: Linear or ReLU.

#![allow(unused_variables)]

use crate::{
    arena::{Arena, PlannedArena},
    model::{Activation, Conv2dSpec, ModelSpec, ParamExponents, ParamLayout},
    rounding::{
        RoundingMode,
        derive_shift_and_scale,
        requant_i32_to_i8,
        saturate_i32,
        shift_round_i32,
    },
};

/// Execute one Conv2D node in scalar mode.
///
/// Safety/Unchecked assumptions for M1 skeleton:
/// - Shape consistency has been validated by the macro (compile‑time) and by the planner (runtime
///   invariants on elements offsets).
/// - Offsets and lengths are within the arena bounds (planner produced them).
pub fn conv2d_scalar(
    plan: &PlannedArena,
    arena: &mut Arena<'_>,
    spec: &ModelSpec,
    node: &Conv2dSpec,
    round: RoundingMode,
) {
    let input_meta = spec.values[node.input as usize];
    let output_meta = spec.values[node.output as usize];

    // Locate weight initializer and interpret layout/exponents.
    let w_init = spec
        .initializers
        .iter()
        .find(|ini| ini.id == node.weights)
        .expect("weight initializer missing");

    match w_init.layout {
        ParamLayout::WeightsI8OHWI { oc, kh, kw, ic } => {
            let oc = oc as usize;
            let kh = kh as usize;
            let kw = kw as usize;
            let ic = ic as usize;

            // View input/output regions in arena using safe disjoint borrows.
            let in_off = plan.offset_of(node.input);
            let out_off = plan.offset_of(node.output);
            let in_elems = input_meta.shape.elements();
            let out_elems = output_meta.shape.elements();
            let (input, output) = arena.io_slices(in_off, in_elems, out_off, out_elems);

            // Weight and bias blobs
            let w_bytes = w_init.data;
            let (bias_bytes, bias_exp_opt): (Option<&[u8]>, Option<i8>) = match node.bias {
                Some(bid) => {
                    if let Some(b) = spec.initializers.iter().find(|ini| ini.id == bid) {
                        (
                            Some(b.data),
                            match b.exponents {
                                ParamExponents::PerTensor(e) => Some(e),
                                _ => None,
                            },
                        )
                    } else {
                        (None, None)
                    }
                }
                None => (None, None),
            };

            // Helper to read i32 bias in little-endian
            let read_bias = |oc_idx: usize| -> i32 {
                if let Some(b) = bias_bytes {
                    let base = oc_idx * 4;
                    if base + 4 <= b.len() {
                        let arr = [b[base], b[base + 1], b[base + 2], b[base + 3]];
                        i32::from_le_bytes(arr)
                    } else {
                        0
                    }
                } else {
                    0
                }
            };

            // Exponents
            let w_exp_tensor: Option<i8> = match w_init.exponents {
                ParamExponents::PerTensor(e) => Some(e),
                _ => None,
            };
            let w_exp_per_oc: Option<&[i8]> = match w_init.exponents {
                ParamExponents::PerChannel(slice) => Some(slice),
                _ => None,
            };
            let in_exp = input_meta.exp;
            let out_exp = output_meta.exp;

            // Shapes
            let n = input_meta.shape.n as usize;
            let ih = input_meta.shape.h as usize;
            let iw = input_meta.shape.w as usize;
            let ic_in = input_meta.shape.c as usize;
            let oh = output_meta.shape.h as usize;
            let ow = output_meta.shape.w as usize;
            let oc_out = output_meta.shape.c as usize;

            debug_assert_eq!(ic_in, ic, "IC mismatch NHWC vs OHWI");
            debug_assert_eq!(oc_out, oc, "OC mismatch NHWC vs OHWI");
            debug_assert_eq!(n as u16, output_meta.shape.n, "Batch mismatch");

            let stride_h = node.strides_hw[0] as usize;
            let stride_w = node.strides_hw[1] as usize;
            let dil_h = node.dilations_hw[0] as usize;
            let dil_w = node.dilations_hw[1] as usize;
            let pad_top = node.pads_hw[0] as isize;
            let pad_left = node.pads_hw[1] as isize;

            // Convolution loops: N, OH, OW, OC
            for bn in 0..n {
                for oy in 0..oh {
                    for ox in 0..ow {
                        // For each output channel, compute accumulator, add bias, requant, fuse
                        // activation, store
                        for oo in 0..oc {
                            let mut acc: i32 = 0;
                            // Window over KH×KW
                            for ky in 0..kh {
                                let in_y = oy * stride_h + ky * dil_h;
                                // account for top padding
                                let in_y = in_y as isize - pad_top;
                                if in_y < 0 || in_y >= ih as isize {
                                    continue;
                                }
                                let in_y = in_y as usize;
                                for kx in 0..kw {
                                    let in_x = ox * stride_w + kx * dil_w;
                                    let in_x = in_x as isize - pad_left;
                                    if in_x < 0 || in_x >= iw as isize {
                                        continue;
                                    }
                                    let in_x = in_x as usize;

                                    // Input base index for this pixel
                                    let in_base = (((bn * ih + in_y) * iw + in_x) * ic_in) as usize;
                                    // Weight base for this (oo, ky, kx) in OHWI layout
                                    let w_base = (((oo * kh + ky) * kw + kx) * ic) as usize;

                                    // Dot over channels
                                    for ci in 0..ic_in {
                                        let x = input[in_base + ci] as i8 as i32; // sign‑extended
                                        let w = w_bytes[w_base + ci] as i8 as i32;
                                        acc = acc.wrapping_add(x * w);
                                    }
                                }
                            }

                            // Add bias with rescaling to accumulator domain if needed.
                            // Accumulator domain exponent is e_acc = in_exp + w_e (per OC when
                            // provided).
                            let mut acc32 = acc;
                            if bias_bytes.is_some() {
                                let b = read_bias(oo);
                                // Use per-OC weight exponent when available for accumulator domain
                                let w_e_for_acc = w_exp_per_oc
                                    .and_then(|s| s.get(oo))
                                    .copied()
                                    .or(w_exp_tensor)
                                    .unwrap_or(0);
                                let e_acc = in_exp.saturating_add(w_e_for_acc);
                                // Rescale bias from its exponent into accumulator domain.
                                if let Some(e_b) = bias_exp_opt {
                                    let delta = (e_b as i32) - (e_acc as i32);
                                    let b_adj: i32 = if delta == 0 {
                                        b
                                    } else if delta > 0 {
                                        // Left shift with saturation when bias has coarser scale.
                                        let widened = (b as i64) << (delta as u32);
                                        saturate_i32(widened)
                                    } else {
                                        // Right shift with rounding when bias has finer scale.
                                        let sh = (-delta) as i32;
                                        shift_round_i32(b, sh, round)
                                    };
                                    let sum = (acc32 as i64) + (b_adj as i64);
                                    acc32 = saturate_i32(sum);
                                } else {
                                    // If bias exponent is unspecified, assume it's already in
                                    // accumulator domain.
                                    let sum = (acc32 as i64) + (b as i64);
                                    acc32 = saturate_i32(sum);
                                }
                            }

                            // Compute requant per OC if needed
                            let w_e = w_exp_per_oc
                                .and_then(|s| s.get(oo))
                                .copied()
                                .or(w_exp_tensor)
                                .unwrap_or(0);
                            let (shift, scale) =
                                derive_shift_and_scale(in_exp.saturating_add(w_e), out_exp);
                            let mut y = requant_i32_to_i8(acc32, shift, scale, round);

                            if let Activation::ReLU = node.activation {
                                if y < 0 {
                                    y = 0;
                                }
                            }

                            // Store to output
                            let out_index = (((bn * oh + oy) * ow + ox) * oc_out + oo) as usize;
                            output[out_index] = y;
                        }
                    }
                }
            }
        }
        ParamLayout::WeightsI8BlockedN16HWC16 { oc, kh, kw, ic } => {
            let oc = oc as usize;
            let kh = kh as usize;
            let kw = kw as usize;
            let ic = ic as usize;

            let in_off = plan.offset_of(node.input);
            let out_off = plan.offset_of(node.output);
            let in_elems = input_meta.shape.elements();
            let out_elems = output_meta.shape.elements();
            let (input, output) = arena.io_slices(in_off, in_elems, out_off, out_elems);

            let w_bytes = w_init.data;
            // Trust macro layout: require proper tiling and length for blocked layout.
            debug_assert!(oc % 16 == 0, "BlockedN16HWC16 requires oc % 16 == 0");
            let tiles = oc / 16;
            debug_assert!(
                w_bytes.len() == kh * kw * ic * 16 * tiles,
                "blocked weights length mismatch"
            );
            let (bias_bytes, bias_exp_opt): (Option<&[u8]>, Option<i8>) = match node.bias {
                Some(bid) => {
                    if let Some(b) = spec.initializers.iter().find(|ini| ini.id == bid) {
                        (
                            Some(b.data),
                            match b.exponents {
                                ParamExponents::PerTensor(e) => Some(e),
                                _ => None,
                            },
                        )
                    } else {
                        (None, None)
                    }
                }
                None => (None, None),
            };
            let read_bias = |oc_idx: usize| -> i32 {
                if let Some(b) = bias_bytes {
                    let base = oc_idx * 4;
                    if base + 4 <= b.len() {
                        let arr = [b[base], b[base + 1], b[base + 2], b[base + 3]];
                        i32::from_le_bytes(arr)
                    } else {
                        0
                    }
                } else {
                    0
                }
            };

            let w_exp_tensor: Option<i8> = match w_init.exponents {
                ParamExponents::PerTensor(e) => Some(e),
                _ => None,
            };
            let w_exp_per_oc: Option<&[i8]> = match w_init.exponents {
                ParamExponents::PerChannel(slice) => Some(slice),
                _ => None,
            };
            let in_exp = input_meta.exp;
            let out_exp = output_meta.exp;

            let n = input_meta.shape.n as usize;
            let ih = input_meta.shape.h as usize;
            let iw = input_meta.shape.w as usize;
            let ic_in = input_meta.shape.c as usize;
            let oh = output_meta.shape.h as usize;
            let ow = output_meta.shape.w as usize;
            let oc_out = output_meta.shape.c as usize;

            debug_assert_eq!(ic_in, ic, "IC mismatch NHWC vs HWC16");
            debug_assert_eq!(oc_out, oc, "OC mismatch NHWC vs HWC16");
            debug_assert_eq!(n as u16, output_meta.shape.n, "Batch mismatch");

            let stride_h = node.strides_hw[0] as usize;
            let stride_w = node.strides_hw[1] as usize;
            let dil_h = node.dilations_hw[0] as usize;
            let dil_w = node.dilations_hw[1] as usize;
            let pad_top = node.pads_hw[0] as isize;
            let pad_left = node.pads_hw[1] as isize;

            let tiles = oc / 16;

            for bn in 0..n {
                for oy in 0..oh {
                    for ox in 0..ow {
                        for oo in 0..oc {
                            let mut acc: i32 = 0;
                            for ky in 0..kh {
                                let in_y = oy * stride_h + ky * dil_h;
                                let in_y = in_y as isize - pad_top;
                                if in_y < 0 || in_y >= ih as isize {
                                    continue;
                                }
                                let in_y = in_y as usize;
                                for kx in 0..kw {
                                    let in_x = ox * stride_w + kx * dil_w;
                                    let in_x = in_x as isize - pad_left;
                                    if in_x < 0 || in_x >= iw as isize {
                                        continue;
                                    }
                                    let in_x = in_x as usize;
                                    let in_base = (((bn * ih + in_y) * iw + in_x) * ic_in) as usize;

                                    // Blocked layout index only (macro guarantees correctness)
                                    let tile = oo / 16;
                                    let inner = oo % 16;
                                    // Order: [tile][kh][kw][ic][inner]
                                    let w_base =
                                        ((((tile * kh + ky) * kw + kx) * ic) * 16) as usize;
                                    for ci in 0..ic_in {
                                        let x = input[in_base + ci] as i8 as i32;
                                        let w = w_bytes[w_base + ci * 16 + inner] as i8 as i32;
                                        acc = acc.wrapping_add(x * w);
                                    }
                                }
                            }

                            // Bias + requant + activation identical to OHWI case
                            let mut acc32 = acc;
                            if bias_bytes.is_some() {
                                let b = read_bias(oo);
                                let w_e_for_acc = w_exp_per_oc
                                    .and_then(|s| s.get(oo))
                                    .copied()
                                    .or(w_exp_tensor)
                                    .unwrap_or(0);
                                let e_acc = in_exp.saturating_add(w_e_for_acc);
                                if let Some(e_b) = bias_exp_opt {
                                    let delta = (e_b as i32) - (e_acc as i32);
                                    let b_adj: i32 = if delta == 0 {
                                        b
                                    } else if delta > 0 {
                                        let widened = (b as i64) << (delta as u32);
                                        saturate_i32(widened)
                                    } else {
                                        let sh = (-delta) as i32;
                                        shift_round_i32(b, sh, round)
                                    };
                                    let sum = (acc32 as i64) + (b_adj as i64);
                                    acc32 = saturate_i32(sum);
                                } else {
                                    let sum = (acc32 as i64) + (b as i64);
                                    acc32 = saturate_i32(sum);
                                }
                            }

                            let w_e = w_exp_per_oc
                                .and_then(|s| s.get(oo))
                                .copied()
                                .or(w_exp_tensor)
                                .unwrap_or(0);
                            let (shift, scale) =
                                derive_shift_and_scale(in_exp.saturating_add(w_e), out_exp);
                            let mut y = requant_i32_to_i8(acc32, shift, scale, round);
                            if let Activation::ReLU = node.activation {
                                if y < 0 {
                                    y = 0;
                                }
                            }
                            let out_index = (((bn * oh + oy) * ow + ox) * oc_out + oo) as usize;
                            output[out_index] = y;
                        }
                    }
                }
            }
        }
        _ => {
            // Non‑Conv parameter bound where Conv weights expected.
            #[cfg(debug_assertions)]
            panic!("Conv2D expects WeightsI8OHWI initializer");
        }
    }

    #[cfg(feature = "trace")]
    {
        let out_meta = spec.values[node.output as usize];
        let out_off = plan.offset_of(node.output);
        let out_slice = arena.value_slice(out_off, out_meta.shape.elements());
        crate::trace::inspect::log_value_i8("Conv2D", node.output, out_meta, out_slice);
    }
}
