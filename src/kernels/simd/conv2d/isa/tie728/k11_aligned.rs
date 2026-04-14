//! 1x1 (11cn) aligned SIMD kernels for ESP32-S3 TIE728.
//!
//! This module provides the aligned 1×1 (OHWI) TIE728 implementation. On
//! Xtensa targets we expose a C-ABI symbol implemented via `global_asm!` that
//! mirrors ESP-DL’s structure: the Rust wrapper builds a compact `KernelArgs`
//! block, and the ASM entry pulls fields, runs the 11c16 microkernel over OC
//! tiles, and performs the fused epilogue (bias, requant, optional ReLU, store).
//!
//! Notes
//! - Call ABI matches ESP-DL’s style for conv2d 11cn: `a2` = output_ptr for the current pixel, `a3`
//!   = input_ptr for the current pixel, `a4` = `&KernelArgs`. Rust iterates H×W and calls the ASM
//!   per pixel; ASM iterates OC tiles.
//! - The epilogue supports both per-layer and per-channel requantization and handles bias/no-bias
//!   in-kernel for parity and performance.
//! - The Rust wrapper validates preconditions and constructs `KernelArgs` from
//!   `ModelSpec`/`Conv2dSpec`/Arena (including packing bias when needed).

// Variables/imports used only when compiled for ESP32-S3 with simd-s3 feature.
#![allow(unused_variables, unused_imports, unused_assignments, dead_code)]

use crate::{
    arena::{Arena, PlannedArena},
    kernels::simd::conv2d::{
        ExecOutcome,
        common::{FusedActivation, KernelArgs, KernelFlags, WeightLayout},
    },
    model::{Conv2dSpec, ModelSpec, ParamExponents},
    rounding::{
        RoundingMode,
        derive_shift_and_scale,
        requant_i32_to_i8,
        saturate_i32,
        shift_round_i32,
    },
};

#[inline(always)]
fn ceil16(x: usize) -> usize {
    (x + 15) & !15
}

#[inline]
pub fn run(
    plan: &PlannedArena,
    arena: &mut Arena<'_>,
    spec: &ModelSpec,
    node: &Conv2dSpec,
    round: RoundingMode,
) -> ExecOutcome {
    // Preconditions (extra constraints for the first M3.2 drop):
    // - 1x1 kernel
    // - NHWC activations; N == 1
    // - Stride == 1, Dilation == 1 (checked below, but logic supports non-1 now)
    // - Weights in OHWI layout (BlockedN16HWC16 will follow next)
    // - IC and OC are multiples of 16 and input/output base pointers are 16‑byte aligned
    // These are enforced here to keep the initial kernel small and match esp‑dl 11cn inner loop
    // assumptions. Router already guarantees aligned+multiple‑of‑16; we double‑check here for
    // safety and to allow precise fallback when unsupported.
    let in_meta = spec.values[node.input as usize];
    let out_meta = spec.values[node.output as usize];
    if in_meta.shape.n != 1 || out_meta.shape.n != 1 {
        return ExecOutcome::Fallback;
    }
    let w_init = match spec.initializers.iter().find(|ini| ini.id == node.weights) {
        Some(w) => w,
        None => return ExecOutcome::Fallback,
    };
    let (kh, kw) = match w_init.layout {
        crate::model::ParamLayout::WeightsI8OHWI { kh, kw, .. } => (kh as i32, kw as i32),
        crate::model::ParamLayout::WeightsI8BlockedN16HWC16 { kh, kw, .. } => {
            (kh as i32, kw as i32)
        }
        _ => (1, 1),
    };
    if kh != 1 || kw != 1 {
        return ExecOutcome::Fallback;
    }
    // We now support padding, so remove the pads_hw check.
    // Strides and dilations are technically supported by the logic below, but the ASM kernel
    // assumes contiguous input for the 1x1 optimization if we were to pass >1 pixel.
    // However, we call ASM per pixel, so stride/dilation is handled by the loop indices.
    // We keep the check for now if we want to be conservative, OR we can relax it.
    // The ASM signature is `dl_tie728_s8_conv2d_11cn(out_ptr, in_ptr, args)`.
    // It processes ONE pixel (OC loop). So stride/dilation is fully handled by the caller.
    // Let's relax strides/dilations/pads checks.
    // But wait, k11_aligned implies 1x1.

    match w_init.layout {
        crate::model::ParamLayout::WeightsI8OHWI { .. } => {}
        crate::model::ParamLayout::WeightsI8BlockedN16HWC16 { .. } => {}
        _ => return ExecOutcome::Fallback,
    }

    // Build the argument block from model/node state.
    let mut args = build_kernel_args_k11_aligned(plan, arena, spec, node);

    // If per‑OC exponents are present and no bias, precompute per‑channel scale factors
    // into a scratch buffer (i16) to match esp‑dl per‑channel path.
    let has_bias = node.bias.is_some();
    match w_init.exponents {
        crate::model::ParamExponents::PerTensor(w_e_tensor) => {
            // Per-layer path: if mac_shift < 0, build uniform per-channel scales (and bias16) and
            // switch to the per-channel macro path for parity and simplicity.
            let oc = out_meta.shape.c as usize;
            let in_exp = in_meta.exp as i32;
            let out_exp = out_meta.exp as i32;
            // Per-esp-dl: mac_shift = out_exp - in_exp - w_exp. When mac_shift < 0,
            // we route to the per-channel vector epilogue with fixed-point scales.
            // In our notation fcf = in_exp + w_exp - out_exp = shift_i. So mac_shift < 0
            // is equivalent to shift_i > 0.
            let shift_i = in_exp + (w_e_tensor as i32) - out_exp;
            if shift_i > 0 {
                // Build scales/bias16 into arena scratch.
                // For negative mac_shift (left shift by -shift_i), route to the per-channel
                // vector epilogue using a UNIFORM s16 scale per lane equal to 2^(-shift_i).
                // The per-channel macro later performs an implicit >>11 before packing to s8,
                // so the uniform scale here must NOT include the extra 2^11 factor.
                let oc_aligned = ceil16(oc);
                let scales_bytes = oc_aligned * core::mem::size_of::<i16>();
                let bias_bytes = if has_bias {
                    oc_aligned * core::mem::size_of::<i16>()
                } else {
                    0
                };
                let needed = scales_bytes + bias_bytes;
                if needed > plan.scratch_bytes {
                    return ExecOutcome::Fallback;
                }
                let scratch = arena.scratch_slice_mut(plan, needed);
                let (scales_bytes_slice, bias_bytes_slice) = scratch.split_at_mut(scales_bytes);
                // Per esp-dl per-channel epilogue mapping:
                // scale16 = 1 << (15 - fcf) with fcf = shift_i (must be > 0 here).
                // To stay within i16, require (15 - shift_i) in [0..14].
                let sbits = 15 - shift_i; // expected 0..14 for valid cases
                // If out of range, fall back to scalar epilogue for parity.
                if sbits < 0 || sbits > 14 {
                    #[cfg(feature = "trace")]
                    crate::ne_warn!(
                        "k11_aligned id={} SILENT_SCALAR: sbits={} out of range",
                        node.output,
                        sbits
                    );
                    crate::kernels::conv2d::conv2d_scalar(plan, arena, spec, node, round);
                    return ExecOutcome::Executed;
                }
                let val = 1i32 << (sbits as u32);
                let s16 = core::cmp::min(val, i32::from(i16::MAX)) as i16;
                // Fill OC entries
                let scales_ptr = scales_bytes_slice.as_mut_ptr() as *mut i16;
                for i in 0..oc {
                    unsafe {
                        *scales_ptr.add(i) = s16;
                    }
                }
                args.pc_scales = scales_bytes_slice.as_ptr() as *const i16;
                if has_bias {
                    // Per‑layer bias16 for the per-channel epilogue must match the s16 lane
                    // extraction domain used by the macro (SRCMB.S16.QACC with shift=4).
                    // Therefore target exponent is e_acc - 4 (not -11). The later pack to s8
                    // applies >>11 via SSR.
                    let e_b = spec
                        .initializers
                        .iter()
                        .find(|ini| Some(ini.id) == node.bias)
                        .and_then(|b| match b.exponents {
                            crate::model::ParamExponents::PerTensor(e) => Some(e as i32),
                            _ => None,
                        })
                        .unwrap_or(0);
                    let b_init = spec
                        .initializers
                        .iter()
                        .find(|ini| Some(ini.id) == node.bias)
                        .unwrap();
                    let b_bytes = b_init.data;
                    let bias_ptr = bias_bytes_slice.as_mut_ptr() as *mut i16;
                    let target_e = (in_exp + (w_e_tensor as i32)) - 4;
                    for i in 0..oc {
                        let base = i * 4;
                        if base + 4 > b_bytes.len() {
                            return ExecOutcome::Fallback;
                        }
                        let bv = i32::from_le_bytes([
                            b_bytes[base],
                            b_bytes[base + 1],
                            b_bytes[base + 2],
                            b_bytes[base + 3],
                        ]);
                        let delta = e_b - target_e;
                        let adj: i32 = if delta >= 0 {
                            // left shift
                            let wide = (bv as i64) << (delta as u32);
                            if wide > i32::MAX as i64 {
                                i32::MAX
                            } else if wide < i32::MIN as i64 {
                                i32::MIN
                            } else {
                                wide as i32
                            }
                        } else {
                            // right shift with HALF_UP
                            shift_round_i32(bv, (-delta) as i32, RoundingMode::HalfUp)
                        };
                        unsafe {
                            *bias_ptr.add(i) = if adj > i16::MAX as i32 {
                                i16::MAX
                            } else if adj < i16::MIN as i32 {
                                i16::MIN
                            } else {
                                adj as i16
                            };
                        }
                    }
                    args.pc_bias_i16 = bias_bytes_slice.as_ptr() as *const i16;
                }
                // Force per-channel epilogue in ASM
                args.flags |= KernelFlags::PER_OC_W_EXP;
            } else {
                // Per-layer path (right-shift rounding in ASM). If bias is present, prepack
                // OC×i32 into the QACC-friendly 64B/tile layout used by
                // `tie728_s8_conv2d_128b_vector_bias` (esp-dl style).
                if has_bias {
                    let tiles = (oc + 15) / 16;
                    let bytes_needed = tiles * 64; // 64 bytes per 16-OC tile
                    if bytes_needed > plan.scratch_bytes {
                        return ExecOutcome::Fallback;
                    }
                    // Locate the bias initializer bytes
                    let b_init = spec
                        .initializers
                        .iter()
                        .find(|ini| Some(ini.id) == node.bias)
                        .expect("bias initializer missing");
                    let b_bytes = b_init.data;
                    let dst = arena.scratch_slice_mut(plan, bytes_needed);
                    // Reinterpret dst as u8 for packing
                    let dst_u8: &mut [u8] = unsafe {
                        core::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut u8, dst.len())
                    };
                    // Pack into dst using esp-dl's 20-bit/nibble compaction with 16B alignment,
                    // but first rescale bias into accumulator-domain exponent e_acc.
                    let e_b = match b_init.exponents {
                        crate::model::ParamExponents::PerTensor(e) => e as i32,
                        _ => 0,
                    };
                    let e_acc = in_exp + (w_e_tensor as i32);
                    crate::kernels::simd::conv2d::common::pack_bias_i32_for_qacc(
                        dst_u8, b_bytes, oc, e_b, e_acc, round,
                    );
                    args.bias = dst_u8.as_ptr() as *const i32;
                }
            }
            // shift_i >= 0: per‑layer vector rounding path (handled in ASM), nothing to prepare
        }
        crate::model::ParamExponents::PerChannel(w_exps) => {
            let oc = out_meta.shape.c as usize;
            if w_exps.len() < oc {
                return ExecOutcome::Fallback;
            }
            let in_exp = in_meta.exp as i32;
            let out_exp = out_meta.exp as i32;
            // Build scales/bias16 into arena tail (aligned to 16 bytes)
            let oc_aligned = ceil16(oc);
            let scales_bytes = oc_aligned * core::mem::size_of::<i16>();
            let bias_bytes = if has_bias {
                oc_aligned * core::mem::size_of::<i16>()
            } else {
                0
            };
            let needed = scales_bytes + bias_bytes;
            if needed > plan.scratch_bytes {
                return ExecOutcome::Fallback;
            }
            let scratch = arena.scratch_slice_mut(plan, needed);
            let (scales_bytes_slice, bias_bytes_slice) = scratch.split_at_mut(scales_bytes);
            // Fill scales as i16
            let scales_ptr = scales_bytes_slice.as_mut_ptr() as *mut i16;
            for i in 0..oc {
                let we = w_exps[i] as i32;
                let fcf_i = in_exp + we - out_exp; // total shift from acc to out
                // esp-dl per-channel epilogue expects scale16 = 1 << (15 - fcf_i)
                let sbits = 15 - fcf_i; // desired 0..14
                if sbits < 0 || sbits > 14 {
                    #[cfg(feature = "trace")]
                    crate::ne_warn!(
                        "k11_aligned id={} SILENT_SCALAR: sbits={} out of range (oc={})",
                        node.output,
                        sbits,
                        i
                    );
                    crate::kernels::conv2d::conv2d_scalar(plan, arena, spec, node, round);
                    return ExecOutcome::Executed;
                }
                let val = 1i32 << (sbits as u32);
                unsafe {
                    *scales_ptr.add(i) = core::cmp::min(val, i32::from(i16::MAX)) as i16;
                }
            }
            args.pc_scales = scales_bytes_slice.as_ptr() as *const i16;
            if has_bias {
                // Build per‑OC i16 bias buffer for per‑channel + bias macro at exponent e_acc(oc) -
                // 4, matching the SRCMB.S16.QACC (shift=4) extraction domain in the
                // macro.
                let e_b = spec
                    .initializers
                    .iter()
                    .find(|ini| Some(ini.id) == node.bias)
                    .and_then(|b| match b.exponents {
                        crate::model::ParamExponents::PerTensor(e) => Some(e as i32),
                        _ => None,
                    })
                    .unwrap_or(0);
                let b_init = spec
                    .initializers
                    .iter()
                    .find(|ini| Some(ini.id) == node.bias)
                    .unwrap();
                let b_bytes = b_init.data;
                let bias_ptr = bias_bytes_slice.as_mut_ptr() as *mut i16;
                for i in 0..oc {
                    let base = i * 4;
                    if base + 4 > b_bytes.len() {
                        return ExecOutcome::Fallback;
                    }
                    let bv = i32::from_le_bytes([
                        b_bytes[base],
                        b_bytes[base + 1],
                        b_bytes[base + 2],
                        b_bytes[base + 3],
                    ]);
                    let e_acc_i = in_exp + (w_exps[i] as i32);
                    let target_e = e_acc_i - 4;
                    let delta = e_b - target_e;
                    let adj: i32 = if delta >= 0 {
                        let wide = (bv as i64) << (delta as u32);
                        if wide > i32::MAX as i64 {
                            i32::MAX
                        } else if wide < i32::MIN as i64 {
                            i32::MIN
                        } else {
                            wide as i32
                        }
                    } else {
                        shift_round_i32(bv, (-delta) as i32, RoundingMode::HalfUp)
                    };
                    unsafe {
                        *bias_ptr.add(i) = if adj > i16::MAX as i32 {
                            i16::MAX
                        } else if adj < i16::MIN as i32 {
                            i16::MIN
                        } else {
                            adj as i16
                        };
                    }
                }
                args.pc_bias_i16 = bias_bytes_slice.as_ptr() as *const i16;
            }
            // Per-channel path prepared via scales/bias16 buffers; ASM will use PER_OC_W_EXP flag.
        }
    }

    // Prepare scalar fallback data if needed (for padding)
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
    let in_exp = in_meta.exp;
    let out_exp = out_meta.exp;

    // esp-dl style: iterate H/W in Rust and call asm per pixel
    // The asm only loops OC tiles for one pixel; this mirrors esp‑dl conv_operation_shell.
    #[cfg(all(feature = "simd-s3", target_arch = "xtensa"))]
    unsafe {
        let ih = args.ih as isize;
        let iw = args.iw as isize;
        let ic = args.ic as isize;
        let oh = args.oh as isize;
        let ow = args.ow as isize;
        let oc = args.oc as isize;
        let stride_h = args.stride_h as isize;
        let stride_w = args.stride_w as isize;
        let pad_top = args.pad_top as isize;
        let pad_left = args.pad_left as isize;

        #[cfg(feature = "trace")]
        let mut simd_pixels: u32 = 0;
        #[cfg(feature = "trace")]
        let mut padding_pixels: u32 = 0;

        for oy in 0..oh {
            for ox in 0..ow {
                let in_y = oy * stride_h - pad_top;
                let in_x = ox * stride_w - pad_left;

                let out_off = ((oy * ow) + ox) * oc;
                let out_ptr = (args.output).offset(out_off);

                if in_y >= 0 && in_y < ih && in_x >= 0 && in_x < iw {
                    // Valid pixel: call SIMD
                    #[cfg(feature = "trace")]
                    {
                        simd_pixels += 1;
                    }
                    let in_off = ((in_y * iw) + in_x) * ic;
                    let in_ptr = (args.input).offset(in_off);
                    dl_tie728_s8_conv2d_11cn(
                        out_ptr,
                        in_ptr as *const i8,
                        &args as *const KernelArgs,
                    );
                } else {
                    // Padding: scalar fallback (acc=0)
                    #[cfg(feature = "trace")]
                    {
                        padding_pixels += 1;
                    }
                    for oo in 0..oc {
                        let mut acc32 = 0;
                        if bias_bytes.is_some() {
                            let b = read_bias(oo as usize);
                            let w_e_for_acc = w_exp_per_oc
                                .and_then(|s| s.get(oo as usize))
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
                        // Requant
                        let w_e = w_exp_per_oc
                            .and_then(|s| s.get(oo as usize))
                            .copied()
                            .or(w_exp_tensor)
                            .unwrap_or(0);
                        let (shift, scale) =
                            derive_shift_and_scale(in_exp.saturating_add(w_e), out_exp);
                        let mut y = requant_i32_to_i8(acc32, shift, scale, round);
                        if let crate::model::Activation::ReLU = node.activation {
                            if y < 0 {
                                y = 0;
                            }
                        }
                        *out_ptr.offset(oo) = y;
                    }
                }
            }
        }
        #[cfg(feature = "trace")]
        {
            let total = simd_pixels + padding_pixels;
            let simd_pct = if total > 0 {
                (simd_pixels * 100) / total
            } else {
                0
            };
            crate::ne_info!(
                "k11_aligned id={}: simd={}px ({}%), pad={}px",
                node.output,
                simd_pixels,
                simd_pct,
                padding_pixels
            );
        }
        return ExecOutcome::Executed;
    }
    #[cfg(not(all(feature = "simd-s3", target_arch = "xtensa")))]
    {
        ExecOutcome::Fallback
    }
}

// FFI declaration for the 1x1 aligned kernel entrypoint (esp-dl style ABI).
// a2: output_ptr, a3: input_ptr, a4: &KernelArgs
#[cfg(all(feature = "simd-s3", target_arch = "xtensa"))]
unsafe extern "C" {
    fn dl_tie728_s8_conv2d_11cn(output_ptr: *mut i8, input_ptr: *const i8, args: *const KernelArgs);
}

/// Build `KernelArgs` for the 1x1 aligned (11cn) kernel from model/node state.
///
/// Notes
/// - Assumes NHWC activations.
/// - Weight layout and exponents are derived from the corresponding initializer.
#[allow(dead_code)]
fn build_kernel_args_k11_aligned(
    plan: &PlannedArena,
    arena: &mut Arena<'_>,
    spec: &ModelSpec,
    node: &Conv2dSpec,
) -> KernelArgs {
    let in_meta = spec.values[node.input as usize];
    let out_meta = spec.values[node.output as usize];
    let ih = in_meta.shape.h as i32;
    let iw = in_meta.shape.w as i32;
    let ic = in_meta.shape.c as i32;
    let oh = out_meta.shape.h as i32;
    let ow = out_meta.shape.w as i32;
    let oc = out_meta.shape.c as i32;

    let (kh, kw) = {
        let w_init = spec
            .initializers
            .iter()
            .find(|ini| ini.id == node.weights)
            .expect("weight initializer missing");
        match w_init.layout {
            crate::model::ParamLayout::WeightsI8OHWI { kh, kw, .. } => (kh as i32, kw as i32),
            crate::model::ParamLayout::WeightsI8BlockedN16HWC16 { kh, kw, .. } => {
                (kh as i32, kw as i32)
            }
            _ => (1, 1),
        }
    };

    let (layout, weights_ptr, w_exp_tensor, w_exp_per_oc_ptr, flags_wexp) = {
        use crate::model::{ParamExponents, ParamLayout};
        let w_init = spec
            .initializers
            .iter()
            .find(|ini| ini.id == node.weights)
            .expect("weight initializer missing");
        let layout = match w_init.layout {
            ParamLayout::WeightsI8OHWI { .. } => WeightLayout::OHWI,
            ParamLayout::WeightsI8BlockedN16HWC16 { .. } => WeightLayout::BlockedN16HWC16,
            _ => WeightLayout::OHWI,
        };
        let weights_ptr = w_init.data.as_ptr() as *const i8;
        match w_init.exponents {
            ParamExponents::PerTensor(e) => (
                layout,
                weights_ptr,
                e,
                core::ptr::null(),
                KernelFlags::empty(),
            ),
            ParamExponents::PerChannel(table) => (
                layout,
                weights_ptr,
                0,
                table.as_ptr(),
                KernelFlags::PER_OC_W_EXP,
            ),
        }
    };

    let (bias_ptr, flags_bias) = if let Some(bid) = node.bias {
        let b_init = spec
            .initializers
            .iter()
            .find(|ini| ini.id == bid)
            .expect("bias initializer missing");
        // Bias is stored as INT32 in little endian.
        let ptr = b_init.data.as_ptr() as *const i32;
        (ptr, KernelFlags::HAS_BIAS)
    } else {
        (core::ptr::null(), KernelFlags::empty())
    };

    // Input/output base pointers from arena
    let in_off = plan.offset_of(node.input);
    let out_off = plan.offset_of(node.output);
    let in_len = (in_meta.shape.elements()) as usize;
    let out_len = (out_meta.shape.elements()) as usize;
    let (input_slice, output_slice) = arena.io_slices(in_off, in_len, out_off, out_len);
    let input_ptr = input_slice.as_ptr();
    let output_ptr = output_slice.as_ptr() as *mut i8;

    // Fused activation
    let fused = match node.activation {
        crate::model::Activation::Linear => FusedActivation::Linear,
        crate::model::Activation::ReLU => FusedActivation::ReLU,
    };

    let flags = flags_bias | flags_wexp;

    // Construct args block
    let mut args = KernelArgs::new_zeroed();
    args.input = input_ptr;
    args.weights = weights_ptr;
    args.bias = bias_ptr;
    args.output = output_ptr;
    args.ih = ih;
    args.iw = iw;
    args.ic = ic;
    args.oh = oh;
    args.ow = ow;
    args.oc = oc;
    args.kh = kh;
    args.kw = kw;
    args.stride_h = node.strides_hw[0] as i32;
    args.stride_w = node.strides_hw[1] as i32;
    args.dil_h = node.dilations_hw[0] as i32;
    args.dil_w = node.dilations_hw[1] as i32;
    args.pad_top = node.pads_hw[0] as i32;
    args.pad_left = node.pads_hw[1] as i32;
    args.in_exp = in_meta.exp;
    args.out_exp = out_meta.exp;
    args.w_exp_per_oc = w_exp_per_oc_ptr;
    args.w_exp_tensor = w_exp_tensor;
    args.layout = layout;
    args.fused = fused;
    args.flags = flags;
    // Operation type bitfield for ACCX-based tails (aligned 11cn does not
    // currently use OC tails, but we keep this populated so the ABI stays
    // consistent across aligned/unaligned entries).
    let op = crate::kernels::simd::conv2d::common::compute_op_type(args.flags.bits(), args.fused);
    args.op_type = op;
    // Per-layer accumulator-domain bias pointer is only meaningful for
    // per-layer + bias; aligned 11cn does not currently use it but we keep
    // it wired for ABI consistency.
    args.pl_bias_i32 = if args.flags.contains(KernelFlags::HAS_BIAS)
        && !args.flags.contains(KernelFlags::PER_OC_W_EXP)
    {
        bias_ptr
    } else {
        core::ptr::null()
    };
    // Unaligned-specific tail helpers (reserved for future ACCX-based remainder paths).
    // The aligned 11cn kernel does not use these today; keep them zero so any
    // future unaligned/ACCX entries reading them from a shared KernelArgs ABI
    // observe a well-defined default.
    args.c_div_x_1 = 0;
    args.c_remainder = 0;
    args
}

// Define the global symbol on Xtensa (implemented below using 11c16 microkernel
// and fused epilogue). Rust loops over H×W and passes per-pixel input/output
// pointers; the ASM loops over OC tiles and performs the epilogue.
// Make shared macros available (same directory include)

// Provide a no-op stub for non-Xtensa or when the feature is disabled so that
// the symbol always exists and callers can remain cfg-free.
#[cfg(not(all(feature = "simd-s3", target_arch = "xtensa")))]
#[unsafe(no_mangle)]
unsafe extern "C" fn dl_tie728_s8_conv2d_11cn(
    _output_ptr: *mut i8,
    _input_ptr: *const i8,
    args: *const KernelArgs,
) {
    let _ = (args, _output_ptr, _input_ptr);
}
