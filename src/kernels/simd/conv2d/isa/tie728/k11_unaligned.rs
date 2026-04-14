//! 1x1 (11cn) unaligned SIMD kernels for ESP32-S3 TIE728.
//!
//! This module provides the unaligned 1×1 (OHWI) TIE728 implementation. It mirrors the
//! aligned entry’s ABI and `KernelArgs` layout, but the ASM entry handles input/output
//! misalignment and OC/IC tails using unaligned loads and specialized stores.

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
    // Stage 1 support policy (bring-up):
    // - Handle misaligned output stores while requiring IC and OC be multiples of 16.
    // - Require input base to be 16-byte aligned (unaligned input loads will be added later).
    // Preconditions: 1x1 NHWC, N==1, stride==1, dilation==1, no padding
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

    match w_init.layout {
        crate::model::ParamLayout::WeightsI8OHWI { .. } => {}
        crate::model::ParamLayout::WeightsI8BlockedN16HWC16 { .. } => {}
        _ => return ExecOutcome::Fallback,
    }

    // Build KernelArgs using the same preparation as the aligned path (bias pack or per-channel
    // tables)
    let args = build_kernel_args_k11_unaligned(plan, arena, spec, node);

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

    // Iterate H×W in Rust, call ASM per pixel (esp-dl style). ASM handles
    // unaligned output stores; input loads are assumed aligned in this stage.
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
                    #[cfg(feature = "trace")]
                    {
                        simd_pixels += 1;
                    }
                    let in_off = ((in_y * iw) + in_x) * ic;
                    let in_ptr = (args.input).offset(in_off);

                    dl_tie728_s8_unaligned_conv2d_11cn(
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
                "k11_unaligned id={}: simd={}px ({}%), pad={}px",
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

// a2: output_ptr, a3: input_ptr, a4: &KernelArgs
#[cfg(all(feature = "simd-s3", target_arch = "xtensa"))]
unsafe extern "C" {
    fn dl_tie728_s8_unaligned_conv2d_11cn(
        output_ptr: *mut i8,
        input_ptr: *const i8,
        args: *const KernelArgs,
    );
}

#[allow(dead_code)]
fn build_kernel_args_k11_unaligned(
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

    let (layout, weights_ptr, w_exp_tensor, w_exp_per_oc_ptr, _flags_wexp) = {
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

    let (bias_ptr, _flags_bias) = if let Some(bid) = node.bias {
        let b_init = spec
            .initializers
            .iter()
            .find(|ini| ini.id == bid)
            .expect("bias initializer missing");
        (b_init.data.as_ptr() as *const i32, KernelFlags::HAS_BIAS)
    } else {
        (core::ptr::null(), KernelFlags::empty())
    };

    // Input/output base pointers from arena
    let in_off = plan.offset_of(node.input);
    let out_off = plan.offset_of(node.output);
    let in_len = (in_meta.shape.elements()) as usize;
    let out_len = (out_meta.shape.elements()) as usize;
    let (input_slice, output_slice) =
        crate::arena::Arena::io_slices(arena, in_off, in_len, out_off, out_len);
    let input_ptr = input_slice.as_ptr();
    let output_ptr = output_slice.as_ptr() as *mut i8;

    // Fused activation
    let fused = match node.activation {
        crate::model::Activation::Linear => FusedActivation::Linear,
        crate::model::Activation::ReLU => FusedActivation::ReLU,
    };

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

    // Prepare per-layer vs per-channel buffers (reuse aligned policy)
    let has_bias = node.bias.is_some();
    // Re-fetch exponents for branching
    let w_init_e = spec
        .initializers
        .iter()
        .find(|ini| ini.id == node.weights)
        .expect("weight initializer missing");
    match w_init_e.exponents {
        crate::model::ParamExponents::PerTensor(w_e_tensor) => {
            let oc_u = out_meta.shape.c as usize;
            let in_exp = in_meta.exp as i32;
            let out_exp = out_meta.exp as i32;
            let shift_i = in_exp + (w_e_tensor as i32) - out_exp;
            if shift_i > 0 {
                // per-channel vector epilogue via uniform i16 scale
                let oc_aligned = ceil16(oc_u);
                let scales_bytes = oc_aligned * core::mem::size_of::<i16>();
                let bias_bytes = if has_bias {
                    oc_aligned * core::mem::size_of::<i16>()
                } else {
                    0
                };
                let needed = scales_bytes + bias_bytes;
                if needed <= plan.scratch_bytes {
                    let scratch = arena.scratch_slice_mut(plan, needed);
                    let (scales_bytes_slice, bias_bytes_slice) = scratch.split_at_mut(scales_bytes);
                    let sbits = 15 - shift_i; // expect 0..14
                    if sbits >= 0 && sbits <= 14 {
                        let val = 1i32 << (sbits as u32);
                        let s16 = core::cmp::min(val, i32::from(i16::MAX)) as i16;
                        let scales_ptr = scales_bytes_slice.as_mut_ptr() as *mut i16;
                        for i in 0..oc_u {
                            unsafe {
                                *scales_ptr.add(i) = s16;
                            }
                        }
                        args.pc_scales = scales_bytes_slice.as_ptr() as *const i16;
                        if has_bias {
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
                            let bias_ptr16 = bias_bytes_slice.as_mut_ptr() as *mut i16;
                            let target_e = (in_exp + (w_e_tensor as i32)) - 4;
                            for i in 0..oc_u {
                                let base = i * 4;
                                if base + 4 > b_bytes.len() {
                                    break;
                                }
                                let bv = i32::from_le_bytes([
                                    b_bytes[base],
                                    b_bytes[base + 1],
                                    b_bytes[base + 2],
                                    b_bytes[base + 3],
                                ]);
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
                                    shift_round_i32(
                                        bv,
                                        (-delta) as i32,
                                        crate::rounding::RoundingMode::HalfUp,
                                    )
                                };
                                unsafe {
                                    *bias_ptr16.add(i) =
                                        adj.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
                                }
                            }
                            args.pc_bias_i16 = bias_bytes_slice.as_ptr() as *const i16;
                        }
                        args.flags |= KernelFlags::PER_OC_W_EXP;
                    }
                }
            } else {
                // per-layer; pack bias for QACC if needed
                if has_bias {
                    let oc_u = out_meta.shape.c as usize;
                    let tiles = (oc_u + 15) / 16;
                    let bytes_needed = tiles * 64;
                    if bytes_needed <= plan.scratch_bytes {
                        let b_init = spec
                            .initializers
                            .iter()
                            .find(|ini| Some(ini.id) == node.bias)
                            .expect("bias initializer missing");
                        let b_bytes = b_init.data;
                        let dst = arena.scratch_slice_mut(plan, bytes_needed);
                        let dst_u8: &mut [u8] = unsafe {
                            core::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut u8, dst.len())
                        };
                        let e_b = match b_init.exponents {
                            crate::model::ParamExponents::PerTensor(e) => e as i32,
                            _ => 0,
                        };
                        let e_acc = in_exp + (w_e_tensor as i32);
                        crate::kernels::simd::conv2d::common::pack_bias_i32_for_qacc(
                            dst_u8,
                            b_bytes,
                            oc_u,
                            e_b,
                            e_acc,
                            crate::rounding::DEFAULT_ROUNDING,
                        );
                        args.bias = dst_u8.as_ptr() as *const i32;
                    }
                }
            }
        }
        crate::model::ParamExponents::PerChannel(w_exps) => {
            let oc_u = out_meta.shape.c as usize;
            let in_exp = in_meta.exp as i32;
            let out_exp = out_meta.exp as i32;
            let oc_aligned = ceil16(oc_u);
            let scales_bytes = oc_aligned * core::mem::size_of::<i16>();
            let bias_bytes = if has_bias {
                oc_aligned * core::mem::size_of::<i16>()
            } else {
                0
            };
            let needed = scales_bytes + bias_bytes;
            if needed <= plan.scratch_bytes {
                let scratch = arena.scratch_slice_mut(plan, needed);
                let (scales_bytes_slice, bias_bytes_slice) = scratch.split_at_mut(scales_bytes);
                let scales_ptr = scales_bytes_slice.as_mut_ptr() as *mut i16;
                for i in 0..oc_u {
                    let we = w_exps[i] as i32;
                    let fcf_i = in_exp + we - out_exp;
                    let sbits = 15 - fcf_i;
                    if sbits < 0 || sbits > 14 {
                        continue;
                    }
                    let val = 1i32 << (sbits as u32);
                    unsafe {
                        *scales_ptr.add(i) = core::cmp::min(val, i32::from(i16::MAX)) as i16;
                    }
                }
                args.pc_scales = scales_bytes_slice.as_ptr() as *const i16;
                if has_bias {
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
                    let bias_ptr16 = bias_bytes_slice.as_mut_ptr() as *mut i16;
                    for i in 0..oc_u {
                        let base = i * 4;
                        if base + 4 > b_bytes.len() {
                            break;
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
                            shift_round_i32(
                                bv,
                                (-delta) as i32,
                                crate::rounding::RoundingMode::HalfUp,
                            )
                        };
                        unsafe {
                            *bias_ptr16.add(i) = adj.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
                        }
                    }
                    args.pc_bias_i16 = bias_bytes_slice.as_ptr() as *const i16;
                }
                args.flags |= KernelFlags::PER_OC_W_EXP;
            }
        }
    }

    // Flags baseline (HAS_BIAS already implied via bias_ptr above if needed)
    if node.bias.is_some() {
        args.flags |= KernelFlags::HAS_BIAS;
    }
    // Operation type bitfield for ACCX-based tails
    let op = crate::kernels::simd::conv2d::common::compute_op_type(args.flags.bits(), args.fused);
    args.op_type = op;
    // Per-layer accumulator-domain bias pointer (raw i32 bias table). Only
    // meaningful when we are in the per-layer path (PER_OC_W_EXP clear) and
    // HAS_BIAS is set; otherwise null.
    args.pl_bias_i32 = if args.flags.contains(KernelFlags::HAS_BIAS)
        && !args.flags.contains(KernelFlags::PER_OC_W_EXP)
    {
        bias_ptr
    } else {
        core::ptr::null()
    };
    // Unaligned-specific tail helpers.
    // For the unaligned 11cn kernel we mirror esp-dl's `ArgsType` by
    // precomputing `c_div_x_1` and `c_remainder` here in Rust and passing
    // them into the Xtensa entry. The inner unaligned IC kernel
    // (`tie728_s8_conv2d_1_1_unaligned_c`) and any future ACCX-based
    // OC/N-tail path both rely on these fields, so keeping them in
    // `KernelArgs` avoids re-deriving them in assembly and keeps the ABI
    // aligned with esp-dl.
    let ic_u = in_meta.shape.c as i32;
    args.c_div_x_1 = (ic_u >> 4) - 1; // input_channel / 16 - 1 (esp-dl ArgsType::c_div_x_1)
    args.c_remainder = ic_u & 0xF; // input_channel % 16     (esp-dl ArgsType::c_remainder / sizeof(feature_t))
    // Precompute OC tiling to avoid divisions in ASM (ESP32-S3 lacks DIV32).
    args.oc_tiles = (oc >> 4) as i32; // full 16-lane tiles
    args.oc_tail = (oc & 0xF) as i32; // remainder lanes
    args
}

// Non-Xtensa stub for the symbol to keep linkage stable when simd-s3 is disabled
#[cfg(not(all(feature = "simd-s3", target_arch = "xtensa")))]
#[unsafe(no_mangle)]
unsafe extern "C" fn dl_tie728_s8_unaligned_conv2d_11cn(
    _output_ptr: *mut i8,
    _input_ptr: *const i8,
    _args: *const KernelArgs,
) {
    let _ = (_args, _output_ptr, _input_ptr);
}
