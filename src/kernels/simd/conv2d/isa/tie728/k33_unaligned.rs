//! 3x3 (33cn) unaligned SIMD kernels for ESP32-S3 TIE728.
//!
//! This module provides the unaligned 3x3 (OHWI) TIE728 implementation.
//! It mirrors the aligned 3x3 implementation but handles unaligned pointers
//! and channel tails using the unaligned assembly kernel.

// Variables/imports used only when compiled for ESP32-S3 with simd-s3 feature.
#![allow(unused_variables, unused_imports, unused_assignments, dead_code)]

use crate::{
    arena::{Arena, PlannedArena},
    kernels::simd::conv2d::{
        ExecOutcome,
        common::{FusedActivation, KernelArgs, KernelFlags, WeightLayout},
    },
    model::{Conv2dSpec, ModelSpec},
    rounding::{RoundingMode, shift_round_i32},
};

#[inline(always)]
fn ceil16(x: usize) -> usize {
    (x + 15) & !15
}

/// Compute effective filter dimensions and start offsets for a border pixel.
/// Returns (effective_kh, effective_kw, filter_h_start, filter_w_start).
/// For fully valid pixels, returns (kh, kw, 0, 0).
/// For pixels outside the input entirely, effective dimensions may be <= 0.
#[inline(always)]
fn compute_effective_filter(
    in_y_origin: isize,
    in_x_origin: isize,
    ih: isize,
    iw: isize,
    kh: isize,
    kw: isize,
    dil_h: isize,
    dil_w: isize,
) -> (isize, isize, isize, isize) {
    // Compute how many kernel rows/cols to skip at start (head clipping)
    let filter_h_start = if in_y_origin < 0 {
        (-in_y_origin + dil_h - 1) / dil_h
    } else {
        0
    };
    let filter_w_start = if in_x_origin < 0 {
        (-in_x_origin + dil_w - 1) / dil_w
    } else {
        0
    };

    // Compute how many kernel rows/cols to skip at end (tail clipping)
    let max_y = in_y_origin + (kh - 1) * dil_h;
    let max_x = in_x_origin + (kw - 1) * dil_w;
    let filter_h_end_clip = if max_y >= ih {
        (max_y - ih + dil_h) / dil_h
    } else {
        0
    };
    let filter_w_end_clip = if max_x >= iw {
        (max_x - iw + dil_w) / dil_w
    } else {
        0
    };

    let effective_kh = kh - filter_h_start - filter_h_end_clip;
    let effective_kw = kw - filter_w_start - filter_w_end_clip;

    (effective_kh, effective_kw, filter_h_start, filter_w_start)
}

#[inline]
pub fn run(
    plan: &PlannedArena,
    arena: &mut Arena<'_>,
    spec: &ModelSpec,
    node: &Conv2dSpec,
    round: RoundingMode,
) -> ExecOutcome {
    // Preconditions:
    // - 3x3 kernel
    // - NHWC activations; N == 1
    // - Weights in OHWI layout
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
    if kh != 3 || kw != 3 {
        return ExecOutcome::Fallback;
    }

    // Check supported layout
    match w_init.layout {
        crate::model::ParamLayout::WeightsI8OHWI { .. } => {}
        crate::model::ParamLayout::WeightsI8BlockedN16HWC16 { .. } => {}
        _ => return ExecOutcome::Fallback,
    }

    // Build KernelArgs
    let mut args = build_kernel_args_k33_unaligned(plan, arena, spec, node);

    // Prepare per-layer vs per-channel buffers
    let has_bias = node.bias.is_some();
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
                // Per-channel vector epilogue via uniform i16 scale
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
                    let sbits = 15 - shift_i;
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
                // Per-layer; pack bias for QACC if needed
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

    // Prepare scalar fallback data if needed (for padding)
    let (bias_bytes, bias_exp_opt): (Option<&[u8]>, Option<i8>) = match node.bias {
        Some(bid) => {
            if let Some(b) = spec.initializers.iter().find(|ini| ini.id == bid) {
                (
                    Some(b.data),
                    match b.exponents {
                        crate::model::ParamExponents::PerTensor(e) => Some(e),
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
        crate::model::ParamExponents::PerTensor(e) => Some(e),
        _ => None,
    };
    let w_exp_per_oc: Option<&[i8]> = match w_init.exponents {
        crate::model::ParamExponents::PerChannel(slice) => Some(slice),
        _ => None,
    };
    let in_exp = in_meta.exp;
    let out_exp = out_meta.exp;

    // Pre-compute weight access parameters for scalar fallback (avoids match in inner loop)
    let w_data = w_init.data;
    let w_ptr = w_data.as_ptr() as *const i8;
    let ic_usize = in_meta.shape.c as usize;
    let oc_usize = out_meta.shape.c as usize;

    // Pre-compute strides based on layout (selected once, not per-weight-access)
    let is_blocked = matches!(
        w_init.layout,
        crate::model::ParamLayout::WeightsI8BlockedN16HWC16 { .. }
    );
    // OHWI strides for 3x3
    let ohwi_stride_oc = 9 * ic_usize;
    let ohwi_stride_ky = 3 * ic_usize;
    let ohwi_stride_kx = ic_usize;
    // BlockedN16HWC16 strides for 3x3 (used only for OHWI scalar fallback, not for blocked SIMD
    // border)
    let _blocked_block_size = 9 * ic_usize * 16;
    let _blocked_stride_ky = 3 * ic_usize * 16;
    let _blocked_stride_kx = ic_usize * 16;

    // Pre-compute requant params per OC (avoids repeated calculations in pixel loop)
    let mut requant_shift: [i32; 256] = [0; 256];
    let mut requant_scale: [i32; 256] = [0; 256];
    let mut bias_adj: [i32; 256] = [0; 256];

    for oo in 0..oc_usize.min(256) {
        let w_e = w_exp_per_oc
            .and_then(|s| s.get(oo))
            .copied()
            .or(w_exp_tensor)
            .unwrap_or(0);
        let (shift, scale) =
            crate::rounding::derive_shift_and_scale(in_exp.saturating_add(w_e), out_exp);
        requant_shift[oo] = shift;
        requant_scale[oo] = scale;

        // Pre-compute adjusted bias
        if bias_bytes.is_some() {
            let b = read_bias(oo);
            let e_acc = in_exp.saturating_add(w_e);
            if let Some(e_b) = bias_exp_opt {
                let delta = (e_b as i32) - (e_acc as i32);
                bias_adj[oo] = if delta == 0 {
                    b
                } else if delta > 0 {
                    let widened = (b as i64) << (delta as u32);
                    crate::rounding::saturate_i32(widened)
                } else {
                    let sh = (-delta) as i32;
                    shift_round_i32(b, sh, round)
                };
            } else {
                bias_adj[oo] = b;
            }
        }
    }

    // Iterate H×W in Rust, call ASM per pixel (esp-dl style).
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
        let dil_h = args.dil_h as isize;
        let dil_w = args.dil_w as isize;

        #[cfg(feature = "trace")]
        let mut simd_pixels: u32 = 0;
        #[cfg(feature = "trace")]
        let mut padding_pixels: u32 = 0;
        #[cfg(feature = "trace")]
        let mut simd_cycles: u64 = 0;
        #[cfg(feature = "trace")]
        let mut scalar_cycles: u64 = 0;

        // Dilation offsets are handled inside the kernel macro for the 3x3 window,
        // but we need to pass the base pointer for the top-left corner of the window.
        // The kernel expects `input_ptr` to point to (oy*stride_y, ox*stride_x).

        for oy in 0..oh {
            for ox in 0..ow {
                let in_y_origin = oy * stride_h - pad_top;
                let in_x_origin = ox * stride_w - pad_left;

                // Check if the 3x3 window is fully in bounds
                // Window extent: [origin, origin + 2*dil]
                let max_y = in_y_origin + 2 * dil_h;
                let max_x = in_x_origin + 2 * dil_w;

                let out_off = ((oy * ow) + ox) * oc;
                let out_ptr = (args.output).offset(out_off);

                if in_y_origin >= 0 && in_x_origin >= 0 && max_y < ih && max_x < iw {
                    // Valid pixel: call SIMD
                    #[cfg(feature = "trace")]
                    {
                        simd_pixels += 1;
                    }
                    #[cfg(feature = "trace")]
                    let t0 = crate::trace::read_ccount();

                    let in_off = ((in_y_origin * iw) + in_x_origin) * ic;
                    let in_ptr = (args.input).offset(in_off);
                    tie728_k33_unaligned(out_ptr, in_ptr as *const i8, &args as *const KernelArgs);

                    #[cfg(feature = "trace")]
                    {
                        simd_cycles += (crate::trace::read_ccount().wrapping_sub(t0)) as u64;
                    }
                } else {
                    // Border pixel: use SIMD with reduced filter dimensions (ESP-DL style)
                    #[cfg(feature = "trace")]
                    {
                        padding_pixels += 1;
                    }
                    #[cfg(feature = "trace")]
                    let t0 = crate::trace::read_ccount();

                    // Compute effective filter dimensions for this border pixel (3x3 kernel)
                    let (eff_kh, eff_kw, fh_start, fw_start) = compute_effective_filter(
                        in_y_origin,
                        in_x_origin,
                        ih,
                        iw,
                        3,
                        3,
                        dil_h,
                        dil_w,
                    );

                    if eff_kh > 0 && eff_kw > 0 && is_blocked {
                        // BlockedN16HWC16: use SIMD with reduced filter dimensions
                        let filter_c_n_offset = ic * 16;

                        // Compute filter pointer offset to skip clipped rows/cols at start
                        // For 3x3: Offset = (fh_start * 3 + fw_start) * ic * 16
                        let filter_start_offset = (fh_start * 3 + fw_start) * filter_c_n_offset;

                        // filter_y_offset: skip columns at end of each row (tail width clipping)
                        let border_filter_y_offset = (3 - eff_kw) * filter_c_n_offset;

                        // filter_n_offset: skip rows after processing effective_kh rows
                        let border_filter_n_offset = (3 - eff_kh) * 3 * filter_c_n_offset;

                        // Compute actual input position (after clipping)
                        let actual_y = in_y_origin + fh_start * dil_h;
                        let actual_x = in_x_origin + fw_start * dil_w;
                        let in_off = ((actual_y * iw) + actual_x) * ic;
                        let in_ptr = (args.input).offset(in_off);

                        // Adjust dilation_y_offset for reduced effective filter width
                        let border_dilation_y_offset = ic * (dil_h * iw - 1 - (eff_kw - 1) * dil_w);

                        // Create border args by bitwise copy (KernelArgs doesn't impl Copy)
                        let mut border_args = core::ptr::read(&args);
                        border_args.kh = eff_kh as i32;
                        border_args.kw = eff_kw as i32;
                        border_args.weights = args.weights.offset(filter_start_offset);
                        border_args.filter_y_offset = border_filter_y_offset as i32;
                        border_args.filter_n_offset = border_filter_n_offset as i32;
                        border_args.dilation_y_offset = border_dilation_y_offset as i32;

                        dl_tie728_s8_unaligned_conv2d_hwcn(
                            out_ptr,
                            in_ptr as *const i8,
                            &border_args as *const KernelArgs,
                        );
                    } else if eff_kh > 0 && eff_kw > 0 {
                        // OHWI layout fallback: use scalar for non-blocked weights
                        for oo in 0..oc {
                            let mut acc32: i32 = 0;
                            let oo_usize = oo as usize;

                            for ky in fh_start..fh_start + eff_kh {
                                for kx in fw_start..fw_start + eff_kw {
                                    let iy = in_y_origin + ky * dil_h;
                                    let ix = in_x_origin + kx * dil_w;
                                    let in_off = ((iy * iw) + ix) * ic;
                                    let in_ptr_base = (args.input).offset(in_off);
                                    let ky_usize = ky as usize;
                                    let kx_usize = kx as usize;

                                    for cc in 0..ic {
                                        let val = *in_ptr_base.offset(cc);
                                        let cc_usize = cc as usize;
                                        let offset = oo_usize * ohwi_stride_oc
                                            + ky_usize * ohwi_stride_ky
                                            + kx_usize * ohwi_stride_kx
                                            + cc_usize;
                                        let w = *w_ptr.add(offset);
                                        acc32 += (val as i32) * (w as i32);
                                    }
                                }
                            }

                            // Bias (use pre-computed adjusted bias)
                            if bias_bytes.is_some() {
                                let sum = (acc32 as i64) + (bias_adj[oo as usize] as i64);
                                acc32 = crate::rounding::saturate_i32(sum);
                            }

                            // Requant (use pre-computed shift/scale)
                            let mut y = crate::rounding::requant_i32_to_i8(
                                acc32,
                                requant_shift[oo as usize],
                                requant_scale[oo as usize],
                                round,
                            );

                            // Activation
                            if let crate::model::Activation::ReLU = node.activation {
                                if y < 0 {
                                    y = 0;
                                }
                            }

                            *out_ptr.offset(oo) = y;
                        }
                    }
                    // else: effective filter is 0x0 (output pixel fully outside input),
                    // output buffer was zero-initialized by arena

                    #[cfg(feature = "trace")]
                    {
                        scalar_cycles += (crate::trace::read_ccount().wrapping_sub(t0)) as u64;
                    }
                }
            }
        }
        #[cfg(feature = "trace")]
        {
            let simd_cy_per_px = if simd_pixels > 0 {
                simd_cycles / simd_pixels as u64
            } else {
                0
            };
            let scalar_cy_per_px = if padding_pixels > 0 {
                scalar_cycles / padding_pixels as u64
            } else {
                0
            };
            crate::ne_info!(
                "k33_unaligned id={}: simd={}px ({}cy, {}cy/px), pad={}px ({}cy, {}cy/px)",
                node.output,
                simd_pixels,
                simd_cycles,
                simd_cy_per_px,
                padding_pixels,
                scalar_cycles,
                scalar_cy_per_px
            );
        }
        return ExecOutcome::Executed;
    }
    #[cfg(not(all(feature = "simd-s3", target_arch = "xtensa")))]
    {
        ExecOutcome::Fallback
    }
}

#[cfg(all(feature = "simd-s3", target_arch = "xtensa"))]
unsafe extern "C" {
    fn tie728_k33_unaligned(output: *mut i8, input: *const i8, args: *const KernelArgs);
    fn dl_tie728_s8_unaligned_conv2d_hwcn(
        output_ptr: *mut i8,
        input_ptr: *const i8,
        args: *const KernelArgs,
    );
}

fn build_kernel_args_k33_unaligned(
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

    let (kh, kw) = (3, 3);

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
        (b_init.data.as_ptr() as *const i32, KernelFlags::HAS_BIAS)
    } else {
        (core::ptr::null(), KernelFlags::empty())
    };

    let in_off = plan.offset_of(node.input);
    let out_off = plan.offset_of(node.output);
    let in_len = (in_meta.shape.elements()) as usize;
    let out_len = (out_meta.shape.elements()) as usize;
    let (input_slice, output_slice) = arena.io_slices(in_off, in_len, out_off, out_len);
    let input_ptr = input_slice.as_ptr();
    let output_ptr = output_slice.as_ptr() as *mut i8;

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
    args.flags = flags_bias | flags_wexp;

    // Prepare per-layer vs per-channel buffers
    let has_bias = node.bias.is_some();
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
                // Per-channel vector epilogue via uniform i16 scale
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
                    let sbits = 15 - shift_i;
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
                // Per-layer; pack bias for QACC if needed
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

    // Operation type bitfield for ACCX-based tails
    args.op_type =
        crate::kernels::simd::conv2d::common::compute_op_type(args.flags.bits(), args.fused);

    // Per-layer accumulator-domain bias pointer (raw i32 bias table).
    args.pl_bias_i32 = if args.flags.contains(KernelFlags::HAS_BIAS)
        && !args.flags.contains(KernelFlags::PER_OC_W_EXP)
    {
        bias_ptr
    } else {
        core::ptr::null()
    };

    // Computed fields for ASM
    let ic_u = ic as i32;
    args.c_div_x_1 = (ic_u / 16) - 1;
    args.c_remainder = ic_u % 16;
    args.oc_tiles = oc / 16;
    args.oc_tail = oc % 16;

    // Dilation offsets
    // The 1x1 kernel advances input_ptr by ic.
    // So to move to the next pixel (dil_x away), we need to add (dil_x * ic) - ic.
    // To move to the next row (dil_y away), we need to adjust from the end of the 3rd pixel.
    let dil_x = args.dil_w;
    let dil_y = args.dil_h;
    args.dilation_x_offset = (dil_x * ic) - ic;
    args.dilation_y_offset = ic * (dil_y * iw - 1 - 2 * dil_x);

    args
}
