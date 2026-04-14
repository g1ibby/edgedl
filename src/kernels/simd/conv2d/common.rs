//! Shared structs and helpers for SIMD Conv2D entry points.
//!
//! Purpose
//! - Define a clear, FFI-safe call ABI for Xtensa TIE728 inline assembly kernels.
//! - Keep the Rust ↔ ASM boundary small and stable: a single `repr(C)` args block.
//! - Allow the ASM to load everything from a fixed base pointer (in a2), mirroring esp-dl’s style
//!   where the entry stub pulls fields into registers and jumps into the inner microkernel.
//!
//! Notes
//! - Call convention: our Xtensa ASM entries use the windowed ABI (`entry`/`retw`). Rust extern
//!   calls still place the first argument in `a2`, and the ASM stubs load the `KernelArgs` fields
//!   via `l32i` at the documented offsets.
//! - Alignment: the engine plans activations at 16-byte aligned offsets; the router decides aligned
//!   vs unaligned variants accordingly.
//! - Quantization: exponents use the power-of-two scheme (scale = 2^e). The ASM epilogue adds bias
//!   in the accumulator domain and requantizes to INT8 with HALF_UP (or HALF_EVEN when enabled)
//!   rounding.
//!
//! On non-Xtensa targets this struct compiles but is unused; the `global_asm!`
//! blocks are gated to the `xtensa` architecture.

use crate::rounding::{RoundingMode, saturate_i32, shift_round_i32};

/// Pack linear OC×i32 bias into the QACC-friendly 64-byte-per-16-lanes format
/// expected by `tie728_s8_conv2d_128b_vector_bias` on ESP32‑S3.
///
/// Behavior mirrors esp‑dl's TensorBase::reset_bias_layout (INT8 path) and
/// nn-engine's scalar Conv2D bias handling:
/// - First rescale each bias sample from its own exponent `e_b` into the accumulator domain
///   exponent `e_acc = in_exp + w_exp_tensor`, using the same logic as `conv2d_scalar` (left shift
///   with saturation when `e_b > e_acc`, right shift with HALF_UP when `e_b < e_acc`).
/// - Then narrow each accumulator-domain bias to 20 bits (mask 0x000F_FFFF) and compact two lanes
///   into 5 bytes: 0x000AAAAA 0x000BBBBB -> bytes AAAAABBBBB using the scheme: even: write low8,
///   then low16 of (val >> 8); odd:  OR low4 into previous byte's high nibble, then write low16 of
///   (val >> 4).
/// - After each group of 8 lanes, advance the output pointer to the next 16‑byte boundary (pad with
///   zeros). This yields 64 bytes for 16 lanes.
/// - Lanes >= oc are treated as zeros.
pub fn pack_bias_i32_for_qacc(
    dst: &mut [u8],
    src_le_i32: &[u8],
    oc: usize,
    e_b: i32,
    e_acc: i32,
    round: RoundingMode,
) {
    let tiles = (oc + 15) / 16;
    let need = tiles * 64;
    assert!(dst.len() >= need, "dst buffer too small for QACC bias pack");

    // Zero entire destination up front to simplify padding.
    for b in &mut dst[..need] {
        *b = 0;
    }

    for t in 0..tiles {
        let base_out = t * 64;
        let mut p = 0usize; // byte offset within this 64‑byte tile
        for i in 0..16usize {
            let idx = t * 16 + i;
            let mut v20: u32 = 0;
            if idx < oc {
                let off = idx * 4;
                if off + 4 <= src_le_i32.len() {
                    let b0 = src_le_i32[off] as u32;
                    let b1 = src_le_i32[off + 1] as u32;
                    let b2 = src_le_i32[off + 2] as u32;
                    let b3 = src_le_i32[off + 3] as u32;
                    let v = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
                    let v_i32 = v as i32;
                    // Rescale bias from its exponent into accumulator domain.
                    let delta = e_b - e_acc;
                    let adj: i32 = if delta == 0 {
                        v_i32
                    } else if delta > 0 {
                        // Left shift with saturation when bias has coarser scale.
                        let widened = (v_i32 as i64) << (delta as u32);
                        saturate_i32(widened)
                    } else {
                        // Right shift with rounding when bias has finer scale.
                        shift_round_i32(v_i32, -delta, round)
                    };
                    // Keep low 20 bits (two's complement) for QACC pack.
                    v20 = (adj as u32) & 0x000F_FFFF;
                }
            }

            if (i & 1) == 1 {
                // odd lane: merge low 4 bits into previous byte's high nibble, then write 16 bits
                // of (v >> 4)
                if p == 0 {
                    // Should not happen for i>=1; guard anyway
                    continue;
                }
                let low4 = (v20 & 0xF) as u8;
                dst[base_out + p - 1] |= low4 << 4;
                let rest = v20 >> 4;
                dst[base_out + p] = (rest & 0xFF) as u8;
                dst[base_out + p + 1] = ((rest >> 8) & 0xFF) as u8;
                p += 2;
            } else {
                // even lane: write low 8, then 16 bits of (v >> 8)
                dst[base_out + p] = (v20 & 0xFF) as u8;
                let rest = v20 >> 8;
                p += 1;
                dst[base_out + p] = (rest & 0xFF) as u8;
                dst[base_out + p + 1] = ((rest >> 8) & 0xFF) as u8;
                p += 2;
            }

            // After 8 and 16 lanes, align to next 16‑byte boundary within the tile
            if ((i + 1) % 8) == 0 {
                let mis = (base_out + p) & 0xF;
                if mis != 0 {
                    p += 16 - mis;
                }
            }
        }
        // p should be <= 64; remaining bytes already zeroed
    }
}

#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum WeightLayout {
    OHWI            = 0,
    BlockedN16HWC16 = 1,
}

#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum FusedActivation {
    Linear = 0,
    ReLU   = 1,
}

bitflags::bitflags! {
    /// Flags bitfield for microkernel behavior toggles.
    #[repr(transparent)]
    #[derive(Default)]
    pub struct KernelFlags: u32 {
        const HAS_BIAS        = 1 << 0;
        const PER_OC_W_EXP    = 1 << 1; // weight exponent per-OC table vs per-tensor
        const RESERVED_2      = 1 << 2;
        const RESERVED_3      = 1 << 3;
    }
}

/// Arguments passed to ASM microkernels. Keep field order stable.
#[repr(C)]
pub struct KernelArgs {
    // Pointers (base addresses)
    /// NHWC input activation base pointer (i8)
    pub input: *const i8,
    /// Weights base pointer (i8) in physical layout (OHWI or BlockedN16HWC16)
    pub weights: *const i8,
    /// Bias base pointer (i32), optional; enabled via HAS_BIAS flag
    pub bias: *const i32,
    /// NHWC output activation base pointer (i8)
    pub output: *mut i8,

    // Dimensions (activation + kernel)
    /// Input height/width/channels
    pub ih: i32,
    pub iw: i32,
    pub ic: i32,
    /// Output height/width/channels (OC is full count, not tiled)
    pub oh: i32,
    pub ow: i32,
    pub oc: i32,
    /// Kernel size (KH × KW)
    pub kh: i32,
    pub kw: i32,

    // Stride/dilation/padding (NHWC)
    /// Stride in H/W
    pub stride_h: i32,
    pub stride_w: i32,
    /// Dilation in H/W
    pub dil_h: i32,
    pub dil_w: i32,
    /// Top/left padding (bottom/right derivable from shapes if needed)
    pub pad_top: i32,
    pub pad_left: i32,

    // Quantization exponents
    /// Input activation exponent (scale = 2^in_exp)
    pub in_exp: i8,
    /// Output activation exponent (scale = 2^out_exp)
    pub out_exp: i8,
    pub _pad_q0: u8,
    pub _pad_q1: u8,
    pub _pad_q2: u8,
    /// Pointer to per-OC weight exponents; used when PER_OC_W_EXP flag is set
    pub w_exp_per_oc: *const i8,
    /// Per-tensor weight exponent used when PER_OC_W_EXP is not set
    pub w_exp_tensor: i8,
    pub _pad_q3: u8,
    pub _pad_q4: u8,
    pub _pad_q5: u8,

    // Layout + fused activation + flags
    /// Physical weight layout selector (OHWI or BlockedN16HWC16)
    pub layout: WeightLayout,
    /// Fused activation (Linear or ReLU)
    pub fused: FusedActivation,
    /// Behavioral flags (HAS_BIAS, PER_OC_W_EXP)
    pub flags: KernelFlags,

    // Per‑OC scaling (optional)
    /// Pointer to per‑OC fixed‑point scale factors (i16), 16 values per OC tile.
    /// Used by per‑channel requantization path (`PER_OC_W_EXP`). Layout: for each
    /// tile of 16 output channels, two consecutive 128‑bit loads (32 bytes) —
    /// the kernel computes tile_idx*32 to address the scale block.
    pub pc_scales: *const i16,
    /// Pointer to per‑OC bias values in i16 domain (see ASM per‑channel path).
    /// Layout: per tile 16×i16 stored as 32 bytes contiguous addressed as
    /// (tile_idx*32). Only used when PER_OC_W_EXP and HAS_BIAS are set.
    pub pc_bias_i16: *const i16,

    // Unaligned / tail helpers (ACCX-based remainder path)
    /// Compact operation-type bitfield for ACCX-based tails. We intentionally
    /// use a reduced encoding compared to esp-dl:
    ///   bit 0: 1 = per-channel (PER_OC_W_EXP), 0 = per-layer
    ///   bit 1: 1 = HAS_BIAS, 0 = no bias
    ///   bit 2: 1 = ReLU, 0 = Linear
    /// This is enough for the combinations currently supported by nn-engine
    /// (Linear/ReLU × per-layer/per-channel × bias/no-bias) while keeping the
    /// field extensible if we add more activations later.
    pub op_type: i32,
    /// Pointer to per-layer bias in accumulator domain (i32 per OC), used by
    /// OC-tail per-layer path to mirror the QACC bias preload used for full
    /// 16-lane tiles. Only meaningful when HAS_BIAS is set and PER_OC_W_EXP
    /// is clear; otherwise may be null.
    pub pl_bias_i32: *const i32,
    // IC helpers for unaligned 11cn kernels
    /// For unaligned 11cn kernels, we mirror esp-dl's use of `c_div_x_1`
    /// (input_channel / 16 - 1) and `c_remainder` (input_channel % 16) in the
    /// ACCX-based OC/N-tail path. Keeping them in `KernelArgs` avoids
    /// recomputing them in assembly and keeps the ABI aligned with esp-dl.
    pub c_div_x_1: i32,
    /// Reserved for esp-dl style `c_remainder` (number of residual input
    /// channels in the final partial block; in esp-dl this is stored as
    /// `c_remainder = (input_channel % u) * sizeof(feature_t)`).
    pub c_remainder: i32,

    // OC tiling helpers (unaligned 11cn)
    /// Number of full 16-lane output tiles (oc / 16). We mirror esp-dl by
    /// precomputing this on the host and loading it in ASM to avoid integer
    /// division instructions in the entry stub (unsupported on ESP32-S3).
    pub oc_tiles: i32,
    /// Output-channel tail count (oc % 16) for the scalar OC-tail epilogue.
    pub oc_tail: i32,

    // 3x3 helpers
    /// Input pointer offset for dilation in X (width).
    pub dilation_x_offset: i32,
    /// Input pointer offset for dilation in Y (height).
    pub dilation_y_offset: i32,
    /// Fused activation parameter alpha (for PReLU/LeakyReLU). Appended at the end
    /// to maintain backward compatibility with kernels that don't consume it.
    pub activation_alpha: i32,
    /// Fused activation shift (for ReLU/PReLU), negative to indicate disabled.
    /// Appended at the end to maintain backward compatibility.
    pub activation_shift: i32,

    // Filter offset helpers for hwc16 macro (stored in q6 vector register)
    /// Filter row offset for hwc16 macro. For BlockedN16HWC16 with full filter
    /// (interior pixels), this is 0. For partial filters at edges, this would be
    /// (actual_kw - effective_kw) * ic * 16. Matches esp-dl's filter_y_offset.
    pub filter_y_offset: i32,
    /// Filter OC-tile offset for hwc16 macro. For BlockedN16HWC16 with full filter
    /// (interior pixels), this is 0. Matches esp-dl's filter_n_offset field.
    pub filter_n_offset: i32,
}

impl KernelArgs {
    #[inline]
    pub fn new_zeroed() -> Self {
        // Safe default; caller must initialize all fields
        use core::ptr::null;
        Self {
            input: null(),
            weights: null(),
            bias: null(),
            output: null_mut(),
            ih: 0,
            iw: 0,
            ic: 0,
            oh: 0,
            ow: 0,
            oc: 0,
            kh: 0,
            kw: 0,
            stride_h: 0,
            stride_w: 0,
            dil_h: 0,
            dil_w: 0,
            pad_top: 0,
            pad_left: 0,
            in_exp: 0,
            out_exp: 0,
            _pad_q0: 0,
            _pad_q1: 0,
            _pad_q2: 0,
            w_exp_per_oc: null(),
            w_exp_tensor: 0,
            _pad_q3: 0,
            _pad_q4: 0,
            _pad_q5: 0,
            layout: WeightLayout::OHWI,
            fused: FusedActivation::Linear,
            flags: KernelFlags::empty(),
            pc_scales: null(),
            pc_bias_i16: null(),
            op_type: 0,
            pl_bias_i32: null(),
            c_div_x_1: 0,
            c_remainder: 0,
            oc_tiles: 0,
            oc_tail: 0,
            dilation_x_offset: 0,
            dilation_y_offset: 0,
            activation_alpha: 0,
            activation_shift: -1,
            filter_y_offset: 0,
            filter_n_offset: 0,
        }
    }
}

/// Compute the compact `op_type` bitfield for ACCX-based tails from the
/// higher-level flags (as raw bits) and fused activation.
#[inline]
pub fn compute_op_type(flags_bits: u32, fused: FusedActivation) -> i32 {
    let mut op: i32 = 0;
    if (flags_bits & KernelFlags::PER_OC_W_EXP.bits()) != 0 {
        op |= 1; // per-channel
    }
    if (flags_bits & KernelFlags::HAS_BIAS.bits()) != 0 {
        op |= 1 << 1; // bias present
    }
    if matches!(fused, FusedActivation::ReLU) {
        op |= 1 << 2; // ReLU
    }
    op
}

use core::ptr::null_mut;
