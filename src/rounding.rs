//! no_std-friendly rounding and requantization helpers matching esp-dl semantics.
//! Default rounding for ESP32-S3 (scalar/SIMD) is HALF_UP. A cfg feature can
//! switch to HALF_EVEN to model other targets.

#![allow(dead_code)]

use core::cmp::max;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RoundingMode {
    HalfUp,
    HalfEven,
}

#[cfg(feature = "round-half-even")]
pub const DEFAULT_ROUNDING: RoundingMode = RoundingMode::HalfEven;
#[cfg(not(feature = "round-half-even"))]
pub const DEFAULT_ROUNDING: RoundingMode = RoundingMode::HalfUp;

#[inline]
pub fn saturate_i8(x: i32) -> i8 {
    if x > 127 {
        127
    } else if x < -128 {
        -128
    } else {
        x as i8
    }
}

#[inline]
pub fn saturate_i32(x: i64) -> i32 {
    if x > i32::MAX as i64 {
        i32::MAX
    } else if x < i32::MIN as i64 {
        i32::MIN
    } else {
        x as i32
    }
}

/// Shift-right with rounding-half-up for signed 32-bit integers.
/// If `shift <= 0`, this performs a left shift with saturation (no rounding).
#[inline]
pub fn shift_round_half_up_i32(value: i32, shift: i32) -> i32 {
    if shift <= 0 {
        let sh = (-shift) as u32;
        let widened = (value as i64) << sh;
        return saturate_i32(widened);
    }
    let sh = shift as u32;
    let half: i64 = 1i64 << (sh - 1);
    let widened = value as i64;
    // Matches esp-dl shift_and_round_half_up: (value + half) >> shift
    let shifted = (widened + half) >> sh;
    saturate_i32(shifted)
}

/// Shift-right with rounding-half-even (bankers rounding) for signed 32-bit integers.
/// If `shift <= 0`, this performs a left shift with saturation (no rounding).
#[inline]
pub fn shift_round_half_even_i32(value: i32, shift: i32) -> i32 {
    if shift <= 0 {
        let sh = (-shift) as u32;
        let widened = (value as i64) << sh;
        return saturate_i32(widened);
    }
    let sh = shift as u32;
    let denom: i64 = 1i64 << sh;
    let v = value as i64;
    // Arithmetic floor division via shift for powers of two
    let q = (v >> sh) as i64; // floor toward -inf
    let r = v - (q << sh); // remainder in [0, denom)
    let half = denom >> 1;
    let mut q_adj = q;
    if r > half || (r == half && (q & 1) != 0) {
        q_adj += 1;
    }
    saturate_i32(q_adj)
}

/// Generic shift-right with the selected rounding mode.
#[inline]
pub fn shift_round_i32(value: i32, shift: i32, mode: RoundingMode) -> i32 {
    match mode {
        RoundingMode::HalfUp => shift_round_half_up_i32(value, shift),
        RoundingMode::HalfEven => shift_round_half_even_i32(value, shift),
    }
}

/// Derive (output_shift, output_scale) such that:
/// - If output_shift >= 0: y = shift_round(x, output_shift)
/// - If output_shift < 0:  y = saturate((x * output_scale) << (-output_shift_is_zero))
/// Here we mirror esp-dl’s DL_SCALE/DL_RESCALE usage: if (out_exp - in_exp) < 0,
/// use a multiplicative scale of 1 << (-(out_exp - in_exp)). Otherwise, shift.
#[inline]
pub fn derive_shift_and_scale(e_in: i8, e_out: i8) -> (i32, i32) {
    // Convert from accumulator exponent e_in to output exponent e_out.
    // y = round(x * 2^(e_in - e_out))
    let diff = (e_in as i32) - (e_out as i32);
    if diff < 0 {
        // Right shift by -diff with rounding
        (-diff, 1)
    } else if diff == 0 {
        (0, 1)
    } else {
        // Multiply by 2^diff (exact power-of-two scaling)
        (0, 1i32 << (diff as u32))
    }
}

/// Apply requantization from i32 accumulator domain to i8 with rounding and saturation.
/// If `output_scale` > 1, performs multiplication then (optional) left shift (always exact in M1
/// usage), otherwise performs right shift with rounding.
#[inline]
pub fn requant_i32_to_i8(acc: i32, output_shift: i32, output_scale: i32, mode: RoundingMode) -> i8 {
    if output_scale > 1 {
        let widened = (acc as i64) * (output_scale as i64);
        // No right shift in this branch for M1; saturate to i32 then clamp to i8.
        let acc32 = saturate_i32(widened);
        saturate_i8(acc32)
    } else {
        let shifted = shift_round_i32(acc, max(0, output_shift), mode);
        saturate_i8(shifted)
    }
}
