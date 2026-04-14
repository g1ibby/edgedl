//! Quantization helpers for SIMD paths.
//!
//! These wrap scalar rounding/shift utilities to keep a single source of truth
//! for requantization behavior.

use crate::rounding::{RoundingMode, derive_shift_and_scale, requant_i32_to_i8};

#[allow(dead_code)]
#[inline]
pub fn requant(acc: i32, in_exp: i8, w_exp: i8, out_exp: i8, round: RoundingMode) -> i8 {
    let (shift, scale) = derive_shift_and_scale(in_exp.saturating_add(w_exp), out_exp);
    requant_i32_to_i8(acc, shift, scale, round)
}
