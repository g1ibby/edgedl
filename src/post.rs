//! Post-processing helpers for model outputs (no_std friendly).
//!
//! - Implements softmax over logits and a convenience path to compute probabilities directly from
//!   quantized INT8 outputs with a power-of-two exponent (scale = 2^exp).
//! - Avoids allocation by writing into caller-provided buffers.

use libm::{expf, powf};

/// Dequantize INT8 slice to f32 using power-of-two exponent (scale = 2^exp).
///
/// - `input.len()` must equal `output.len()`.
#[inline]
pub fn dequantize_i8_to_f32(input: &[i8], exp: i8, output: &mut [f32]) {
    assert!(
        input.len() == output.len(),
        "dequantize_i8_to_f32: length mismatch"
    );
    let scale = powf(2.0f32, exp as f32);
    for (i, &q) in input.iter().enumerate() {
        output[i] = (q as f32) * scale;
    }
}

/// Numerically-stable softmax: out[i] = exp(x[i] - max) / sum_j exp(x[j] - max)
///
/// - `input.len()` must equal `output.len()`.
#[inline]
pub fn softmax(input: &[f32], output: &mut [f32]) {
    assert!(input.len() == output.len(), "softmax: length mismatch");
    let n = input.len();
    if n == 0 {
        return;
    }
    // Subtract max for numerical stability
    let mut max_v = input[0];
    for &v in input.iter().skip(1) {
        if v > max_v {
            max_v = v;
        }
    }
    let mut sum = 0.0f32;
    for i in 0..n {
        let e = expf(input[i] - max_v);
        output[i] = e;
        sum += e;
    }
    // Avoid division by zero on degenerate inputs
    let denom = if sum > 0.0 { sum } else { 1.0 };
    for i in 0..n {
        output[i] /= denom;
    }
}

/// Compute probabilities from quantized INT8 logits and exponent (scale = 2^exp)
/// into `probs` using a stable softmax.
///
/// - `logits_q.len()` must equal `probs.len()`.
/// - Uses single-pass dequantization inside the softmax for zero-allocation.
#[inline]
pub fn softmax_from_i8(logits_q: &[i8], exp: i8, probs: &mut [f32]) {
    assert!(
        logits_q.len() == probs.len(),
        "softmax_from_i8: length mismatch"
    );
    let n = logits_q.len();
    if n == 0 {
        return;
    }
    // Find max in dequantized domain: max_f = (max_q as f32) * scale
    let mut max_q = logits_q[0];
    for &q in logits_q.iter().skip(1) {
        if q > max_q {
            max_q = q;
        }
    }
    let scale = powf(2.0f32, exp as f32);
    let max_f = (max_q as f32) * scale;

    // Compute exp((q*scale) - max_f) and sum
    let mut sum = 0.0f32;
    for i in 0..n {
        let v = (logits_q[i] as f32) * scale - max_f;
        let e = expf(v);
        probs[i] = e;
        sum += e;
    }

    // Normalize to probabilities
    let denom = if sum > 0.0 { sum } else { 1.0 };
    for i in 0..n {
        probs[i] /= denom;
    }
}
