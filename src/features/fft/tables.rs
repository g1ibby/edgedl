//! Precomputed twiddle factor tables for 1024-point FFT.
//!
//! These tables are generated at compile time for the radix-2 FFT algorithm.
//!
//! IMPORTANT: The FFT twiddle table (TWIDDLE_1024) must be in BIT-REVERSED order
//! to match what the esp-dl TIE728 FFT assembly expects. This is because esp-dl
//! generates the twiddle table and then applies bit-reversal to it.
//!
//! The RFFT post-processing table (RFFT_POST_1024) stays in natural order since
//! post-processing is applied after bit-reversal.

use core::f32::consts::PI;

/// Aligned wrapper for SIMD compatibility (requires 16-byte alignment for 128-bit loads)
#[repr(C, align(16))]
pub struct AlignedTable<const N: usize>(pub [f32; N]);

/// Twiddle factors for 512-point complex FFT (used in 1024-point RFFT).
/// IMPORTANT: Stored in BIT-REVERSED order for esp-dl TIE728 FFT.
/// Total: 512 pairs = 1024 floats
pub static TWIDDLE_1024: AlignedTable<1024> = AlignedTable(generate_twiddle_512_bitrev());

/// Post-processing twiddle factors for converting 512-point complex FFT to 1024-point RFFT.
/// Format: [cos(2π*0/1024), sin(2π*0/1024), cos(2π*1/1024), sin(2π*1/1024), ...]
/// Total: 512 pairs = 1024 floats (natural order)
pub static RFFT_POST_1024: AlignedTable<1024> = AlignedTable(generate_rfft_post_1024());

/// Bit-reversal permutation table for 512-point FFT (9-bit indices).
/// Entry i contains bit_reverse(i, 9).
/// Used to eliminate runtime bit-reversal calculation in FFT.
pub static BITREV_512: [u16; 512] = generate_bitrev_512();

/// Generate bit-reversal table at compile time.
const fn generate_bitrev_512() -> [u16; 512] {
    let mut table = [0u16; 512];
    let mut i = 0usize;
    while i < 512 {
        table[i] = const_bit_reverse(i, 9) as u16;
        i += 1;
    }
    table
}

/// Bit-reverse an index with the given number of bits.
const fn const_bit_reverse(mut x: usize, bits: usize) -> usize {
    let mut result = 0usize;
    let mut b = 0;
    while b < bits {
        result = (result << 1) | (x & 1);
        x >>= 1;
        b += 1;
    }
    result
}

/// Generate twiddle factors for 512-point complex FFT in BIT-REVERSED order.
/// This matches esp-dl's `dl_gen_fftr2_table_f32` which applies bit-reversal after generation.
///
/// CRITICAL: Use N=512 (complex FFT size), NOT N=1024 (RFFT size)!
/// esp-dl generates twiddles based on the complex FFT size: W_512^k = cos(2πk/512) + j*sin(2πk/512)
/// for k=0..255 (N/2 entries), then applies 8-bit bit-reversal.
const fn generate_twiddle_512_bitrev() -> [f32; 1024] {
    const N: usize = 512; // Complex FFT size (NOT RFFT size!)
    const N_HALF: usize = 256; // Number of twiddle entries (N/2)
    const LOG2N_HALF: usize = 8; // log2(256) = 8 for bit-reversal

    let mut table = [0.0f32; 1024]; // Buffer larger than needed, only first 512 floats used

    // Generate twiddles for k=0..255, stored at bit-reversed positions
    // esp-dl stores (cos(2πk/N), sin(2πk/N)) - POSITIVE sin
    let mut k = 0usize;
    while k < N_HALF {
        let angle = 2.0 * PI * (k as f32) / (N as f32); // 2πk/512
        let cos_val = const_cos(angle);
        let sin_val = const_sin(angle); // POSITIVE to match esp-dl

        // Store at bit-reversed position (8-bit reversal for 256 entries)
        let j = const_bit_reverse(k, LOG2N_HALF);
        table[2 * j] = cos_val;
        table[2 * j + 1] = sin_val;
        k += 1;
    }
    table
}

/// Generate RFFT post-processing twiddle factors.
/// For converting N/2-point complex FFT to N-point real FFT.
/// W_N^k = cos(2πk/N) - j*sin(2πk/N) for k = 0..N/2
const fn generate_rfft_post_1024() -> [f32; 1024] {
    const N: usize = 1024;
    let mut table = [0.0f32; 1024];
    let mut k = 0usize;
    while k < N / 2 {
        let angle = 2.0 * PI * (k as f32) / (N as f32);
        table[2 * k] = const_cos(angle);
        table[2 * k + 1] = const_sin(angle);
        k += 1;
    }
    table
}

/// Const-compatible cosine using Taylor series approximation.
/// Accurate to ~1e-6 for the range we need.
const fn const_cos(x: f32) -> f32 {
    // Normalize angle to [-π, π]
    let x = normalize_angle(x);

    // Taylor series: cos(x) = 1 - x²/2! + x⁴/4! - x⁶/6! + ...
    let x2 = x * x;
    let x4 = x2 * x2;
    let x6 = x4 * x2;
    let x8 = x4 * x4;
    let x10 = x6 * x4;
    let x12 = x6 * x6;
    let x14 = x8 * x6;

    1.0 - x2 / 2.0 + x4 / 24.0 - x6 / 720.0 + x8 / 40320.0 - x10 / 3628800.0 + x12 / 479001600.0
        - x14 / 87178291200.0
}

/// Const-compatible sine using Taylor series approximation.
const fn const_sin(x: f32) -> f32 {
    // Normalize angle to [-π, π]
    let x = normalize_angle(x);

    // Taylor series: sin(x) = x - x³/3! + x⁵/5! - x⁷/7! + ...
    let x2 = x * x;
    let x3 = x2 * x;
    let x5 = x3 * x2;
    let x7 = x5 * x2;
    let x9 = x7 * x2;
    let x11 = x9 * x2;
    let x13 = x11 * x2;
    let x15 = x13 * x2;

    x - x3 / 6.0 + x5 / 120.0 - x7 / 5040.0 + x9 / 362880.0 - x11 / 39916800.0 + x13 / 6227020800.0
        - x15 / 1307674368000.0
}

/// Normalize angle to [-π, π] range for better Taylor series convergence.
const fn normalize_angle(x: f32) -> f32 {
    let two_pi = 2.0 * PI;
    // Reduce to [0, 2π)
    let mut x = x % two_pi;
    if x < 0.0 {
        x += two_pi;
    }
    // Reduce to [-π, π]
    if x > PI {
        x -= two_pi;
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Bit-reverse for test verification (runtime version)
    fn bit_reverse_test(x: usize, bits: usize) -> usize {
        let mut result = 0;
        let mut x = x;
        for _ in 0..bits {
            result = (result << 1) | (x & 1);
            x >>= 1;
        }
        result
    }

    #[test]
    fn test_twiddle_accuracy_bitrev() {
        use libm::{cosf, sinf};

        // Check a few twiddle values against libm
        // Note: TWIDDLE_1024 is in BIT-REVERSED order
        // IMPORTANT: Twiddles are W_512^k (complex FFT size), NOT W_1024^k (RFFT size)
        // Only 256 entries (k=0..255) are generated
        // Tolerance tightened to 1e-6 with higher-order Taylor series
        const TOLERANCE: f32 = 1e-6;

        for k in [0, 1, 64, 128, 255] {
            let angle = 2.0 * PI * (k as f32) / 512.0; // Complex FFT size
            let expected_cos = cosf(angle);
            let expected_sin = sinf(angle); // POSITIVE to match esp-dl

            // Twiddle for natural index k is stored at bit-reversed position (8-bit)
            let j = bit_reverse_test(k, 8);
            let actual_cos = TWIDDLE_1024.0[2 * j];
            let actual_sin = TWIDDLE_1024.0[2 * j + 1];

            assert!(
                (actual_cos - expected_cos).abs() < TOLERANCE,
                "cos mismatch at k={} (bitrev={}): {} vs {}",
                k,
                j,
                actual_cos,
                expected_cos
            );
            assert!(
                (actual_sin - expected_sin).abs() < TOLERANCE,
                "sin mismatch at k={} (bitrev={}): {} vs {}",
                k,
                j,
                actual_sin,
                expected_sin
            );
        }
    }

    #[test]
    fn test_rfft_post_accuracy() {
        use libm::{cosf, sinf};

        // Check RFFT post-processing twiddles
        // Tolerance tightened to 1e-6 with higher-order Taylor series
        const TOLERANCE: f32 = 1e-6;

        for k in [0, 1, 128, 256, 511] {
            let angle = 2.0 * PI * (k as f32) / 1024.0;
            let expected_cos = cosf(angle);
            let expected_sin = sinf(angle);

            let actual_cos = RFFT_POST_1024.0[2 * k];
            let actual_sin = RFFT_POST_1024.0[2 * k + 1];

            assert!(
                (actual_cos - expected_cos).abs() < TOLERANCE,
                "rfft cos mismatch at k={}: {} vs {}",
                k,
                actual_cos,
                expected_cos
            );
            assert!(
                (actual_sin - expected_sin).abs() < TOLERANCE,
                "rfft sin mismatch at k={}: {} vs {}",
                k,
                actual_sin,
                expected_sin
            );
        }
    }
}
