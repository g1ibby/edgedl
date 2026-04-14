//! TIE728 SIMD-accelerated FFT for ESP32-S3.
//!
//! This module provides a real FFT (RFFT) implementation using TIE728 SIMD
//! instructions on ESP32-S3. On other targets, it falls back to microfft.
//!
//! The implementation is ported from esp-dl's `dl_fft2r_fc32_aes3_` assembly.

mod tables;

core::arch::global_asm!(include_str!("tie728_fft2r.S"));

pub use tables::{BITREV_512, RFFT_POST_1024, TWIDDLE_1024};

/// Complex f32 representation for FFT output.
#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub struct Complex32 {
    pub re: f32,
    pub im: f32,
}

impl Complex32 {
    #[inline]
    pub const fn new(re: f32, im: f32) -> Self {
        Self { re, im }
    }
}

// FFI declaration for the TIE728 radix-2 complex FFT.
// Signature: dl_tie728_fft2r_fc32(data: *mut f32, n: i32, twiddle_table: *const f32)
// - data: interleaved complex float array (re0, im0, re1, im1, ...) - in-place
// - n: FFT size (number of complex samples, e.g., 512 for 1024-point RFFT)
// - twiddle_table: precomputed cos/sin pairs
unsafe extern "C" {
    fn dl_tie728_fft2r_fc32(data: *mut f32, n: i32, twiddle_table: *const f32);
}

/// Bit-reverse permutation for 512 complex values in-place using precomputed LUT.
/// Input data is interleaved [re0, im0, re1, im1, ...]
/// Uses BITREV_512 lookup table for O(1) index lookup instead of O(log n) computation.
#[inline]
fn bitrev_fc32_inplace_512(data: &mut [f32]) {
    for i in 0..512 {
        let j = BITREV_512[i] as usize;
        if i < j {
            // Swap complex values at positions i and j
            data.swap(2 * i, 2 * j);
            data.swap(2 * i + 1, 2 * j + 1);
        }
    }
}

/// Perform 1024-point real FFT using TIE728 SIMD on ESP32-S3.
///
/// Input: 1024 real f32 samples (will be modified in-place)
/// Output: 513 complex bins (DC to Nyquist)
///
/// On non-ESP32-S3 targets, falls back to microfft.
#[allow(dead_code)]
pub fn rfft_1024_simd(data: &mut [f32; 1024]) -> [Complex32; 513] {
    // Real FFT algorithm (following esp-dl):
    // 1. Treat N real samples as N/2 complex samples (pack pairs)
    // 2. Perform N/2-point complex FFT
    // 3. Apply bit-reversal to FFT output
    // 4. Apply post-processing to recover N/2+1 unique complex bins

    unsafe {
        // Step 1: Data is already packed as N/2 complex pairs (re[0],re[1] -> complex[0], etc.)
        // The real-to-complex packing interprets data[2k], data[2k+1] as complex[k]

        // Check alignment for SIMD
        assert!(
            (data.as_ptr() as usize) % 16 == 0,
            "Input buffer must be 16-byte aligned for SIMD FFT"
        );

        // Step 2: Perform N/2 = 512-point complex FFT in-place
        dl_tie728_fft2r_fc32(data.as_mut_ptr(), 512, TWIDDLE_1024.0.as_ptr());

        // Step 3: Apply bit-reversal to FFT output using precomputed LUT
        bitrev_fc32_inplace_512(data.as_mut_slice());

        // Step 4: RFFT post-processing to extract N/2+1 complex bins from N/2 complex FFT output
        let mut output = [Complex32::default(); 513];
        rfft_post_process_1024(data, &mut output);
        output
    }
}

/// Perform 1024-point real FFT using TIE728 SIMD, writing to provided output buffer.
///
/// This variant avoids stack allocation of the 513-element output array (~4KB),
/// which is critical for embedded systems with limited stack space.
///
/// Input: 1024 real f32 samples (will be modified in-place)
/// Output: writes 513 complex bins to provided buffer
pub fn rfft_1024_simd_into(data: &mut [f32; 1024], output: &mut [Complex32; 513]) {
    unsafe {
        // Check alignment for SIMD
        assert!(
            (data.as_ptr() as usize) % 16 == 0,
            "Input buffer must be 16-byte aligned for SIMD FFT"
        );

        // Step 1: Perform N/2 = 512-point complex FFT in-place
        dl_tie728_fft2r_fc32(data.as_mut_ptr(), 512, TWIDDLE_1024.0.as_ptr());

        // Step 2: Apply bit-reversal to FFT output
        bitrev_fc32_inplace_512(data.as_mut_slice());

        // Step 3: RFFT post-processing to extract N/2+1 complex bins
        rfft_post_process_1024(data, output);
    }
}

/// RFFT post-processing: convert N/2-point complex FFT to N/2+1 real FFT bins.
///
/// This implements the standard real FFT post-processing:
/// X[k] = 0.5 * (Z[k] + conj(Z[N/2-k])) - 0.5j * W_N^k * (Z[k] - conj(Z[N/2-k]))
///
/// Where Z is the N/2-point complex FFT output and W_N^k are twiddle factors.
/// 2x unrolled for better instruction pipelining.
fn rfft_post_process_1024(data: &[f32; 1024], output: &mut [Complex32; 513]) {
    const N: usize = 1024;
    const N2: usize = N / 2; // 512

    // Interpret data as N2 complex values
    let z = unsafe { core::slice::from_raw_parts(data.as_ptr() as *const Complex32, N2) };

    // DC component: X[0] = Z[0].re + Z[0].im (purely real)
    output[0] = Complex32::new(z[0].re + z[0].im, 0.0);

    // Nyquist component: X[N/2] = Z[0].re - Z[0].im (purely real)
    output[N2] = Complex32::new(z[0].re - z[0].im, 0.0);

    // Middle bins: k = 1..N/2-1 (2x unrolled)
    // Process 2 bins per iteration for better pipelining
    let mut k = 1usize;
    while k + 1 < N2 {
        // --- Process bin k ---
        let zk = z[k];
        let zn_k = z[N2 - k];

        let f1k_re = zk.re + zn_k.re;
        let f1k_im = zk.im - zn_k.im;
        let f2k_re = zk.re - zn_k.re;
        let f2k_im = zk.im + zn_k.im;

        let cos_k = RFFT_POST_1024.0[2 * k];
        let sin_k = RFFT_POST_1024.0[2 * k + 1];
        let c = -sin_k;
        let s = -cos_k;

        let tw_re = c * f2k_re - s * f2k_im;
        let tw_im = s * f2k_re + c * f2k_im;

        output[k] = Complex32::new(0.5 * (f1k_re + tw_re), 0.5 * (f1k_im + tw_im));

        // --- Process bin k+1 ---
        let k1 = k + 1;
        let zk1 = z[k1];
        let zn_k1 = z[N2 - k1];

        let f1k1_re = zk1.re + zn_k1.re;
        let f1k1_im = zk1.im - zn_k1.im;
        let f2k1_re = zk1.re - zn_k1.re;
        let f2k1_im = zk1.im + zn_k1.im;

        let cos_k1 = RFFT_POST_1024.0[2 * k1];
        let sin_k1 = RFFT_POST_1024.0[2 * k1 + 1];
        let c1 = -sin_k1;
        let s1 = -cos_k1;

        let tw1_re = c1 * f2k1_re - s1 * f2k1_im;
        let tw1_im = s1 * f2k1_re + c1 * f2k1_im;

        output[k1] = Complex32::new(0.5 * (f1k1_re + tw1_re), 0.5 * (f1k1_im + tw1_im));

        k += 2;
    }

    // Handle remaining odd bin (k = 511)
    if k < N2 {
        let zk = z[k];
        let zn_k = z[N2 - k];

        let f1k_re = zk.re + zn_k.re;
        let f1k_im = zk.im - zn_k.im;
        let f2k_re = zk.re - zn_k.re;
        let f2k_im = zk.im + zn_k.im;

        let cos_k = RFFT_POST_1024.0[2 * k];
        let sin_k = RFFT_POST_1024.0[2 * k + 1];
        let c = -sin_k;
        let s = -cos_k;

        let tw_re = c * f2k_re - s * f2k_im;
        let tw_im = s * f2k_re + c * f2k_im;

        output[k] = Complex32::new(0.5 * (f1k_re + tw_re), 0.5 * (f1k_im + tw_im));
    }
}
