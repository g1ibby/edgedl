#![allow(clippy::needless_range_loop)]

use core::f32::consts::PI;

use libm::{floorf, log10f, logf, powf, roundf, sqrtf};
use microfft::real::rfft_1024;

// SIMD-specific imports and declarations for ESP32-S3
#[cfg(all(feature = "simd-s3", target_arch = "xtensa"))]
use crate::features::fft::{Complex32, rfft_1024_simd, rfft_1024_simd_into};

#[cfg(all(feature = "simd-s3", target_arch = "xtensa"))]
core::arch::global_asm!(include_str!("tie728_mel.S"));

#[cfg(all(feature = "simd-s3", target_arch = "xtensa"))]
unsafe extern "C" {
    #[allow(dead_code)]
    fn dl_tie728_apply_window_i16_f32(wave: *const i16, window: *const f32, out: *mut f32, n: i32);
    fn dl_tie728_apply_window_i16_f32_x4(
        wave: *const i16,
        window: *const f32,
        out: *mut f32,
        n: i32,
    );
    fn dl_tie728_power_spectrum_f32(complex_in: *const f32, power_out: *mut f32, n_bins: i32);
    fn dl_tie728_mel_accumulate(power: *const f32, count: i32, start_w: f32, w_step: f32) -> f32;
    fn dl_tie728_normalize_f32(data: *mut f32, count: i32, mean: f32, inv_std: f32);
    fn dl_tie728_log_f32(data: *mut f32, count: i32, eps: f32);
}

/// Scratch buffers for mel spectrogram computation.
/// Allocate statically or on heap to reduce stack usage from ~13KB to ~1KB.
#[cfg(all(feature = "simd-s3", target_arch = "xtensa"))]
#[repr(C, align(16))]
pub struct MelScratch {
    /// Windowed samples for FFT (4 KB)
    pub frame: [f32; 1024],
    /// Power spectrum output (2 KB)
    pub power: [f32; 513],
    /// PCM samples with padding (2 KB)
    pub wave_buf: [i16; 1024],
    /// FFT complex output (4 KB)
    pub fft_out: [Complex32; 513],
}

#[cfg(all(feature = "simd-s3", target_arch = "xtensa"))]
impl MelScratch {
    /// Create a new zeroed MelScratch buffer.
    pub const fn new() -> Self {
        Self {
            frame: [0.0; 1024],
            power: [0.0; 513],
            wave_buf: [0; 1024],
            fft_out: [Complex32 { re: 0.0, im: 0.0 }; 513],
        }
    }
}

/// Precomputed Hann window for N=1024.
/// This avoids recomputing cosf() 1024 times per function call.
/// Made public for parity testing.
pub static HANN_1024: [f32; 1024] = generate_hann_1024();

/// Generate Hann window at compile time.
const fn generate_hann_1024() -> [f32; 1024] {
    let mut hann = [0.0f32; 1024];
    let mut n = 0usize;
    while n < 1024 {
        // Hann: 0.5 - 0.5*cos(2πn/(N-1))
        let angle = 2.0 * PI * (n as f32) / 1023.0;
        hann[n] = 0.5 - 0.5 * const_cos(angle);
        n += 1;
    }
    hann
}

/// Const-compatible cosine using Taylor series.
const fn const_cos(x: f32) -> f32 {
    // Normalize to [-π, π] for better convergence
    let x = const_normalize_angle(x);
    let x2 = x * x;
    let x4 = x2 * x2;
    let x6 = x4 * x2;
    let x8 = x4 * x4;
    let x10 = x6 * x4;
    1.0 - x2 / 2.0 + x4 / 24.0 - x6 / 720.0 + x8 / 40320.0 - x10 / 3628800.0
}

const fn const_normalize_angle(x: f32) -> f32 {
    let two_pi = 2.0 * PI;
    let mut x = x % two_pi;
    if x < 0.0 {
        x += two_pi;
    }
    if x > PI {
        x -= two_pi;
    }
    x
}

/// Quantize mel spectrogram values by a fixed exponent.
///
/// The inner loop uses `black_box(T)` to prevent LLVM from pre-computing
/// `M * T * sizeof(f32)` as a constant. The Xtensa LLVM backend (esp-clang
/// >= 20.1.1) cannot select certain large negative constants produced by
/// array-stride folding.
#[inline(never)]
pub fn quantize_by_engine_exp<const M: usize, const T: usize>(
    mel: &[[f32; T]; M],
    e_in: i8,
    out: &mut [i8],
) {
    assert_eq!(out.len(), M * T);
    let scale = powf(2.0f32, e_in as f32);
    let mut idx = 0usize;
    for m in 0..M {
        for t in 0..core::hint::black_box(T) {
            let v = mel[m][t] / scale;
            let q = roundf(v) as i32;
            out[idx] = q.clamp(-128, 127) as i8;
            idx += 1;
        }
    }
}

#[inline]
fn reflect_get_i16(x: &[i16], idx: isize) -> i16 {
    // Reflect pad indexing over [0, len)
    let n = x.len() as isize;
    if n == 0 {
        return 0;
    }
    let mut i = idx;
    if i < 0 {
        i = -i; // reflect across -0.5
    }
    let period = 2 * n;
    let mut i_mod = i % period;
    if i_mod < 0 {
        i_mod += period;
    }
    let j = if i_mod < n { i_mod } else { period - 1 - i_mod };
    x[j as usize]
}

#[inline]
fn hz_to_mel_htk(f: f32) -> f32 {
    2595.0 * log10f(1.0 + f / 700.0)
}

#[inline]
fn mel_to_hz_htk(m: f32) -> f32 {
    700.0 * (powf(10.0, m / 2595.0) - 1.0)
}

/// Compute simd log-mel spectrogram with per-example normalization.
///
/// - wave_i16: mono PCM i16 at sample rate `sr`
/// - sr: sample rate (expected 32000)
/// - n_fft: FFT size (only 1024 supported here)
/// - hop_length: hop in samples (e.g., 320 for 10 ms @ 32 kHz)
/// - fmin_hz..fmax_hz: mel band edges
/// - log_eps: epsilon added before log to prevent -inf
/// - center: reflect-pad by n_fft/2 on both sides when true
/// - out[M][T]: destination buffer (M mels by T frames)
#[cfg(all(feature = "simd-s3", target_arch = "xtensa"))]
pub fn compute_log_mel_simd<const M: usize, const T: usize>(
    wave_i16: &[i16],
    sr: usize,
    n_fft: usize,
    hop_length: usize,
    fmin_hz: f32,
    fmax_hz: f32,
    log_eps: f32,
    center: bool,
    out: &mut [[f32; T]; M],
) {
    assert!(n_fft == 1024, "only n_fft=1024 supported in MCU frontend");
    let n_fft_usize: usize = 1024;

    // Guard: compiled for a specific maximum mel bands
    const MAX_MELS: usize = 64;
    debug_assert!(M <= MAX_MELS, "M={} exceeds MAX_MELS={}", M, MAX_MELS);

    // Use precomputed Hann window (eliminates 1024 cosf calls per invocation)
    let hann = &HANN_1024;

    // PCM i16 -> f32 [-1,1]
    let len = wave_i16.len();

    // Compute frames count consistent with Python reference (center reflect padding)
    let pad = if center { n_fft_usize / 2 } else { 0 };
    let total = len + 2 * pad;
    let frames = if total >= n_fft_usize {
        1 + (total - n_fft_usize) / hop_length
    } else {
        1
    };
    assert!(frames == T, "frame count mismatch: {} != {}", frames, T);

    // Prepare mel bin edges using HTK mel scale
    let mut bins_storage = [0usize; MAX_MELS + 2];
    let m_min = hz_to_mel_htk(fmin_hz);
    let m_max = hz_to_mel_htk(fmax_hz);
    // Linspace over mels
    for i in 0..(M + 2) {
        let t = i as f32 / ((M + 1) as f32);
        let mel = m_min + t * (m_max - m_min);
        let hz = mel_to_hz_htk(mel);
        let bin = floorf(((n_fft_usize + 1) as f32) * (hz / (sr as f32))) as isize;
        let clamped = if bin < 0 { 0 } else { bin };
        bins_storage[i] = clamped as usize;
    }

    // Work buffers
    #[repr(C, align(16))]
    struct AlignedFrame([f32; 1024]);
    let mut frame = AlignedFrame([0f32; 1024]);
    let mut power = [0f32; 513];

    // Precompute mel bin edges (same for all frames)
    let last_valid = 512;
    let mut b0 = [0usize; MAX_MELS + 2];
    for i in 0..(M + 2) {
        let mut v = bins_storage[i];
        if v > last_valid {
            v = last_valid;
        }
        b0[i] = v;
    }
    for mm in 1..(M + 1) {
        if b0[mm - 1] == b0[mm] {
            b0[mm] = core::cmp::min(b0[mm] + 1, last_valid);
        }
        if b0[mm] == b0[mm + 1] {
            b0[mm + 1] = core::cmp::min(b0[mm + 1] + 1, last_valid);
        }
    }

    // Compute mel energies per frame
    for t in 0..T {
        // Gather frame with reflect padding
        let base: isize = (t * hop_length) as isize - (pad as isize);

        // Use a temporary buffer for the gathered i16 samples to enable SIMD windowing
        let mut wave_buf = [0i16; 1024];

        // Fast path: when frame is fully within buffer bounds (no padding needed)
        if base >= 0 && (base as usize + n_fft_usize) <= len {
            // Direct slice copy - eliminates ~97% of reflect_get_i16 calls
            wave_buf.copy_from_slice(&wave_i16[base as usize..base as usize + n_fft_usize]);
        } else {
            // Slow path: reflect padding (only ~4 frames at edges)
            for n in 0..n_fft_usize {
                let idx = base + (n as isize);
                wave_buf[n] = reflect_get_i16(wave_i16, idx);
            }
        }

        unsafe {
            // Use 4x unrolled windowing for better throughput
            dl_tie728_apply_window_i16_f32_x4(
                wave_buf.as_ptr(),
                hann.as_ptr(),
                frame.0.as_mut_ptr(),
                1024,
            );
        }

        // SIMD FFT returns 513 unpacked bins
        let spec = rfft_1024_simd(&mut frame.0);

        // SIMD power spectrum
        unsafe {
            dl_tie728_power_spectrum_f32(spec.as_ptr() as *const f32, power.as_mut_ptr(), 513);
        }

        // Triangular mel filters (SIMD)
        for m in 0..M {
            let f_m_minus = b0[m];
            let f_m = b0[m + 1];
            let f_m_plus = b0[m + 2];

            let mut acc = 0.0f32;

            // Part 1: Increasing slope (f_m_minus to f_m)
            if f_m_minus < f_m {
                let count = (f_m - f_m_minus) as i32;
                let denom = count as f32;
                let inv_denom = 1.0 / denom;

                unsafe {
                    acc += dl_tie728_mel_accumulate(
                        power.as_ptr().add(f_m_minus),
                        count,
                        0.0,       // start_w
                        inv_denom, // w_step
                    );
                }
            }

            // Part 2: Decreasing slope (f_m to f_m_plus)
            if f_m < f_m_plus {
                let count = (f_m_plus - f_m) as i32;
                let denom = count as f32;
                let inv_denom = 1.0 / denom;

                unsafe {
                    acc += dl_tie728_mel_accumulate(
                        power.as_ptr().add(f_m),
                        count,
                        1.0,        // start_w
                        -inv_denom, // w_step
                    );
                }
            }
            out[m][t] = acc;
        }
    }

    // Log scaling and per-example normalization
    let eps = log_eps.max(1e-12);

    // 1. Logarithm in assembly (in-place)
    unsafe {
        dl_tie728_log_f32(out.as_mut_ptr() as *mut f32, (M * T) as i32, eps);
    }

    // 2. Compute mean/std in Rust (scalar loop)
    let mut sum = 0.0f32;
    let mut sumsq = 0.0f32;
    // Flattened iteration
    for m in 0..M {
        for t in 0..T {
            let v = out[m][t];
            sum += v;
            sumsq += v * v;
        }
    }
    let count = (M * T) as f32;
    let mean = sum / count;
    let var = (sumsq / count) - mean * mean;
    let std = sqrtf(var.max(0.0)).max(1e-6);
    let inv_std = 1.0f32 / std;

    // 3. Normalization in assembly (SIMD)
    for m in 0..M {
        unsafe {
            dl_tie728_normalize_f32(out[m].as_mut_ptr(), T as i32, mean, inv_std);
        }
    }
}

/// Compute log-mel spectrogram using pre-allocated scratch buffers.
///
/// This variant uses external scratch buffers instead of stack-allocated arrays,
/// reducing stack usage from ~13KB to ~1KB. Critical for embedded systems with
/// limited stack space (e.g., ESP32-S3 dual-core with separate stack per core).
///
/// Arguments are identical to `compute_log_mel_simd`, plus:
/// - scratch: pre-allocated MelScratch buffer (can be static or heap-allocated)
#[cfg(all(feature = "simd-s3", target_arch = "xtensa"))]
pub fn compute_log_mel_simd_with_scratch<const M: usize, const T: usize>(
    wave_i16: &[i16],
    sr: usize,
    n_fft: usize,
    hop_length: usize,
    fmin_hz: f32,
    fmax_hz: f32,
    log_eps: f32,
    center: bool,
    out: &mut [[f32; T]; M],
    scratch: &mut MelScratch,
) {
    assert!(n_fft == 1024, "only n_fft=1024 supported in MCU frontend");
    let n_fft_usize: usize = 1024;

    const MAX_MELS: usize = 64;
    debug_assert!(M <= MAX_MELS, "M={} exceeds MAX_MELS={}", M, MAX_MELS);

    let hann = &HANN_1024;
    let len = wave_i16.len();

    let pad = if center { n_fft_usize / 2 } else { 0 };
    let total = len + 2 * pad;
    let frames = if total >= n_fft_usize {
        1 + (total - n_fft_usize) / hop_length
    } else {
        1
    };
    assert!(frames == T, "frame count mismatch: {} != {}", frames, T);

    // Prepare mel bin edges using HTK mel scale
    let mut bins_storage = [0usize; MAX_MELS + 2];
    let m_min = hz_to_mel_htk(fmin_hz);
    let m_max = hz_to_mel_htk(fmax_hz);
    for i in 0..(M + 2) {
        let t = i as f32 / ((M + 1) as f32);
        let mel = m_min + t * (m_max - m_min);
        let hz = mel_to_hz_htk(mel);
        let bin = floorf(((n_fft_usize + 1) as f32) * (hz / (sr as f32))) as isize;
        let clamped = if bin < 0 { 0 } else { bin };
        bins_storage[i] = clamped as usize;
    }

    // Precompute mel bin edges
    let last_valid = 512;
    let mut b0 = [0usize; MAX_MELS + 2];
    for i in 0..(M + 2) {
        let mut v = bins_storage[i];
        if v > last_valid {
            v = last_valid;
        }
        b0[i] = v;
    }
    for mm in 1..(M + 1) {
        if b0[mm - 1] == b0[mm] {
            b0[mm] = core::cmp::min(b0[mm] + 1, last_valid);
        }
        if b0[mm] == b0[mm + 1] {
            b0[mm + 1] = core::cmp::min(b0[mm + 1] + 1, last_valid);
        }
    }

    // Compute mel energies per frame using scratch buffers
    for t in 0..T {
        let base: isize = (t * hop_length) as isize - (pad as isize);

        // Fast path: when frame is fully within buffer bounds
        if base >= 0 && (base as usize + n_fft_usize) <= len {
            scratch
                .wave_buf
                .copy_from_slice(&wave_i16[base as usize..base as usize + n_fft_usize]);
        } else {
            // Slow path: reflect padding
            for n in 0..n_fft_usize {
                let idx = base + (n as isize);
                scratch.wave_buf[n] = reflect_get_i16(wave_i16, idx);
            }
        }

        unsafe {
            dl_tie728_apply_window_i16_f32_x4(
                scratch.wave_buf.as_ptr(),
                hann.as_ptr(),
                scratch.frame.as_mut_ptr(),
                1024,
            );
        }

        // SIMD FFT into scratch buffer (avoids 4KB stack allocation)
        rfft_1024_simd_into(&mut scratch.frame, &mut scratch.fft_out);

        // SIMD power spectrum
        unsafe {
            dl_tie728_power_spectrum_f32(
                scratch.fft_out.as_ptr() as *const f32,
                scratch.power.as_mut_ptr(),
                513,
            );
        }

        // Triangular mel filters (SIMD)
        for m in 0..M {
            let f_m_minus = b0[m];
            let f_m = b0[m + 1];
            let f_m_plus = b0[m + 2];

            let mut acc = 0.0f32;

            if f_m_minus < f_m {
                let count = (f_m - f_m_minus) as i32;
                let denom = count as f32;
                let inv_denom = 1.0 / denom;

                unsafe {
                    acc += dl_tie728_mel_accumulate(
                        scratch.power.as_ptr().add(f_m_minus),
                        count,
                        0.0,
                        inv_denom,
                    );
                }
            }

            if f_m < f_m_plus {
                let count = (f_m_plus - f_m) as i32;
                let denom = count as f32;
                let inv_denom = 1.0 / denom;

                unsafe {
                    acc += dl_tie728_mel_accumulate(
                        scratch.power.as_ptr().add(f_m),
                        count,
                        1.0,
                        -inv_denom,
                    );
                }
            }
            out[m][t] = acc;
        }
    }

    // Log scaling and per-example normalization
    let eps = log_eps.max(1e-12);

    unsafe {
        dl_tie728_log_f32(out.as_mut_ptr() as *mut f32, (M * T) as i32, eps);
    }

    let mut sum = 0.0f32;
    let mut sumsq = 0.0f32;
    for m in 0..M {
        for t in 0..T {
            let v = out[m][t];
            sum += v;
            sumsq += v * v;
        }
    }
    let count = (M * T) as f32;
    let mean = sum / count;
    let var = (sumsq / count) - mean * mean;
    let std = sqrtf(var.max(0.0)).max(1e-6);
    let inv_std = 1.0f32 / std;

    for m in 0..M {
        unsafe {
            dl_tie728_normalize_f32(out[m].as_mut_ptr(), T as i32, mean, inv_std);
        }
    }
}

/// Scalar-only version of compute_log_mel_inplace (always uses microfft).
/// This is used for parity testing against SIMD implementation.
pub fn compute_log_mel_scalar<const M: usize, const T: usize>(
    wave_i16: &[i16],
    sr: usize,
    n_fft: usize,
    hop_length: usize,
    fmin_hz: f32,
    fmax_hz: f32,
    log_eps: f32,
    center: bool,
    out: &mut [[f32; T]; M],
) {
    assert!(n_fft == 1024, "only n_fft=1024 supported in MCU frontend");
    let n_fft_usize: usize = 1024;

    const MAX_MELS: usize = 64;
    debug_assert!(M <= MAX_MELS, "M={} exceeds MAX_MELS={}", M, MAX_MELS);

    let hann = &HANN_1024;
    let len = wave_i16.len();

    let pad = if center { n_fft_usize / 2 } else { 0 };
    let total = len + 2 * pad;
    let frames = if total >= n_fft_usize {
        1 + (total - n_fft_usize) / hop_length
    } else {
        1
    };
    assert!(frames == T, "frame count mismatch: {} != {}", frames, T);

    let mut bins_storage = [0usize; MAX_MELS + 2];
    let m_min = hz_to_mel_htk(fmin_hz);
    let m_max = hz_to_mel_htk(fmax_hz);
    for i in 0..(M + 2) {
        let t = i as f32 / ((M + 1) as f32);
        let mel = m_min + t * (m_max - m_min);
        let hz = mel_to_hz_htk(mel);
        let bin = floorf(((n_fft_usize + 1) as f32) * (hz / (sr as f32))) as isize;
        let clamped = if bin < 0 { 0 } else { bin };
        bins_storage[i] = clamped as usize;
    }

    let mut frame = [0f32; 1024];
    let mut power = [0f32; 513];

    for t in 0..T {
        let base: isize = (t * hop_length) as isize - (pad as isize);
        for n in 0..n_fft_usize {
            let idx = base + (n as isize);
            let s = reflect_get_i16(wave_i16, idx);
            let x = (s as f32) / 32768.0;
            frame[n] = x * hann[n];
        }

        // Always use microfft (scalar reference)
        // microfft returns 512 elements with DC/Nyquist packed:
        // spec[0].re = DC component (bin 0, purely real)
        // spec[0].im = Nyquist component (bin 512, purely real)
        // spec[1..512] = bins 1..511
        let spec = rfft_1024(&mut frame);

        // DC bin (k=0) - purely real
        power[0] = spec[0].re * spec[0].re;

        // Nyquist bin (k=512) - packed in spec[0].im, purely real
        power[512] = spec[0].im * spec[0].im;

        // Middle bins (k=1..512)
        for k in 1..512 {
            let re = spec[k].re;
            let im = spec[k].im;
            power[k] = re * re + im * im;
        }

        let last_valid = 512;

        let mut b0 = [0usize; MAX_MELS + 2];
        for i in 0..(M + 2) {
            let mut v = bins_storage[i];
            if v > last_valid {
                v = last_valid;
            }
            b0[i] = v;
        }
        for mm in 1..(M + 1) {
            if b0[mm - 1] == b0[mm] {
                b0[mm] = core::cmp::min(b0[mm] + 1, last_valid);
            }
            if b0[mm] == b0[mm + 1] {
                b0[mm + 1] = core::cmp::min(b0[mm + 1] + 1, last_valid);
            }
        }

        for m in 0..M {
            let f_m_minus = b0[m];
            let f_m = b0[m + 1];
            let f_m_plus = b0[m + 2];

            let mut acc = 0.0f32;
            if f_m_minus < f_m {
                let denom = (f_m - f_m_minus) as f32;
                for k in f_m_minus..f_m {
                    let w = (k - f_m_minus) as f32 / denom.max(1e-6);
                    acc += w * power[k];
                }
            }
            if f_m < f_m_plus {
                let denom = (f_m_plus - f_m) as f32;
                for k in f_m..f_m_plus {
                    let w = (f_m_plus - k) as f32 / denom.max(1e-6);
                    acc += w * power[k];
                }
            }
            out[m][t] = acc;
        }
    }

    let eps = log_eps.max(1e-12);
    let mut sum = 0.0f32;
    let mut sumsq = 0.0f32;
    for m in 0..M {
        for t in 0..T {
            let v = logf((out[m][t]).max(eps));
            out[m][t] = v;
            sum += v;
            sumsq += v * v;
        }
    }
    let count = (M * T) as f32;
    let mean = sum / count;
    let var = (sumsq / count) - mean * mean;
    let std = sqrtf(var.max(0.0)).max(1e-6);
    let inv_std = 1.0f32 / std;
    for m in 0..M {
        for t in 0..T {
            out[m][t] = (out[m][t] - mean) * inv_std;
        }
    }
}
