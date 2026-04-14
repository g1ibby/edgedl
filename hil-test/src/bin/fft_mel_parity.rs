//! HIL: Parity tests for SIMD FFT and Mel spectrogram computation
//!
//! These tests validate that the TIE728 SIMD FFT produces results identical
//! to the reference microfft implementation.
//% CHIPS: esp32s3
//% FEATURES: defmt trace simd-s3

#![no_std]
#![no_main]

use edgedl::features::{
    fft::rfft_1024_simd,
    mel::{HANN_1024, compute_log_mel_scalar, compute_log_mel_simd},
};
use hil_test as _;

// Real audio samples and constants
#[path = "../../../tests/noise.rs"]
mod sample;

#[embedded_test::tests(default_timeout = 60, executor = hil_test::Executor::new())]
mod tests {
    use super::*;

    struct Ctx;

    #[init]
    fn init() -> Ctx {
        let _p = esp_hal::init(
            esp_hal::Config::default().with_cpu_clock(esp_hal::clock::CpuClock::max()),
        );
        Ctx
    }

    /// Test FFT parity: SIMD FFT vs microfft on a single frame from real audio
    #[test]
    fn fft_simd_vs_microfft_parity(_ctx: Ctx) {
        // Extract first frame from real audio (windowed)
        // Extract first frame from real audio (windowed)
        #[repr(C, align(16))]
        struct AlignedFrame([f32; 1024]);
        let mut frame_simd = AlignedFrame([0f32; 1024]);
        let mut frame_ref = [0f32; 1024];

        for n in 0..1024 {
            let s = sample::PUMP_OFF_WAV[n] as f32 / 32768.0;
            frame_simd.0[n] = s * HANN_1024[n];
            frame_ref[n] = s * HANN_1024[n];
        }

        // Run SIMD FFT - returns 513 unpacked bins
        let spec_simd = rfft_1024_simd(&mut frame_simd.0);

        // Run microfft reference - returns 512 elements with DC/Nyquist packed:
        // spec_ref[0].re = DC (bin 0), spec_ref[0].im = Nyquist (bin 512)
        // spec_ref[1..512] = bins 1..511
        let spec_ref = microfft::real::rfft_1024(&mut frame_ref);

        // Track max difference for reporting
        let mut max_diff_re = 0.0f32;
        let mut max_diff_im = 0.0f32;
        let mut max_diff_bin = 0usize;
        let mut fail_count = 0u32;

        const TOLERANCE: f32 = 1e-3; // Allow small numerical differences

        // Helper to check and track a single bin comparison
        let mut check_bin = |k: usize, simd_re: f32, simd_im: f32, ref_re: f32, ref_im: f32| {
            let diff_re = (simd_re - ref_re).abs();
            let diff_im = (simd_im - ref_im).abs();

            if diff_re > max_diff_re {
                max_diff_re = diff_re;
                max_diff_bin = k;
            }
            if diff_im > max_diff_im {
                max_diff_im = diff_im;
            }

            if diff_re > TOLERANCE || diff_im > TOLERANCE {
                fail_count += 1;
                if fail_count <= 5 {
                    #[cfg(feature = "defmt")]
                    defmt::warn!(
                        "FFT bin {} mismatch: simd=({}, {}) ref=({}, {}) diff=({}, {})",
                        k,
                        simd_re,
                        simd_im,
                        ref_re,
                        ref_im,
                        diff_re,
                        diff_im
                    );
                }
            }
        };

        // DC bin (k=0): microfft packs DC real in spec_ref[0].re, imaginary should be 0
        check_bin(0, spec_simd[0].re, spec_simd[0].im, spec_ref[0].re, 0.0);

        // Middle bins (k=1..511): direct comparison
        for k in 1..512 {
            check_bin(
                k,
                spec_simd[k].re,
                spec_simd[k].im,
                spec_ref[k].re,
                spec_ref[k].im,
            );
        }

        // Nyquist bin (k=512): microfft packs Nyquist real in spec_ref[0].im, imaginary should be 0
        check_bin(
            512,
            spec_simd[512].re,
            spec_simd[512].im,
            spec_ref[0].im,
            0.0,
        );

        #[cfg(feature = "defmt")]
        defmt::info!(
            "FFT parity: max_diff_re={} max_diff_im={} at bin {} fails={}",
            max_diff_re,
            max_diff_im,
            max_diff_bin,
            fail_count
        );

        assert_eq!(
            fail_count, 0,
            "FFT parity failed: {} bins exceed tolerance {}",
            fail_count, TOLERANCE
        );
    }

    /// Test Mel spectrogram parity: SIMD vs scalar implementation
    #[test]
    fn mel_simd_vs_scalar_parity(_ctx: Ctx) {
        const M: usize = sample::N_MELS; // 64
        const T: usize = sample::N_FRAMES; // 101

        // Compute with SIMD FFT (current implementation)
        let mut mel_simd: [[f32; T]; M] = [[0.0; T]; M];
        compute_log_mel_simd::<M, T>(
            &sample::PUMP_OFF_WAV,
            sample::SAMPLE_RATE,
            sample::N_FFT,
            sample::HOP_LENGTH,
            sample::FMIN_HZ,
            sample::FMAX_HZ,
            sample::LOG_EPS,
            sample::CENTER,
            &mut mel_simd,
        );

        // Compute with scalar FFT (reference - always uses microfft)
        let mut mel_ref: [[f32; T]; M] = [[0.0; T]; M];
        compute_log_mel_scalar::<M, T>(
            &sample::PUMP_OFF_WAV,
            sample::SAMPLE_RATE,
            sample::N_FFT,
            sample::HOP_LENGTH,
            sample::FMIN_HZ,
            sample::FMAX_HZ,
            sample::LOG_EPS,
            sample::CENTER,
            &mut mel_ref,
        );

        // Track max difference for reporting
        let mut max_diff = 0.0f32;
        let mut max_diff_m = 0usize;
        let mut max_diff_t = 0usize;
        let mut fail_count = 0u32;

        const TOLERANCE: f32 = 2e-2; // Allow small numerical differences (relaxed for SIMD vs microfft)

        // Compare all M×T values
        for m in 0..M {
            for t in 0..T {
                let diff = (mel_simd[m][t] - mel_ref[m][t]).abs();

                if diff > max_diff {
                    max_diff = diff;
                    max_diff_m = m;
                    max_diff_t = t;
                }

                if diff > TOLERANCE {
                    fail_count += 1;
                    if fail_count <= 5 {
                        // Print first few failures
                        #[cfg(feature = "defmt")]
                        defmt::warn!(
                            "Mel[{}][{}] mismatch: simd={} ref={} diff={}",
                            m,
                            t,
                            mel_simd[m][t],
                            mel_ref[m][t],
                            diff
                        );
                    }
                }
            }
        }

        #[cfg(feature = "defmt")]
        defmt::info!(
            "Mel parity: max_diff={} at [{},{}] fails={}/{}",
            max_diff,
            max_diff_m,
            max_diff_t,
            fail_count,
            M * T
        );

        assert_eq!(
            fail_count, 0,
            "Mel parity failed: {} values exceed tolerance {}",
            fail_count, TOLERANCE
        );
    }

    /// Test that scalar mel produces correct classification (sanity check)
    #[test]
    fn mel_scalar_classification_sanity(_ctx: Ctx) {
        const M: usize = sample::N_MELS;
        const T: usize = sample::N_FRAMES;

        // Compute OFF sample with scalar
        let mut mel_off: [[f32; T]; M] = [[0.0; T]; M];
        compute_log_mel_scalar::<M, T>(
            &sample::PUMP_OFF_WAV,
            sample::SAMPLE_RATE,
            sample::N_FFT,
            sample::HOP_LENGTH,
            sample::FMIN_HZ,
            sample::FMAX_HZ,
            sample::LOG_EPS,
            sample::CENTER,
            &mut mel_off,
        );

        // Check that mel values are in reasonable range (normalized)
        let mut min_val = f32::INFINITY;
        let mut max_val = f32::NEG_INFINITY;
        for m in 0..M {
            for t in 0..T {
                if mel_off[m][t] < min_val {
                    min_val = mel_off[m][t];
                }
                if mel_off[m][t] > max_val {
                    max_val = mel_off[m][t];
                }
            }
        }

        #[cfg(feature = "defmt")]
        defmt::info!("Mel scalar range: min={} max={}", min_val, max_val);

        // After normalization, values should roughly be in [-3, 3] range
        assert!(
            min_val > -10.0 && max_val < 10.0,
            "Mel values out of expected range: [{}, {}]",
            min_val,
            max_val
        );
    }
}
