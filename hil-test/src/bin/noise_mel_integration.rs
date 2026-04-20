//! HIL: compute log-mel from real audio, quantize, run model and check labels
//% CHIPS: esp32s3
//% FEATURES: defmt trace simd-s3

#![no_std]
#![no_main]

use edgedl::{
    Aligned16,
    features::mel::{compute_log_mel_simd, quantize_by_engine_exp},
};
use hil_test as _;
use static_cell::ConstStaticCell;

extern crate edgedl_macros;

// Bind model from the edgedl tests folder
#[edgedl_macros::espdl_model(path = "../tests/noise_model.espdl")]
struct __ModelBind;

// Real audio samples and constants
#[path = "../../../tests/noise.rs"]
mod sample;

// Use ConstStaticCell for arena to avoid StaticCell reinit issues with embedded-test
static ARENA: ConstStaticCell<__ModelBindArena> = ConstStaticCell::new(__ModelBind::new_arena());

// Frontend + model inference is heavier; allow more time.
#[embedded_test::tests(default_timeout = 60, executor = hil_test::Executor::new())]
mod tests {
    use super::*;

    struct Ctx;

    #[init]
    fn init() -> Ctx {
        hil_test::init_rtt();
        let _p = esp_hal::init(
            esp_hal::Config::default().with_cpu_clock(esp_hal::clock::CpuClock::max()),
        );
        Ctx
    }

    fn argmax(xs: &[f32]) -> usize {
        let mut best_i = 0usize;
        let mut best_v = f32::NEG_INFINITY;
        for (i, &v) in xs.iter().enumerate() {
            if v > best_v {
                best_v = v;
                best_i = i;
            }
        }
        best_i
    }

    #[test]
    fn real_audio_off_on_predictions(_ctx: Ctx) {
        // Validate dimensions against model spec
        assert!(!__ModelBind::SPEC.inputs.is_empty());
        let in_id = __ModelBind::SPEC.inputs[0];
        let in_meta = __ModelBind::SPEC.values[in_id as usize];
        assert_eq!(in_meta.shape.n as usize, 1);
        assert_eq!(in_meta.shape.c as usize, 1);
        assert_eq!(in_meta.shape.h as usize, sample::N_MELS);
        assert_eq!(in_meta.shape.w as usize, sample::N_FRAMES);

        let out0_id = __ModelBind::SPEC.outputs[0];
        let out0_meta = __ModelBind::SPEC.values[out0_id as usize];
        assert_eq!(out0_meta.shape.elements(), 2);

        const M: usize = sample::N_MELS;
        const T: usize = sample::N_FRAMES;
        const IN_ELEMS: usize = M * T;

        // OFF sample -> mel -> quant -> predict
        let mut mel_off: [[f32; T]; M] = [[0.0; T]; M];
        compute_log_mel_simd::<M, T>(
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
        let mut input_off: Aligned16<[i8; IN_ELEMS]> = Aligned16([0; IN_ELEMS]);
        quantize_by_engine_exp::<M, T>(&mel_off, in_meta.exp, &mut input_off.0);

        // ON sample -> mel -> quant -> predict
        let mut mel_on: [[f32; T]; M] = [[0.0; T]; M];
        compute_log_mel_simd::<M, T>(
            &sample::PUMP_ON_WAV,
            sample::SAMPLE_RATE,
            sample::N_FFT,
            sample::HOP_LENGTH,
            sample::FMIN_HZ,
            sample::FMAX_HZ,
            sample::LOG_EPS,
            sample::CENTER,
            &mut mel_on,
        );
        let mut input_on: Aligned16<[i8; IN_ELEMS]> = Aligned16([0; IN_ELEMS]);
        quantize_by_engine_exp::<M, T>(&mel_on, in_meta.exp, &mut input_on.0);

        // Predict probabilities with static arena to avoid stack blowup
        // Arena must be 16-byte aligned for TIE728 SIMD operations
        let mut probs: [f32; 2] = [0.0; 2];
        // Use try_take() - embedded-test framework may run initialization twice
        let arena = ARENA
            .try_take()
            .expect("ARENA already taken - reset device and run again");
        let mut rt = __ModelBind::new(&mut arena.0[..]).expect("runtime new");
        rt.predict_simd(&input_off.0, &mut probs)
            .expect("predict off");
        let off_label = argmax(&probs);
        rt.predict_simd(&input_on.0, &mut probs)
            .expect("predict on");
        let on_label = argmax(&probs);

        #[cfg(feature = "defmt")]
        defmt::info!(
            "mel_off_on: exp_in={}, exp_out={} labels off={} on={}",
            in_meta.exp,
            out0_meta.exp,
            off_label,
            on_label
        );

        assert_eq!(off_label, 0);
        assert_eq!(on_label, 1);
    }
}
