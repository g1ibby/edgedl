//! Integration test: compute log-mel from real audio, quantize to NHWC INT8,
//! run the model end-to-end, and assert predicted OFF/ON labels.

extern crate edgedl_macros;

#[cfg(not(all(feature = "simd-s3", target_arch = "xtensa")))]
use edgedl::features::mel::compute_log_mel_scalar;
#[cfg(all(feature = "simd-s3", target_arch = "xtensa"))]
use edgedl::features::mel::compute_log_mel_simd;
use edgedl::features::mel::quantize_by_engine_exp;

// Bind the model from the embedded .espdl file
#[edgedl_macros::espdl_model(path = "tests/noise_model.espdl")]
struct __ModelBind;

// Bring in the real audio and constants generated for firmware
#[path = "noise.rs"]
mod sample;

use std::sync::Once;
static INIT: Once = Once::new();
fn init_logging() {
    INIT.call_once(|| {
        let _ = env_logger::builder()
            .is_test(true)
            .filter_module("edgedl", log::LevelFilter::Debug)
            .try_init();
    });
}

fn argmax_f32(xs: &[f32]) -> usize {
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

fn stats(name: &str, x: &[[f32; sample::N_FRAMES]; sample::N_MELS]) {
    let mut min_v = f32::INFINITY;
    let mut max_v = f32::NEG_INFINITY;
    let mut sum = 0.0f32;
    let mut sumsq = 0.0f32;
    let mut cnt = 0usize;
    for row in x.iter() {
        for &v in row.iter() {
            if v < min_v {
                min_v = v;
            }
            if v > max_v {
                max_v = v;
            }
            sum += v;
            sumsq += v * v;
            cnt += 1;
        }
    }
    let mean = sum / (cnt as f32);
    let var = (sumsq / (cnt as f32)) - mean * mean;
    let std = var.max(0.0).sqrt();
    eprintln!(
        "{} mel stats: min={:.3} max={:.3} mean={:.3} std={:.3}",
        name, min_v, max_v, mean, std
    );
}

fn stats_i8(name: &str, x: &[i8]) {
    let mut min_v = i8::MAX;
    let mut max_v = i8::MIN;
    for &v in x {
        if v < min_v {
            min_v = v;
        }
        if v > max_v {
            max_v = v;
        }
    }
    eprintln!("{} i8 stats: min={} max={}", name, min_v, max_v);
}

#[test]
fn real_audio_off_on_predictions() {
    init_logging();
    // Validate input layout and expected dimensions
    assert!(
        __ModelBind::SPEC.inputs.len() == 1,
        "expected single-input model"
    );
    let in_id = __ModelBind::SPEC.inputs[0];
    let in_meta = __ModelBind::SPEC.values[in_id as usize];
    assert!(in_meta.shape.n as usize == 1, "N != 1");
    assert!(in_meta.shape.c as usize == 1, "C != 1");
    assert!(in_meta.shape.h as usize == sample::N_MELS, "H != N_MELS");
    assert!(
        in_meta.shape.w as usize == sample::N_FRAMES,
        "W != N_FRAMES"
    );

    // OFF sample
    let mut mel_off: [[f32; sample::N_FRAMES]; sample::N_MELS] =
        [[0.0; sample::N_FRAMES]; sample::N_MELS];
    #[cfg(all(feature = "simd-s3", target_arch = "xtensa"))]
    compute_log_mel_simd::<{ sample::N_MELS }, { sample::N_FRAMES }>(
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
    #[cfg(not(all(feature = "simd-s3", target_arch = "xtensa")))]
    compute_log_mel_scalar::<{ sample::N_MELS }, { sample::N_FRAMES }>(
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
    stats("OFF", &mel_off);
    let mut input_off = vec![0i8; in_meta.shape.elements()];
    quantize_by_engine_exp::<{ sample::N_MELS }, { sample::N_FRAMES }>(
        &mel_off,
        in_meta.exp,
        &mut input_off,
    );
    stats_i8("OFF(exp)", &input_off);
    let out0_id = __ModelBind::SPEC.outputs[0];
    let out0_meta = __ModelBind::SPEC.values[out0_id as usize];
    let mut off_probs = vec![0.0f32; out0_meta.shape.elements()];
    // Allocate arena and runtime once; reuse across predictions
    let mut arena = vec![0i8; __ModelBind::ARENA_SIZE];
    let mut rt = __ModelBind::new(&mut arena[..]).expect("runtime new");
    rt.predict(&input_off, &mut off_probs).expect("predict off");
    let off_label = argmax_f32(&off_probs);
    eprintln!("OFF probs  (len={}): {:?}", off_probs.len(), off_probs);

    eprintln!(
        "Input meta: N={} H={} W={} C={} exp(e_in)={} | Output exp(e_out)={}",
        in_meta.shape.n,
        in_meta.shape.h,
        in_meta.shape.w,
        in_meta.shape.c,
        in_meta.exp,
        out0_meta.exp
    );

    // ON sample
    let mut mel_on: [[f32; sample::N_FRAMES]; sample::N_MELS] =
        [[0.0; sample::N_FRAMES]; sample::N_MELS];
    #[cfg(all(feature = "simd-s3", target_arch = "xtensa"))]
    compute_log_mel_simd::<{ sample::N_MELS }, { sample::N_FRAMES }>(
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
    #[cfg(not(all(feature = "simd-s3", target_arch = "xtensa")))]
    compute_log_mel_scalar::<{ sample::N_MELS }, { sample::N_FRAMES }>(
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
    stats("ON", &mel_on);
    let mut input_on = vec![0i8; in_meta.shape.elements()];
    quantize_by_engine_exp::<{ sample::N_MELS }, { sample::N_FRAMES }>(
        &mel_on,
        in_meta.exp,
        &mut input_on,
    );
    stats_i8("ON(exp)", &input_on);
    let mut on_probs = vec![0.0f32; out0_meta.shape.elements()];
    rt.predict(&input_on, &mut on_probs).expect("predict on");
    let on_label = argmax_f32(&on_probs);
    eprintln!("ON probs   (len={}): {:?}", on_probs.len(), on_probs);

    // Assertions last
    assert_eq!(out0_meta.shape.elements(), 2, "expected 2-class logits");
    assert_eq!(off_label, 0, "OFF sample should predict class 0");
    assert_eq!(on_label, 1, "ON sample should predict class 1");
}
