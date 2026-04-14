//! Noise classification example using edgedl inference engine.
//!
//! This example demonstrates end-to-end audio classification:
//! - Computes log-mel spectrograms from raw audio using SIMD
//! - Quantizes features to INT8
//! - Runs neural network inference on ESP32-S3

#![no_std]
#![no_main]

#[cfg(feature = "defmt")]
use defmt_rtt as _;
use edgedl::features::mel::{compute_log_mel_simd, quantize_by_engine_exp};
use esp_backtrace as _;
use esp_hal::{clock::CpuClock, main, time::Instant};
use static_cell::ConstStaticCell;

// Add ESP-IDF app descriptor for espflash compatibility
esp_bootloader_esp_idf::esp_app_desc!();

extern crate edgedl_macros;

// Bind the same model used by HIL tests
// Path is relative to this crate's Cargo.toml directory
#[edgedl_macros::espdl_model(path = "../../tests/noise_model.espdl")]
struct __ModelBind;

// Reuse the real audio samples and constants from edgedl/tests
// Path is relative to this file (src/main.rs)
#[path = "../../../tests/noise.rs"]
mod sample;

const IN_ELEMS: usize = sample::N_MELS * sample::N_FRAMES;
static MEL: ConstStaticCell<[[f32; sample::N_FRAMES]; sample::N_MELS]> =
    ConstStaticCell::new([[0.0; sample::N_FRAMES]; sample::N_MELS]);
static INPUT: ConstStaticCell<edgedl::Aligned16<[i8; IN_ELEMS]>> =
    ConstStaticCell::new(edgedl::Aligned16([0; IN_ELEMS]));
static ARENA: ConstStaticCell<__ModelBindArena> = ConstStaticCell::new(__ModelBind::new_arena());

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

#[main]
fn main() -> ! {
    let config = esp_hal::Config::default().with_cpu_clock(CpuClock::max());
    let peripherals = esp_hal::init(config);

    edgedl::mem::probe_stack();
    esp_println::println!("mem(boot): {}", edgedl::mem::report());
    esp_println::println!(
        "buffers: mel={}B input={}B arena={}B",
        core::mem::size_of::<[[f32; sample::N_FRAMES]; sample::N_MELS]>(),
        core::mem::size_of::<edgedl::Aligned16<[i8; IN_ELEMS]>>(),
        core::mem::size_of::<__ModelBindArena>(),
    );
    #[cfg(feature = "arena-report")]
    esp_println::println!("{}", __ModelBind::ARENA_REPORT);
    let psram = esp_hal::psram::Psram::new(peripherals.PSRAM, Default::default());
    let (_psram_start, psram_size) = psram.raw_parts();
    esp_println::println!("psram: {}B", psram_size);

    // Validate dimensions against model spec
    let in_id = __ModelBind::SPEC.inputs[0];
    let in_meta = __ModelBind::SPEC.values[in_id as usize];
    let out0_id = __ModelBind::SPEC.outputs[0];
    let out0_meta = __ModelBind::SPEC.values[out0_id as usize];

    assert_eq!(in_meta.shape.n as usize, 1);
    assert_eq!(in_meta.shape.c as usize, 1);
    assert_eq!(in_meta.shape.h as usize, sample::N_MELS);
    assert_eq!(in_meta.shape.w as usize, sample::N_FRAMES);
    assert_eq!(out0_meta.shape.elements(), 2);

    const M: usize = sample::N_MELS;
    const T: usize = sample::N_FRAMES;

    // Take ownership of static buffers once per boot
    let mel: &mut [[f32; T]; M] = MEL.take();
    let input = INPUT.take();
    let arena = ARENA.take();

    // Predict with static-backed buffers
    let mut probs: [f32; 2] = [0.0; 2];
    let (
        off_label,
        off_probs,
        off_us,
        off_mel_us,
        off_quant_us,
        on_label,
        on_probs,
        on_us,
        on_mel_us,
        on_quant_us,
    ) = {
        // OFF sample -> mel -> quant -> predict
        let t_mel_off = Instant::now();
        compute_log_mel_simd::<M, T>(
            &sample::PUMP_OFF_WAV,
            sample::SAMPLE_RATE,
            sample::N_FFT,
            sample::HOP_LENGTH,
            sample::FMIN_HZ,
            sample::FMAX_HZ,
            sample::LOG_EPS,
            sample::CENTER,
            mel,
        );
        let off_mel_us = t_mel_off.elapsed().as_micros() as u32;

        let t_quant_off = Instant::now();
        quantize_by_engine_exp::<M, T>(&*mel, in_meta.exp, &mut input.0);
        let off_quant_us = t_quant_off.elapsed().as_micros() as u32;

        let mut rt = __ModelBind::new(&mut arena.0[..]).expect("runtime new");
        let t_off = Instant::now();
        rt.predict_simd(&input.0, &mut probs).expect("predict off");
        let off_us = t_off.elapsed().as_micros() as u32;
        let off_probs = probs;
        let off = argmax(&off_probs);

        // ON sample -> mel -> quant -> predict (reuse buffers)
        let t_mel_on = Instant::now();
        compute_log_mel_simd::<M, T>(
            &sample::PUMP_ON_WAV,
            sample::SAMPLE_RATE,
            sample::N_FFT,
            sample::HOP_LENGTH,
            sample::FMIN_HZ,
            sample::FMAX_HZ,
            sample::LOG_EPS,
            sample::CENTER,
            mel,
        );
        let on_mel_us = t_mel_on.elapsed().as_micros() as u32;

        let t_quant_on = Instant::now();
        quantize_by_engine_exp::<M, T>(&*mel, in_meta.exp, &mut input.0);
        let on_quant_us = t_quant_on.elapsed().as_micros() as u32;

        let t_on = Instant::now();
        rt.predict_simd(&input.0, &mut probs).expect("predict on");
        let on_us = t_on.elapsed().as_micros() as u32;
        let on_probs = probs;
        let on = argmax(&on_probs);
        (
            off,
            off_probs,
            off_us,
            off_mel_us,
            off_quant_us,
            on,
            on_probs,
            on_us,
            on_mel_us,
            on_quant_us,
        )
    };

    let off_s = off_us as f32 / 1_000_000.0;
    let off_ms = off_us as f32 / 1_000.0;
    let off_mel_ms = off_mel_us as f32 / 1_000.0;
    let off_quant_ms = off_quant_us as f32 / 1_000.0;
    let on_s = on_us as f32 / 1_000_000.0;
    let on_ms = on_us as f32 / 1_000.0;
    let on_mel_ms = on_mel_us as f32 / 1_000.0;
    let on_quant_ms = on_quant_us as f32 / 1_000.0;

    esp_println::println!(
        "mel_off: exp_in={} exp_out={} label:{} probs=[{:.3},{:.3}] time={:.3}s ({:.1}ms, mel={:.1}ms, quant={:.1}ms)",
        in_meta.exp,
        out0_meta.exp,
        off_label,
        off_probs[0],
        off_probs[1],
        off_s,
        off_ms,
        off_mel_ms,
        off_quant_ms,
    );
    esp_println::println!(
        "mel_on: exp_in={} exp_out={} label:{} probs=[{:.3},{:.3}] time={:.3}s ({:.1}ms, mel={:.1}ms, quant={:.1}ms)",
        in_meta.exp,
        out0_meta.exp,
        on_label,
        on_probs[0],
        on_probs[1],
        on_s,
        on_ms,
        on_mel_ms,
        on_quant_ms,
    );

    esp_println::println!("mem(done): {}", edgedl::mem::report());

    assert_eq!(off_label, 0);
    assert_eq!(on_label, 1);

    loop {}
}
