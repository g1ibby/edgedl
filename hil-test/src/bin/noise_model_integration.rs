//! HIL: run `noise_model.espdl` end-to-end using golden INT8 inputs
//% CHIPS: esp32s3
//% FEATURES: defmt trace simd-s3

#![no_std]
#![no_main]

use edgedl::Aligned16;
use hil_test as _;

extern crate edgedl_macros;

// Bind model from the edgedl tests folder
#[edgedl_macros::espdl_model(path = "../tests/noise_model.espdl")]
struct __ModelBind;

// Golden OFF/ON inputs (already engine-quantized INT8, NHWC order)
#[path = "../../../tests/golden_inputs.rs"]
mod golden;

// Scalar kernels on MCU are slow; allow more time for two inferences.
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
    fn noise_model_off_on_from_golden_inputs(_ctx: Ctx) {
        // Validate shapes
        assert!(!__ModelBind::SPEC.inputs.is_empty());
        assert!(!__ModelBind::SPEC.outputs.is_empty());

        let in_id = __ModelBind::SPEC.inputs[0];
        let in_meta = __ModelBind::SPEC.values[in_id as usize];
        let need = in_meta.shape.elements();
        assert_eq!(golden::GOLDEN_OFF_INPUT_I8.len(), need);
        assert_eq!(golden::GOLDEN_ON_INPUT_I8.len(), need);

        let out0_id = __ModelBind::SPEC.outputs[0];
        let out0_meta = __ModelBind::SPEC.values[out0_id as usize];
        assert_eq!(out0_meta.shape.n as usize, 1);
        assert_eq!(out0_meta.shape.elements(), 2);

        let mut probs: [f32; 2] = [0.0; 2];
        // Use a static arena buffer to avoid large stack allocations on MCU
        // Arena must be 16-byte aligned for TIE728 SIMD operations
        let arena: &mut Aligned16<[i8; __ModelBind::ARENA_SIZE]> = hil_test::mk_static!(
            Aligned16<[i8; __ModelBind::ARENA_SIZE]>,
            Aligned16([0; __ModelBind::ARENA_SIZE])
        );
        let mut rt = __ModelBind::new(&mut arena.0[..]).expect("runtime new");

        // Copy golden inputs into 16-byte aligned buffers (required by TIE728 SIMD)
        let input_off: &mut Aligned16<[i8; 6464]> =
            hil_test::mk_static!(Aligned16<[i8; 6464]>, Aligned16([0; 6464]));
        input_off.0.copy_from_slice(&golden::GOLDEN_OFF_INPUT_I8);

        let input_on: &mut Aligned16<[i8; 6464]> =
            hil_test::mk_static!(Aligned16<[i8; 6464]>, Aligned16([0; 6464]));
        input_on.0.copy_from_slice(&golden::GOLDEN_ON_INPUT_I8);

        // OFF -> expect class 0
        rt.predict_simd(&input_off.0, &mut probs)
            .expect("predict off");
        let off_label = argmax(&probs);

        // ON -> expect class 1
        rt.predict_simd(&input_on.0, &mut probs)
            .expect("predict on");
        let on_label = argmax(&probs);

        #[cfg(feature = "defmt")]
        defmt::info!("noise_model_off_on: off={} on={}", off_label, on_label);

        assert_eq!(off_label, 0);
        assert_eq!(on_label, 1);
    }
}
