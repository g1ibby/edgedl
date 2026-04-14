#![allow(dead_code)]

extern crate edgedl_macros;

use edgedl::{
    error::Result as EngineResult,
    model::ValueMeta,
};

// Bind the real noise model from edgedl/tests
#[edgedl_macros::espdl_model(path = "../tests/noise_model.espdl")]
struct NoiseModelBind;

// Golden OFF/ON inputs (already engine-quantized INT8, NHWC order)
#[path = "../../tests/golden_inputs.rs"]
mod golden;

/// Compare scalar vs SIMD predictions for the full noise model using the
/// real .espdl and golden inputs. This is expected to fail today and serves
/// as an end-to-end parity regression for the real model data.
pub fn run_full_model_parity() {
    // Validate basic I/O
    assert!(!NoiseModelBind::SPEC.inputs.is_empty());
    assert!(!NoiseModelBind::SPEC.outputs.is_empty());

    let in_id = NoiseModelBind::SPEC.inputs[0];
    let in_meta: ValueMeta = NoiseModelBind::SPEC.values[in_id as usize];
    let need = in_meta.shape.elements();
    assert_eq!(golden::GOLDEN_OFF_INPUT_I8.len(), need);
    assert_eq!(golden::GOLDEN_ON_INPUT_I8.len(), need);

    let out0_id = NoiseModelBind::SPEC.outputs[0];
    let out0_meta: ValueMeta = NoiseModelBind::SPEC.values[out0_id as usize];
    assert_eq!(out0_meta.shape.n as usize, 1);
    let out_need = out0_meta.shape.elements();
    assert_eq!(out_need, 2, "expected 2-class logits");

    // Scalar runtime
    let arena_scalar: &mut [i8; NoiseModelBind::ARENA_SIZE] =
        hil_test::mk_static!([i8; NoiseModelBind::ARENA_SIZE], [0; NoiseModelBind::ARENA_SIZE]);
    let mut rt_scalar = NoiseModelBind::new(&mut arena_scalar[..]).expect("scalar runtime new");

    // SIMD runtime
    let arena_simd: &mut [i8; NoiseModelBind::ARENA_SIZE] =
        hil_test::mk_static!([i8; NoiseModelBind::ARENA_SIZE], [0; NoiseModelBind::ARENA_SIZE]);
    let mut rt_simd = NoiseModelBind::new(&mut arena_simd[..]).expect("simd runtime new");

    // OFF sample
    let mut probs_scalar_off: [f32; 2] = [0.0; 2];
    let mut probs_simd_off: [f32; 2] = [0.0; 2];

    rt_scalar
        .predict(&golden::GOLDEN_OFF_INPUT_I8, &mut probs_scalar_off)
        .expect("scalar predict off");

    #[cfg(target_arch = "xtensa")]
    {
        rt_simd
            .predict_simd(&golden::GOLDEN_OFF_INPUT_I8, &mut probs_simd_off)
            .expect("simd predict off");
    }
    #[cfg(not(target_arch = "xtensa"))]
    {
        // On non-Xtensa targets, SIMD entry is not generated; reuse scalar.
        rt_simd
            .predict(&golden::GOLDEN_OFF_INPUT_I8, &mut probs_simd_off)
            .expect("scalar-as-simd predict off");
    }

    // ON sample
    let mut probs_scalar_on: [f32; 2] = [0.0; 2];
    let mut probs_simd_on: [f32; 2] = [0.0; 2];

    rt_scalar
        .predict(&golden::GOLDEN_ON_INPUT_I8, &mut probs_scalar_on)
        .expect("scalar predict on");

    #[cfg(target_arch = "xtensa")]
    {
        rt_simd
            .predict_simd(&golden::GOLDEN_ON_INPUT_I8, &mut probs_simd_on)
            .expect("simd predict on");
    }
    #[cfg(not(target_arch = "xtensa"))]
    {
        rt_simd
            .predict(&golden::GOLDEN_ON_INPUT_I8, &mut probs_simd_on)
            .expect("scalar-as-simd predict on");
    }

    // Compare logits element-wise for both OFF and ON samples.
    let eps: f32 = 1e-3;
    for i in 0..2 {
        let d_off = (probs_scalar_off[i] - probs_simd_off[i]).abs();
        if d_off > eps {
            edgedl::ne_error!(
                "noise_model_full_parity: OFF mismatch at i={} scalar={} simd={} (assert will fail)",
                i,
                probs_scalar_off[i],
                probs_simd_off[i]
            );
            assert!(
                d_off <= eps,
                "OFF mismatch at {}: scalar={} simd={}",
                i,
                probs_scalar_off[i],
                probs_simd_off[i]
            );
        }

        let d_on = (probs_scalar_on[i] - probs_simd_on[i]).abs();
        if d_on > eps {
            edgedl::ne_error!(
                "noise_model_full_parity: ON mismatch at i={} scalar={} simd={} (assert will fail)",
                i,
                probs_scalar_on[i],
                probs_simd_on[i]
            );
            assert!(
                d_on <= eps,
                "ON mismatch at {}: scalar={} simd={}",
                i,
                probs_scalar_on[i],
                probs_simd_on[i]
            );
        }
    }
}

