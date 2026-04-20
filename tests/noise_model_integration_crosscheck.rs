//! Host diff for the `noise_model_integration` HIL test.
//!
//! Invoked by `cargo xtask run full-tests esp32s3 --test noise_model_integration`
//! with `EDGEDL_DUMP_DIR` pointing at a temp dir populated by the probe driver.
//! For each `simd_off` / `simd_on` run the device captured, we re-run scalar
//! inference on the host against the same golden INT8 inputs and compare each
//! node's scalar output to the device's SIMD dump byte-by-byte.
//!
//! The `#[ignore]` attribute keeps `cargo test` green when there's no dump dir;
//! the `crosscheck` feature gate keeps the file off the default build where
//! `edgedl::engine::NodeHook` isn't emitted.

#![cfg(feature = "crosscheck")]

extern crate edgedl_macros;

#[edgedl_macros::espdl_model(path = "tests/noise_model.espdl")]
struct __ModelBind;

#[path = "golden_inputs.rs"]
mod golden;

use std::fs;
use std::path::{Path, PathBuf};

const TOLERANCE: i32 = 1;

#[test]
#[ignore = "requires EDGEDL_DUMP_DIR from `cargo xtask run full-tests`"]
fn host_scalar_vs_device_simd() {
    let dump_dir = PathBuf::from(
        std::env::var("EDGEDL_DUMP_DIR")
            .expect("EDGEDL_DUMP_DIR must be set; run via `cargo xtask run full-tests`"),
    );
    assert!(
        dump_dir.is_dir(),
        "dump dir {} is not a directory",
        dump_dir.display()
    );

    let mut arena = vec![0i8; __ModelBind::ARENA_SIZE];
    let mut rt = __ModelBind::new(&mut arena).expect("runtime new");
    let mut probs = [0.0f32; 2];

    let mut total = Totals::default();
    for (tag, input) in [
        ("simd_off", &golden::GOLDEN_OFF_INPUT_I8[..]),
        ("simd_on", &golden::GOLDEN_ON_INPUT_I8[..]),
    ] {
        eprintln!("\n== run '{tag}' ==");
        let mut hook = DiffHook {
            run_tag: tag,
            dump_dir: &dump_dir,
            totals: Totals::default(),
        };
        rt.predict_scalar_hooked(input, &mut probs, &mut hook)
            .unwrap_or_else(|e| panic!("scalar inference failed for {tag}: {e:?}"));
        total.passed += hook.totals.passed;
        total.failed += hook.totals.failed;
        total.missing += hook.totals.missing;
    }

    eprintln!(
        "\nnoise_model_integration_crosscheck: {} pass, {} fail, {} missing (tolerance={})",
        total.passed, total.failed, total.missing, TOLERANCE
    );

    assert_eq!(total.failed, 0, "{} nodes exceeded tolerance", total.failed);
    assert_eq!(
        total.missing, 0,
        "{} nodes missing device dumps",
        total.missing
    );
}

#[derive(Default)]
struct Totals {
    passed: usize,
    failed: usize,
    missing: usize,
}

struct DiffHook<'a> {
    run_tag: &'static str,
    dump_dir: &'a Path,
    totals: Totals,
}

impl<'a> edgedl::engine::NodeHook for DiffHook<'a> {
    fn after_node(
        &mut self,
        idx: usize,
        node: &edgedl::model::NodeSpec,
        _meta: &edgedl::model::ValueMeta,
        scalar_bytes: &[i8],
    ) {
        let name = format!(
            "{}_n{}_{}_v{}",
            self.run_tag,
            idx,
            node.op.tag(),
            node.op.output()
        );
        let path = self.dump_dir.join(format!("{name}.bin"));
        let device = match fs::read(&path) {
            Ok(v) => v,
            Err(_) => {
                eprintln!(
                    "MISS {:<40} elems={} (no device dump found)",
                    name,
                    scalar_bytes.len()
                );
                self.totals.missing += 1;
                return;
            }
        };
        if device.len() != scalar_bytes.len() {
            eprintln!(
                "FAIL {:<40} elems={} device_len={} (size mismatch)",
                name,
                scalar_bytes.len(),
                device.len()
            );
            self.totals.failed += 1;
            return;
        }
        let mut max_abs = 0i32;
        let mut offenders = 0usize;
        for (s, d) in scalar_bytes.iter().zip(device.iter()) {
            let diff = (*s as i32) - (*d as i8 as i32);
            let abs = diff.abs();
            if abs > max_abs {
                max_abs = abs;
            }
            if abs > TOLERANCE {
                offenders += 1;
            }
        }
        if offenders == 0 {
            eprintln!(
                "PASS {:<40} elems={:<6} max|Δ|={}",
                name,
                scalar_bytes.len(),
                max_abs
            );
            self.totals.passed += 1;
        } else {
            eprintln!(
                "FAIL {:<40} elems={:<6} max|Δ|={} offenders={}",
                name,
                scalar_bytes.len(),
                max_abs,
                offenders
            );
            self.totals.failed += 1;
        }
    }
}
