extern crate edgedl_macros;

#[edgedl_macros::espdl_model(path = "tests/noise_model.espdl")]
struct __NoiseBind;

fn parse_report_usize(report: &str, key: &str) -> usize {
    for line in report.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix(key) {
            return rest
                .trim()
                .parse::<usize>()
                .unwrap_or_else(|_| panic!("failed to parse {key} from '{line}'"));
        }
    }
    panic!("missing key '{key}' in report");
}

#[test]
fn arena_size_drops_with_liveness_reuse() {
    // Old full-residency layout for this model (pre-liveness planner) was ~138,896 bytes
    // (138,640 activations + 256 scratch).
    let activation_bytes = parse_report_usize(__NoiseBind::ARENA_REPORT, "size_bytes=");
    let scratch_bytes = __NoiseBind::ARENA_SIZE - activation_bytes;

    eprintln!("activation_bytes={activation_bytes} scratch_bytes={scratch_bytes}");

    assert_eq!(scratch_bytes, 256, "unexpected scratch for noise model");
    const {
        assert!(
            __NoiseBind::ARENA_SIZE < 138_896,
            "arena reuse did not reduce total arena size"
        );
        assert!(
            __NoiseBind::ARENA_SIZE <= 60_000,
            "unexpectedly large arena after liveness planning"
        );
    }
}
