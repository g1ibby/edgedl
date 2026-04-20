//! HIL: Conv2D 3x3 (33cn) SIMD vs scalar parity on ESP32-S3
//% CHIPS: esp32s3
//% FEATURES: defmt trace

#![no_std]
#![no_main]

use hil_test as _;

// Keep modules outside of src/bin so xtask can scan src/bin as files only
#[path = "../conv33_parity/dilated.rs"]
mod dilated;
#[path = "../conv33_parity/ic_tail.rs"]
mod ic_tail;
#[path = "../conv33_parity/noise_like.rs"]
mod noise_like;
#[path = "../conv33_parity/oc_tail.rs"]
mod oc_tail;
#[path = "../conv33_parity/ohwi.rs"]
mod ohwi;
#[path = "../conv33_parity/padding.rs"]
mod padding;
#[path = "../conv33_parity/strided.rs"]
mod strided;
#[path = "../conv33_parity/util.rs"]
mod util;

#[embedded_test::tests(default_timeout = 60, executor = hil_test::Executor::new())]
mod tests {
    use super::*;

    pub struct Ctx;

    #[init]
    fn init() -> Ctx {
        hil_test::init_rtt();
        let _p = esp_hal::init(
            esp_hal::Config::default().with_cpu_clock(esp_hal::clock::CpuClock::max()),
        );
        Ctx
    }

    #[test]
    fn conv33_ohwi_parity(_ctx: Ctx) {
        ohwi::run_group();
    }

    #[test]
    fn conv33_strided_parity(_ctx: Ctx) {
        strided::run_group();
    }

    #[test]
    fn conv33_dilated_parity(_ctx: Ctx) {
        dilated::run_group();
    }

    #[test]
    fn conv33_ic_tail_parity(_ctx: Ctx) {
        ic_tail::run_group();
    }

    #[test]
    fn conv33_oc_tail_parity(_ctx: Ctx) {
        oc_tail::run_group();
    }

    #[test]
    fn conv33_padding_parity(_ctx: Ctx) {
        padding::test_padding_3333_linear_pt();
        padding::test_padding_3333_relu_bias_pt();
        padding::test_padding_3333_relu_bias_pc();
        padding::test_padding_blocked_relu_bias_pt();
        padding::test_padding_stride2();
        padding::test_padding_dilation2();
    }

    #[test]
    fn conv33_noise_like_parity(_ctx: Ctx) {
        noise_like::run_group();
    }
}
