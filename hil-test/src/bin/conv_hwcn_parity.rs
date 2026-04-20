//! HIL: Conv2D HWCN 5x5 SIMD vs scalar parity on ESP32-S3
//% CHIPS: esp32s3
//% FEATURES: defmt trace

#![no_std]
#![no_main]

use hil_test as _;

// Keep modules outside of src/bin so xtask can scan src/bin as files only
#[path = "../conv_hwcn_parity/aligned.rs"]
mod aligned;
#[path = "../conv_hwcn_parity/unaligned.rs"]
mod unaligned;
#[path = "../conv_hwcn_parity/util.rs"]
mod util;

#[path = "../conv_hwcn_parity/noise_like.rs"]
mod noise_like;
#[path = "../conv_hwcn_parity/padding.rs"]
mod padding;

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
    fn conv_hwcn_aligned_parity(_ctx: Ctx) {
        aligned::run_group();
    }

    #[test]
    fn conv_hwcn_unaligned_parity(_ctx: Ctx) {
        unaligned::run_group();
    }

    #[test]
    fn conv_hwcn_padding_parity(_ctx: Ctx) {
        padding::run_group();
    }

    #[test]
    fn conv_hwcn_noise_like_parity(_ctx: Ctx) {
        noise_like::run_group();
    }
}
