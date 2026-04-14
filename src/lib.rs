#![cfg_attr(not(feature = "std"), no_std)]
// Enable experimental inline/global asm on Xtensa when building SIMD kernels or trace
// (read_ccount).
#![cfg_attr(
    all(
        any(feature = "simd-s3", feature = "trace", feature = "stack-probe"),
        target_arch = "xtensa"
    ),
    feature(asm_experimental_arch)
)]
#![allow(clippy::all)]

// Runtime is no_std by default. Keep std-only code behind features.

// no alloc usage in runtime

// Rounding helpers and requantization utilities (no_std)
pub mod rounding;

// Temporarily keep espdl reader unexposed in runtime to preserve no_std. When
// needed for tooling, enable via the `std` feature and gate code accordingly.
// #[cfg(feature = "std")]
// pub mod espdl;

// FlatBuffers schema/bindings move to the macros crate. Do not re-export here.
pub mod arena;
pub mod engine;
pub mod error;
pub mod features;
pub mod kernels;
pub mod mem;
pub mod model;
pub mod post;
pub mod trace;

/// Wrapper to ensure 16-byte alignment for SIMD operations.
/// esp-dl TIE728 kernels require all tensor base pointers to be 16-byte aligned.
#[repr(C, align(16))]
pub struct Aligned16<T>(pub T);
