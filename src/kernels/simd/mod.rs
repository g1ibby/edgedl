//! SIMD kernels scaffolding.
//!
//! M3 introduces a SIMD execution path for Conv2D on ESP32-S3. This module
//! provides a router and typed entry points. During initial scaffolding the
//! router always requests a scalar fallback.

pub mod conv2d;
