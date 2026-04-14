//! ISA-specific SIMD kernels.
//!
//! Modules here are always present to keep the API surface stable across
//! targets. Architecture-specific assembly is gated inside the modules,
//! while callers perform high-level capability checks.

pub mod tie728;
