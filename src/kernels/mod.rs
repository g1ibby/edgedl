//! Operator kernels: scalar (portable) implementations.
//!
//! M1 provides scalar skeletons that validate shapes and document the
//! contract. Implementations will fill these in incrementally.

pub mod conv2d;
pub mod linear;
pub mod pad;
pub mod reduce_mean;
pub mod relu;

// SIMD kernels and router entry points (feature-gated in later milestones).
// For M3 scaffolding, we always compile the module; router returns Fallback
// until ISA wrappers are implemented. The module stays no_std-friendly.
pub mod simd;
