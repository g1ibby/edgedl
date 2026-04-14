//! ESP32-S3 TIE728 SIMD kernels.

// Keep all Conv2D TIE728 assembly in one ordered block so helper macros are
// defined before the kernel files that invoke them. Relying on separate
// `global_asm!` items is not stable across build graphs.
#[cfg(all(feature = "simd-s3", target_arch = "xtensa"))]
core::arch::global_asm!(concat!(
    include_str!("tie728_shared_macros.S"),
    "\n",
    include_str!("tie728_k11_aligned.S"),
    "\n",
    include_str!("tie728_k11_unaligned.S"),
    "\n",
    include_str!("tie728_k33_aligned.S"),
    "\n",
    include_str!("tie728_k33_unaligned.S"),
    "\n",
    include_str!("tie728_hwcn_aligned.S"),
    "\n",
    include_str!("tie728_hwcn_unaligned.S"),
));

pub mod hwcn_aligned;
pub mod hwcn_unaligned;
pub mod k11_aligned;
pub mod k11_unaligned;
pub mod k33_aligned;
pub mod k33_unaligned;
