//! Portable logging facade for edgedl.
//!
//! Single feature `trace` controls inclusion of all instrumentation.
//! Backend selection is automatic:
//! - When built without `std` (embedded), uses `defmt`.
//! - When built with `std` (host), uses `log`.
//!
//! In non-`trace` builds, all macros are no-ops and helper functions are
//! compiled out via `#[cfg(feature = "trace")]`.

#[cfg(feature = "trace")]
pub mod inspect;

// ─────────────────────────────────────────────────────────────────────────────
// Timing: CPU cycle counter
// ─────────────────────────────────────────────────────────────────────────────

/// Read CPU cycle counter (CCOUNT register on Xtensa).
/// Returns raw cycle count for precise timing measurements.
#[cfg(all(feature = "trace", target_arch = "xtensa"))]
#[inline(always)]
pub fn read_ccount() -> u32 {
    let val: u32;
    unsafe { core::arch::asm!("rsr.ccount {}", out(reg) val) };
    val
}

/// Fallback for non-Xtensa (host tests): returns 0.
#[cfg(all(feature = "trace", not(target_arch = "xtensa")))]
#[inline(always)]
pub fn read_ccount() -> u32 {
    0
}

/// No-op when trace disabled.
#[cfg(not(feature = "trace"))]
#[inline(always)]
pub fn read_ccount() -> u32 {
    0
}

// Macros: ne_trace!, ne_debug!, ne_info!, ne_warn!, ne_error!

// defmt backend (no_std)
#[cfg(all(feature = "trace", not(feature = "std")))]
#[macro_export]
macro_rules! ne_trace {
    ($($t:tt)*) => { defmt::trace!($($t)*) };
}
#[cfg(all(feature = "trace", not(feature = "std")))]
#[macro_export]
macro_rules! ne_debug {
    ($($t:tt)*) => { defmt::debug!($($t)*) };
}
#[cfg(all(feature = "trace", not(feature = "std")))]
#[macro_export]
macro_rules! ne_info {
    ($($t:tt)*) => { defmt::info!($($t)*) };
}
#[cfg(all(feature = "trace", not(feature = "std")))]
#[macro_export]
macro_rules! ne_warn {
    ($($t:tt)*) => { defmt::warn!($($t)*) };
}
#[cfg(all(feature = "trace", not(feature = "std")))]
#[macro_export]
macro_rules! ne_error {
    ($($t:tt)*) => { defmt::error!($($t)*) };
}

// log backend (std)
#[cfg(all(feature = "trace", feature = "std"))]
#[macro_export]
macro_rules! ne_trace {
    ($($t:tt)*) => { ::log::trace!($($t)*) };
}
#[cfg(all(feature = "trace", feature = "std"))]
#[macro_export]
macro_rules! ne_debug {
    ($($t:tt)*) => { ::log::debug!($($t)*) };
}
#[cfg(all(feature = "trace", feature = "std"))]
#[macro_export]
macro_rules! ne_info {
    ($($t:tt)*) => { ::log::info!($($t)*) };
}
#[cfg(all(feature = "trace", feature = "std"))]
#[macro_export]
macro_rules! ne_warn {
    ($($t:tt)*) => { ::log::warn!($($t)*) };
}
#[cfg(all(feature = "trace", feature = "std"))]
#[macro_export]
macro_rules! ne_error {
    ($($t:tt)*) => { ::log::error!($($t)*) };
}

// No-op backend when `trace` is disabled
#[cfg(not(feature = "trace"))]
#[macro_export]
macro_rules! ne_trace {
    ($($t:tt)*) => {};
}
#[cfg(not(feature = "trace"))]
#[macro_export]
macro_rules! ne_debug {
    ($($t:tt)*) => {};
}
#[cfg(not(feature = "trace"))]
#[macro_export]
macro_rules! ne_info {
    ($($t:tt)*) => {};
}
#[cfg(not(feature = "trace"))]
#[macro_export]
macro_rules! ne_warn {
    ($($t:tt)*) => {};
}
#[cfg(not(feature = "trace"))]
#[macro_export]
macro_rules! ne_error {
    ($($t:tt)*) => {};
}
