//! Lightweight tensor inspection helpers used by kernels/engine instrumentation.
//! Compiled only when the `trace` feature is enabled.

use crate::model::{Shape4dNHWC, ValueId, ValueMeta};

#[inline]
fn stats_i8(xs: &[i8]) -> (i8, i8, f32, usize, usize) {
    let mut min_v: i8 = i8::MAX;
    let mut max_v: i8 = i8::MIN;
    let mut sum: i64 = 0;
    let mut sat_lo = 0usize;
    let mut sat_hi = 0usize;
    for &v in xs {
        if v < min_v {
            min_v = v;
        }
        if v > max_v {
            max_v = v;
        }
        sum += v as i64;
        if v == -128 {
            sat_lo += 1;
        }
        if v == 127 {
            sat_hi += 1;
        }
    }
    let elems = xs.len() as f32;
    let mean = if elems > 0.0 {
        (sum as f32) / elems
    } else {
        0.0
    };
    (min_v, max_v, mean, sat_lo, sat_hi)
}

/// Log a one-line summary for an INT8 value slice with its metadata.
#[allow(unused_variables)]
pub fn log_value_i8(label: &str, id: ValueId, meta: ValueMeta, data: &[i8]) {
    let Shape4dNHWC { n, h, w, c } = meta.shape;
    let (min_v, max_v, mean, sat_lo, sat_hi) = stats_i8(data);
    // Use crate-level facade macros; expand to defmt or log depending on target.
    crate::ne_info!(
        "{} id={} shape N{}x{}x{}x{} e={} stats: min={} max={} mean={} sat_lo={} sat_hi={}",
        label,
        id,
        n,
        h,
        w,
        c,
        meta.exp,
        min_v,
        max_v,
        mean,
        sat_lo,
        sat_hi
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Conv2D performance statistics
// ─────────────────────────────────────────────────────────────────────────────

/// Per-layer Conv2D execution statistics for profiling.
#[derive(Debug, Clone, Copy)]
pub struct Conv2dStats {
    /// Kernel name (e.g., "k33_aligned", "hwcn_unaligned")
    pub kernel_name: &'static str,
    /// Output value ID
    pub output_id: ValueId,
    /// CPU cycles taken
    pub cycles: u32,
    /// Number of output pixels computed via SIMD
    pub simd_pixels: u32,
    /// Number of output pixels computed via scalar (padding fallback)
    pub padding_pixels: u32,
    /// True if fell back to full scalar due to sbits out of range
    pub silent_scalar: bool,
}

impl Conv2dStats {
    /// Create new stats with zeroed counters.
    #[inline]
    pub fn new(kernel_name: &'static str, output_id: ValueId) -> Self {
        Self {
            kernel_name,
            output_id,
            cycles: 0,
            simd_pixels: 0,
            padding_pixels: 0,
            silent_scalar: false,
        }
    }

    /// Log the statistics summary.
    pub fn log(&self) {
        let total = self.simd_pixels + self.padding_pixels;
        let simd_pct = if total > 0 {
            (self.simd_pixels as u32 * 100) / total
        } else {
            0
        };

        if self.silent_scalar {
            crate::ne_warn!(
                "{} id={}: {}cy [SILENT_SCALAR]",
                self.kernel_name,
                self.output_id,
                self.cycles
            );
        } else {
            crate::ne_info!(
                "{} id={}: {}cy, simd={}px ({}%), pad={}px",
                self.kernel_name,
                self.output_id,
                self.cycles,
                self.simd_pixels,
                simd_pct,
                self.padding_pixels
            );
        }
    }
}
