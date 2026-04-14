//! SIMD Conv2D entry points and routing.
//!
//! - Router selects between 1x1 (11cn), 3x3 (33cn), and general (hwcn) families, and between
//!   aligned and unaligned variants.

mod common;
mod isa;
mod quant;
mod router;

use router::{AlignClass, Family};

use crate::{
    arena::{Arena, PlannedArena},
    model::{Conv2dSpec, ModelSpec},
    rounding::RoundingMode,
};

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ExecOutcome {
    /// SIMD path executed (future phases).
    Executed,
    /// Router declined; the caller should use the scalar kernel.
    Fallback,
}

/// Attempt to execute Conv2D via SIMD. Returns `Fallback` when unsupported.
#[inline]
pub fn try_conv2d(
    plan: &PlannedArena,
    arena: &mut Arena<'_>,
    spec: &ModelSpec,
    node: &Conv2dSpec,
    round: RoundingMode,
) -> ExecOutcome {
    // Consult the router and, when eligible and supported, invoke the
    // architecture-specific path. Otherwise, fall back to scalar.
    let decision = router::decide(spec, node, plan, arena);
    #[cfg(feature = "trace")]
    crate::ne_info!("SIMD Conv2D decision: {:?}", decision);

    // Aligned 1x1 (11cn) fast path. The ISA module decides at build time
    // whether an optimized path is available; otherwise `run` returns Fallback.
    if decision.family == Family::K11 {
        if decision.align == AlignClass::Aligned16 {
            return isa::tie728::k11_aligned::run(plan, arena, spec, node, round);
        } else {
            return isa::tie728::k11_unaligned::run(plan, arena, spec, node, round);
        }
    }

    // Aligned 3x3 (33cn) fast path.
    if decision.family == Family::K33 {
        if decision.align == AlignClass::Aligned16 {
            return isa::tie728::k33_aligned::run(plan, arena, spec, node, round);
        } else {
            return isa::tie728::k33_unaligned::run(plan, arena, spec, node, round);
        }
    }

    // Generic HWCN fast path.
    if decision.family == Family::Khw {
        if decision.align == AlignClass::Aligned16 {
            return isa::tie728::hwcn_aligned::run(plan, arena, spec, node, round);
        } else {
            return isa::tie728::hwcn_unaligned::run(plan, arena, spec, node, round);
        }
    }

    // Unsupported or not eligible: ask caller to use scalar.
    ExecOutcome::Fallback
}
