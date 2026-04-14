use edgedl::model::Activation;

use super::util::{Case, Layout, WeightExps, run_parity};

pub fn run_group() {
    // Large IC/OC (32x32) - requires updating util.rs or handling dynamic shapes?
    // util.rs has fixed consts. We can add a `large` flag to Case and handle it in util.rs
    // or just use the existing shapes if they are enough?
    // Existing IC=16, OC=16.
    // To test tile loop, we need OC > 16.

    // Let's stick to the Case struct and maybe add a `large_channels` option.
    // But `util.rs` uses const generics/statics for shapes.
    // We might need a separate `util_large.rs` or make `util.rs` more flexible.
    // For now, let's skip "large" if it requires significant refactoring of the test harness,
    // or we can try to reuse the existing harness but with a different "large" util if we want.

    // Actually, `util.rs` defines `IC=16`, `OC=16`.
    // If we want to test OC=32, we need different statics.
    // Let's create `util_large.rs` for this.
}
