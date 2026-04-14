use edgedl::model::Activation;

use super::util::{Case, Layout, WeightExps, run_parity};

pub fn run_group() {
    // Blocked N16 HWC16 (K=1), per-tensor exponents, no bias
    run_parity(&Case {
        layout: Layout::BlockedN16HWC16,
        w_exps: WeightExps::PerTensor0,
        has_bias: false,
        activation: Activation::Linear,
        out_exp: 0,
    });
}
