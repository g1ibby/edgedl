use edgedl::model::Activation;

use super::util::{Case, Layout, WeightExps, run_parity};

pub fn run_group() {
    // Per-tensor exponents (no bias): Linear and ReLU
    run_parity(&Case {
        layout: Layout::OHWI,
        w_exps: WeightExps::PerTensor0,
        has_bias: false,
        activation: Activation::Linear,
        out_exp: 0,
    });
    run_parity(&Case {
        layout: Layout::OHWI,
        w_exps: WeightExps::PerTensor0,
        has_bias: false,
        activation: Activation::ReLU,
        out_exp: 0,
    });

    // Per-tensor with bias
    run_parity(&Case {
        layout: Layout::OHWI,
        w_exps: WeightExps::PerTensor0,
        has_bias: true,
        activation: Activation::Linear,
        out_exp: 0,
    });

    // Per-tensor with bias and negative mac_shift (out_exp > in+weight)
    run_parity(&Case {
        layout: Layout::OHWI,
        w_exps: WeightExps::PerTensor0,
        has_bias: true,
        activation: Activation::Linear,
        out_exp: 2,
    });

    // Per-channel exponents (no bias / with bias)
    run_parity(&Case {
        layout: Layout::OHWI,
        w_exps: WeightExps::PerChannel,
        has_bias: false,
        activation: Activation::Linear,
        out_exp: 0,
    });
    run_parity(&Case {
        layout: Layout::OHWI,
        w_exps: WeightExps::PerChannel,
        has_bias: true,
        activation: Activation::Linear,
        out_exp: 0,
    });
}
