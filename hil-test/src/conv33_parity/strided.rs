use edgedl::model::Activation;

use super::util::{Case, Layout, WeightExps, run_parity};

pub fn run_group() {
    // Stride 2x2
    run_parity(&Case {
        layout: Layout::OHWI,
        w_exps: WeightExps::PerTensor0,
        has_bias: false,
        activation: Activation::Linear,
        out_exp: 0,
        stride: 2,
        dilation: 1,
    });

    // Stride 2x2 with bias
    run_parity(&Case {
        layout: Layout::OHWI,
        w_exps: WeightExps::PerTensor0,
        has_bias: true,
        activation: Activation::Linear,
        out_exp: 0,
        stride: 2,
        dilation: 1,
    });
}
