use edgedl::model::Activation;

use super::util::{Case, Layout, WeightExps, run_parity};

pub fn run_group() {
    // Dilation 2x2
    // Input needs to be large enough:
    // KH_eff = (KH-1)*dil + 1 = (2)*2 + 1 = 5
    // H=5 is just enough for OH=1 if padding=0.
    // Our util.rs has H=5, W=5.
    // OH = (H - KH_eff)/stride + 1 = (5-5)/1 + 1 = 1.
    run_parity(&Case {
        layout: Layout::OHWI,
        w_exps: WeightExps::PerTensor0,
        has_bias: false,
        activation: Activation::Linear,
        out_exp: 0,
        stride: 1,
        dilation: 2,
    });

    // Dilation 2x2 with bias
    run_parity(&Case {
        layout: Layout::OHWI,
        w_exps: WeightExps::PerTensor0,
        has_bias: true,
        activation: Activation::Linear,
        out_exp: 0,
        stride: 1,
        dilation: 2,
    });
}
