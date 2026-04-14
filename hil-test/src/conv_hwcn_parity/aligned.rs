use edgedl::model::Activation;

use super::util::*;

pub fn run_group() {
    // Aligned cases: IC % 16 == 0, OC % 16 == 0
    // We use IC=16, OC=16 from util.rs which satisfies this.

    // 1. Basic Linear
    run_parity(&Case {
        layout: Layout::OHWI,
        w_exps: WeightExps::PerTensor0,
        has_bias: false,
        activation: Activation::Linear,
    });

    // 2. Basic ReLU
    run_parity(&Case {
        layout: Layout::OHWI,
        w_exps: WeightExps::PerTensor0,
        has_bias: false,
        activation: Activation::ReLU,
    });

    // 3. With Bias Linear
    run_parity(&Case {
        layout: Layout::OHWI,
        w_exps: WeightExps::PerTensor0,
        has_bias: true,
        activation: Activation::Linear,
    });

    // 4. With Bias ReLU
    run_parity(&Case {
        layout: Layout::OHWI,
        w_exps: WeightExps::PerTensor0,
        has_bias: true,
        activation: Activation::ReLU,
    });

    // 5. Per-channel weights Linear
    run_parity(&Case {
        layout: Layout::OHWI,
        w_exps: WeightExps::PerChannel,
        has_bias: false,
        activation: Activation::Linear,
    });

    // 6. Per-channel weights + Bias
    run_parity(&Case {
        layout: Layout::OHWI,
        w_exps: WeightExps::PerChannel,
        has_bias: true,
        activation: Activation::Linear,
    });
}
