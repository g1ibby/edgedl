//! Router for SIMD Conv2D.
//!
//! Decides kernel family and alignment eligibility. For now, this only
//! computes a decision structure used for diagnostics; execution is not
//! implemented yet and the caller should fall back to scalar.

use crate::{
    arena::{Arena, PlannedArena},
    model::{Activation, Conv2dSpec, ModelSpec, ParamLayout},
};

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "trace", derive(defmt::Format))]
pub enum Family {
    K11,
    K33,
    Khw,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "trace", derive(defmt::Format))]
pub enum AlignClass {
    Aligned16,
    Unaligned,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "trace", derive(defmt::Format))]
pub enum FusedAct {
    Linear,
    ReLU,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "trace", derive(defmt::Format))]
pub enum WeightPhy {
    OHWI,
    BlockedN16HWC16,
}

// Fields are read by SIMD kernels on ESP32-S3.
#[allow(dead_code)]
#[derive(Copy, Clone, Debug)]
#[cfg_attr(feature = "trace", derive(defmt::Format))]
pub struct Decision {
    pub family: Family,
    pub align: AlignClass,
    pub fused: FusedAct,
    pub weights: WeightPhy,
    pub ic: usize,
    pub oc: usize,
    pub kh: usize,
    pub kw: usize,
    pub has_bias: bool,
    /// Strides in [H, W]
    pub strides_hw: [u16; 2],
    /// Dilations in [H, W]
    pub dilations_hw: [u16; 2],
    /// Pads as [top, left, bottom, right]
    pub pads_hw: [u16; 4],
    /// Group count (1 = regular conv, C = depthwise)
    pub groups: u16,
}

#[allow(unused_variables)]
pub fn decide(
    spec: &ModelSpec,
    node: &Conv2dSpec,
    plan: &PlannedArena,
    arena: &mut Arena<'_>,
) -> Decision {
    // Weight physical layout from initializer
    let w_init = spec
        .initializers
        .iter()
        .find(|ini| ini.id == node.weights)
        .expect("weight initializer missing");
    let weights = match w_init.layout {
        ParamLayout::WeightsI8OHWI { .. } => WeightPhy::OHWI,
        ParamLayout::WeightsI8BlockedN16HWC16 { .. } => WeightPhy::BlockedN16HWC16,
        _ => WeightPhy::OHWI,
    };

    // Family by kernel size (from weight initializer dims)
    let (kh, kw) = match w_init.layout {
        ParamLayout::WeightsI8OHWI { kh, kw, .. } => (kh as usize, kw as usize),
        ParamLayout::WeightsI8BlockedN16HWC16 { kh, kw, .. } => (kh as usize, kw as usize),
        _ => (1usize, 1usize),
    };
    let family = if kh == 1 && kw == 1 {
        Family::K11
    } else if kh == 3 && kw == 3 {
        Family::K33
    } else {
        Family::Khw
    };

    // Activation fuse
    let fused = match node.activation {
        Activation::Linear => FusedAct::Linear,
        Activation::ReLU => FusedAct::ReLU,
    };

    // Alignment check (16-byte) for inputs/outputs via arena offsets.
    // Also require channel multiples of 16 for aligned fast path.
    let in_meta = spec.values[node.input as usize];
    let out_meta = spec.values[node.output as usize];
    let ic = in_meta.shape.c as usize;
    let oc = out_meta.shape.c as usize;
    let in_off = plan.offset_of(node.input);
    let out_off = plan.offset_of(node.output);
    let aligned = (in_off % 16 == 0) && (out_off % 16 == 0) && (ic % 16 == 0) && (oc % 16 == 0);
    let align = if aligned {
        AlignClass::Aligned16
    } else {
        AlignClass::Unaligned
    };

    let has_bias = node.bias.is_some();

    let dec = Decision {
        family,
        align,
        fused,
        weights,
        ic,
        oc,
        kh,
        kw,
        has_bias,
        strides_hw: node.strides_hw,
        dilations_hw: node.dilations_hw,
        pads_hw: node.pads_hw,
        groups: node.groups,
    };

    dec
}
