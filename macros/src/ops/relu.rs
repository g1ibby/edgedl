use std::collections::BTreeMap;

use quote::quote;

use crate::{context::ValMeta, dl_generated::dl};

pub fn emit(
    n: dl::Node,
    _fb_graph: dl::Graph,
    name_to_id: &BTreeMap<String, u16>,
    val_meta_map: &BTreeMap<String, ValMeta>,
    _init_info_map: &BTreeMap<String, crate::context::InitInfo>,
) -> Result<proc_macro2::TokenStream, String> {
    // Resolve inputs/outputs
    let inputs: Vec<u16> = if let Some(v) = n.input() {
        let mut ids = Vec::with_capacity(v.len());
        for j in 0..v.len() {
            let nm = v.get(j);
            let id = *name_to_id
                .get(nm)
                .ok_or_else(|| format!("Relu input '{}' missing id", nm))?;
            ids.push(id);
        }
        ids
    } else {
        Vec::new()
    };
    let outputs: Vec<u16> = if let Some(v) = n.output() {
        let mut ids = Vec::with_capacity(v.len());
        for j in 0..v.len() {
            let nm = v.get(j);
            let id = *name_to_id
                .get(nm)
                .ok_or_else(|| format!("Relu output '{}' missing id", nm))?;
            ids.push(id);
        }
        ids
    } else {
        Vec::new()
    };

    let input = *inputs
        .first()
        .ok_or_else(|| "Relu missing input".to_string())?;
    let output = *outputs
        .first()
        .ok_or_else(|| "Relu missing output".to_string())?;

    // Validate shapes/exponents exist and are compatible (N/H/W/C must match)
    let in_name = n
        .input()
        .and_then(|v| if !v.is_empty() { Some(v.get(0)) } else { None })
        .unwrap_or("");
    let out_name = n
        .output()
        .and_then(|v| if !v.is_empty() { Some(v.get(0)) } else { None })
        .unwrap_or("");
    let in_meta_v = val_meta_map
        .get(in_name)
        .ok_or_else(|| "Relu input meta missing".to_string())?;
    let out_meta_v = val_meta_map
        .get(out_name)
        .ok_or_else(|| "Relu output meta missing".to_string())?;
    if in_meta_v.n != out_meta_v.n
        || in_meta_v.h != out_meta_v.h
        || in_meta_v.w != out_meta_v.w
        || in_meta_v.c != out_meta_v.c
    {
        return Err("Relu must preserve N/H/W/C dims".to_string());
    }
    // Exponents must be present; we keep output exponent equal to input exponent
    if in_meta_v.exp.is_none() || out_meta_v.exp.is_none() {
        return Err("Relu requires input/output exponents in value_info".to_string());
    }

    let node = quote! { edgedl::model::NodeOp::ReLU(edgedl::model::ActivationSpec{ input: #input, output: #output }) };
    Ok(quote! { edgedl::model::NodeSpec { op: #node } })
}
