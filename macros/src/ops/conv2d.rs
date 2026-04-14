use std::collections::BTreeMap;

use quote::quote;

use crate::{
    context::{InitInfo, ValMeta},
    dl_generated::dl,
    util::i64_to_u16_checked,
};

pub fn emit(
    n: dl::Node,
    _fb_graph: dl::Graph,
    name_to_id: &BTreeMap<String, u16>,
    val_meta_map: &BTreeMap<String, ValMeta>,
    init_info_map: &BTreeMap<String, InitInfo>,
) -> Result<(proc_macro2::TokenStream, usize), String> {
    let inputs: Vec<u16> = if let Some(v) = n.input() {
        let mut ids = Vec::with_capacity(v.len());
        for j in 0..v.len() {
            let nm = v.get(j);
            let id = *name_to_id
                .get(nm)
                .ok_or_else(|| format!("Conv input '{}' missing id", nm))?;
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
                .ok_or_else(|| format!("Conv output '{}' missing id", nm))?;
            ids.push(id);
        }
        ids
    } else {
        Vec::new()
    };

    let input = *inputs
        .first()
        .ok_or_else(|| "Conv missing input".to_string())?;
    let weights = *inputs
        .get(1)
        .ok_or_else(|| "Conv missing weights".to_string())?;
    let bias_opt = inputs.get(2).copied();
    let input_name = n
        .input()
        .and_then(|v| if !v.is_empty() { Some(v.get(0)) } else { None })
        .unwrap_or("");
    let weights_name = n
        .input()
        .and_then(|v| if v.len() >= 2 { Some(v.get(1)) } else { None })
        .unwrap_or("");
    let output_name = n
        .output()
        .and_then(|v| if !v.is_empty() { Some(v.get(0)) } else { None })
        .unwrap_or("");

    // Attributes
    let mut strides = [1i64, 1];
    let mut dil = [1i64, 1];
    let mut pads = [0i64, 0, 0, 0];
    let mut groups = 1i64;
    if let Some(attrs) = n.attribute() {
        for a_idx in 0..attrs.len() {
            let a = attrs.get(a_idx);
            if a.name() == Some("strides") {
                if let Some(v) = a.ints() {
                    if v.len() >= 2 {
                        strides = [v.get(0), v.get(1)];
                    }
                }
            }
            if a.name() == Some("dilations") {
                if let Some(v) = a.ints() {
                    if v.len() >= 2 {
                        dil = [v.get(0), v.get(1)];
                    }
                }
            }
            if a.name() == Some("pads") {
                if let Some(v) = a.ints() {
                    if v.len() >= 4 {
                        pads = [v.get(0), v.get(1), v.get(2), v.get(3)];
                    }
                }
            }
            if a.name() == Some("group") {
                if let Some(iv) = a.i() {
                    groups = iv.i();
                }
            }
        }
    }
    let strides_hw = [
        i64_to_u16_checked(strides[0], "stride_h")?,
        i64_to_u16_checked(strides[1], "stride_w")?,
    ];
    let dil_hw = [
        i64_to_u16_checked(dil[0], "dil_h")?,
        i64_to_u16_checked(dil[1], "dil_w")?,
    ];
    let pads_hw = [
        i64_to_u16_checked(pads[0], "pad_top")?,
        i64_to_u16_checked(pads[1], "pad_left")?,
        i64_to_u16_checked(pads[2], "pad_bottom")?,
        i64_to_u16_checked(pads[3], "pad_right")?,
    ];
    let groups_u16 = i64_to_u16_checked(groups, "groups")?;
    let out = *outputs
        .first()
        .ok_or_else(|| "Conv missing output".to_string())?;
    if groups_u16 != 1 {
        return Err("Conv groups != 1 not supported in M1".to_string());
    }

    // Validations
    let in_meta_v = val_meta_map
        .get(input_name)
        .ok_or_else(|| "Conv input meta missing".to_string())?;
    let out_meta_v = val_meta_map
        .get(output_name)
        .ok_or_else(|| "Conv output meta missing".to_string())?;
    let w_info = init_info_map
        .get(weights_name)
        .ok_or_else(|| "Conv weights meta missing".to_string())?;
    if w_info.dtype != dl::TensorDataType::INT8 {
        return Err("Conv weights must be INT8".to_string());
    }
    // Validate weight dims vs input/output channels
    if let Some((oc_w, _kh, _kw, ic_w)) = w_info.dims4 {
        if ic_w != in_meta_v.c {
            return Err("Conv weights IC must match input channels".to_string());
        }
        if oc_w != out_meta_v.c {
            return Err("Conv weights OC must match output channels".to_string());
        }
    } else {
        return Err("Conv weights dims missing".to_string());
    }
    if in_meta_v.n != out_meta_v.n {
        return Err("Conv batch N must be preserved".to_string());
    }
    if in_meta_v.exp.is_none() {
        return Err("Conv input lacks exponent in value_info".to_string());
    }
    if out_meta_v.exp.is_none() {
        return Err("Conv output lacks exponent in value_info".to_string());
    }
    if w_info.exps_len == 0 {
        return Err("Conv weights lack exponents (need per-tensor or per-OC)".to_string());
    }

    // Activation attribute (fused). Exporters may encode it as STRING ("Relu") or INT
    // (dl::activation_type_t).
    let mut act = quote! { edgedl::model::Activation::Linear };
    if let Some(attrs) = n.attribute() {
        for a_idx in 0..attrs.len() {
            let a = attrs.get(a_idx);
            if a.name() == Some("activation") {
                // Prefer string value when present: "Relu" or "ReLU" => ReLU
                if let Some(bytes_vec) = a.s() {
                    // bytes stored as FlatBuffers Vector<u8>
                    let mut raw = Vec::with_capacity(bytes_vec.len());
                    for j in 0..bytes_vec.len() {
                        raw.push(bytes_vec.get(j));
                    }
                    let s = String::from_utf8(raw).unwrap_or_default();
                    let s_lower = s.to_ascii_lowercase();
                    if s_lower == "relu" {
                        act = quote! { edgedl::model::Activation::ReLU };
                    } else {
                        act = quote! { edgedl::model::Activation::Linear };
                    }
                } else if let Some(iv) = a.i() {
                    // Fallback: integer enum (1 => ReLU)
                    let code = iv.i();
                    act = if code == 1 {
                        quote! { edgedl::model::Activation::ReLU }
                    } else {
                        quote! { edgedl::model::Activation::Linear }
                    };
                }
            }
        }
    }
    let bias_expr = if let Some(b) = bias_opt {
        quote! { Some(#b) }
    } else {
        quote! { None }
    };
    let conv = quote! { edgedl::model::NodeOp::Conv2d(edgedl::model::Conv2dSpec{
        input: #input, weights: #weights, bias: #bias_expr, output: #out,
        strides_hw: [#(#strides_hw),*], dilations_hw: [#(#dil_hw),*], pads_hw: [#(#pads_hw),*],
        activation: #act, groups: #groups_u16,
    })};
    // SIMD scratch estimate for per-channel path (max across model):
    // If weights have per-OC exponents (exps_len == oc), we may need scales (oc_aligned*i16)
    // and bias16 when bias is present (oc_aligned*i16). This is a conservative over-approx;
    // the runtime carves exact bytes per node at execution.
    let mut scratch: usize = 0;
    if let Some(wi) = init_info_map.get(weights_name) {
        if let Some((_oc_w, _kh, _kw, _ic_w)) = wi.dims4 {
            let oc = val_meta_map.get(output_name).unwrap().c as usize;
            let oc_aligned = (oc + 15) & !15usize;
            // Reserve scratch when per-OC exponents are present (scales and optional bias16)
            // and conservatively also when per-tensor (we may need per-layer negative left-shift,
            // which we handle via the per-channel macro with uniform scales).
            if wi.exps_len == oc || wi.exps_len == 1 {
                let mut need = oc_aligned * 2; // scales i16
                if bias_opt.is_some() {
                    need += oc_aligned * 2; // bias16 i16
                }
                if need > scratch {
                    scratch = need;
                }
            }
        }
    }
    Ok((quote! { edgedl::model::NodeSpec { op: #conv } }, scratch))
}
