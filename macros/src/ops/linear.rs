use std::collections::BTreeMap;

use quote::quote;

use crate::{
    context::{InitInfo, ValMeta},
    dl_generated::dl,
};

pub fn emit(
    n: dl::Node,
    fb_graph: dl::Graph,
    name_to_id: &BTreeMap<String, u16>,
    val_meta_map: &BTreeMap<String, ValMeta>,
    init_info_map: &BTreeMap<String, InitInfo>,
) -> Result<proc_macro2::TokenStream, String> {
    let inputs: Vec<u16> = if let Some(v) = n.input() {
        let mut ids = Vec::with_capacity(v.len());
        for j in 0..v.len() {
            let nm = v.get(j);
            let id = *name_to_id
                .get(nm)
                .ok_or_else(|| format!("Gemm input '{}' missing id", nm))?;
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
                .ok_or_else(|| format!("Gemm output '{}' missing id", nm))?;
            ids.push(id);
        }
        ids
    } else {
        Vec::new()
    };

    let input = *inputs
        .first()
        .ok_or_else(|| "Gemm missing input".to_string())?;
    let weights = *inputs
        .get(1)
        .ok_or_else(|| "Gemm missing weights".to_string())?;
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

    // Attributes: alpha, beta, transA, transB, activation
    let mut alpha_i64: i64 = 1;
    let mut beta_i64: i64 = 1;
    let mut trans_a: i64 = 0;
    let mut _trans_b: i64 = 1; // potentially unused
    let mut act = quote! { edgedl::model::Activation::Linear };
    if let Some(attrs) = n.attribute() {
        for a_idx in 0..attrs.len() {
            let a = attrs.get(a_idx);
            if a.name() == Some("alpha") {
                if let Some(fv) = a.f() {
                    alpha_i64 = fv.f() as i64;
                }
                if let Some(iv) = a.i() {
                    alpha_i64 = iv.i();
                }
            }
            if a.name() == Some("beta") {
                if let Some(fv) = a.f() {
                    beta_i64 = fv.f() as i64;
                }
                if let Some(iv) = a.i() {
                    beta_i64 = iv.i();
                }
            }
            if a.name() == Some("transA") {
                if let Some(iv) = a.i() {
                    trans_a = iv.i();
                }
            }
            if a.name() == Some("transB") {
                if let Some(iv) = a.i() {
                    _trans_b = iv.i();
                }
            }
            if a.name() == Some("activation") {
                // Prefer string attribute if present, e.g., "Relu"
                if let Some(bytes_vec) = a.s() {
                    let mut raw = Vec::with_capacity(bytes_vec.len());
                    for j in 0..bytes_vec.len() {
                        raw.push(bytes_vec.get(j));
                    }
                    let s = String::from_utf8(raw).unwrap_or_default();
                    if s.eq_ignore_ascii_case("relu") {
                        act = quote! { edgedl::model::Activation::ReLU };
                    } else {
                        act = quote! { edgedl::model::Activation::Linear };
                    }
                } else if let Some(iv) = a.i() {
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
    if alpha_i64 != 1 || beta_i64 != 1 {
        return Err("Gemm: only alpha=1 and beta=1 are supported in M1".to_string());
    }
    if trans_a != 0 {
        return Err("Gemm: transA != 0 not supported in M1".to_string());
    }

    let in_meta_v = val_meta_map
        .get(input_name)
        .ok_or_else(|| "Gemm input meta missing".to_string())?;
    let out_meta_v = val_meta_map
        .get(output_name)
        .ok_or_else(|| "Gemm output meta missing".to_string())?;
    let w_info = init_info_map
        .get(weights_name)
        .ok_or_else(|| "Gemm weights meta missing".to_string())?;
    if w_info.dtype != dl::TensorDataType::INT8 {
        return Err("Gemm weights must be INT8".to_string());
    }
    if let Some((oc_w, _kh, _kw, ic_w)) = w_info.dims4 {
        if ic_w != in_meta_v.c {
            return Err("Gemm weights IC must match input channels".to_string());
        }
        if oc_w != out_meta_v.c {
            return Err("Gemm weights OC must match output channels".to_string());
        }
    } else {
        return Err("Gemm weights dims missing".to_string());
    }
    if in_meta_v.n != out_meta_v.n || in_meta_v.h != out_meta_v.h || in_meta_v.w != out_meta_v.w {
        return Err("Gemm must preserve N/H/W dims".to_string());
    }
    if in_meta_v.exp.is_none() {
        return Err("Gemm input lacks exponent in value_info".to_string());
    }
    if out_meta_v.exp.is_none() {
        return Err("Gemm output lacks exponent in value_info".to_string());
    }
    if w_info.exps_len != 1 {
        return Err("Gemm weights must have per-tensor exponent".to_string());
    }
    if let Some(_bid) = bias_opt {
        // Verify bias initializer exists (basic presence check)
        let mut found = false;
        if let Some(inits) = fb_graph.initializer() {
            if let Some(name) = n
                .input()
                .and_then(|v| if v.len() >= 3 { Some(v.get(2)) } else { None })
            {
                for j in 0..inits.len() {
                    let t = inits.get(j);
                    if t.name() == Some(name) {
                        found = true;
                        break;
                    }
                }
            }
        }
        if !found {
            return Err("Gemm bias initializer missing".to_string());
        }
    }

    let out = *outputs
        .first()
        .ok_or_else(|| "Gemm missing output".to_string())?;
    let bias_expr = if let Some(b) = bias_opt {
        quote! { Some(#b) }
    } else {
        quote! { None }
    };
    let node = quote! { edgedl::model::NodeOp::Linear(edgedl::model::LinearSpec{
        input: #input, weights: #weights, bias: #bias_expr, output: #out, activation: #act
    })};
    Ok(quote! { edgedl::model::NodeSpec { op: #node } })
}
