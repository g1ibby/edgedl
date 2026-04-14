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
    _init_info_map: &BTreeMap<String, InitInfo>,
) -> Result<proc_macro2::TokenStream, String> {
    let inputs: Vec<u16> = if let Some(v) = n.input() {
        let mut ids = Vec::with_capacity(v.len());
        for j in 0..v.len() {
            let nm = v.get(j);
            let id = *name_to_id
                .get(nm)
                .ok_or_else(|| format!("Pad input '{}' missing id", nm))?;
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
                .ok_or_else(|| format!("Pad output '{}' missing id", nm))?;
            ids.push(id);
        }
        ids
    } else {
        Vec::new()
    };

    let input = *inputs
        .first()
        .ok_or_else(|| "Pad missing input".to_string())?;
    let out = *outputs
        .first()
        .ok_or_else(|| "Pad missing output".to_string())?;
    let input_name = n
        .input()
        .and_then(|v| if !v.is_empty() { Some(v.get(0)) } else { None })
        .unwrap_or("");
    let output_name = n
        .output()
        .and_then(|v| if !v.is_empty() { Some(v.get(0)) } else { None })
        .unwrap_or("");

    let mut mode_variant = quote! { edgedl::model::PaddingMode::Constant };
    let mut has_const = false;
    let mut const_i8: i8 = 0;
    let mut pads_vec: Vec<i64> = Vec::new();
    if let Some(attrs) = n.attribute() {
        for a_idx in 0..attrs.len() {
            let a = attrs.get(a_idx);
            if a.name() == Some("mode") {
                if let Some(iv) = a.i() {
                    let m = iv.i();
                    mode_variant = if m == 0 {
                        quote! { edgedl::model::PaddingMode::Constant }
                    } else if m == 1 {
                        quote! { edgedl::model::PaddingMode::Edge }
                    } else {
                        quote! { edgedl::model::PaddingMode::Reflect }
                    };
                }
            }
            if a.name() == Some("pads") {
                if let Some(v) = a.ints() {
                    pads_vec.clear();
                    for k in 0..v.len() {
                        pads_vec.push(v.get(k));
                    }
                }
            }
            if a.name() == Some("value") {
                if let Some(iv) = a.i() {
                    const_i8 = i64_to_u16_checked(iv.i(), "pad_value")
                        .map_err(|_| "pad value out of range".to_string())?
                        as i8;
                    has_const = true;
                }
            }
        }
    }
    if pads_vec.len() != 8 {
        return Err("Pad pads must have length 8 (NHWC)".to_string());
    }
    let pads_nhwc = pads_vec
        .iter()
        .map(|&x| quote! { #x as i64 })
        .collect::<Vec<_>>();
    // Simple reflect feasibility check against value info
    let in_meta_v = val_meta_map
        .get(input_name)
        .ok_or_else(|| "Pad input meta missing".to_string())?;
    let out_meta_v = val_meta_map
        .get(output_name)
        .ok_or_else(|| "Pad output meta missing".to_string())?;
    if let Some(v) = n
        .attribute()
        .and_then(|v| {
            (0..v.len())
                .map(|i| v.get(i))
                .find(|a| a.name() == Some("mode"))
        })
        .and_then(|a| a.i())
    {
        let m = v.i();
        if m == 2 {
            // reflect
            let n0 = pads_vec[0] as u16;
            let n1 = pads_vec[4] as u16;
            let h0 = pads_vec[1] as u16;
            let h1 = pads_vec[5] as u16;
            let w0 = pads_vec[2] as u16;
            let w1 = pads_vec[6] as u16;
            let c0 = pads_vec[3] as u16;
            let c1 = pads_vec[7] as u16;
            let n = in_meta_v.n;
            let h = in_meta_v.h;
            let w = in_meta_v.w;
            let c = in_meta_v.c;
            if n0 >= n || n1 >= n {
                return Err("Pad reflect pads must be < N".to_string());
            }
            if h0 >= h || h1 >= h {
                return Err("Pad reflect pads must be < H".to_string());
            }
            if w0 >= w || w1 >= w {
                return Err("Pad reflect pads must be < W".to_string());
            }
            if c0 >= c || c1 >= c {
                return Err("Pad reflect pads must be < C".to_string());
            }
        }
    }
    // Output shape validation: out = in + pads
    let oh_calc = in_meta_v.h as i64 + pads_vec[1] + pads_vec[5];
    let ow_calc = in_meta_v.w as i64 + pads_vec[2] + pads_vec[6];
    let on_calc = in_meta_v.n as i64 + pads_vec[0] + pads_vec[4];
    let oc_calc = in_meta_v.c as i64 + pads_vec[3] + pads_vec[7];
    if oh_calc != out_meta_v.h as i64
        || ow_calc != out_meta_v.w as i64
        || on_calc != out_meta_v.n as i64
        || oc_calc != out_meta_v.c as i64
    {
        return Err("Pad output shape does not match pads and input shape".to_string());
    }
    let pad = quote! { edgedl::model::NodeOp::Pad(edgedl::model::PadSpec{
        input: #input, output: #out, pads_nhwc: [#( #pads_nhwc ),*], mode: #mode_variant, has_const: #has_const, const_i8: #const_i8
    })};
    Ok(quote! { edgedl::model::NodeSpec { op: #pad } })
}
