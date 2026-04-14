use std::collections::BTreeMap;

use quote::quote;

use crate::{
    context::{InitInfo, ValMeta},
    dl_generated::dl,
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
                .ok_or_else(|| format!("ReduceMean input '{}' missing id", nm))?;
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
                .ok_or_else(|| format!("ReduceMean output '{}' missing id", nm))?;
            ids.push(id);
        }
        ids
    } else {
        Vec::new()
    };

    let input = *inputs
        .first()
        .ok_or_else(|| "ReduceMean missing input".to_string())?;
    let out = *outputs
        .first()
        .ok_or_else(|| "ReduceMean missing output".to_string())?;
    // Names for shape inference (if needed)
    let input_name = n
        .input()
        .and_then(|v| if !v.is_empty() { Some(v.get(0)) } else { None })
        .unwrap_or("");
    let output_name = n
        .output()
        .and_then(|v| if !v.is_empty() { Some(v.get(0)) } else { None })
        .unwrap_or("");

    let mut keepdims = true; // ONNX default keepdims=1
    let mut axes_bits: u8 = 0; // bitmap over NHWC (0=N,1=H,2=W,3=C)
    let mut axes_found = false;
    if let Some(attrs) = n.attribute() {
        for a_idx in 0..attrs.len() {
            let a = attrs.get(a_idx);
            if a.name() == Some("keepdims") {
                if let Some(iv) = a.i() {
                    keepdims = iv.i() != 0;
                }
            }
            if a.name() == Some("axes") {
                if let Some(v) = a.ints() {
                    for k in 0..v.len() {
                        let mut ax = v.get(k);
                        // Support negative axes (ONNX semantics): wrap by rank 4
                        if ax < 0 {
                            ax += 4;
                        }
                        if !(0..=3).contains(&ax) {
                            return Err(
                                "ReduceMean axes must be within [-4,3] for NHWC".to_string()
                            );
                        }
                        axes_bits |= 1u8 << (ax as u8);
                        axes_found = true;
                    }
                }
            }
        }
    }
    // Fallback: if axes not provided, infer from input/output shapes (NHWC) when possible.
    if !axes_found {
        if let (Some(in_vm), Some(out_vm)) =
            (val_meta_map.get(input_name), val_meta_map.get(output_name))
        {
            if in_vm.n > out_vm.n {
                axes_bits |= 0b0001;
            }
            if in_vm.h > out_vm.h {
                axes_bits |= 0b0010;
            }
            if in_vm.w > out_vm.w {
                axes_bits |= 0b0100;
            }
            if in_vm.c > out_vm.c {
                axes_bits |= 0b1000;
            }
            // If still zero and shapes equal, leave as 0 (no-op reduction) —
            // this matches bad exporters but keeps macro robust.
        }
    }
    let rm = quote! { edgedl::model::NodeOp::ReduceMean(edgedl::model::ReduceMeanSpec{ input: #input, output: #out, axes_bitmap_nhwc: #axes_bits, keepdims: #keepdims }) };
    Ok(quote! { edgedl::model::NodeSpec { op: #rm } })
}
