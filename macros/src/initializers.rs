use std::collections::BTreeMap;

use quote::{format_ident, quote};

use crate::{
    context::InitInfo,
    dl_generated::dl,
    util::{bytes_from_tensor_full, i64_to_i8_checked, i64_to_u16_checked},
};

pub struct InitTokens {
    pub statics: Vec<proc_macro2::TokenStream>,
    pub table: Vec<proc_macro2::TokenStream>,
}

/// Scan initializers, update `init_info_map` and `val_meta_map`, and emit static
/// data + initializer table tokens.
pub fn emit_initializers(
    fb_graph: dl::Graph,
    name_to_id: &BTreeMap<String, u16>,
    init_info_map: &mut BTreeMap<String, InitInfo>,
    manifest_dir: &str,
) -> Result<InitTokens, String> {
    struct InitTmp {
        id: u16,
        layout: proc_macro2::TokenStream,
        data: Vec<u8>,
        exponents: Vec<i8>,
    }

    let mut inits_tmp: Vec<InitTmp> = Vec::new();
    if let Some(v) = fb_graph.initializer() {
        for i in 0..v.len() {
            let t = v.get(i);
            let name = t.name().unwrap_or("");
            let id = match name_to_id.get(name) {
                Some(v) => *v,
                None => return Err(format!("initializer '{}' missing id in name map", name)),
            };
            let dt = t.data_type();
            let dims = t.dims();
            if dt == dl::TensorDataType::INT8 {
                // Weights: exporter provides HWIO [KH, KW, IC, OC]; map to OHWI and reorder bytes
                // accordingly.
                let (oc, kh, kw, ic) = if let Some(dv) = dims {
                    if dv.len() == 4 {
                        let kh = i64_to_u16_checked(dv.get(0), "kh")?;
                        let kw = i64_to_u16_checked(dv.get(1), "kw")?;
                        let ic = i64_to_u16_checked(dv.get(2), "ic")?;
                        let oc = i64_to_u16_checked(dv.get(3), "oc")?;
                        (oc, kh, kw, ic)
                    } else {
                        return Err(format!(
                            "initializer '{}' INT8 weights must be rank 4 HWIO, got rank {}",
                            name,
                            dv.len()
                        ));
                    }
                } else {
                    return Err(format!("initializer '{}' INT8 weights missing dims", name));
                };
                // Inspect doc_string for packed layout hints (ESP-DL blocked HWC16)
                let mut use_blocked_n16 = false;
                if let Some(ds) = t.doc_string() {
                    let dsl = ds.to_ascii_lowercase();
                    let hints_blocked = dsl.contains("(n/16)hwc16");
                    let unaligned = dsl.contains("unaligned");
                    if hints_blocked && !unaligned && (oc % 16 == 0) {
                        use_blocked_n16 = true;
                    }
                }
                // 1x1 Gemm heads with small OC are usually plain OHWI
                if kh == 1 && kw == 1 && oc < 16 {
                    use_blocked_n16 = false;
                }
                // Exporter provides bytes already in the runtime-consumed memory order
                let data = bytes_from_tensor_full(&t)?;
                // Decide final layout: use blocked only if explicitly hinted by doc_string.
                // Fallback to OHWI otherwise.
                let layout = if use_blocked_n16 {
                    quote! { edgedl::model::ParamLayout::WeightsI8BlockedN16HWC16 { oc: #oc, kh: #kh, kw: #kw, ic: #ic } }
                } else {
                    quote! { edgedl::model::ParamLayout::WeightsI8OHWI { oc: #oc, kh: #kh, kw: #kw, ic: #ic } }
                };
                let mut exps = Vec::<i8>::new();
                if let Some(es) = t.exponents() {
                    for j in 0..es.len() {
                        exps.push(i64_to_i8_checked(es.get(j))?);
                    }
                }
                inits_tmp.push(InitTmp {
                    id,
                    layout,
                    data,
                    exponents: exps,
                });
                // Record weights info for validations
                let dims4 = Some((oc, kh, kw, ic));
                init_info_map.insert(
                    name.to_string(),
                    InitInfo {
                        dtype: dt,
                        dims4,
                        exps_len: t.exponents().map(|v| v.len()).unwrap_or(0),
                    },
                );
            } else if dt == dl::TensorDataType::INT32 {
                // Bias: [OC]
                let mut oc: u16 = 0;
                if let Some(dv) = dims {
                    if !dv.is_empty() {
                        oc = i64_to_u16_checked(dv.get(0), "oc")?;
                    }
                }
                let layout = quote! { edgedl::model::ParamLayout::BiasI32 { oc: #oc } };
                let data = bytes_from_tensor_full(&t)?;
                let mut exps = Vec::<i8>::new();
                if let Some(es) = t.exponents() {
                    for j in 0..es.len() {
                        exps.push(i64_to_i8_checked(es.get(j))?);
                    }
                }
                inits_tmp.push(InitTmp {
                    id,
                    layout,
                    data,
                    exponents: exps,
                });
                init_info_map.insert(
                    name.to_string(),
                    InitInfo {
                        dtype: dt,
                        dims4: None,
                        exps_len: t.exponents().map(|v| v.len()).unwrap_or(0),
                    },
                );
            } else {
                // Other types not supported in M1; retain info for diagnostics
                init_info_map.insert(
                    name.to_string(),
                    InitInfo {
                        dtype: dt,
                        dims4: None,
                        exps_len: 0,
                    },
                );
            }
        }
    }

    // Write initializer blobs to binary files and reference them via
    // include_bytes!(). This prevents the Xtensa LLVM backend from attempting
    // (and failing) to instruction-select large constants from thousands of
    // inline literals (esp-clang >= 20.1.1).
    let data_dir = format!("{}/target/edgedl-data", manifest_dir);
    std::fs::create_dir_all(&data_dir)
        .map_err(|e| format!("failed to create {}: {}", data_dir, e))?;

    let mut init_statics_tokens = Vec::<proc_macro2::TokenStream>::new();
    // Define a 16-byte aligned wrapper to ensure parameter blobs meet alignment requirements.
    init_statics_tokens.push(quote! { #[repr(align(16))] struct __NN_ALIGN16<T>(T); });
    let mut init_array_tokens = Vec::<proc_macro2::TokenStream>::new();
    for (idx, it) in inits_tmp.iter().enumerate() {
        let ident = format_ident!("__NN_INIT_{}", idx);
        let ident_exps = format_ident!("__NN_INIT_{}_EXPS", idx);
        let bytes = &it.data;
        let len = bytes.len();

        let data_path = format!("{}/init_{}.bin", data_dir, idx);
        std::fs::write(&data_path, bytes)
            .map_err(|e| format!("failed to write {}: {}", data_path, e))?;

        let exps = &it.exponents;
        let exps_len = exps.len();
        let mut exp_elems = Vec::<proc_macro2::TokenStream>::with_capacity(exps_len);
        for e in exps.iter() {
            exp_elems.push(quote! { #e });
        }
        let layout = &it.layout;
        init_statics_tokens.push(quote! {
            #[cfg_attr(target_os = "none", link_section = ".data")]
            static #ident: __NN_ALIGN16<[u8; #len]> = __NN_ALIGN16(*include_bytes!(#data_path));
            #[cfg_attr(target_os = "none", link_section = ".data")]
            static #ident_exps: __NN_ALIGN16<[i8; #exps_len ]> = __NN_ALIGN16([ #( #exp_elems ),* ]);
        });
        // Build exponents enum variant
        let exps_variant = if exps.is_empty() {
            quote! { edgedl::model::ParamExponents::PerTensor(0) }
        } else if exps.len() == 1 {
            quote! { edgedl::model::ParamExponents::PerTensor(#ident_exps.0[0]) }
        } else {
            quote! { edgedl::model::ParamExponents::PerChannel(&#ident_exps.0) }
        };
        let id = it.id;
        init_array_tokens.push(quote! {
            edgedl::model::Initializer { id: #id, layout: #layout, data: &#ident.0, exponents: #exps_variant }
        });
    }

    Ok(InitTokens {
        statics: init_statics_tokens,
        table: init_array_tokens,
    })
}
