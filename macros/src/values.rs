use std::collections::BTreeMap;

use proc_macro_error::abort_call_site;
use quote::quote;

use crate::{context::ValMeta, dl_generated::dl, util::i64_to_u16_checked};

/// Build `values` array tokens and input/output id arrays. Complements `val_meta_map`
/// by inferring shapes from initializer dims when missing in value_info.
pub fn emit_values_and_io(
    fb_graph: dl::Graph,
    name_to_id: &BTreeMap<String, u16>,
    name_by_id: &[String],
    val_meta_map: &mut BTreeMap<String, ValMeta>,
) -> (Vec<proc_macro2::TokenStream>, Vec<u16>, Vec<u16>) {
    // Build tokens strictly in ValueId order using name_by_id
    let mut values_tokens: Vec<proc_macro2::TokenStream> = Vec::with_capacity(name_by_id.len());

    for name in name_by_id.iter() {
        // Seed from val_meta_map (from value_info), then patch missing shape from initializer.
        let mut n: u16 = 1;
        let mut h: u16 = 0;
        let mut w: u16 = 0;
        let mut c: u16 = 0;
        let mut exp_i8: i8 = 0;

        if let Some(vm) = val_meta_map.get(name.as_str()) {
            n = vm.n;
            h = vm.h;
            w = vm.w;
            c = vm.c;
            if let Some(e) = vm.exp {
                exp_i8 = e;
            }
        }

        // If shape is missing/unknown, derive from initializer dims (parameters commonly lack
        // value_info)
        if h == 0 || w == 0 || c == 0 {
            if let Some(v) = fb_graph.initializer() {
                for i in 0..v.len() {
                    let t = v.get(i);
                    if t.name() == Some(name.as_str()) {
                        if let Some(dv) = t.dims() {
                            let len = dv.len();
                            let (n2, h2, w2, c2) = if len == 4 {
                                (
                                    i64_to_u16_checked(dv.get(0), "N").unwrap_or_else(|e| {
                                        abort_call_site!("initializer {}: {}", name, e)
                                    }),
                                    i64_to_u16_checked(dv.get(1), "H").unwrap_or_else(|e| {
                                        abort_call_site!("initializer {}: {}", name, e)
                                    }),
                                    i64_to_u16_checked(dv.get(2), "W").unwrap_or_else(|e| {
                                        abort_call_site!("initializer {}: {}", name, e)
                                    }),
                                    i64_to_u16_checked(dv.get(3), "C").unwrap_or_else(|e| {
                                        abort_call_site!("initializer {}: {}", name, e)
                                    }),
                                )
                            } else if len == 3 {
                                (
                                    1,
                                    i64_to_u16_checked(dv.get(0), "H").unwrap_or_else(|e| {
                                        abort_call_site!("initializer {}: {}", name, e)
                                    }),
                                    i64_to_u16_checked(dv.get(1), "W").unwrap_or_else(|e| {
                                        abort_call_site!("initializer {}: {}", name, e)
                                    }),
                                    i64_to_u16_checked(dv.get(2), "C").unwrap_or_else(|e| {
                                        abort_call_site!("initializer {}: {}", name, e)
                                    }),
                                )
                            } else if len == 1 {
                                (
                                    1,
                                    1,
                                    1,
                                    i64_to_u16_checked(dv.get(0), "C").unwrap_or_else(|e| {
                                        abort_call_site!("initializer {}: {}", name, e)
                                    }),
                                )
                            } else {
                                (n, h, w, c)
                            };
                            n = n2;
                            h = h2;
                            w = w2;
                            c = c2;
                        }
                        // Patch val_meta_map entry without inventing exponents
                        match val_meta_map.get_mut(name) {
                            Some(vm) => {
                                if vm.h == 0 || vm.w == 0 || vm.c == 0 {
                                    vm.n = n;
                                    vm.h = h;
                                    vm.w = w;
                                    vm.c = c;
                                }
                            }
                            None => {
                                val_meta_map.insert(
                                    name.clone(),
                                    ValMeta {
                                        n,
                                        h,
                                        w,
                                        c,
                                        exp: None,
                                    },
                                );
                            }
                        }
                        break;
                    }
                }
            }
        }

        // Enforce that all values have a concrete NHWC shape at this point. If still unknown,
        // abort with a clear diagnostic rather than emitting zeros that could mislead the engine.
        if h == 0 || w == 0 || c == 0 {
            abort_call_site!(
                "missing NHWC shape for value '{}': not in value_info as 4D/3D/2D/1D and no initializer dims found",
                name
            );
        }

        let val_ts = quote! {
            edgedl::model::ValueMeta {
                shape: edgedl::model::Shape4dNHWC { n: #n, h: #h, w: #w, c: #c },
                exp: #exp_i8
            }
        };
        values_tokens.push(val_ts);
    }

    // Inputs/Outputs arrays (by id order of names listed in graph inputs/outputs)
    let mut inputs_ids = Vec::<u16>::new();
    if let Some(v) = fb_graph.input() {
        for i in 0..v.len() {
            if let Some(n) = v.get(i).name() {
                inputs_ids.push(
                    *name_to_id
                        .get(n)
                        .unwrap_or_else(|| abort_call_site!("graph input '{}' missing id", n)),
                );
            }
        }
    }
    let mut outputs_ids = Vec::<u16>::new();
    if let Some(v) = fb_graph.output() {
        for i in 0..v.len() {
            if let Some(n) = v.get(i).name() {
                outputs_ids.push(
                    *name_to_id
                        .get(n)
                        .unwrap_or_else(|| abort_call_site!("graph output '{}' missing id", n)),
                );
            }
        }
    }

    (values_tokens, inputs_ids, outputs_ids)
}
