use std::collections::{BTreeMap, BTreeSet};

use crate::{
    context::{InitInfo, ValMeta},
    dl_generated::dl,
};

mod conv2d;
mod linear;
mod pad;
mod reduce_mean;
mod relu;

pub fn emit_nodes(
    fb_graph: dl::Graph,
    name_to_id: &BTreeMap<String, u16>,
    val_meta_map: &BTreeMap<String, ValMeta>,
    init_info_map: &BTreeMap<String, InitInfo>,
) -> Result<(Vec<proc_macro2::TokenStream>, BTreeSet<String>, usize), String> {
    let mut plan_tokens = Vec::<proc_macro2::TokenStream>::new();
    let mut ops_set: BTreeSet<String> = BTreeSet::new();
    let mut simd_scratch_max: usize = 0;

    if let Some(nodes) = fb_graph.node() {
        for i in 0..nodes.len() {
            let n = nodes.get(i);
            let op = n.op_type().unwrap_or("");
            ops_set.insert(op.to_string());
            // Inputs/outputs are resolved within each op emitter with contextual errors.

            match op {
                "Conv" | "ConvInteger" => {
                    let (ts, scratch) =
                        conv2d::emit(n, fb_graph, name_to_id, val_meta_map, init_info_map)?;
                    plan_tokens.push(ts);
                    if scratch > simd_scratch_max {
                        simd_scratch_max = scratch;
                    }
                }
                "Gemm" => {
                    let ts = linear::emit(n, fb_graph, name_to_id, val_meta_map, init_info_map)?;
                    plan_tokens.push(ts);
                }
                "Pad" => {
                    let ts = pad::emit(n, fb_graph, name_to_id, val_meta_map, init_info_map)?;
                    plan_tokens.push(ts);
                }
                "ReduceMean" => {
                    let ts =
                        reduce_mean::emit(n, fb_graph, name_to_id, val_meta_map, init_info_map)?;
                    plan_tokens.push(ts);
                }
                "Relu" => {
                    let ts = relu::emit(n, fb_graph, name_to_id, val_meta_map, init_info_map)?;
                    plan_tokens.push(ts);
                }
                _ => return Err(format!("unsupported op: {}", op)),
            }
        }
    }

    Ok((plan_tokens, ops_set, simd_scratch_max))
}
