use std::collections::BTreeMap;

use crate::{
    dl_generated::dl,
    util::{i64_to_i8_checked, i64_to_u16_checked},
};

#[derive(Clone, Copy, Debug)]
pub struct ValMeta {
    pub n: u16,
    pub h: u16,
    pub w: u16,
    pub c: u16,
    pub exp: Option<i8>,
}

#[derive(Clone, Copy, Debug)]
pub struct InitInfo {
    pub dtype: dl::TensorDataType,
    pub dims4: Option<(u16, u16, u16, u16)>,
    pub exps_len: usize,
}

/// Build both maps: name->id and id->name (ordered by assigned id).
pub fn build_name_maps(graph: dl::Graph) -> (BTreeMap<String, u16>, Vec<String>) {
    let mut name_to_id: BTreeMap<String, u16> = BTreeMap::new();
    let mut name_by_id: Vec<String> = Vec::new();
    let mut push_name = |s: &str| {
        if !name_to_id.contains_key(s) {
            let id = name_by_id.len() as u16;
            name_to_id.insert(s.to_string(), id);
            name_by_id.push(s.to_string());
        }
    };
    if let Some(v) = graph.input() {
        for i in 0..v.len() {
            if let Some(n) = v.get(i).name() {
                push_name(n);
            }
        }
    }
    if let Some(v) = graph.output() {
        for i in 0..v.len() {
            if let Some(n) = v.get(i).name() {
                push_name(n);
            }
        }
    }
    if let Some(v) = graph.value_info() {
        for i in 0..v.len() {
            if let Some(n) = v.get(i).name() {
                push_name(n);
            }
        }
    }
    if let Some(v) = graph.initializer() {
        for i in 0..v.len() {
            if let Some(n) = v.get(i).name() {
                push_name(n);
            }
        }
    }
    (name_to_id, name_by_id)
}

/// Seed value metadata (NHWC + exp) from value_info tables.
pub fn seed_val_meta_from_value_info(
    graph: dl::Graph,
) -> Result<BTreeMap<String, ValMeta>, String> {
    let mut val_meta_map: BTreeMap<String, ValMeta> = BTreeMap::new();
    if let Some(v) = graph.value_info() {
        for i in 0..v.len() {
            let vi = v.get(i);
            if let Some(name) = vi.name() {
                let mut dims_i64: Vec<i64> = Vec::new();
                if let Some(tp) = vi.value_info_type() {
                    if let Some(tensor) = tp.value_as_tensor_type() {
                        if let Some(shape) = tensor.shape() {
                            if let Some(dims) = shape.dim() {
                                for d in 0..dims.len() {
                                    let dd = dims.get(d);
                                    if let Some(val) = dd.value() {
                                        dims_i64.push(val.dim_value());
                                    }
                                }
                            }
                        }
                    }
                }
                let (n, h, w, c) = match dims_i64.len() {
                    4 => (
                        i64_to_u16_checked(dims_i64[0], "N")?,
                        i64_to_u16_checked(dims_i64[1], "H")?,
                        i64_to_u16_checked(dims_i64[2], "W")?,
                        i64_to_u16_checked(dims_i64[3], "C")?,
                    ),
                    3 => (
                        1,
                        i64_to_u16_checked(dims_i64[0], "H")?,
                        i64_to_u16_checked(dims_i64[1], "W")?,
                        i64_to_u16_checked(dims_i64[2], "C")?,
                    ),
                    2 => (
                        // Expand [N, C] to NHWC as [N, 1, 1, C]
                        i64_to_u16_checked(dims_i64[0], "N")?,
                        1,
                        1,
                        i64_to_u16_checked(dims_i64[1], "C")?,
                    ),
                    1 => (
                        // Expand [C] to NHWC as [1, 1, 1, C]
                        1,
                        1,
                        1,
                        i64_to_u16_checked(dims_i64[0], "C")?,
                    ),
                    _ => (1, 0, 0, 0),
                };
                let mut exp: Option<i8> = None;
                if let Some(exps) = vi.exponents() {
                    if !exps.is_empty() {
                        exp = Some(i64_to_i8_checked(exps.get(0))?);
                    }
                }
                val_meta_map.insert(name.to_string(), ValMeta { n, h, w, c, exp });
            }
        }
    }
    Ok(val_meta_map)
}
