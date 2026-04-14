use std::collections::{BTreeMap, BTreeSet};

use crate::{
    context::{InitInfo, ValMeta},
    dl_generated::dl,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Block {
    off: usize,
    len: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct ArenaPlan {
    pub offsets: Vec<usize>,
    pub size_bytes: usize,
    pub report: String,
}

#[derive(Clone, Copy, Debug)]
struct LiveRange {
    start: i32,
    end: i32,
}

#[derive(Clone, Debug)]
struct AllocItem<'a> {
    id: usize,
    name: &'a str,
    bytes: usize,
    range: LiveRange,
}

fn align_up(x: usize, align: usize) -> usize {
    debug_assert!(align.is_power_of_two());
    (x + (align - 1)) & !(align - 1)
}

fn insert_free_and_coalesce(free: &mut Vec<Block>, block: Block) {
    if block.len == 0 {
        return;
    }
    free.push(block);
    free.sort_by_key(|b| b.off);
    let mut out: Vec<Block> = Vec::with_capacity(free.len());
    for b in free.drain(..) {
        if let Some(last) = out.last_mut() {
            if last.off + last.len == b.off {
                last.len += b.len;
                continue;
            }
        }
        out.push(b);
    }
    *free = out;
}

fn best_fit_index(free: &[Block], need: usize) -> Option<usize> {
    let mut best: Option<(usize, usize, usize)> = None; // (idx, len, off)
    for (idx, b) in free.iter().enumerate() {
        if b.len < need {
            continue;
        }
        match best {
            None => best = Some((idx, b.len, b.off)),
            Some((_, best_len, best_off)) => {
                if b.len < best_len || (b.len == best_len && b.off < best_off) {
                    best = Some((idx, b.len, b.off));
                }
            }
        }
    }
    best.map(|(idx, _, _)| idx)
}

fn remove_active_expired(active: &mut Vec<(i32, usize, Block)>, start: i32, free: &mut Vec<Block>) {
    // `active` is sorted by (end, id).
    // Drain all entries with end < start in one pass to avoid O(n^2) shifting.
    let expire_count = active.partition_point(|(end, _, _)| *end < start);
    for (_, _, block) in active.drain(0..expire_count) {
        insert_free_and_coalesce(free, block);
    }
}

fn insert_active(active: &mut Vec<(i32, usize, Block)>, entry: (i32, usize, Block)) {
    let key = (entry.0, entry.1);
    let pos = active
        .binary_search_by_key(&key, |(end, id, _)| (*end, *id))
        .unwrap_or_else(|p| p);
    active.insert(pos, entry);
}

#[allow(clippy::type_complexity)]
fn build_io(
    fb_graph: dl::Graph,
    name_to_id: &BTreeMap<String, u16>,
    init_info_map: &BTreeMap<String, InitInfo>,
) -> Result<Vec<(Vec<usize>, Vec<usize>)>, String> {
    let mut out: Vec<(Vec<usize>, Vec<usize>)> = Vec::new();
    let nodes = match fb_graph.node() {
        Some(v) => v,
        None => return Ok(out),
    };
    for ni in 0..nodes.len() {
        let n = nodes.get(ni);
        let mut inputs: Vec<usize> = Vec::new();
        if let Some(v) = n.input() {
            for j in 0..v.len() {
                let nm = v.get(j);
                if nm.is_empty() {
                    continue;
                }
                if init_info_map.contains_key(nm) {
                    continue;
                }
                let id = *name_to_id
                    .get(nm)
                    .ok_or_else(|| format!("node {} input '{}' missing id", ni, nm))?;
                inputs.push(id as usize);
            }
        }
        let mut outputs: Vec<usize> = Vec::new();
        if let Some(v) = n.output() {
            for j in 0..v.len() {
                let nm = v.get(j);
                if nm.is_empty() {
                    continue;
                }
                if init_info_map.contains_key(nm) {
                    continue;
                }
                let id = *name_to_id
                    .get(nm)
                    .ok_or_else(|| format!("node {} output '{}' missing id", ni, nm))?;
                outputs.push(id as usize);
            }
        }
        out.push((inputs, outputs));
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn plan_arena_liveness(
    fb_graph: dl::Graph,
    name_to_id: &BTreeMap<String, u16>,
    name_by_id: &[String],
    val_meta_map: &BTreeMap<String, ValMeta>,
    init_info_map: &BTreeMap<String, InitInfo>,
    inputs_ids: &[u16],
    outputs_ids: &[u16],
    align: usize,
) -> Result<ArenaPlan, String> {
    if !align.is_power_of_two() {
        return Err("arena alignment must be a power of two".to_string());
    }

    let values_len = name_by_id.len();
    let mut offsets: Vec<usize> = vec![0; values_len];
    let mut is_param: Vec<bool> = vec![false; values_len];
    let mut bytes_aligned: Vec<usize> = vec![0; values_len];

    for (id, name) in name_by_id.iter().enumerate() {
        if init_info_map.contains_key(name) {
            is_param[id] = true;
            continue;
        }
        let vm = val_meta_map
            .get(name)
            .ok_or_else(|| format!("missing value metadata for '{}'", name))?;
        let elems = (vm.n as usize) * (vm.h as usize) * (vm.w as usize) * (vm.c as usize);
        bytes_aligned[id] = align_up(elems, align);
    }

    // Compute producer(start) and last use(end) in node-index time.
    let io = build_io(fb_graph, name_to_id, init_info_map)?;
    let plan_len_i32: i32 = io.len().try_into().unwrap_or(i32::MAX);

    let mut start: Vec<Option<i32>> = vec![None; values_len];
    let inputs_set: BTreeSet<usize> = inputs_ids.iter().map(|&v| v as usize).collect();
    for &id in inputs_set.iter() {
        if id < values_len && !is_param[id] {
            start[id] = Some(-1);
        }
    }
    for (node_idx, (_ins, outs)) in io.iter().enumerate() {
        let node_idx_i32: i32 = node_idx.try_into().unwrap_or(i32::MAX);
        for &out_id in outs.iter() {
            if out_id >= values_len || is_param[out_id] {
                continue;
            }
            start[out_id] = Some(node_idx_i32);
        }
    }

    let mut last_use: Vec<i32> = vec![-1; values_len];
    for (node_idx, (ins, _outs)) in io.iter().enumerate() {
        let node_idx_i32: i32 = node_idx.try_into().unwrap_or(i32::MAX);
        for &in_id in ins.iter() {
            if in_id >= values_len || is_param[in_id] {
                continue;
            }
            last_use[in_id] = last_use[in_id].max(node_idx_i32);
        }
    }
    // Pin model outputs through the end of the plan so output views remain valid.
    let pin_end = (plan_len_i32 - 1).max(-1);
    for &out_id_u16 in outputs_ids {
        let out_id = out_id_u16 as usize;
        if out_id < values_len && !is_param[out_id] {
            last_use[out_id] = last_use[out_id].max(pin_end);
        }
    }

    let mut items: Vec<AllocItem<'_>> = Vec::new();
    for (id, name) in name_by_id.iter().enumerate() {
        if is_param[id] {
            offsets[id] = 0;
            continue;
        }
        let Some(start_i) = start[id] else {
            return Err(format!(
                "activation value '{}' (id={}) has no producer and is not a model input",
                name, id
            ));
        };
        let end_i = last_use[id].max(start_i);
        items.push(AllocItem {
            id,
            name,
            bytes: bytes_aligned[id],
            range: LiveRange {
                start: start_i,
                end: end_i,
            },
        });
    }

    items.sort_by_key(|it| (it.range.start, it.id));

    let mut cursor: usize = 0;
    let mut active: Vec<(i32, usize, Block)> = Vec::new(); // (end, id, block) sorted by (end,id)
    let mut free: Vec<Block> = Vec::new(); // sorted by off (coalesced)

    for it in items.iter() {
        remove_active_expired(&mut active, it.range.start, &mut free);

        let need = it.bytes;
        let assigned = if let Some(idx) = best_fit_index(&free, need) {
            let block = free.remove(idx);
            let off = block.off;
            let remain = Block {
                off: off + need,
                len: block.len - need,
            };
            insert_free_and_coalesce(&mut free, remain);
            off
        } else {
            let off = align_up(cursor, align);
            cursor = off + need;
            off
        };

        offsets[it.id] = assigned;
        insert_active(
            &mut active,
            (
                it.range.end,
                it.id,
                Block {
                    off: assigned,
                    len: need,
                },
            ),
        );
    }

    let size_bytes = align_up(cursor, align);

    let mut report = String::new();
    report.push_str("arena(liveness):\n");
    report.push_str(&format!("  values={} plan_len={}\n", values_len, io.len()));
    report.push_str(&format!("  size_bytes={}\n", size_bytes));
    report.push_str("  id,start,end,bytes,off,name\n");
    // Emit in ValueId order for stable diffs.
    let mut by_id = items.clone();
    by_id.sort_by_key(|it| it.id);
    for it in by_id.iter() {
        report.push_str(&format!(
            "  {},{},{},{},{},{}\n",
            it.id, it.range.start, it.range.end, it.bytes, offsets[it.id], it.name
        ));
    }

    Ok(ArenaPlan {
        offsets,
        size_bytes,
        report,
    })
}
