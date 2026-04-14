//! Scalar Linear (Gemm/Dense) kernel for INT8.
//!
//! Contract
//! - Inputs: A[M,K] flattened from NHWC as M = N×H×W, K = C.
//! - Weights: B[N,K] stored as OHWI with kh=kw=1 in the `Initializer` (ParamLayout::WeightsI8OHWI {
//!   oc: N, kh: 1, kw: 1, ic: K }).
//! - Bias: optional INT32 of length N, broadcast on M.
//! - Output: Y[M,N] reshaped as NHWC [N,H,W,N].
//! - Quantization: accumulate in i32 with exponent e_acc = e_in + e_w (e_w per‑tensor), rescale
//!   bias to e_acc, then requantize to e_out using configured rounding.
//! - Fused activation: Linear or ReLU (post‑requant clamp).

#![allow(unused_variables)]

use crate::{
    arena::{Arena, PlannedArena},
    model::{Activation, ModelSpec, ParamExponents, ParamLayout},
    rounding::{
        RoundingMode,
        derive_shift_and_scale,
        requant_i32_to_i8,
        saturate_i32,
        shift_round_i32,
    },
};

#[inline]
fn read_i32_le(bytes: &[u8], idx: usize) -> i32 {
    let base = idx * 4;
    let arr = [
        bytes[base],
        bytes[base + 1],
        bytes[base + 2],
        bytes[base + 3],
    ];
    i32::from_le_bytes(arr)
}

/// Execute one Linear (Gemm) node in scalar mode.
pub fn linear_scalar(
    plan: &PlannedArena,
    arena: &mut Arena<'_>,
    spec: &ModelSpec,
    node: &crate::model::LinearSpec,
    round: RoundingMode,
) {
    // Resolve input/output metadata and arena regions
    let in_meta = spec.values[node.input as usize];
    let out_meta = spec.values[node.output as usize];

    // Shapes (NHWC)
    let n = in_meta.shape.n as usize;
    let ih = in_meta.shape.h as usize;
    let iw = in_meta.shape.w as usize;
    let k = in_meta.shape.c as usize; // K

    let on = out_meta.shape.n as usize;
    let oh = out_meta.shape.h as usize;
    let ow = out_meta.shape.w as usize;
    let nout = out_meta.shape.c as usize; // N (output features)

    debug_assert_eq!(n, on, "batch must be preserved");
    debug_assert_eq!(ih, oh, "height must be preserved");
    debug_assert_eq!(iw, ow, "width must be preserved");

    // Resolve weights initializer and interpret as OHWI with kh=kw=1
    let w_init = spec
        .initializers
        .iter()
        .find(|ini| ini.id == node.weights)
        .expect("Linear weights initializer missing");
    let (oc, kh, kw, ic, blocked_n16) = match w_init.layout {
        ParamLayout::WeightsI8OHWI { oc, kh, kw, ic } => {
            (oc as usize, kh as usize, kw as usize, ic as usize, false)
        }
        ParamLayout::WeightsI8BlockedN16HWC16 { oc, kh, kw, ic } => {
            (oc as usize, kh as usize, kw as usize, ic as usize, true)
        }
        _ => panic!("Linear weights must be INT8 OHWI or BlockedN16HWC16 layout"),
    };
    debug_assert_eq!(kh, 1, "Linear expects kh=1");
    debug_assert_eq!(kw, 1, "Linear expects kw=1");
    debug_assert_eq!(ic, k, "Weights IC must equal input K");
    debug_assert_eq!(oc, nout, "Weights OC must equal output C (N_out)");

    // Exponents
    let e_in = in_meta.exp;
    let e_out = out_meta.exp;
    let e_w = match w_init.exponents {
        ParamExponents::PerTensor(e) => e,
        ParamExponents::PerChannel(_) => {
            panic!("Linear weights: only PerTensor exponent supported in M1")
        }
    };
    let e_acc = e_in.saturating_add(e_w);

    // Optional bias
    let (bias_bytes_opt, e_b_opt, bias_len) = match node.bias {
        Some(bid) => {
            let b = spec
                .initializers
                .iter()
                .find(|ini| ini.id == bid)
                .expect("Linear bias initializer missing");
            let len = match b.layout {
                ParamLayout::BiasI32 { oc } => oc as usize,
                _ => 0,
            };
            if len != nout {
                panic!("Linear bias length {} != N_out {}", len, nout);
            }
            // esp-dl stores INT32 bias in accumulator domain for quantized Gemm/Conv.
            // When exponents are present but equal to 0, treat as "already in accumulator domain".
            let e_b = match b.exponents {
                ParamExponents::PerTensor(e) => Some(e),
                ParamExponents::PerChannel(_) => None,
            };
            (Some(b.data), e_b, len)
        }
        None => (None, None, 0usize),
    };

    // Precompute bias rescale to accumulator domain if present
    // bias_acc = bias_raw scaled from exponent e_b into accumulator exponent e_acc.
    // If e_b > e_acc (coarser scale), left shift; if e_b < e_acc (finer scale), right shift with
    // rounding.
    let rescale_bias = |raw: i32| -> i32 {
        if let Some(e_b) = e_b_opt {
            // Treat e_b == 0 as "already in accumulator domain" to match esp-dl
            if e_b == 0 {
                return raw;
            }
            let diff = (e_b as i32) - (e_acc as i32);
            if diff == 0 {
                raw
            } else if diff > 0 {
                let sh = diff as u32;
                saturate_i32((raw as i64) << sh)
            } else {
                let sh = (-diff) as i32;
                shift_round_i32(raw, sh, round)
            }
        } else {
            raw
        }
    };

    // Requantization from accumulator to output exponent
    let (out_shift, out_scale) = derive_shift_and_scale(e_acc, e_out);

    // Arena views
    let in_off = plan.offset_of(node.input);
    let out_off = plan.offset_of(node.output);
    let (input, output) = arena.io_slices(
        in_off,
        in_meta.shape.elements(),
        out_off,
        out_meta.shape.elements(),
    );

    let w = w_init.data; // raw INT8 bytes
    // For OHWI with kh=kw=1: per-OC contiguous K. For BlockedN16HWC16: tiles of 16 OC, then
    // IC-major.
    let k_block = k;

    // Main loops: over M (flattened NHW across batch) and N_out
    let m = n * ih * iw;
    for m_idx in 0..m {
        let a_base = m_idx * k;
        for oc_idx in 0..nout {
            // Dot product over K
            let mut acc: i32 = 0;
            if !blocked_n16 {
                let w_base = oc_idx * k_block; // oc-major contiguous K
                for j in 0..k {
                    let a = input[a_base + j] as i8 as i32;
                    let wij = w[w_base + j] as i8 as i32;
                    acc += a * wij;
                }
            } else {
                // Blocked N/16 HWC16 layout with kh=kw=1
                let tile = oc_idx / 16;
                let inner = oc_idx % 16;
                let w_base = (tile * k * 16) as usize;
                for j in 0..k {
                    let a = input[a_base + j] as i8 as i32;
                    let wij = w[w_base + j * 16 + inner] as i8 as i32;
                    acc += a * wij;
                }
            }
            // Bias add (in accumulator domain)
            if let Some(bb) = bias_bytes_opt {
                let raw = read_i32_le(bb, oc_idx);
                acc = acc.saturating_add(rescale_bias(raw));
            }
            // Requantize to INT8 with configured rounding
            let mut y = requant_i32_to_i8(acc, out_shift, out_scale, round);
            // Fused activation
            if let Activation::ReLU = node.activation {
                if y < 0 {
                    y = 0;
                }
            }
            output[m_idx * nout + oc_idx] = y;
        }
    }

    #[cfg(feature = "trace")]
    {
        let out_view = arena.value_slice(out_off, out_meta.shape.elements());
        crate::trace::inspect::log_value_i8("Linear", node.output, out_meta, out_view);
    }
}
