//! Structured tensor dump over RTT up-channel 1.
//!
//! Wire format lives in the `crosscheck-proto` crate — this file only constructs a
//! `FrameHeader` from `edgedl` types and pumps the bytes into the RTT up-channel.

use crosscheck_proto::{
    FrameHeader, OP_CONV2D, OP_LINEAR, OP_PAD, OP_REDUCE_MEAN, OP_RELU, write_frame,
};
use edgedl::model::{NodeOp, NodeSpec, ValueMeta};
use rtt_target::UpChannel;

// Single-threaded access: embedded-test runs test bodies sequentially on one core,
// and we never call the writer from an interrupt. Matches rtt-target's own defmt
// module pattern (see rtt-target/src/defmt.rs).
static mut TENSOR_CHANNEL: Option<UpChannel> = None;

pub(crate) fn set_channel(ch: UpChannel) {
    unsafe {
        *core::ptr::addr_of_mut!(TENSOR_CHANNEL) = Some(ch);
    }
}

fn op_type_byte(op: &NodeOp) -> u8 {
    match op {
        NodeOp::Conv2d(_) => OP_CONV2D,
        NodeOp::Pad(_) => OP_PAD,
        NodeOp::ReduceMean(_) => OP_REDUCE_MEAN,
        NodeOp::Linear(_) => OP_LINEAR,
        NodeOp::ReLU(_) => OP_RELU,
    }
}

/// `edgedl::engine::NodeHook` implementation that writes each node's output
/// tensor to RTT channel 1 using the structured frame format above. The
/// `run_tag` distinguishes multiple inference runs (e.g. simd_off vs simd_on)
/// within a single test so the host tool can group frames by run.
pub struct ProbeHook {
    run_tag: &'static str,
}

impl ProbeHook {
    pub fn new(run_tag: &'static str) -> Self {
        Self { run_tag }
    }
}

impl edgedl::engine::NodeHook for ProbeHook {
    fn after_node(
        &mut self,
        idx: usize,
        node: &NodeSpec,
        meta: &ValueMeta,
        output_bytes: &[i8],
    ) {
        let data: &[u8] = unsafe {
            core::slice::from_raw_parts(output_bytes.as_ptr() as *const u8, output_bytes.len())
        };
        write_tensor_frame(self.run_tag, idx as u16, node, meta, data);
    }
}

/// Emit a single structured frame to the tensor RTT channel. No-op if the
/// channel hasn't been installed via [`set_channel`].
pub fn write_tensor_frame(
    run_tag: &str,
    step: u16,
    node: &NodeSpec,
    meta: &ValueMeta,
    data: &[u8],
) {
    let header = FrameHeader {
        step,
        value_id: node.op.output(),
        op_type: op_type_byte(&node.op),
        exp: meta.exp,
        shape_n: meta.shape.n,
        shape_h: meta.shape.h,
        shape_w: meta.shape.w,
        shape_c: meta.shape.c,
    };

    unsafe {
        let ch_ptr = core::ptr::addr_of_mut!(TENSOR_CHANNEL);
        if let Some(ch) = (*ch_ptr).as_mut() {
            write_frame(run_tag.as_bytes(), &header, data, |chunk| {
                ch.write(chunk);
            });
        }
    }
}
