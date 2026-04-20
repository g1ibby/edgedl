//! Core runtime types for edgedl (no_std).
//!
//! Purpose
//! - These types define the stable, Rust-first binary surface that the `edgedl-macros` proc-macro
//!   populates from a `.espdl` model at compile time. The runtime owns these types; the macro only
//!   instantiates constant values of them.
//!
//! Scope and Constraints (M1)
//! - INT8 activations and weights, INT32 bias, symmetric power-of-two quantization (scale = 2^e).
//!   Fused activations: Linear and ReLU.
//! - Layouts: NHWC for activations, OHWI for convolution weights.
//! - Single aligned arena (no inplace aliasing in M1); 16-byte alignment.
//! - no_std; only `core` (and `alloc` for the arena implementation).
//!
//! On the use of `u16`
//! - Many fields (dims, counts, indices) use `u16` to shrink the constant model metadata footprint
//!   in flash/rodata and improve cache density on ESP32‑S3. Typical embedded models have dimensions
//!   and channel counts well within 65,535. Keeping these as 16-bit reduces total constant size.
//! - The proc-macro validates at compile time that exported shapes/IDs fit in `u16`; models
//!   exceeding these limits will be rejected with a clear error. If we need larger models later, we
//!   can widen selectively.

/// Identifier for a graph value (tensor) within a ModelSpec.
///
/// - This is an index into `ModelSpec.values` (0..values.len()).
/// - Chosen as `u16` to keep tables compact; the macro ensures bounds.
pub type ValueId = u16;

/// 4D shape for NHWC activations (batch, height, width, channels).
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct Shape4dNHWC {
    /// Batch (N)
    pub n: u16,
    /// Height (H)
    pub h: u16,
    /// Width (W)
    pub w: u16,
    /// Channels (C)
    pub c: u16,
}

impl Shape4dNHWC {
    #[inline]
    pub const fn new(n: u16, h: u16, w: u16, c: u16) -> Self {
        Self { n, h, w, c }
    }
    /// Total element count (N × H × W × C).
    #[inline]
    pub const fn elements(&self) -> usize {
        (self.n as usize) * (self.h as usize) * (self.w as usize) * (self.c as usize)
    }
}

/// Fused activation type applied to op outputs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Activation {
    Linear,
    ReLU,
}

/// Padding mode for Pad.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PaddingMode {
    Constant,
    Edge,
    Reflect,
}

/// Quantization exponent(s) associated with a parameter tensor.
///
/// - PerTensor: a single exponent `e` means scale = 2^e for the entire tensor.
/// - PerChannel: exponents per output channel (for Conv weights), slice length equals OC.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ParamExponents {
    PerTensor(i8),
    PerChannel(&'static [i8]),
}

/// Parameter layout descriptor; tells the runtime what a parameter blob
/// encodes without requiring FlatBuffers at runtime.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ParamLayout {
    /// INT8 convolution weights laid out as [OC, KH, KW, IC] (OHWI).
    /// The raw bytes are interpreted as `i8` with the given dimensions.
    WeightsI8OHWI { oc: u16, kh: u16, kw: u16, ic: u16 },
    /// INT8 convolution weights packed as (N/16)HWC16 (ESP-DL blocked layout):
    /// outer tiles of 16 output channels, then H, W, C, and the inner 16 OC.
    /// Dims are provided as HWIO, but bytes are stored in blocked HWC16 order.
    WeightsI8BlockedN16HWC16 { oc: u16, kh: u16, kw: u16, ic: u16 },
    /// INT32 bias, length equals OC.
    BiasI32 { oc: u16 },
}

/// A graph initializer (parameter tensor) embedded in the model.
///
/// - `id` is the value ID this initializer binds to (matches a value in `values`).
/// - `layout` describes how to interpret the raw bytes in `data` at runtime.
/// - `data` is a big-endian-agnostic raw byte view of the parameter payload emitted by the exporter
///   (we interpret as little-endian where needed).
/// - `exponents` are the quant exponents for this parameter (per tensor or per OC).
#[derive(Clone, Copy, Debug)]
pub struct Initializer {
    pub id: ValueId,
    pub layout: ParamLayout,
    pub data: &'static [u8],
    pub exponents: ParamExponents,
}

/// Runtime metadata for a value (tensor) in the graph.
///
/// - `shape` is 4D NHWC for activations. For parameters, the macro emits a separate `Initializer`
///   with its own layout.
/// - `exp` is the activation exponent (scale = 2^exp) for quantized tensors.
#[derive(Clone, Copy, Debug)]
pub struct ValueMeta {
    pub shape: Shape4dNHWC,
    pub exp: i8,
}

/// Static Conv2D descriptor.
///
/// - `input`/`output` are value IDs referring to entries in `values`.
/// - `weights`/`bias` refer to initializer IDs bound to parameter blobs.
/// - `strides_hw` and `dilations_hw` are [H, W].
/// - `pads_hw` is [top, left, bottom, right] (H-before-W ordering).
/// - `activation` is fused post-op activation.
/// - `groups` allows grouped/depthwise conv (groups == input channels for depthwise).
#[derive(Clone, Copy, Debug)]
pub struct Conv2dSpec {
    pub input: ValueId,
    pub weights: ValueId,
    pub bias: Option<ValueId>,
    pub output: ValueId,
    pub strides_hw: [u16; 2],
    pub dilations_hw: [u16; 2],
    /// Pads as [top, left, bottom, right]
    pub pads_hw: [u16; 4],
    pub activation: Activation,
    pub groups: u16,
}

/// Static Linear (Gemm/Dense) descriptor.
///
/// Semantics (M1): Y = A × B + C with alpha=1, beta=1 and B stored as [N, K]
/// (i.e., ONNX Gemm transB=1). Only INT8×INT8→INT32 is supported, with optional
/// INT32 bias and a fused activation (Linear/ReLU). Shapes:
/// - A: [N, H, W, K] (NHWC activation)
/// - B: [N_out, 1, 1, K] encoded as OHWI in `Initializer::WeightsI8OHWI`
/// - C: [N_out] (optional), broadcast across M = N×H×W
/// - Y: [N, H, W, N_out]
#[derive(Clone, Copy, Debug)]
pub struct LinearSpec {
    pub input: ValueId,
    pub weights: ValueId,
    pub bias: Option<ValueId>,
    pub output: ValueId,
    pub activation: Activation,
}

/// Static Pad descriptor.
///
/// - `pads_nhwc` follows ONNX semantics: [n0,h0,w0,c0,n1,h1,w1,c1].
/// - `mode` selects Constant/Edge/Reflect; for Constant the `const_i8` scalar is used when
///   `has_const == true`.
#[derive(Clone, Copy, Debug)]
pub struct PadSpec {
    pub input: ValueId,
    pub output: ValueId,
    /// NHWC pads in ONNX order: [n0,h0,w0,c0,n1,h1,w1,c1]
    pub pads_nhwc: [i32; 8],
    pub mode: PaddingMode,
    pub has_const: bool,
    pub const_i8: i8,
}

/// Static ReduceMean descriptor.
///
/// - `axes_bitmap_nhwc` is a compact 4-bit mask over NHWC axes (bit 0 = N, 1 = H, 2 = W, 3 = C).
///   This avoids storing a variable-length axes vector.
/// - `keepdims` mirrors ONNX keepdims.
#[derive(Clone, Copy, Debug)]
pub struct ReduceMeanSpec {
    pub input: ValueId,
    pub output: ValueId,
    /// Axes bitmap over NHWC (bit 0=N, 1=H, 2=W, 3=C)
    pub axes_bitmap_nhwc: u8,
    pub keepdims: bool,
}

/// A single node in the execution plan.
#[derive(Clone, Copy, Debug)]
pub enum NodeOp {
    Conv2d(Conv2dSpec),
    Pad(PadSpec),
    ReduceMean(ReduceMeanSpec),
    Linear(LinearSpec),
    /// Standalone ReLU activation (y = max(0, x))
    ReLU(ActivationSpec),
}

impl NodeOp {
    /// The `ValueId` this node writes to.
    pub fn output(&self) -> ValueId {
        match self {
            NodeOp::Conv2d(c) => c.output,
            NodeOp::Pad(p) => p.output,
            NodeOp::ReduceMean(rm) => rm.output,
            NodeOp::Linear(l) => l.output,
            NodeOp::ReLU(r) => r.output,
        }
    }

    /// Short op tag — useful for host-side dump filenames.
    pub fn tag(&self) -> &'static str {
        match self {
            NodeOp::Conv2d(_) => "conv2d",
            NodeOp::Pad(_) => "pad",
            NodeOp::ReduceMean(_) => "reduce_mean",
            NodeOp::Linear(_) => "linear",
            NodeOp::ReLU(_) => "relu",
        }
    }
}

/// Standalone activation spec (currently only ReLU)
#[derive(Clone, Copy, Debug)]
pub struct ActivationSpec {
    pub input: ValueId,
    pub output: ValueId,
}

/// Node wrapper; small handle to the operation.
#[derive(Clone, Copy, Debug)]
pub struct NodeSpec {
    pub op: NodeOp,
}

/// The fully materialized model description emitted at compile time.
///
/// - `alignment` requests the arena base address alignment in bytes (typically 16).
/// - `inputs`/`outputs` are slices of value IDs that identify graph I/O.
/// - `values` holds each graph value’s static shape and exponent.
/// - `initializers` binds value IDs to parameter blobs and their layouts.
/// - `plan` is a topologically sorted list of nodes to execute.
#[derive(Clone, Copy, Debug)]
pub struct ModelSpec {
    pub alignment: u8,
    pub inputs: &'static [ValueId],
    pub outputs: &'static [ValueId],
    pub values: &'static [ValueMeta],
    pub initializers: &'static [Initializer],
    pub plan: &'static [NodeSpec],
}
