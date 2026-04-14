//! Engine: binds a ModelSpec to an Arena and drives execution.
//!
//! M1 Scope
//! - Full‑residency arena planning without inplace.
//! - Scalar execution entrypoint that iterates the plan and calls scalar kernels.
//! - Parameters are looked up from `spec.initializers` as needed per node.
//! - Kernels are stubs in M1 scaffolding; they will be filled iteratively.

use crate::{
    arena::{Arena, PlannedArena},
    kernels::{
        conv2d::conv2d_scalar,
        linear::linear_scalar,
        pad::pad_scalar,
        reduce_mean::reduce_mean_scalar,
    },
    model::{ModelSpec, NodeOp, ValueId},
    rounding::DEFAULT_ROUNDING,
};

pub struct Engine<'m, 'p, 'a> {
    spec: &'m ModelSpec,
    plan: &'p PlannedArena,
    arena: Arena<'a>,
}

impl<'m, 'p, 'a> Engine<'m, 'p, 'a> {
    /// Construct an Engine over a `ModelSpec` using a preallocated arena buffer and a preplanned
    /// arena.
    pub fn from_parts(spec: &'m ModelSpec, plan: &'p PlannedArena, buf: &'a mut [i8]) -> Self {
        let need = plan.size_bytes + plan.scratch_bytes;
        assert!(
            buf.len() >= need,
            "arena buffer too small (need {} bytes incl. scratch, got {})",
            need,
            buf.len()
        );
        let arena = Arena::from_buf(buf);
        Self { spec, plan, arena }
    }

    /// Execute the model in scalar mode. In a future revision, inputs/outputs will
    /// be passed/returned via typed views; M1 focuses on the execution loop skeleton.
    pub fn infer_scalar(&mut self) {
        let round = DEFAULT_ROUNDING;
        for node in self.spec.plan.iter() {
            match &node.op {
                NodeOp::Conv2d(conv) => {
                    conv2d_scalar(self.plan, &mut self.arena, self.spec, conv, round)
                }
                NodeOp::Pad(p) => pad_scalar(self.plan, &mut self.arena, self.spec, p),
                NodeOp::ReduceMean(rm) => {
                    reduce_mean_scalar(self.plan, &mut self.arena, self.spec, rm, round)
                }
                NodeOp::Linear(l) => linear_scalar(self.plan, &mut self.arena, self.spec, l, round),
                NodeOp::ReLU(r) => {
                    crate::kernels::relu::relu_scalar(self.plan, &mut self.arena, self.spec, r)
                }
            }
        }
    }

    /// Execute the model, preferring SIMD kernels when available and
    /// supported, with automatic fallback to scalar.
    pub fn infer_simd(&mut self) {
        let round = DEFAULT_ROUNDING;
        for node in self.spec.plan.iter() {
            match &node.op {
                NodeOp::Conv2d(conv) => {
                    // Prefer SIMD on supported targets when the feature is enabled; otherwise
                    // run the scalar kernel directly.
                    #[cfg(feature = "trace")]
                    let _start_cycles = crate::trace::read_ccount();

                    #[cfg(all(feature = "simd-s3", target_arch = "xtensa"))]
                    {
                        let outcome = crate::kernels::simd::conv2d::try_conv2d(
                            self.plan,
                            &mut self.arena,
                            self.spec,
                            conv,
                            round,
                        );
                        if matches!(outcome, crate::kernels::simd::conv2d::ExecOutcome::Fallback) {
                            #[cfg(feature = "trace")]
                            crate::ne_warn!("Conv2D id={} FALLBACK to scalar", conv.output);
                            conv2d_scalar(self.plan, &mut self.arena, self.spec, conv, round)
                        }
                    }
                    #[cfg(not(all(feature = "simd-s3", target_arch = "xtensa")))]
                    {
                        conv2d_scalar(self.plan, &mut self.arena, self.spec, conv, round)
                    }

                    #[cfg(feature = "trace")]
                    {
                        let elapsed = crate::trace::read_ccount().wrapping_sub(_start_cycles);
                        // ESP32-S3 @ 240MHz: 1 cycle = 1/240_000 ms
                        let us_total = elapsed / 240;
                        let ms = us_total / 1000;
                        let us_frac = us_total % 1000;
                        crate::ne_info!(
                            "Conv2D id={} total: {}cy ({}.{:03}ms)",
                            conv.output,
                            elapsed,
                            ms,
                            us_frac
                        );
                    }
                }
                NodeOp::Pad(p) => pad_scalar(self.plan, &mut self.arena, self.spec, p),
                NodeOp::ReduceMean(rm) => {
                    reduce_mean_scalar(self.plan, &mut self.arena, self.spec, rm, round)
                }
                NodeOp::Linear(l) => linear_scalar(self.plan, &mut self.arena, self.spec, l, round),
                NodeOp::ReLU(r) => {
                    crate::kernels::relu::relu_scalar(self.plan, &mut self.arena, self.spec, r)
                }
            }
        }
    }

    /// Write bytes into the arena region for a given input `ValueId`.
    ///
    /// Panics if `id` is not a declared model input, if `id` is out of range,
    /// or if `data.len()` does not match the input tensor's element count.
    pub fn write_input_id(&mut self, id: ValueId, data: &[i8]) {
        // Ensure id is one of the declared inputs
        let is_input = self.spec.inputs.iter().any(|&v| v == id);
        assert!(
            is_input,
            "write_input_id: value {} is not a model input",
            id
        );

        let vidx = id as usize;
        assert!(
            vidx < self.spec.values.len(),
            "write_input_id: id out of range"
        );
        let meta = self.spec.values[vidx];
        let elems = meta.shape.elements();
        assert!(
            data.len() == elems,
            "write_input_id: length mismatch (got {}, expected {})",
            data.len(),
            elems
        );

        let off = self.plan.offset_of(id);
        let dst = self.arena.value_slice_mut(off, elems);
        dst.copy_from_slice(data);
    }

    /// Read a view of the arena region for a given output `ValueId`.
    ///
    /// Panics if `id` is not a declared model output or is out of range.
    pub fn read_output_id(&self, id: ValueId) -> &[i8] {
        let is_output = self.spec.outputs.iter().any(|&v| v == id);
        assert!(
            is_output,
            "read_output_id: value {} is not a model output",
            id
        );

        let vidx = id as usize;
        assert!(
            vidx < self.spec.values.len(),
            "read_output_id: id out of range"
        );
        let meta = self.spec.values[vidx];
        let elems = meta.shape.elements();

        let off = self.plan.offset_of(id);
        self.arena.value_slice(off, elems)
    }

    /// Write input by input index (into `spec.inputs`).
    ///
    /// Panics if `index` is out of range or length mismatches.
    pub fn write_input_index(&mut self, index: usize, data: &[i8]) {
        assert!(
            index < self.spec.inputs.len(),
            "write_input_index: index out of range"
        );
        let id = self.spec.inputs[index];
        self.write_input_id(id, data);
    }

    /// Read output by output index (into `spec.outputs`).
    ///
    /// Panics if `index` is out of range.
    pub fn read_output_index(&self, index: usize) -> &[i8] {
        assert!(
            index < self.spec.outputs.len(),
            "read_output_index: index out of range"
        );
        let id = self.spec.outputs[index];
        self.read_output_id(id)
    }
}
