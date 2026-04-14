//! Arena planning and storage (zero-alloc, no_std).
//!
//! Design
//! - We keep a single, contiguous allocation for all intermediate activations ("arena").
//! - M1 uses a simple full‑residency plan without inplace aliasing: each value gets a distinct,
//!   aligned offset. This is easy to reason about and aligns with esp‑dl’s greedy planner behavior
//!   when inplace is disabled.
//! - We avoid requiring the underlying allocation to be aligned; instead we over‑allocate and
//!   select an aligned base inside the buffer. All value offsets are relative to this aligned base.
//! - Parameters (weights, bias) are not placed in the arena; they live as statics via `Initializer`
//!   and are fetched by id.

#![allow(dead_code)]

use crate::model::ValueId;

/// Planned arena: non‑overlapping offsets for each value.
pub struct PlannedArena {
    /// Required arena size in bytes (activations only; no scratch).
    pub size_bytes: usize,
    /// Per‑value offset (in bytes) from the base of the arena buffer.
    pub offsets: &'static [usize],
    /// Scratch capacity in bytes reserved after `size_bytes`.
    /// Used by SIMD kernels for per‑channel tables, temporary tiles, etc.
    pub scratch_bytes: usize,
}

impl PlannedArena {
    #[inline]
    pub fn offset_of(&self, id: ValueId) -> usize {
        self.offsets[id as usize]
    }

    #[inline]
    pub fn scratch_capacity(&self) -> usize {
        self.scratch_bytes
    }
}

/// Backing storage for intermediate activations: borrowed mutable slice.
pub struct Arena<'a> {
    buf: &'a mut [i8],
}

impl<'a> Arena<'a> {
    /// Wrap an external buffer. Length must be at least `plan.size_bytes`.
    pub fn from_buf(buf: &'a mut [i8]) -> Self {
        Self { buf }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.buf.len()
    }

    /// Get a mutable slice view for a value region with a given element count.
    /// Panics if out of bounds.
    pub fn value_slice_mut(&mut self, off_bytes: usize, elem_count: usize) -> &mut [i8] {
        let start = off_bytes;
        let end = start + elem_count;
        assert!(end <= self.buf.len(), "Arena slice out of range");
        &mut self.buf[start..end]
    }

    /// Get an immutable slice view for a value region with a given element count.
    pub fn value_slice(&self, off_bytes: usize, elem_count: usize) -> &[i8] {
        let start = off_bytes;
        let end = start + elem_count;
        assert!(end <= self.buf.len(), "Arena slice out of range");
        &self.buf[start..end]
    }

    /// Borrow input (immutable) and output (mutable) non-overlapping regions together.
    /// Panics if regions overlap or are out of bounds.
    pub fn io_slices(
        &mut self,
        in_off: usize,
        in_len: usize,
        out_off: usize,
        out_len: usize,
    ) -> (&[i8], &mut [i8]) {
        let in_end = in_off + in_len;
        let out_end = out_off + out_len;
        assert!(
            in_end <= self.buf.len() && out_end <= self.buf.len(),
            "Arena slice out of range"
        );
        // Ensure non-overlap
        let non_overlap = in_end <= out_off || out_end <= in_off;
        assert!(non_overlap, "Arena io_slices: regions must not overlap");

        if in_off <= out_off {
            let (head, tail) = self.buf.split_at_mut(out_off);
            let input = &head[in_off..in_off + in_len];
            let output = &mut tail[..out_len];
            (input, output)
        } else {
            let (head, tail) = self.buf.split_at_mut(in_off);
            let output = &mut head[out_off..out_off + out_len];
            let input = &tail[..in_len];
            (input, output)
        }
    }

    /// Borrow a mutable scratch region from the tail of the arena buffer.
    /// The range starts at `plan.size_bytes` and must fit within `plan.scratch_bytes`.
    pub fn scratch_slice_mut(&mut self, plan: &PlannedArena, bytes: usize) -> &mut [i8] {
        let start = plan.size_bytes;
        let end = start + bytes;
        assert!(
            bytes <= plan.scratch_bytes,
            "scratch request exceeds reserved capacity"
        );
        assert!(end <= self.buf.len(), "Arena scratch out of range");
        &mut self.buf[start..end]
    }
}
