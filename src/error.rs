#![allow(clippy::all)]

// Lightweight, no_std-friendly error type for public APIs.
// Keep variants minimal and stable; prefer explicit lengths for diagnostics.

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Error {
    NoInputs,
    NoOutputs,
    ArenaTooSmall { expected: usize, got: usize },
    InputLenMismatch { expected: usize, got: usize },
    OutputLenMismatch { expected: usize, got: usize },
}

pub type Result<T> = core::result::Result<T, Error>;
