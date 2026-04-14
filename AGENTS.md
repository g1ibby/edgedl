# AGENTS.md

This file provides guidance to coding agents working in this repository.

## Project Overview

`edgedl` is a Rust-based, `no_std`-capable neural network inference engine targeting ESP32-S3. It mirrors `esp-dl` semantics and supports two execution modes:

- Scalar: portable reference path for desktop development and golden output generation
- SIMD: ESP32-S3-optimized path using Xtensa TIE728 DSP instructions

## Build And Test Commands

### Run example on device

```bash
cd examples/noise
source ~/export-esp.sh && cargo run --release --features "trace,defmt"
```

Requires an ESP32-S3 connected via USB-Serial-JTAG.

### Run hardware-in-loop tests

```bash
source ~/export-esp.sh && cargo xtask run tests esp32s3 --test noise_model_integration
source ~/export-esp.sh && cargo xtask run tests esp32s3 --test noise_mel_integration
```

Requires an ESP32-S3 connected via USB-Serial-JTAG.

### Run desktop tests

```bash
cargo test
```

### Linting

```bash
cargo fmt --all
cargo clippy --all-targets -- -D warnings
```

### Build for size analysis

```bash
xtensa-esp32s3-elf-size -A target/xtensa-esp32s3-none-elf/release/noise_mel_integration
```

## Repository Structure

- `src/`: runtime crate; `no_std`; no FlatBuffers dependency
- `macros/`: proc-macro crate; parses `.espdl` at compile time; owns `Dl.fbs` and FlatBuffers dependencies
- `hil-test/`: hardware-in-loop tests using `embedded-test`
- `examples/noise/`: example firmware for noise model inference
- `xtask/`: build automation and test orchestration

## Architecture Notes

### Execution Modes

- Scalar: safe Rust reference implementation; portable across targets; used for desktop development
- SIMD: Xtensa ESP32-S3 kernels using `asm!`; requires 16-byte alignment for the fast path and falls back when shapes or alignment are unsupported

### Type System

- INT8 symmetric quantization with integer exponent `e` where scale = `2^e`
- Accumulator domain is INT32
- Rounding defaults to HALF_UP to match `esp-dl`; `round-half-even` switches to HALF_EVEN
- Activations use NHWC layout
- Weights use OHWI layout
- Tensor storage should remain 16-byte aligned for SIMD

### Core Runtime Types

Core struct layouts live in the runtime crate and must remain the source of truth. The macros crate should instantiate these types with model-specific constants and must not redefine them.

Important types include:

- `TensorI8`
- `WeightsI8`
- `BiasI32`
- `NodeSpec`
- `ModelSpec`
- `Arena`
- `Initializer`
- `ValueMeta`

### Codegen Boundary

The macros crate generates model-specific constants, descriptors, and aliases. It must not introduce alternate struct layouts for runtime-owned types.

## Cargo Features

- `simd-s3`: enable SIMD routing and kernels for ESP32-S3
- `trace`: instrumentation logging; uses `defmt` on `no_std` and `log` on `std`
- `stack-probe`: stack usage probing on Xtensa
- `round-half-even`: use HALF_EVEN rounding instead of HALF_UP
- `arena-report`: emit compile-time arena layout diagnostics
- `std`: enable std-only helpers

## Testing Guidance

- Desktop tests in `tests/` cover tensor math, requantization, shape helpers, and end-to-end scalar inference
- HIL tests in `hil-test/` cover scalar vs SIMD parity and kernel probes on device
- Expected output parity is `|simd - scalar| <= 1` after requantization to INT8

## Important Notes

- Keep `lto = false` in release profiles when working with TIE728 SIMD assembly
- Parameters remain in static storage; arena allocations are for intermediates
- `probe-rs` is used for flashing and HIL workflows
- The device connection expectation is USB-Serial-JTAG on ESP32-S3

## Agent Working Rules

### Think Before Coding

- State assumptions explicitly when they matter
- Surface ambiguity instead of silently picking an interpretation
- Prefer the simpler approach when multiple options exist
- Stop and ask if a key requirement is unclear

### Simplicity First

- Implement only what was requested
- Avoid speculative abstractions, configuration, or extensibility
- Do not add handling for impossible scenarios
- If the solution feels overbuilt, simplify it

### Surgical Changes

- Touch only code required for the task
- Do not refactor unrelated areas
- Match the existing local style
- Remove only the unused code introduced by your own changes
- If unrelated issues are noticed, mention them separately instead of fixing them opportunistically

### Goal-Driven Execution

- Translate each task into verifiable success criteria
- Prefer tests or checks that prove the requested behavior
- For multi-step work, keep a short plan and verify each step
