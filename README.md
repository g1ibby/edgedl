# edgedl

A Rust-based, `no_std`-capable INT8 neural network inference engine for ESP32-S3 microcontrollers.

## Overview

edgedl runs `.espdl` model files exported from [esp-dl](https://github.com/espressif/esp-dl). Models are parsed at compile-time via proc-macro and embedded directly into flash. No runtime parsing or dynamic allocation required.

## Features

- **INT8 only** - Symmetric quantization with per-tensor or per-channel scales
- **Compile-time model embedding** - `.espdl` files parsed by proc-macro, weights stored in flash
- **Dual execution paths** - Scalar (portable Rust) and SIMD (TIE728 assembly)
- **Deterministic memory** - Arena allocator for intermediate tensors
- **`no_std` compatible** - Runs on bare-metal without heap allocator

## Target Hardware

| Target | Status |
|--------|--------|
| ESP32-S3 (Xtensa TIE728) | Supported |
| ESP32-P4 | Planned |

## Getting Started

### Desktop Development (Scalar Mode)

```bash
# Run tests
cargo test

# Check compilation
cargo check
```

### ESP32-S3 (SIMD Mode)

Requires ESP32-S3 connected via USB-Serial-JTAG and `probe-rs` installed.

```bash
# Install probe-rs
cargo install probe-rs-tools

# Run example on device
cd examples/noise
source ~/export-esp.sh && cargo run --release --features "trace,defmt"

# Run hardware-in-loop tests (fast path: on-device assertions only)
source ~/export-esp.sh && cargo xtask run tests esp32s3 --test noise_model_integration
```

### HIL testing: two modes

The same HIL binary can be driven two ways:

**`run tests`** — fast. Flashes via `probe-rs run`, embedded-test asserts on device, pass/fail propagates through semihosting. Use for CI and smoke tests.

```bash
source ~/export-esp.sh && cargo xtask run tests esp32s3 --test noise_model_integration
```

**`run full-tests`** — full-fidelity. Same device test, but xtask also captures every intermediate tensor from RTT channel 1 to a temp directory, then runs the sibling host crosscheck (`tests/<name>_crosscheck.rs`) which re-computes the same inference with scalar kernels on the host and compares byte-by-byte. Use when debugging SIMD kernel divergences.

```bash
source ~/export-esp.sh && cargo xtask run full-tests esp32s3 --test noise_model_integration
```

On success xtask prints the dump directory path. On failure (device test, probe driver, or host diff) it prints the path too so you can inspect the frames. Requires `espflash` on PATH.

Adding a host diff for a new HIL test:

1. Create `hil-test/src/bin/<name>.rs` with the device test. Do **not** put `crosscheck` in `//% FEATURES` — `run full-tests` injects it automatically. Leaving it out keeps `run tests` clean (channel 1 stays closed, terminal stays readable).
2. Create `tests/<name>_crosscheck.rs` as a `#[cfg(feature = "crosscheck")]` + `#[ignore]`'d `#[test]`. Read `EDGEDL_DUMP_DIR`, do whatever analysis the test needs — full host inference via `predict_scalar_hooked`, or just shape/checksum assertions on the dumped files.

If no sibling `_crosscheck.rs` exists, `run full-tests` still flashes and captures the dump, and prints where it went.

## Supported Operations

### Layers

| Operation | Scalar | SIMD (ESP32-S3) | Notes |
|-----------|:------:|:---------------:|-------|
| Conv2D | Yes | Yes | 1x1, 3x3 optimized; arbitrary kernel sizes supported |
| Depthwise Conv2D | Yes | - | Scalar only |
| Linear (Dense) | Yes | - | Scalar only |
| Pad | Yes | - | Scalar only |
| ReduceMean | Yes | - | Scalar only |
| ReLU | Yes | - | Standalone scalar; fused in SIMD Conv2D |

### Conv2D SIMD Kernels

| Kernel | Aligned | Unaligned | Description |
|--------|:-------:|:---------:|-------------|
| K11 | Yes | Yes | Optimized 1x1 convolution |
| K33 | Yes | Yes | Optimized 3x3 convolution |
| HWCN | Yes | Yes | Generic kernel (5x5, 7x7, arbitrary) |

### Quantization

| Feature | Supported |
|---------|:---------:|
| INT8 per-tensor scale | Yes |
| INT8 per-channel scale | Yes |
| INT32 bias | Yes |

Rounding: HALF_UP by default (matches esp-dl). HALF_EVEN available via feature flag.

### SIMD Constraints

- Input/output channels must be multiples of 16
- Batch size = 1
- 16-byte buffer alignment preferred (unaligned fallback available)

## Project Structure

```
edgedl/
├── src/           # Runtime crate (no_std, no FlatBuffers)
├── macros/        # Proc-macro crate (#[espdl_model] attribute)
├── hil-test/      # Hardware-in-loop tests for ESP32-S3
├── examples/      # Example firmware
├── xtask/         # Build automation
```

## Cargo Features

| Feature | Description |
|---------|-------------|
| `simd-s3` | Enable SIMD kernels for ESP32-S3 |
| `trace` | Instrumentation logging (defmt on device, log on desktop) |
| `stack-probe` | Stack usage probing on Xtensa |
| `round-half-even` | Use HALF_EVEN rounding (default is HALF_UP) |
| `std` | Enable std-only helpers |

## Development

The project uses `cargo-xtask` for build automation. Run commands from the workspace root.

### Common Commands

```bash
# Format all packages (requires nightly)
cargo xtask fmt-packages

# Lint all packages
cargo xtask lint-packages

# Type check for specific chip
cargo xtask check-packages --chips esp32s3

# Run host tests
cargo xtask host-tests

# Run full CI pipeline
cargo xtask ci esp32s3

# Clean build artifacts
cargo xtask clean
```

### Requirements

- Rust stable toolchain (desktop)
- Rust nightly toolchain (formatting)
- ESP Rust toolchain (ESP32-S3 targets)
- `probe-rs` (hardware testing)

## Limitations

- INT8 models only (no INT16 or FP32)
- Batch size = 1 for SIMD paths
- Channel count must be multiple of 16 for SIMD
- No LSTM/GRU, Softmax, Sigmoid, Resize operators

## Roadmap

### Phase 1: Core Completeness
- Depthwise Conv2D SIMD (MobileNet/EfficientNet support)
- MaxPool2D / AvgPool2D SIMD
- Linear SIMD

### Phase 2: Additional Features
- Element-wise Add SIMD (residual connections)
- GlobalAvgPool
- ESP32-P4 support

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Related Projects

- [esp-dl](https://github.com/espressif/esp-dl) - Espressif's C++ deep learning library for ESP32
