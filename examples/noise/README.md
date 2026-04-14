# `example-noise` memory usage

## Runtime RAM (stack + static)

This example prints memory info on boot and after inference:
- `static`: `.data + .bss` bytes (always resident in RAM)
- `stack_*`: current/max stack usage (requires `edgedl` feature `stack-probe`)

## Build-time RAM/Flash from the ELF

After building, inspect section sizes:

```sh
xtensa-esp32s3-elf-size -A target/xtensa-esp32s3-none-elf/release/example-noise
```

Quick rule of thumb:
- RAM (static): `.data` + `.bss` (+ any `.noinit`)
- Flash/mapped: `.text` + `.rodata`

