    .section .iram1, "ax"
    .align 4

    ################################################################################################
    # TIE728 SIMD conv2d (1x1 unaligned, OHWI)
    #
    # Entry ABI mirrors aligned variant:
    # - a2: output_ptr (current pixel)
    # - a3: input_ptr  (current pixel)
    # - a4: &KernelArgs
    #
    # This implementation mirrors esp‑dl’s unaligned 11cn structure with a few
    # intentional differences noted here:
    #  - Avoid integer division in ASM: ESP32‑S3 lacks DIV32. We precompute
    #    `oc_tiles` (oc/16) and `oc_tail` (oc%16) in KernelArgs and load them.
    #  - Maintain a rolling filter_ptr (a11). The inner unaligned C kernel
    #    advances the filter stream across the C dimension, so successive tiles
    #    naturally consume contiguous weight blocks.
    #  - Use a14 as the dedicated GPR scratch in unaligned stores so we do not
    #    clobber a10 (mac_shift) across tiles.
    ################################################################################################

    # Offsets into KernelArgs (repr(C)) — must match common.rs
    .set KA_input,        0
    .set KA_weights,      4
    .set KA_bias,         8
    .set KA_output,      12
    .set KA_ih,          16
    .set KA_iw,          20
    .set KA_ic,          24
    .set KA_oh,          28
    .set KA_ow,          32
    .set KA_oc,          36
    .set KA_kh,          40
    .set KA_kw,          44
    .set KA_stride_h,    48
    .set KA_stride_w,    52
    .set KA_dil_h,       56
    .set KA_dil_w,       60
    .set KA_pad_top,     64
    .set KA_pad_left,    68
    .set KA_in_exp,      72
    .set KA_out_exp,     73
    # padding bytes at 74..76
    .set KA_wexp_ptr,    80
    .set KA_wexp_tensor, 84
    # padding bytes at 85..87
    .set KA_layout,      88
    .set KA_fused,       92
    .set KA_flags,       96
    .set KA_pc_scales,   100
    .set KA_pc_bias16,   104
    .set KA_op_type,     108
    .set KA_pl_bias_i32, 112
    .set KA_c_div_x_1,   116
    .set KA_c_remainder, 120
    .set KA_oc_tiles,    124
    .set KA_oc_tail,     128

    # 11c16 microkernel comes from shared macros include

    .global dl_tie728_s8_unaligned_conv2d_11cn
    .type   dl_tie728_s8_unaligned_conv2d_11cn, @function
dl_tie728_s8_unaligned_conv2d_11cn:
    .align 4
    entry sp, 160
    # a2: output_ptr (this pixel)
    # a3: input_ptr  (this pixel)
    # a4: &KernelArgs

    # Load esp-dl style IC helpers and precomputed OC tiling counters from
    # KernelArgs. We avoid integer division in ASM (ESP32-S3 lacks DIV32).
    # ic is only needed later in the OC tail loop; reload there to keep regs free
    l32i a7,  a4, KA_oc_tiles     # a7 = oc_tiles (full 16-lane tiles)
    l32i a13, a4, KA_oc_tail      # a13 = oc_tail  (1..15 or 0)
    l32i a6,  a4, KA_c_div_x_1    # c_div_x_1 (ic/16 - 1, may be <0 when ic<16)
    l32i a8,  a4, KA_c_remainder  # a8 = c_remainder (ic % 16, esp-dl ArgsType::c_remainder)
    # mac_shift = out_exp - in_exp - w_exp_tensor
    l8ui a10, a4, KA_out_exp
    l8ui a11, a4, KA_in_exp
    l8ui a5,  a4, KA_wexp_tensor
    sext a10, a10, 7
    sext a11, a11, 7
    sext a5,  a5,  7
    sub  a10, a10, a11
    sub  a10, a10, a5

    # Load weights base and set the rolling filter_ptr for the first tile
    l32i a5,  a4, KA_weights      # weights base
    mov  a11, a5                  # a11 = current filter_ptr (advanced by inner macro)
    movi a15, 0                   # a15 = tile_idx (0..oc_tiles-1)

    # Tile loop
    # Tile loop
.L_tile_loop_u:
    beqz a7, 8f
    # Accumulator init: per-layer bias preload or zero; per-channel skips preload

    # Accumulator init: per-layer bias preload or zero; per-channel skips preload
    l32i a12, a4, KA_flags
    # Replace bit-branch with AND+branch to avoid any core/assembler quirks.
    # if (flags & (1<<1)) != 0 -> skip preload
    movi a14, 2
    and  a14, a12, a14
    bnez a14, 1f
    # if ((flags & (1<<0)) == 0) -> zero accumulators
    movi a14, 1
    and  a14, a12, a14
    beqz a14, 0f
    EE.ZERO.QACC                          # Zero QACC before bias preload (20-bit bias doesn't fill 40-bit lanes)
    l32i a14, a4, KA_bias         # bias base (packed per 16 lanes)
    slli a9,  a15, 6              # byte offset = tile_idx * 64
    add  a14, a14, a9
    tie728_s8_conv2d_128b_vector_bias a14
    j 1f
0:
    EE.ZERO.QACC
1:
    # MAC over IC blocks (reload input base per tile).
    # We pass the esp-dl style `c_div_x_1` and `c_remainder` counters into
    # the shared unaligned inner macro so full IC blocks and IC tails follow
    # the same front/back merge and pointer movement as esp-dl.
    mov a14, a3                     # reload input_ptr (this pixel)
    # Inner unaligned C MAC: pass c_div_x_1 and c_remainder from args.
    # Use a9 as the macro scratch (temporary) to keep a10 (mac_shift) intact.
    # Pass the preserved c_remainder in a8 (do not clobber it with other uses).
    tie728_s8_conv2d_1_1_unaligned_c a14, a11, a6, a8, q0, q1, q2, q3, a9, a4

    # Requantization: per-channel vs per-layer
    # (Re-load flags into a12 if needed; keep a8 intact for c_remainder semantics)
    l32i a12, a4, KA_flags
    bbsi a12, 1, .L_pc
.L_pl:
    tie728_s8_vector_round_result q0, a10, a14, q3
    j .L_epilogue
.L_pc:
    # scale ptr = base + tile_idx*32, bias16 ptr similar when HAS_BIAS
    l32i a5,  a4, KA_pc_scales
    slli a9,  a15, 5              # *32 bytes per tile
    add  a5,  a5,  a9
    # Near-branch trampoline for HAS_BIAS to avoid long-range bbsi
    l32i a14, a4, KA_flags
    bbsi a14, 0, 1f               # if HAS_BIAS set → near trampoline
    j 2f                          # else jump to no-bias far block via near j
1:
    j 3f                          # with-bias far block via near j
2:
    # no-bias far block
    tie728_s8_conv2d_per_channel_result q0, q1, a5, a10, q2
    j .L_epilogue
3:
    # with-bias far block
    l32i a14, a4, KA_pc_bias16
    slli a9,  a15, 5              # *32 bytes per tile
    add  a14, a14, a9
    tie728_s8_conv2d_per_channel_with_bias_result q0, q1, a5, a14, a10, q2

.L_epilogue:
    # Fused ReLU if requested
    l32i a14, a4, KA_fused
    addi a14, a14, -1
    bnez a14, 5f
    movi a12, 0
    tie728_s8_conv2d_relu q0, a12, a12
5:
    # Unaligned store selection via pointer low-nibble (no SAR dependency)
    movi a14, 15
    and  a14, a2, a14             # a14 = misalignment (0..15)
    beqz a14, 6f                  # aligned
    movi a9, 8
    beq  a14, a9, 7f              # half-aligned
    # General unaligned store path: spill via GPR stores using a14 scratch
    dl_tie728_s8_unaligned_store0 q0, a2, a14

    addi a7, a7, -1
    addi a15, a15, 1              # tile_idx++ (filter_ptr already advanced inside macro)
    bnez a7, .L_tile_loop_u
    j 8f
6:
    EE.VST.128.IP q0, a2, 16

    addi a7, a7, -1
    addi a15, a15, 1              # tile_idx++ (filter_ptr already advanced inside macro)
    bnez a7, .L_tile_loop_u
    j 8f
7:
    dl_tie728_s8_unaligned_store1 q0, a2

    addi a7, a7, -1
    addi a15, a15, 1              # tile_idx++ (filter_ptr already advanced inside macro)
    bnez a7, .L_tile_loop_u
    j 8f

8:
    # Handle OC remainder (1..15): per-element accumulation and store
    beqz a13, 10f                 # if no tail, done
    # a11 already points at weights + (oc_tiles * ic * 16)
    # Compute per-channel scale/bias base for tail tiles when needed: (oc_tiles * 32)
    l32i a8,  a4, KA_pc_scales    # a8 = scale base (may be null)
    beqz a8, 11f
    slli a9,  a15, 5              # *32 bytes per completed tile (tile_idx == oc_tiles)
    add  a8,  a8,  a9             # a8 = scale base for tail
11:
    l32i a12, a4, KA_pc_bias16
    beqz a12, 12f
    slli a9,  a15, 5              # *32 bytes per completed tile
    add  a12, a12, a9             # a12 = bias16 base for tail
12:
    # a2 points after last full-tile store; write tail bytes here
    # Restore output_ptr into a2 after using a2 above (re-read from args)
    # l32i a2, a4, KA_output
    # Advance a2 by (oc_tiles * 16)
    # slli a14, a15, 4
    # add  a2,  a2,  a14
    # Loop j = 0..oc_tail-1
    mov  a15, a13                 # a15 = remaining tail count
.L_tail_loop:
    beqz a15, 10f
    # Compute filter_ptr for this oc: a14 + (j * ic)
    # Keep j = (oc_tail - a15)
    sub  a6,  a13, a15            # a6 = j_idx
    l32i a9,  a4, KA_ic
    mull a6,  a6,  a9             # j * ic
    add  a6,  a6,  a11            # filter_ptr = base_for_tail + j*ic
    # Accumulate into ACCX over C via the shared unaligned remainder macro.
    # We pass c_div_x_1 and c_remainder from KernelArgs so the inner loop
    # can mirror esp-dl's front/back/tail handling over the C dimension.
    mov  a14, a3                  # a14 = input_ptr for this pixel
    l32i a8,  a4, KA_c_div_x_1    # a8 = c_div_x_1 (ic/16 - 1)
    l32i a9,  a4, KA_c_remainder  # a9 = c_remainder (ic % 16)
    tie728_s8_conv2d_1_1_c_n_remainder a5, a14, a6, a8, a9, a10, a7

    # Convert ACCX to a scalar accumulator in the accumulator exponent
    # domain. For per-layer tails we will add per-layer bias in this domain
    # before applying mac_shift; for per-channel tails we will feed this
    # value into the per-channel i16 bias/scale epilogue.
    movi a14, 0
    EE.SRS.ACCX a5, a14, 0        # a5 = acc

    # Compute tail index j_idx = (oc_tail - remaining), reused below.
    sub  a6,  a13, a15            # a6 = j_idx (0..oc_tail-1)

    # Load op_type to distinguish per-layer/per-channel + bias/no-bias.
    l32i a7,  a4, KA_op_type      # a7 = op_type bitfield

    # Per-layer path (bit0 == 0). Optionally add per-layer bias in the
    # accumulator domain using the raw i32 bias table.
    # Use a near-branch trampoline to reach the far .L_tail_pc_path label.
    bbci a7, 0, 90f               # if per-layer (bit0==0) fall through; else jump
    j .L_tail_pc_path
90:

    # Only add bias when HAS_BIAS (bit1) and a per-layer bias table exists.
    # Near-branch trampoline for far target .L_tail_pl_no_bias
    bbsi a7, 1, 91f               # if HAS_BIAS set, skip the far jump
    j .L_tail_pl_no_bias
91:
    l32i a14, a4, KA_pl_bias_i32
    beqz a14, .L_tail_pl_no_bias
    slli a11, a6, 2               # byte offset = j_idx * 4
    add  a14, a14, a11
    l32i a11, a14, 0
    add  a5,  a5,  a11            # acc += bias_i32[j_idx]
.L_tail_pl_no_bias:
    # mac_shift = out_exp - in_exp - w_exp_tensor
    l8ui a8,  a4, KA_out_exp
    l8ui a9,  a4, KA_in_exp
    l8ui a10, a4, KA_wexp_tensor
    sext a8,  a8,  7
    sext a9,  a9,  7
    sext a10, a10, 7
    sub  a8,  a8, a9
    sub  a8,  a8, a10             # a8 = mac_shift
    # For per-layer path, scale_base/bias_base/j are unused in the epilogue.
    movi a11, 0
    movi a10, 0
    movi a6,  0
    tie728_s8_conv2d_1_1_tail_epilogue a7, a2, a8, a10, a11, a6, a5, a10, a11
    addi a15, a15, -1
    j .L_tail_loop

.L_tail_pc_path:
    # Per-channel path: use per-OC i16 bias/scale tables. Compute j = 2*j_idx
    # (byte offset into the tables) and derive mac_shift as above.
    slli a6,  a6,  1              # j = j_idx * 2
    l8ui a8,  a4, KA_out_exp
    l8ui a9,  a4, KA_in_exp
    l8ui a10, a4, KA_wexp_tensor
    sext a8,  a8,  7
    sext a9,  a9,  7
    sext a10, a10, 7
    sub  a8,  a8, a9
    sub  a8,  a8, a10             # a8 = mac_shift
    # a2: output_ptr, a8: scale_base, a12: bias_base, a6: j, a5: acc,
    # a10/a11: scratch
    tie728_s8_conv2d_1_1_tail_epilogue a7, a2, a8, a8, a12, a6, a5, a10, a11
    addi a15, a15, -1
    j .L_tail_loop

10:
    retw
