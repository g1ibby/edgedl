    .section .iram1, "ax"
    .align 4

    ################################################################################################
    # TIE728 SIMD conv2d (1x1 aligned, OHWI)
    #
    # This implementation mirrors esp‑dl's 11cn microkernel structure while adapting the ABI:
    # - a2: output_ptr (current pixel), a3: input_ptr (current pixel), a4: &KernelArgs
    # - Rust iterates OH×OW and calls this entry; the ASM iterates OC tiles (16‑wide).
    # - Supports per‑layer (mac_shift >= 0) and per‑channel (PER_OC_W_EXP) requantization,
    #   with bias/no‑bias handled in‑kernel; optional ReLU fuse.
    # - Weight layout: OHWI (BlockedN16HWC16 handled in a follow‑up).
    #
    # The inner dot‑product is built from esp‑dl's tie728_s8_conv2d_11c16 macro, provided in
    # the shared include (tie728_shared_macros.S) to avoid duplication across aligned/unaligned.
    # Vector instructions used:
    # - EE.VLD.128.IP: load 16 bytes (advance ptr)
    # - EE.VSMULAS.S8.QACC[.LD.INCP]: s8*s8 accumulate broadcasted lane into QACC
    # - EE.ZERO.QACC: zero accumulators
    # - EE.SRCMB.S8.QACC: extract rounded bytes from accumulators (vector requant helper macro)
    # - EE.VRELU.S8: vector ReLU/leaky ReLU (here used as ReLU with alpha=0)
    # - EE.VST.128.IP: store 16 bytes (advance ptr)

    # Offsets into KernelArgs (repr(C))
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

    ################################################################################################
    # Shared helper macros are included via tie728_shared_macros.S; no local 11c16 macro here.

    # Per‑channel requantization using precomputed fixed‑point scale factors (i16)
    .global dl_tie728_s8_conv2d_11cn
    .type   dl_tie728_s8_conv2d_11cn, @function
dl_tie728_s8_conv2d_11cn:
    .align 4
    entry sp, 128
    # a2: output_ptr (this pixel)
    # a3: input_ptr  (this pixel)
    # a4: &KernelArgs

    # Compute oc_tiles (n_rs3) and c_div_x_1 from args
    l32i a12, a4, KA_oc           # oc
    l32i a9,  a4, KA_ic           # ic
    movi a15, 16
    remu a13, a12, a15            # oc % 16
    bnez a13, 9f
    quou a7,  a12, a15            # a7 = oc_tiles
    srai a6,  a9,  4              # a6 = ic/16
    addi a6,  a6, -1              # c_div_x_1

    # Load per-layer mac_shift = out_exp - in_exp - w_exp_tensor (esp-dl semantics)
    l8ui a10, a4, KA_out_exp
    l8ui a11, a4, KA_in_exp
    l8ui a5,  a4, KA_wexp_tensor
    sext a10, a10, 7
    sext a11, a11, 7
    sext a5,  a5,  7
    sub  a10, a10, a11
    sub  a10, a10, a5

    # Load weights base
    l32i a5,  a4, KA_weights

    # OC tile loop
.L_tile_loop:
    beqz a7, 8f

    # Compute filter_ptr for tile: weights + (tile_idx * ic * 16)
    l32i a12, a4, KA_oc
    movi a15, 16
    quou a12, a12, a15            # oc_tiles
    sub  a13, a12, a7             # tile_idx
    mull a13, a13, a9             # tile_idx * ic
    slli a13, a13, 4              # * 16 bytes
    add  a13, a13, a5             # filter_ptr

    # Accumulator init: per-layer bias preload vs zero
    l32i a15, a4, KA_flags
    bbsi a15, 1, 1f               # if PER_OC_W_EXP set, skip preload
    bbci a15, 0, 0f               # if no HAS_BIAS, zero
    EE.ZERO.QACC                          # Zero QACC before bias preload (20-bit bias doesn't fill 40-bit lanes)
    l32i a14, a4, KA_bias         # bias base
    l32i a12, a4, KA_oc
    movi a5, 16
    quou a12, a12, a5
    sub  a12, a12, a7             # tile_idx
    slli a12, a12, 6              # *64 bytes
    add  a14, a14, a12
    tie728_s8_conv2d_128b_vector_bias a14
    j 1f
0:
    EE.ZERO.QACC
1:
    # MAC over IC blocks (reload input base per tile like esp-dl)
    mov a15, a3
    tie728_s8_conv2d_11c16 q0, a15, q1, q2, a13, a6

    # Requantization selection: branch to a near label to avoid crossing large macros
    l32i a15, a4, KA_flags
    bbsi a15, 1, .L_per_channel   # if PER_OC_W_EXP set → per-channel
.L_per_layer:
    tie728_s8_vector_round_result q0, a10, a11, q3
    j .L_after_requant

.L_per_channel:
    # Compute scale ptr = base + tile_idx*32
    l32i a5,  a4, KA_pc_scales
    l32i a12, a4, KA_oc
    movi a14, 16
    quou a12, a12, a14
    sub  a12, a12, a7
    slli a12, a12, 5              # *32 bytes
    add  a5,  a5,  a12            # scale ptr for tile
    # Bias or no-bias per-channel epilogue (mirror esp-dl control flow)
    # Use local numeric labels to guarantee short-branch reachability.
    # Bias/no-bias selection using short-branch trampolines to keep range small.
    l32i a14, a4, KA_flags
    bbsi a14, 0, 1f                 # if HAS_BIAS set → branch to 1f (near)
    j 2f                            # else jump to no-bias far label via near trampoline
1:
    j 3f                            # with-bias far label via near trampoline
2:
    # no-bias far block
    tie728_s8_conv2d_per_channel_result q0, q1, a5, a10, q2
    j .L_after_requant
3:
    # with-bias far block
    l32i a14, a4, KA_pc_bias16
    add  a14, a14, a12              # bias16 ptr for tile
    tie728_s8_conv2d_per_channel_with_bias_result q0, q1, a5, a14, a10, q2
    j .L_after_requant

.L_after_requant:
    # Fused ReLU if requested
    l32i a11, a4, KA_fused
    addi a11, a11, -1
    bnez a11, 5f
    movi a12, 0
    tie728_s8_conv2d_relu q0, a12, a12
5:
    EE.VST.128.IP q0, a2, 16
    addi a7, a7, -1
    j .L_tile_loop

8:
    retw

9:
    retw
