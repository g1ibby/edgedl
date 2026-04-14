    .section .iram1, "ax"
    .align 4

    ################################################################################################
    # TIE728 SIMD conv2d (HWCN aligned, OHWI)
    #
    # Entry ABI:
    # - a2: output_ptr (current pixel)
    # - a3: input_ptr  (current pixel)
    # - a4: &KernelArgs
    #
    # Register Usage:
    # - a2: output_ptr (preserved/advanced)
    # - a3: input_ptr (preserved/reloaded)
    # - a4: args pointer (preserved)
    # - a5: filter_ptr (rolling)
    # - a6: c_div_x_1
    # - a7: loop counter (oc_tiles)
    # - a8: mac_shift / scratch
    # - a9: dilation_x_offset
    # - a10: dilation_y_offset
    # - a11: bias_ptr (rolling) / scratch
    # - a12: flags / scratch
    # - a13: scratch
    # - a14: scratch
    # - a15: tile_idx / scratch
    #
    # Vector Registers:
    # - q0: Accumulator / Output
    # - q1: Input
    # - q2: Filter
    # - q3: Filter / Scratch
    # - q6: Scratch (filter_offset_q)
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
    .set KA_dil_x_off,   132
    .set KA_dil_y_off,   136
    # activation_alpha at 140, activation_shift at 144
    .set KA_filter_y_off, 148
    .set KA_filter_n_off, 152

    .global dl_tie728_s8_conv2d_hwcn
    .type   dl_tie728_s8_conv2d_hwcn, @function
dl_tie728_s8_conv2d_hwcn:
    .align 4
    entry sp, 160
    # a2: output_ptr
    # a3: input_ptr
    # a4: &KernelArgs

    # Load constants
    l32i a7,  a4, KA_oc_tiles     # a7 = oc_tiles
    l32i a6,  a4, KA_c_div_x_1    # a6 = c_div_x_1
    l32i a9,  a4, KA_dil_x_off    # a9 = dilation_x_offset
    l32i a10, a4, KA_dil_y_off    # a10 = dilation_y_offset

    # mac_shift = out_exp - in_exp - w_exp_tensor
    l8ui a8,  a4, KA_out_exp
    l8ui a11, a4, KA_in_exp
    l8ui a5,  a4, KA_wexp_tensor
    sext a8,  a8, 7
    sext a11, a11, 7
    sext a5,  a5,  7
    sub  a8,  a8, a11
    sub  a8,  a8, a5              # a8 = mac_shift

    # Load weights base and set rolling filter_ptr
    l32i a5,  a4, KA_weights      # weights base
    mov  a11, a5                  # a11 = current filter_ptr (advanced by macro)
    movi a15, 0                   # a15 = tile_idx

    # Initialize filter_offset_q (q6) with filter_y_offset and filter_n_offset.
    # The hwc16 macro extracts these from q6 slots 1 and 2 and adds to filter_ptr.
    # For BlockedN16HWC16 interior pixels, both are 0 (weights are contiguous).
    # Without this initialization, q6 contains garbage from previous operations,
    # corrupting filter_ptr and causing massive slowdowns (500-1500x) for high IC/OC.
    # See: esp-dl dl_tie728_s8_conv2d.S:1936-1937 for reference initialization.
    l32i a13, a4, KA_filter_y_off
    l32i a14, a4, KA_filter_n_off
    EE.MOVI.32.Q q6, a13, 1    # filter_y_offset -> slot 1
    EE.MOVI.32.Q q6, a14, 2    # filter_n_offset -> slot 2

    # Tile loop
.L_hwcn_tile_loop:
    beqz a7, .L_hwcn_done

    # Accumulator init
    l32i a12, a4, KA_flags
    
    movi a14, 2
    and  a14, a12, a14
    bnez a14, 1f                  # Per-channel -> zero accumulators
    
    movi a14, 1
    and  a14, a12, a14
    beqz a14, 1f                  # No bias -> zero accumulators

    # Per-layer with bias -> preload
    EE.ZERO.QACC                          # Zero QACC before bias preload (20-bit bias doesn't fill 40-bit lanes)
    l32i a14, a4, KA_bias
    slli a13, a15, 6              # tile_idx * 64
    add  a14, a14, a13
    tie728_s8_conv2d_128b_vector_bias a14
    j 2f

1:
    EE.ZERO.QACC
2:
    # HWCN Convolution
    mov a14, a3                   # reload input_ptr
    # Macro arguments: input_v0, input_ptr, filter_v0, filter_v1, filter_ptr, c_div_x_1, dil_x, dil_y, filter_h, filter_w, filter_offset_q, args
    # Using q1 as input_v0, q2/q3 as filter vectors, q6 as filter_offset_q
    # a12, a13 as scratch for filter_h, filter_w
    tie728_s8_conv2d_hwc16 q1, a14, q2, q3, a11, a6, a9, a10, a12, a13, q6, a4

    # Requantization
    l32i a12, a4, KA_flags
    bbsi a12, 1, .L_hwcn_pc            # Bit 1 set -> Per-channel

.L_hwcn_pl:
    # Per-layer epilogue
    tie728_s8_vector_round_result q0, a8, a14, q3
    j .L_hwcn_epilogue

.L_hwcn_pc:
    # Per-channel epilogue
    l32i a5,  a4, KA_pc_scales
    slli a13, a15, 5              # tile_idx * 32
    add  a5,  a5,  a13            # scale_ptr
    
    l32i a12, a4, KA_flags
    bbci a12, 0, .L_hwcn_pc_no_bias # Bit 0 clear -> No bias
    j .L_hwcn_pc_has_bias           # Bit 0 set -> Has bias

.L_hwcn_pc_no_bias:
    # No bias
    tie728_s8_conv2d_per_channel_result q0, q1, a5, a8, q2
    j .L_hwcn_epilogue

.L_hwcn_pc_has_bias:
    # With bias
    l32i a14, a4, KA_pc_bias16
    slli a13, a15, 5              # tile_idx * 32
    add  a14, a14, a13            # bias16_ptr
    tie728_s8_conv2d_per_channel_with_bias_result q0, q1, a5, a14, a8, q2

.L_hwcn_epilogue:
    # Fused ReLU
    l32i a14, a4, KA_fused
    addi a14, a14, -1
    bnez a14, 4f
    movi a12, 0
    tie728_s8_conv2d_relu q0, a12, a12
4:
    # Store result (aligned 16 bytes)
    EE.VST.128.IP q0, a2, 16

    addi a7, a7, -1
    addi a15, a15, 1
    bnez a7, .L_hwcn_tile_loop

.L_hwcn_done:
    retw
