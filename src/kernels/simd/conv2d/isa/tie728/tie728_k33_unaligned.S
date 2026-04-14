    .section .iram1, "ax"
    .align 4

    ################################################################################################
    # TIE728 SIMD conv2d (3x3 unaligned, OHWI)
    #
    # Entry ABI:
    # - a2: output_ptr (current pixel)
    # - a3: input_ptr  (current pixel)
    # - a4: &KernelArgs
    #
    # Register Usage:
    # - a2: output_ptr (modified in loop)
    # - a3: input_ptr (preserved/reloaded)
    # - a4: args pointer (preserved)
    # - a5: filter_ptr (preserved)
    # - a6: c_div_x_1 (preserved)
    # - a7: loop counter (oc_tiles or oc_tail)
    # - a8: mac_shift / channel_factor
    # - a9: dilation_x_offset
    # - a10: dilation_y_offset
    # - a11: bias_ptr (vector) / pl_bias_i32 (scalar)
    # - a12: activation_alpha / alpha_ptr / op_type / scratch
    # - a13: activation_shift / c_remainder
    # - a14: scratch (input_sar, operation_type, acc)
    # - a15: moving_input_ptr (scratch)
    #
    # Vector Registers:
    # - q0-q6: scratch vectors for macros
    # - q7:    operation_type (q7[0])
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

    .global tie728_k33_unaligned
    .type   tie728_k33_unaligned, @function
    .align  4

tie728_k33_unaligned:
    entry sp, 48

    # Load arguments
    l32i a7,  a4, KA_oc_tiles
    l32i a5,  a4, KA_weights
    l32i a6,  a4, KA_c_div_x_1
    l32i a9,  a4, KA_dil_x_off
    l32i a10, a4, KA_dil_y_off

    # Load common fields
    l32i a14, a4, KA_op_type
    EE.MOVI.32.Q q7, a14, 0 # q7[0] = op_type

    # Calculate mac_shift (a8)
    l8ui a12, a4, KA_in_exp
    l8ui a13, a4, KA_out_exp
    l8ui a14, a4, KA_wexp_tensor
    add  a12, a12, a14 # in + w
    sub  a8,  a13, a12 # out - (in + w) -> mac_shift

    l32i a11, a4, KA_bias
    l32i a13, a4, KA_c_remainder

    # Check if we have any OC tiles
    bnez a7, .L_loop_unaligned_out

    # Determine output alignment (SAR byte)


.L_loop_unaligned_out:
    mov a15, a3 # reload input_ptr
    EE.ZERO.QACC

    beqz a11, 1f
    tie728_s8_conv2d_128b_vector_bias a11
1:
    # Perform 3x3 convolution
    tie728_s8_conv2d_3_3_unaligned_c q0, q1, q2, a15, q3, q5, a5, a6, a9, a10, a13, a14, a4

    # Result processing (epilogue)
    # Rounding:
    tie728_s8_vector_round_result q0, a8, a15, q3

    # Activation (ReLU/PReLU)
    l32i a12, a4, KA_op_type # reload op_type
    bbci a12, 2, 2f
    # ReLU
    movi a15, 0
    movi a12, 0 # shift 0
    tie728_s8_conv2d_relu q0, a15, a12
2:
    # Store unaligned
    dl_tie728_s8_unaligned_store0 q0, a2, a15

    addi a7, a7, -1
    beqz a7, 1f
    j .L_loop_unaligned_out
1:

.L_tail_processing:
    # Handle OC tail (scalar loop)
    l32i a7, a4, KA_oc_tail
    beqz a7, .L_end

    # Load pl_bias_i32 into a11 (repurposing a11)
    l32i a11, a4, KA_pl_bias_i32

.L_k33_unaligned_tail_loop:
    mov a15, a3 # reload input_ptr
    EE.ZERO.ACCX

    # 1
    tie728_s8_conv2d_1_1_c_n_remainder_accumulate a14, a15, a5, a6, a13, a12, a8
    add a15, a15, a9
    # 2
    tie728_s8_conv2d_1_1_c_n_remainder_accumulate a14, a15, a5, a6, a13, a12, a8
    add a15, a15, a9
    # 3
    tie728_s8_conv2d_1_1_c_n_remainder_accumulate a14, a15, a5, a6, a13, a12, a8
    add a15, a15, a10
    # 4
    tie728_s8_conv2d_1_1_c_n_remainder_accumulate a14, a15, a5, a6, a13, a12, a8
    add a15, a15, a9
    # 5
    tie728_s8_conv2d_1_1_c_n_remainder_accumulate a14, a15, a5, a6, a13, a12, a8
    add a15, a15, a9
    # 6
    tie728_s8_conv2d_1_1_c_n_remainder_accumulate a14, a15, a5, a6, a13, a12, a8
    add a15, a15, a10
    # 7
    tie728_s8_conv2d_1_1_c_n_remainder_accumulate a14, a15, a5, a6, a13, a12, a8
    add a15, a15, a9
    # 8
    tie728_s8_conv2d_1_1_c_n_remainder_accumulate a14, a15, a5, a6, a13, a12, a8
    add a15, a15, a9
    # 9
    tie728_s8_conv2d_1_1_c_n_remainder_accumulate a14, a15, a5, a6, a13, a12, a8

    # Extract ACCX to scalar
    movi a14, 0
    EE.SRS.ACCX a14, a14, 0 # acc = a14

    # Epilogue
    l32i a12, a4, KA_op_type # reload op_type

    # Per-layer bias handling
    bbci a12, 0, 3f # Jump if per-layer
    j 4f # Per-channel (skip)

3:  # Per-layer
    bbci a12, 1, 5f # Jump if no bias
    # Add bias
    beqz a11, 5f
    l32i a15, a11, 0 # load bias
    add a14, a14, a15
    addi a11, a11, 4 # advance bias ptr
5:
    # Recalculate mac_shift (a8) since we clobbered it
    l8ui a15, a4, KA_in_exp
    l8ui a13, a4, KA_out_exp
    l8ui a8,  a4, KA_wexp_tensor # use a8 as temp for wexp
    add  a15, a15, a8 # in + w
    sub  a8,  a13, a15 # out - (in + w) -> mac_shift

    # Call epilogue with bias_base=0 (since we added it manually)
    movi a15, 0
    tie728_s8_conv2d_1_1_tail_epilogue a12, a2, a8, a15, a15, a15, a14, a15, a13

    addi a7, a7, -1
    beqz a7, 1f
    j .L_k33_unaligned_tail_loop
1:
    j .L_end

4: # Per-channel (placeholder)
    # ...
    addi a7, a7, -1
    bnez a7, .L_k33_unaligned_tail_loop

.L_end:
    retw
