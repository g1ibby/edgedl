    .section .iram1, "ax"
    .align 4

    ################################################################################################
    # TIE728 SIMD conv2d (HWCN unaligned, generic KH×KW)
    #
    # Entry ABI:
    # - a2: output_ptr (current pixel)
    # - a3: input_ptr  (current pixel)
    # - a4: &KernelArgs
    #
    # Register Usage:
    # - a2: output_ptr (preserved/advanced)
    # - a3: base input_ptr (preserved)
    # - a4: args pointer (preserved)
    # - a5: filter_base (weights)
    # - a6: c_div_x_1
    # - a7: oc_tiles loop counter
    # - a8: mac_shift
    # - a9: dilation_x_offset
    # - a10: dilation_y_offset
    # - a11: rolling filter_ptr (advanced by inner macros)
    # - a12: flags / scratch
    # - a13: c_remainder (ic % 16)
    # - a14: scratch
    # - a15: tile_idx
    #
    # Vector Registers:
    # - q0/q1: input scratch (front/back aligned) for unaligned inner macro
    # - q2/q3: filter scratch
    # - q6   : scratch (unused here but reserved for consistency)
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
    .set KA_act_alpha,   140
    .set KA_act_shift,   144

    ################################################################################################
    # HWCN H×W traversal using unaligned 1×1 inner microkernel
    #
    # Exact port of esp-dl's `tie728_s8_conv2d_h_w_unaligned_c`, adapted to our
    # KernelArgs layout and inner 1×1 unaligned signature. Walks KH×KW taps,
    # advancing input_ptr by dilation_x/y offsets and maintaining filter_ptr so
    # it points just past this OC tile’s (kh*kw*ic) block on return.
    #
    # Arguments:
    #   input_v0        : vector (merged input)
    #   input_front     : vector (front-aligned input)
    #   input_back      : vector (back-aligned input)
    #   input_ptr       : GPR (modified)
    #   filter_v0       : vector
    #   filter_v1       : vector
    #   filter_ptr      : GPR (modified)
    #   c_div_x_1       : GPR ((ic/16) - 1)
    #   filter_h        : GPR (scratch)
    #   filter_w        : GPR (scratch)
    #   args            : GPR (&KernelArgs)
    #   rem_c           : GPR (ic % 16)
    #   tmp             : GPR scratch
    #   filter_offset_q : vector scratch
    ################################################################################################
    .macro tie728_s8_conv2d_h_w_unaligned_c input_v0 input_front input_back input_ptr filter_v0 filter_v1 filter_ptr c_div_x_1 filter_h filter_w args rem_c tmp filter_offset_q
        # Load filter_width
        l32i \filter_w, \args, KA_kw
        # Fix: branch range is small, trampoline to width==1 path
        bgei \filter_w, 2, .L_hwun_w_ge_2
        j .L_hwun_w_eq_1

.L_hwun_w_ge_2:
        # General width path (filter_w >= 2)
        l32i \filter_h, \args, KA_kh
.L_hwun_h_loop:
            l32i \filter_w, \args, KA_kw
.L_hwun_w_loop:
                # 1x1 unaligned inner at (ky,kx)
                tie728_s8_conv2d_1_1_unaligned_c \input_ptr, \filter_ptr, \c_div_x_1, \rem_c, \input_v0, \input_front, \filter_v0, \filter_v1, \tmp, \args
                # Advance along width by dilation_x_offset
                l32i \tmp, \args, KA_dil_x_off
                add \input_ptr, \input_ptr, \tmp

                addi \filter_w, \filter_w, -1
                # If more columns remain in this row, continue
                blti \filter_w, 2, .L_hwun_row_last
                j .L_hwun_w_loop

.L_hwun_row_last:
            # Final column in this row
            tie728_s8_conv2d_1_1_unaligned_c \input_ptr, \filter_ptr, \c_div_x_1, \rem_c, \input_v0, \input_front, \filter_v0, \filter_v1, \tmp, \args
            # Move to next row: input_ptr += dilation_y_offset
            l32i \tmp, \args, KA_dil_y_off
            add \input_ptr, \input_ptr, \tmp
            # Note: filter_ptr is already advanced correctly by the inner 1x1 macro
            # (no fixup needed for contiguous OHWI layout)

            addi \filter_h, \filter_h, -1
        # Fix: bnez range is small (~128 bytes), two inner macros expand to 1000+ bytes
        # Use beqz+j trampoline for far backward branch
        beqz \filter_h, .L_hwun_done
        j .L_hwun_h_loop

    # filter_w == 1 path
.L_hwun_w_eq_1:
        l32i \filter_h, \args, KA_kh
.L_hwun_kw1_loop:
            tie728_s8_conv2d_1_1_unaligned_c \input_ptr, \filter_ptr, \c_div_x_1, \rem_c, \input_v0, \input_front, \filter_v0, \filter_v1, \tmp, \args

            l32i \tmp, \args, KA_dil_y_off
            add \input_ptr, \input_ptr, \tmp
            # Note: filter_ptr already advanced by inner macro

            addi \filter_h, \filter_h, -1
        # Fix: bnez range is small (~128 bytes), inner macro expands to 500+ bytes
        # Use beqz+j trampoline for far backward branch
        beqz \filter_h, .L_hwun_done
        j .L_hwun_kw1_loop

.L_hwun_done:
        # Note: filter_ptr now points just past this tile's (kh*kw*ic*16) block
    .endm

    ################################################################################################
    # HWCN H×W traversal for OC tails (ACCX remainder path)
    #
    # Port of esp-dl's `tie728_s8_conv2d_h_w_c_n_remainder`, adapted to our
    # KernelArgs layout and `tie728_s8_conv2d_1_1_c_n_remainder_accumulate`.
    # Filter traversal mirrors the non-tail macro so filter_ptr lands just past
    # this OC’s (kh*kw*ic) block on return.
    ################################################################################################
    .macro tie728_s8_conv2d_h_w_c_n_remainder input_ptr filter_ptr c_div_x_1 rem_c args filter_h filter_w tmp0 tmp1 filter_offset_q
        # Load filter_width
        l32i \filter_w, \args, KA_kw
        # Zero ACCX once at the beginning.
        EE.ZERO.ACCX

        # Fix: blti/bgei range is small, use trampoline
        bgei \filter_w, 2, .L_hwun_rem_w_ge_2
        j .L_hwun_rem_w_eq_1

.L_hwun_rem_w_ge_2:
        # General width path (filter_w >= 2)
        l32i \filter_h, \args, KA_kh
.L_hwun_rem_h_loop:
            l32i \filter_w, \args, KA_kw
.L_hwun_rem_w_loop:
                tie728_s8_conv2d_1_1_c_n_remainder_accumulate a5, \input_ptr, \filter_ptr, \c_div_x_1, \rem_c, \tmp0, \tmp1
                # dilation_x_offset from KernelArgs
                l32i \tmp0, \args, KA_dil_x_off
                add \input_ptr, \input_ptr, \tmp0

                addi \filter_w, \filter_w, -1
                # Fix: bgei range is small, use trampoline
                blti \filter_w, 2, .L_hwun_rem_row_last
                j .L_hwun_rem_w_loop

.L_hwun_rem_row_last:
            # Final column in this row
            tie728_s8_conv2d_1_1_c_n_remainder_accumulate a5, \input_ptr, \filter_ptr, \c_div_x_1, \rem_c, \tmp0, \tmp1
            # dilation_y_offset from KernelArgs
            l32i \tmp0, \args, KA_dil_y_off
            add \input_ptr, \input_ptr, \tmp0
            # Note: filter_ptr already advanced by inner macro

            addi \filter_h, \filter_h, -1
        # Fix: bnez range is small, use beqz+j trampoline for far backward branch
        beqz \filter_h, .L_hwun_rem_done
        j .L_hwun_rem_h_loop

    # filter_w == 1 path
.L_hwun_rem_w_eq_1:
        l32i \filter_h, \args, KA_kh
.L_hwun_rem_kw1_loop:
            tie728_s8_conv2d_1_1_c_n_remainder_accumulate a5, \input_ptr, \filter_ptr, \c_div_x_1, \rem_c, \tmp0, \tmp1
            l32i \tmp0, \args, KA_dil_y_off
            add \input_ptr, \input_ptr, \tmp0
            # Note: filter_ptr already advanced by inner macro

            addi \filter_h, \filter_h, -1
        # Fix: bnez range is small, use beqz+j trampoline for far backward branch
        beqz \filter_h, .L_hwun_rem_done
        j .L_hwun_rem_kw1_loop

.L_hwun_rem_done:
        # Note: filter_ptr now points just past this OC's (kh*kw*ic) block
    .endm

    .global dl_tie728_s8_unaligned_conv2d_hwcn
    .type   dl_tie728_s8_unaligned_conv2d_hwcn, @function
dl_tie728_s8_unaligned_conv2d_hwcn:
    .align 4
    entry sp, 160

    # a2: output_ptr
    # a3: input_ptr (base for this pixel)
    # a4: &KernelArgs

    # Load tiling helpers and dilation offsets
    l32i a7,  a4, KA_oc_tiles     # oc_tiles
    l32i a6,  a4, KA_c_div_x_1    # c_div_x_1
    l32i a13, a4, KA_c_remainder  # c_remainder
    l32i a9,  a4, KA_dil_x_off    # dilation_x_offset
    l32i a10, a4, KA_dil_y_off    # dilation_y_offset

    # mac_shift = out_exp - in_exp - w_exp_tensor
    l8ui a8,  a4, KA_out_exp
    l8ui a12, a4, KA_in_exp
    l8ui a5,  a4, KA_wexp_tensor
    sext a8,  a8,  7
    sext a12, a12, 7
    sext a5,  a5,  7
    sub  a8,  a8,  a12
    sub  a8,  a8,  a5              # a8 = mac_shift

    # Load weights base and init rolling filter_ptr
    l32i a5,  a4, KA_weights      # weights base
    mov  a11, a5                  # current filter_ptr
    movi a15, 0                   # tile_idx

    # Tile loop
 .L_hwcn_u_tile_loop:
    # Branch range trampoline: if a7==0 jump to done via near branch + far jump
    beqz a7, 1f
    j 2f
1:
    j .L_hwcn_u_tiles_done
2:

    # Accumulator init (mirror k11/k33 patterns):
    # If PER_OC_W_EXP set → zero QACC (bias applied later via scale table)
    # Else if HAS_BIAS set → preload bias; otherwise zero QACC.
    l32i a12, a4, KA_flags
    movi a14, 2
    and  a14, a12, a14
    bnez a14, 0f                  # per-channel → zero QACC
    movi a14, 1
    and  a14, a12, a14
    beqz a14, 0f                  # no bias → zero QACC
    # per-layer with bias: preload tile bias into QACC
    EE.ZERO.QACC                          # Zero QACC before bias preload (20-bit bias doesn't fill 40-bit lanes)
    l32i a14, a4, KA_bias
    slli a5,  a15, 6              # tile_idx * 64
    add  a14, a14, a5
    tie728_s8_conv2d_128b_vector_bias a14
    j 1f
0:
    EE.ZERO.QACC
1:
    # H×W convolution using unaligned 1×1 inner
    mov a14, a3                   # reload input_ptr for this pixel
    # args: input_v0=q1, input_front=q0, input_back=q4, input_ptr=a14,
    #       filter_v0=q2, filter_v1=q3, filter_ptr=a11,
    #       c_div_x_1=a6, filter_h=a5, filter_w=a12, args=a4,
    #       rem_c=a13, tmp=a9, filter_offset_q=q6
    tie728_s8_conv2d_h_w_unaligned_c q1, q0, q4, a14, q2, q3, a11, a6, a5, a12, a4, a13, a9, q6

    # Epilogue: per-layer vs per-channel (use trampoline for far targets)
    l32i a12, a4, KA_flags
    bbsi a12, 1, 3f
    j .L_hwcn_u_pl
3:
    j .L_hwcn_u_pc
.L_hwcn_u_pl:
    tie728_s8_vector_round_result q0, a8, a14, q3
    j .L_hwcn_u_store

.L_hwcn_u_pc:
    # per-channel: compute table bases for this tile
    l32i a5,  a4, KA_pc_scales
    slli a14, a15, 5              # tile_idx * 32
    add  a5,  a5,  a14            # scale_ptr
    l32i a12, a4, KA_flags
    # If HAS_BIAS bit is set, go to with-bias path via near branch
    bbsi a12, 0, .L_hwcn_u_pc_with_bias
    j .L_hwcn_u_pc_no_bias
.L_hwcn_u_pc_with_bias:
    l32i a14, a4, KA_pc_bias16
    slli a12, a15, 5
    add  a14, a14, a12            # bias16_ptr
    tie728_s8_conv2d_per_channel_with_bias_result q0, q1, a5, a14, a8, q2
    j .L_hwcn_u_store
.L_hwcn_u_pc_no_bias:
    tie728_s8_conv2d_per_channel_result q0, q1, a5, a8, q2

.L_hwcn_u_store:
    # Fused ReLU
    l32i a14, a4, KA_fused
    addi a14, a14, -1
    bnez a14, 4f
    movi a12, 0
    tie728_s8_conv2d_relu q0, a12, a12
4:
    # Select unaligned store strategy based on output_ptr low nibble
    movi a14, 15
    and  a14, a2, a14
    beqz a14, 5f
    movi a5, 8
    beq  a14, a5, 6f
    dl_tie728_s8_unaligned_store0 q0, a2, a5
    j 7f
5:
    EE.VST.128.IP q0, a2, 16
    j 7f
6:
    dl_tie728_s8_unaligned_store1 q0, a2
7:
    addi a7, a7, -1
    addi a15, a15, 1
    beqz a7, 5f
    j .L_hwcn_u_tile_loop
5:

.L_hwcn_u_tiles_done:
    # OC tail (1..15): accumulate into ACCX over H×W with remainder macro
    l32i a7,  a4, KA_oc_tail      # remaining count
    beqz a7, .L_hwcn_u_done

    # Base pointers for per-channel scales/bias for the tail region
    l32i a5,  a4, KA_pc_scales    # a5 = scale_base (may be null)
    beqz a5, 8f
    slli a12, a15, 5              # tile_idx == oc_tiles
    add  a5,  a5,  a12            # scale_base for tail
8:
    l32i a12, a4, KA_pc_bias16    # a12 = bias16_base (may be null)
    beqz a12, 9f
    slli a14, a15, 5
    add  a12, a12, a14            # bias16_base for tail
9:
    # Tail loop: j from 0..oc_tail-1
.L_hwcn_u_tail_loop:
    beqz a7, .L_hwcn_u_done
    # Compute j_idx = oc_tail - remaining
    l32i a14, a4, KA_oc_tail
    sub  a6,  a14,  a7            # a6 = j_idx
    # filter_ptr_tail = filter_base_after_tiles + j_idx*ic
    l32i a14, a4, KA_ic
    mull a6,  a6,  a14
    add  a6,  a6,  a11            # a6 = tail filter_ptr

    # Accumulate over H×W into ACCX using remainder macro
    mov  a14, a3                  # a14 = input_ptr for this pixel
    l32i a15,  a4, KA_c_div_x_1   # a15 = c_div_x_1
    # Save base input_ptr (a3) to stack since macro uses a3 as tmp0
    s32i a3, sp, 0
    # filter_h=a10, filter_w=a9 (scratch), tmp0=a3, tmp1=a8, filter_offset_q=q6
    tie728_s8_conv2d_h_w_c_n_remainder a14, a6, a15, a13, a4, a10, a9, a3, a8, q6
    # Restore base input_ptr
    l32i a3, sp, 0

    # Extract scalar accumulator to a10 (preserving a5=scale_base)
    movi a6, 0
    EE.SRS.ACCX a10, a6, 0        # a10 = acc

    # Tail epilogue: uses op_type; per-layer path may add pl_bias_i32 here
    l32i a6,  a4, KA_op_type      # a6 = op_type

    # Per-layer optional bias add path (bit0 == 0)
    bbci a6, 0, .L_hwcn_tail_pl_path
    j .L_hwcn_tail_pl_after
.L_hwcn_tail_pl_path:
    # If HAS_BIAS (bit1) and pl_bias_i32 table present, add bias in acc domain
    bbsi a6, 1, .L_hwcn_tail_pl_with_bias
    j .L_hwcn_tail_pl_no_bias
.L_hwcn_tail_pl_with_bias:
    l32i a11, a4, KA_pl_bias_i32
    beqz a11, .L_hwcn_tail_pl_no_bias
    # offset = j_idx*4 (do not clobber a2)
    l32i a15, a4, KA_oc_tail
    sub  a15, a15, a7             # a15 = oc_tail - remaining = j_idx
    slli a15, a15, 2              # a15 = j_idx * 4
    add  a11, a11, a15
    l32i a15, a11, 0
    add  a10, a10, a15            # acc += bias[j_idx]
.L_hwcn_tail_pl_no_bias:
    # Recompute mac_shift in a8
    l8ui a11, a4, KA_out_exp
    l8ui a15, a4, KA_in_exp
    # NOTE: do not clobber a2 (output_ptr); use a14 as scratch.
    l8ui a14,  a4, KA_wexp_tensor
    sext a11, a11, 7
    sext a15, a15, 7
    sext a14,  a14,  7
    sub  a11, a11, a15
    sub  a8,  a11, a14            # a8 = mac_shift
.L_hwcn_tail_pl_after:

    # Build j = 2*j_idx for per-channel
    l32i a14, a4, KA_oc_tail
    sub  a15,  a14,  a7           # a15 = j_idx
    slli a15,  a15,  1            # j = 2*j_idx

    # Call epilogue: op_type=a6, output_ptr=a2, mac_shift=a8, scale_base=a5, bias_base=a12, j=a15, acc=a10
    tie728_s8_conv2d_1_1_tail_epilogue a6, a2, a8, a5, a12, a15, a10, a3, a9

    addi a7, a7, -1
    j .L_hwcn_u_tail_loop

.L_hwcn_u_done:
    retw

