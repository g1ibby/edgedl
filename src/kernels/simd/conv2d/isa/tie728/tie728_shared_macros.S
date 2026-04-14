################################################################################################
# Shared TIE728 helper macros for Conv2D kernels (ESP32‑S3)
#
# These macros are reused by both aligned and unaligned 1x1 (11cn) kernels.
# Keep this file in the same directory as the Rust modules and include once from
# the ISA module (mod.rs) with:
#   core::arch::global_asm!(include_str!("tie728_shared_macros.S"));
################################################################################################

################################################################################################
# Shared content:
#  - tie728_s8_vector_round_result
#  - tie728_s8_conv2d_relu
#  - tie728_s8_conv2d_128b_vector_bias
#  - tie728_s8_conv2d_per_channel_result
#  - tie728_s8_conv2d_per_channel_with_bias_result
#  - tie728_s8_conv2d_hwc16
################################################################################################
################################################################################################
# 11c16 inner microkernel (shared)
#
# Accumulates over one IC block (16 channels) and advances filter/input pointers.
# Copied from esp‑dl (tie728_s8_conv2d_11c16) and used by both aligned and
# unaligned entries. Keeps stream-friendly interleaving of filter loads with
# broadcasted input lane MACs.
.macro tie728_s8_conv2d_11c16 input_v0 input_ptr filter_v0 filter_v1 filter_ptr c_div_x_1
    # Prime the streams: load first input block and first two filter vectors
    EE.VLD.128.IP \input_v0,  \input_ptr,  16
    EE.VLD.128.IP \filter_v0, \filter_ptr, 16
    EE.VLD.128.IP \filter_v1, \filter_ptr, 16

    # Loop across remaining IC blocks (>=0). Each iteration performs 16 MACs using
    # broadcasted input lanes 0..15, interleaved with streamed filter loads.
    loopgtz \c_div_x_1, 0f
    EE.VSMULAS.S8.QACC.LD.INCP \filter_v0, \filter_ptr, \filter_v0, \input_v0, 0
    EE.VSMULAS.S8.QACC.LD.INCP \filter_v1, \filter_ptr, \filter_v1, \input_v0, 1
    EE.VSMULAS.S8.QACC.LD.INCP \filter_v0, \filter_ptr, \filter_v0, \input_v0, 2
    EE.VSMULAS.S8.QACC.LD.INCP \filter_v1, \filter_ptr, \filter_v1, \input_v0, 3
    EE.VSMULAS.S8.QACC.LD.INCP \filter_v0, \filter_ptr, \filter_v0, \input_v0, 4
    EE.VSMULAS.S8.QACC.LD.INCP \filter_v1, \filter_ptr, \filter_v1, \input_v0, 5
    EE.VSMULAS.S8.QACC.LD.INCP \filter_v0, \filter_ptr, \filter_v0, \input_v0, 6
    EE.VSMULAS.S8.QACC.LD.INCP \filter_v1, \filter_ptr, \filter_v1, \input_v0, 7
    EE.VSMULAS.S8.QACC.LD.INCP \filter_v0, \filter_ptr, \filter_v0, \input_v0, 8
    EE.VSMULAS.S8.QACC.LD.INCP \filter_v1, \filter_ptr, \filter_v1, \input_v0, 9
    EE.VSMULAS.S8.QACC.LD.INCP \filter_v0, \filter_ptr, \filter_v0, \input_v0, 10
    EE.VSMULAS.S8.QACC.LD.INCP \filter_v1, \filter_ptr, \filter_v1, \input_v0, 11
    EE.VSMULAS.S8.QACC.LD.INCP \filter_v0, \filter_ptr, \filter_v0, \input_v0, 12
    EE.VSMULAS.S8.QACC.LD.INCP \filter_v1, \filter_ptr, \filter_v1, \input_v0, 13
    EE.VSMULAS.S8.QACC.LD.INCP \filter_v0, \filter_ptr, \filter_v0, \input_v0, 14
    # Last step of the block loads the next input vector for the subsequent block
    EE.VSMULAS.S8.QACC.LD.INCP \input_v0,  \input_ptr,  \filter_v1, \input_v0, 15
    EE.VLD.128.IP               \filter_v1, \filter_ptr, 16
0:
    EE.VSMULAS.S8.QACC.LD.INCP \filter_v0, \filter_ptr, \filter_v0, \input_v0, 0
    EE.VSMULAS.S8.QACC.LD.INCP \filter_v1, \filter_ptr, \filter_v1, \input_v0, 1
    EE.VSMULAS.S8.QACC.LD.INCP \filter_v0, \filter_ptr, \filter_v0, \input_v0, 2
    EE.VSMULAS.S8.QACC.LD.INCP \filter_v1, \filter_ptr, \filter_v1, \input_v0, 3
    EE.VSMULAS.S8.QACC.LD.INCP \filter_v0, \filter_ptr, \filter_v0, \input_v0, 4
    EE.VSMULAS.S8.QACC.LD.INCP \filter_v1, \filter_ptr, \filter_v1, \input_v0, 5
    EE.VSMULAS.S8.QACC.LD.INCP \filter_v0, \filter_ptr, \filter_v0, \input_v0, 6
    EE.VSMULAS.S8.QACC.LD.INCP \filter_v1, \filter_ptr, \filter_v1, \input_v0, 7
    EE.VSMULAS.S8.QACC.LD.INCP \filter_v0, \filter_ptr, \filter_v0, \input_v0, 8
    EE.VSMULAS.S8.QACC.LD.INCP \filter_v1, \filter_ptr, \filter_v1, \input_v0, 9
    EE.VSMULAS.S8.QACC.LD.INCP \filter_v0, \filter_ptr, \filter_v0, \input_v0, 10
    EE.VSMULAS.S8.QACC.LD.INCP \filter_v1, \filter_ptr, \filter_v1, \input_v0, 11
    EE.VSMULAS.S8.QACC.LD.INCP \filter_v0, \filter_ptr, \filter_v0, \input_v0, 12
    EE.VSMULAS.S8.QACC.LD.INCP \filter_v1, \filter_ptr, \filter_v1, \input_v0, 13
    EE.VSMULAS.S8.QACC                                  \filter_v0, \input_v0, 14
    EE.VSMULAS.S8.QACC                                  \filter_v1, \input_v0, 15
.endm

# Vector rounding from QACC to int8 with HALF_UP behavior when mac_shift > 0
# Inputs:
#   output_v: vector register for result bytes
#   mac_shift: number of right shift bits (>= 0)
#   tmp: GPR scratch
#   tmp_q1: vector scratch
.macro tie728_s8_vector_round_result output_v mac_shift tmp tmp_q1
    beqz  \mac_shift, 501f                   # no shift → just extract

    # Build 0x00010001 pattern in a vector to add 0.5 ulp before final shift
    MOVI  \tmp, 257
    EE.MOVI.32.Q  \output_v, \tmp, 0
    EE.MOVI.32.Q  \output_v, \tmp, 1
    EE.MOVI.32.Q  \output_v, \tmp, 2
    EE.MOVI.32.Q  \output_v, \tmp, 3
    movi.n  \tmp, 16
    wsr.sar  \tmp
    EE.VSL.32  \tmp_q1, \output_v
    EE.ORQ  \tmp_q1, \tmp_q1, \output_v

    addi  \tmp, \mac_shift, -1
    EE.SRCMB.S8.QACC  \output_v, \tmp, 0

    movi.n  \tmp, 1
    EE.MOVI.32.Q  \output_v, \tmp, 0
    EE.VSMULAS.S8.QACC  \tmp_q1, \output_v, 0   # qacc += round
    EE.SRCMB.S8.QACC  \output_v, \tmp, 0
    j  502f
501:
    EE.SRCMB.S8.QACC  \output_v, \mac_shift, 0
502:
.endm


# Vector ReLU (alpha=0, shift=0 → plain ReLU)
.macro tie728_s8_conv2d_relu output_v activation_alpha activation_shift
    EE.VRELU.S8 \output_v, \activation_alpha, \activation_shift
.endm

# Dummy PReLU (user requested to disable/ignore for now)
.macro tie728_s8_conv2d_prelu output_v tmp_q activation_alpha activation_shift
    # No-op for now
.endm

# Element rounding from ACCX (scalar) to int8
# Used in n_remainder path.
# Arguments:
#   acc: GPR holding scalar accumulated value (from EE.SRS.ACCX)
#   mac_shift: GPR holding shift amount
#   tmp: scratch GPR
#   tmp_q: scratch vector (unused here but kept for signature compat)
.macro tie728_s8_element_round_result acc mac_shift tmp tmp_q
    beqz \mac_shift, 503f
    
    # Add 1 << (mac_shift - 1) for rounding
    addi \tmp, \mac_shift, -1
    ssl \tmp
    movi \tmp, 1
    sll \tmp, \tmp # tmp = 1 << (mac_shift - 1)
    
    add \acc, \acc, \tmp
    
    ssr \mac_shift
    sra \acc, \acc
    
    j 504f
503:
    # No shift
504:
.endm

# Preload QACC with 16×i32 bias values (esp-dl style), bias_ptr advances by 64 bytes
.macro tie728_s8_conv2d_128b_vector_bias bias_ptr
    EE.LD.QACC_L.L.128.IP  \bias_ptr, 16
    EE.LD.QACC_L.H.32.IP   \bias_ptr, 16
    EE.LD.QACC_H.L.128.IP  \bias_ptr, 16
    EE.LD.QACC_H.H.32.IP   \bias_ptr, 16
.endm

# Per-channel result without bias
# Expects:
#  - QACC holds 16×i32 accumulators
#  - scale_factor points to 32 bytes (16×i16) for this tile; the macro advances it by 32
#  - Uses `a1` (sp) as scratch to shuffle QACC halves
.macro tie728_s8_conv2d_per_channel_result output_v scale_q scale_factor tmp tmp_q1
    movi \tmp, 4
    EE.SRCMB.S16.QACC \output_v, \tmp, 0
    mov \tmp, a1
    EE.ST.QACC_L.L.128.IP \tmp, 16
    EE.ST.QACC_L.H.32.IP \tmp, 16
    EE.ST.QACC_H.L.128.IP \tmp, 16
    EE.ST.QACC_H.H.32.IP \tmp, 4
    movi \tmp, 20
    EE.SRCMB.S16.QACC \tmp_q1, \tmp, 0
    addi \tmp, a1, 5
    l16si \tmp, \tmp, 0
    s16i \tmp, a1, 2
    addi \tmp, a1, 10
    l16si \tmp, \tmp, 0
    s16i \tmp, a1, 4
    addi \tmp, a1, 15
    l16si \tmp, \tmp, 0
    s16i \tmp, a1, 6
    addi \tmp, a1, 32
    l16si \tmp, \tmp, 0
    s16i \tmp, a1, 8
    addi \tmp, a1, 37
    l16si \tmp, \tmp, 0
    s16i \tmp, a1, 10
    addi \tmp, a1, 42
    l16si \tmp, \tmp, 0
    s16i \tmp, a1, 12
    addi \tmp, a1, 47
    l16si \tmp, \tmp, 0
    s16i \tmp, a1, 14
    EE.VLD.128.IP \output_v, a1, 0
    EE.VZIP.16 \output_v, \tmp_q1
    movi \tmp, 11
    ssr \tmp
    EE.VLD.128.IP \scale_q, \scale_factor, 16
    movi \tmp, 0
    EE.VMUL.S16 \output_v, \output_v, \scale_q
    EE.VLD.128.IP \scale_q, \scale_factor, 16
    EE.MOV.S16.QACC \output_v
    EE.VMUL.S16 \tmp_q1, \tmp_q1, \scale_q
    EE.SRCMB.S8.QACC \output_v, \tmp, 0
    EE.MOV.S16.QACC \tmp_q1
    EE.SRCMB.S8.QACC \tmp_q1, \tmp, 0
    EE.VUNZIP.8 \output_v, \tmp_q1
.endm

################################################################################################
# Unaligned helpers (ported from esp-dl semantics)
#
# These macros implement the s8 1x1 "unaligned" Conv2D behavior from
# `dl_tie728_s8_conv2d.S` for the ESP32-S3 TIE728 ISA. Where we diverge from
# esp-dl, comments below call it out explicitly (e.g., use of a fixed scratch
# vector register instead of a macro parameter).
################################################################################################

# Inner unaligned MAC over IC (handles full blocks + IC remainder).
# This is a direct port of esp-dl's `tie728_s8_conv2d_1_1_unaligned_c`, with two
# tiny structural differences called out below:
#  - We pass only `input_ptr`/`filter_ptr` plus the esp-dl style counters
#    `c_div_x_1` and `c_remainder` instead of every vector register.
#  - We dedicate `q4` as the local "input_back_aligned" scratch instead of
#    taking it as a macro parameter. This keeps the call sites smaller while
#    preserving the instruction sequence.
# Expects:
#  - input_ptr (GPR) and filter_ptr (GPR)
#  - c_div_x_1 = (ic/16) - 1  (may be <0 when ic < 16)
#  - c_remainder in a register (0..15) — number of remaining channels after
#    16*floor(ic/16); this mirrors esp-dl's `ArgsType::c_remainder` field.
#  - Accumulators in QACC zeroed or preloaded with bias by caller.
#  - Vector temps:
#      in_q0  : reused as "input_v0" (merged input)
#      in_q1  : "input_front_aligned"
#      filt_q0: "filter_v0"
#      filt_q1: "filter_v1"
#    We also use q4 internally as "input_back_aligned".
#    (This use of q4 is a minor deviation from esp-dl, which passes all
#     vector registers as macro parameters.)
#
.macro tie728_s8_conv2d_1_1_unaligned_c input_ptr filter_ptr c_div_x_1 c_remainder in_q0 in_q1 filt_q0 filt_q1 tmp args
    # Input path (esp-dl mirror).
    # Local naming to keep the mapping clear:
    #   in_q0  -> input_v0
    #   in_q1  -> input_front_aligned
    #   q4     -> input_back_aligned (extra scratch vs esp-dl macro args)

    # If ic/16 - 1 < 0, we enter the "input_channel < 16" tail path (label 17f),
    # matching esp-dl's early branch on c_div_x_1. Use a bgez + long jump
    # trampoline to avoid blti immediate form issues in LLVM's Xtensa asm.
    bgez \c_div_x_1, 90f
    j 17f
90:

    # Load first "front" aligned chunk via USAR to seed the front/back pair.
    EE.LD.128.USAR.IP \in_q1, \input_ptr, 16             # input_front_aligned

    # Prime filter streams and back-aligned input
    EE.VLD.128.IP \filt_q0, \filter_ptr, 16              # filter_v0
    EE.VLD.128.IP \filt_q1, \filter_ptr, 16              # filter_v1

    # Safety check for aligned end-of-buffer read
    # If aligned (SAR=0) AND last block (c_div_x_1=0) AND no remainder (c_remainder=0),
    # then we don't need q4 and loading it might fault.
    rsr.sar \tmp
    bnez \tmp, 200f
    bnez \c_div_x_1, 200f
    bnez \c_remainder, 200f
    
    # Skip load, just advance pointer
    addi \input_ptr, \input_ptr, 16
    j 201f
    
200:
    EE.LD.128.USAR.IP q4, \input_ptr, 16                 # input_back_aligned (uses q4)
201:

    # Full 16-channel blocks: merge front/back into input_v0, advance
    # back-aligned and filter streams, and accumulate into QACC.
    loopgtz \c_div_x_1, 0f
        EE.SRC.Q.QUP \in_q0, \in_q1, q4                  # input_v0 = merge(front, back)
        EE.LD.128.USAR.IP q4, \input_ptr, 16             # update back-aligned

        EE.VSMULAS.S8.QACC.LD.INCP \filt_q0, \filter_ptr, \filt_q0, \in_q0, 0
        EE.VSMULAS.S8.QACC.LD.INCP \filt_q1, \filter_ptr, \filt_q1, \in_q0, 1
        EE.VSMULAS.S8.QACC.LD.INCP \filt_q0, \filter_ptr, \filt_q0, \in_q0, 2
        EE.VSMULAS.S8.QACC.LD.INCP \filt_q1, \filter_ptr, \filt_q1, \in_q0, 3
        EE.VSMULAS.S8.QACC.LD.INCP \filt_q0, \filter_ptr, \filt_q0, \in_q0, 4
        EE.VSMULAS.S8.QACC.LD.INCP \filt_q1, \filter_ptr, \filt_q1, \in_q0, 5
        EE.VSMULAS.S8.QACC.LD.INCP \filt_q0, \filter_ptr, \filt_q0, \in_q0, 6
        EE.VSMULAS.S8.QACC.LD.INCP \filt_q1, \filter_ptr, \filt_q1, \in_q0, 7
        EE.VSMULAS.S8.QACC.LD.INCP \filt_q0, \filter_ptr, \filt_q0, \in_q0, 8
        EE.VSMULAS.S8.QACC.LD.INCP \filt_q1, \filter_ptr, \filt_q1, \in_q0, 9
        EE.VSMULAS.S8.QACC.LD.INCP \filt_q0, \filter_ptr, \filt_q0, \in_q0, 10
        EE.VSMULAS.S8.QACC.LD.INCP \filt_q1, \filter_ptr, \filt_q1, \in_q0, 11
        EE.VSMULAS.S8.QACC.LD.INCP \filt_q0, \filter_ptr, \filt_q0, \in_q0, 12
        EE.VSMULAS.S8.QACC.LD.INCP \filt_q1, \filter_ptr, \filt_q1, \in_q0, 13
        EE.VSMULAS.S8.QACC.LD.INCP \filt_q0, \filter_ptr, \filt_q0, \in_q0, 14
        EE.VSMULAS.S8.QACC.LD.INCP \filt_q1, \filter_ptr, \filt_q1, \in_q0, 15
0:
    # Final full 16-channel block (same pattern, no USAR refresh at the end).
    EE.SRC.Q.QUP \in_q0, \in_q1, q4

    # At this point esp-dl's upstream macro keeps `input_front_aligned` and
    # `input_back_aligned` latched and then:
    #   EE.SRC.Q.QUP input_v0, input_front_aligned, input_back_aligned
    #   addi input_ptr, input_ptr, -16
    #   add  input_ptr, input_ptr, remainder_c
    # (there is a commented-out EE.LD.128.USAR.XP variant in esp-dl, but the
    # active code uses the pointer arithmetic shown above). We mirror that
    # active implementation exactly, using `c_remainder` instead of
    # `remainder_c` and reusing q4 as the back-aligned vector.

    addi \input_ptr, \input_ptr, -16
    add  \input_ptr, \input_ptr, \c_remainder

    EE.VSMULAS.S8.QACC.LD.INCP \filt_q0, \filter_ptr, \filt_q0, \in_q0, 0
    EE.VSMULAS.S8.QACC.LD.INCP \filt_q1, \filter_ptr, \filt_q1, \in_q0, 1
    EE.VSMULAS.S8.QACC.LD.INCP \filt_q0, \filter_ptr, \filt_q0, \in_q0, 2
    EE.VSMULAS.S8.QACC.LD.INCP \filt_q1, \filter_ptr, \filt_q1, \in_q0, 3
    EE.VSMULAS.S8.QACC.LD.INCP \filt_q0, \filter_ptr, \filt_q0, \in_q0, 4
    EE.VSMULAS.S8.QACC.LD.INCP \filt_q1, \filter_ptr, \filt_q1, \in_q0, 5
    EE.VSMULAS.S8.QACC.LD.INCP \filt_q0, \filter_ptr, \filt_q0, \in_q0, 6
    EE.VSMULAS.S8.QACC.LD.INCP \filt_q1, \filter_ptr, \filt_q1, \in_q0, 7
    EE.VSMULAS.S8.QACC.LD.INCP \filt_q0, \filter_ptr, \filt_q0, \in_q0, 8
    EE.VSMULAS.S8.QACC.LD.INCP \filt_q1, \filter_ptr, \filt_q1, \in_q0, 9
    EE.VSMULAS.S8.QACC.LD.INCP \filt_q0, \filter_ptr, \filt_q0, \in_q0, 10
    EE.VSMULAS.S8.QACC.LD.INCP \filt_q1, \filter_ptr, \filt_q1, \in_q0, 11
    EE.VSMULAS.S8.QACC.LD.INCP \filt_q0, \filter_ptr, \filt_q0, \in_q0, 12
    EE.VSMULAS.S8.QACC.LD.INCP \filt_q1, \filter_ptr, \filt_q1, \in_q0, 13
    EE.VSMULAS.S8.QACC \filt_q0, \in_q0, 14
    EE.VSMULAS.S8.QACC \filt_q1, \in_q0, 15

    # If there is no IC remainder, we are done; pointer has been advanced to
    # the end of the C dimension, matching esp-dl's contract.
    beqz \c_remainder, 3f

    # For ic < 16 (input_channel < 16) we jump to the dedicated ESP-DL tail
    # path at 17f, which uses USAR.XP and a bitfield over c_remainder to
    # control how many lanes participate. For ic >= 16, we emulate the
    # `remainder_c` section exactly, using the same merged input_v0 and
    # filter streams.

    # Load next filter block for the remainder section (esp-dl does this
    # unconditionally when remainder_c != 0).
    EE.VLD.128.IP \filt_q0, \filter_ptr, 16

    # The original esp-dl code optionally restores the SAR byte here via
    # rur.sar_byte/wur.sar_byte and a USAR.IP load. We instead use a plain
    # VLD into q4 and merge front/back with EE.SRC.Q, which produces the
    # same byte ordering for the purposes of the remainder MAC.
    EE.VLD.128.IP q4, \input_ptr, 0
    EE.SRC.Q \in_q0, \in_q1, q4
    j 15f

17: # input_channel < 16 (ic_div16 == 0)
    # This branch mirrors esp-dl's small-IC path: we use USAR.XP with
    # c_remainder to synthesize the merged vector, then run a lane-masked MAC
    # controlled by the bitfield in c_remainder.
    EE.VLD.128.IP \filt_q0, \filter_ptr, 16

    EE.LD.128.USAR.XP \in_q1, \input_ptr, \c_remainder
    EE.VLD.128.IP q4, \input_ptr, 0
    EE.SRC.Q \in_q0, \in_q1, q4

15: # Remainder-c lane mask (c_remainder in 0..15), ported from esp-dl.
    # We use the same bitfield structure as esp-dl (highest bits first) and
    # interpret `c_remainder` exactly like esp-dl's `remainder_c`.
    # Use near-branch trampolines for far targets to stay within 8-bit range
    bbsi \c_remainder, 3, 91f
    j 7f
91:

    EE.VLD.128.IP \filt_q1, \filter_ptr, 16

    bbsi \c_remainder, 2, 92f
    j 11f
92:
    bbsi \c_remainder, 1, 93f
    j 13f
93:
    bbsi \c_remainder, 0, 94f
    j 14f
94:
    EE.VSMULAS.S8.QACC.LD.INCP \filt_q0, \filter_ptr, \filt_q0, \in_q0, 0
    EE.VSMULAS.S8.QACC.LD.INCP \filt_q1, \filter_ptr, \filt_q1, \in_q0, 1
    EE.VSMULAS.S8.QACC.LD.INCP \filt_q0, \filter_ptr, \filt_q0, \in_q0, 2
    EE.VSMULAS.S8.QACC.LD.INCP \filt_q1, \filter_ptr, \filt_q1, \in_q0, 3
    EE.VSMULAS.S8.QACC.LD.INCP \filt_q0, \filter_ptr, \filt_q0, \in_q0, 4
    EE.VSMULAS.S8.QACC.LD.INCP \filt_q1, \filter_ptr, \filt_q1, \in_q0, 5
    EE.VSMULAS.S8.QACC.LD.INCP \filt_q0, \filter_ptr, \filt_q0, \in_q0, 6
    EE.VSMULAS.S8.QACC.LD.INCP \filt_q1, \filter_ptr, \filt_q1, \in_q0, 7
    EE.VSMULAS.S8.QACC.LD.INCP \filt_q0, \filter_ptr, \filt_q0, \in_q0, 8
    EE.VSMULAS.S8.QACC.LD.INCP \filt_q1, \filter_ptr, \filt_q1, \in_q0, 9
    EE.VSMULAS.S8.QACC.LD.INCP \filt_q0, \filter_ptr, \filt_q0, \in_q0, 10
    EE.VSMULAS.S8.QACC.LD.INCP \filt_q1, \filter_ptr, \filt_q1, \in_q0, 11
    EE.VSMULAS.S8.QACC.LD.INCP \filt_q0, \filter_ptr, \filt_q0, \in_q0, 12
    EE.VSMULAS.S8.QACC  \filt_q1, \in_q0, 13
    EE.VSMULAS.S8.QACC  \filt_q0, \in_q0, 14
    j 16f

14:
    EE.VSMULAS.S8.QACC.LD.INCP \filt_q0, \filter_ptr, \filt_q0, \in_q0, 0
    EE.VSMULAS.S8.QACC.LD.INCP \filt_q1, \filter_ptr, \filt_q1, \in_q0, 1
    EE.VSMULAS.S8.QACC.LD.INCP \filt_q0, \filter_ptr, \filt_q0, \in_q0, 2
    EE.VSMULAS.S8.QACC.LD.INCP \filt_q1, \filter_ptr, \filt_q1, \in_q0, 3
    EE.VSMULAS.S8.QACC.LD.INCP \filt_q0, \filter_ptr, \filt_q0, \in_q0, 4
    EE.VSMULAS.S8.QACC.LD.INCP \filt_q1, \filter_ptr, \filt_q1, \in_q0, 5
    EE.VSMULAS.S8.QACC.LD.INCP \filt_q0, \filter_ptr, \filt_q0, \in_q0, 6
    EE.VSMULAS.S8.QACC.LD.INCP \filt_q1, \filter_ptr, \filt_q1, \in_q0, 7
    EE.VSMULAS.S8.QACC.LD.INCP \filt_q0, \filter_ptr, \filt_q0, \in_q0, 8
    EE.VSMULAS.S8.QACC.LD.INCP \filt_q1, \filter_ptr, \filt_q1, \in_q0, 9
    EE.VSMULAS.S8.QACC.LD.INCP \filt_q0, \filter_ptr, \filt_q0, \in_q0, 10
    EE.VSMULAS.S8.QACC.LD.INCP \filt_q1, \filter_ptr, \filt_q1, \in_q0, 11
    EE.VSMULAS.S8.QACC  \filt_q0, \in_q0, 12
    EE.VSMULAS.S8.QACC  \filt_q1, \in_q0, 13
    j 16f

13:
    bbsi \c_remainder, 0, 95f
    j 12f
95:
    EE.VSMULAS.S8.QACC.LD.INCP \filt_q0, \filter_ptr, \filt_q0, \in_q0, 0
    EE.VSMULAS.S8.QACC.LD.INCP \filt_q1, \filter_ptr, \filt_q1, \in_q0, 1
    EE.VSMULAS.S8.QACC.LD.INCP \filt_q0, \filter_ptr, \filt_q0, \in_q0, 2
    EE.VSMULAS.S8.QACC.LD.INCP \filt_q1, \filter_ptr, \filt_q1, \in_q0, 3
    EE.VSMULAS.S8.QACC.LD.INCP \filt_q0, \filter_ptr, \filt_q0, \in_q0, 4
    EE.VSMULAS.S8.QACC.LD.INCP \filt_q1, \filter_ptr, \filt_q1, \in_q0, 5
    EE.VSMULAS.S8.QACC.LD.INCP \filt_q0, \filter_ptr, \filt_q0, \in_q0, 6
    EE.VSMULAS.S8.QACC.LD.INCP \filt_q1, \filter_ptr, \filt_q1, \in_q0, 7
    EE.VSMULAS.S8.QACC.LD.INCP \filt_q0, \filter_ptr, \filt_q0, \in_q0, 8
    EE.VSMULAS.S8.QACC.LD.INCP \filt_q1, \filter_ptr, \filt_q1, \in_q0, 9
    EE.VSMULAS.S8.QACC.LD.INCP \filt_q0, \filter_ptr, \filt_q0, \in_q0, 10
    EE.VSMULAS.S8.QACC.LD.INCP \filt_q1, \filter_ptr, \filt_q1, \in_q0, 11
    EE.VSMULAS.S8.QACC.LD.INCP \filt_q0, \filter_ptr, \filt_q0, \in_q0, 12
    EE.VSMULAS.S8.QACC  \filt_q1, \in_q0, 13
    EE.VSMULAS.S8.QACC  \filt_q0, \in_q0, 14
    j 16f

12:
    EE.VSMULAS.S8.QACC.LD.INCP \filt_q0, \filter_ptr, \filt_q0, \in_q0, 0
    EE.VSMULAS.S8.QACC.LD.INCP \filt_q1, \filter_ptr, \filt_q1, \in_q0, 1
    EE.VSMULAS.S8.QACC.LD.INCP \filt_q0, \filter_ptr, \filt_q0, \in_q0, 2
    EE.VSMULAS.S8.QACC.LD.INCP \filt_q1, \filter_ptr, \filt_q1, \in_q0, 3
    EE.VSMULAS.S8.QACC.LD.INCP \filt_q0, \filter_ptr, \filt_q0, \in_q0, 4
    EE.VSMULAS.S8.QACC.LD.INCP \filt_q1, \filter_ptr, \filt_q1, \in_q0, 5
    EE.VSMULAS.S8.QACC.LD.INCP \filt_q0, \filter_ptr, \filt_q0, \in_q0, 6
    EE.VSMULAS.S8.QACC.LD.INCP \filt_q1, \filter_ptr, \filt_q1, \in_q0, 7
    EE.VSMULAS.S8.QACC.LD.INCP \filt_q0, \filter_ptr, \filt_q0, \in_q0, 8
    EE.VSMULAS.S8.QACC.LD.INCP \filt_q1, \filter_ptr, \filt_q1, \in_q0, 9
    EE.VSMULAS.S8.QACC  \filt_q0, \in_q0, 10
    EE.VSMULAS.S8.QACC  \filt_q1, \in_q0, 11
    j 16f

11:
    bbsi \c_remainder, 1, 96f
    j 9f
96:
    bbsi \c_remainder, 0, 97f
    j 10f
97:
    EE.VSMULAS.S8.QACC.LD.INCP \filt_q0, \filter_ptr, \filt_q0, \in_q0, 0
    EE.VSMULAS.S8.QACC.LD.INCP \filt_q1, \filter_ptr, \filt_q1, \in_q0, 1
    EE.VSMULAS.S8.QACC.LD.INCP \filt_q0, \filter_ptr, \filt_q0, \in_q0, 2
    EE.VSMULAS.S8.QACC.LD.INCP \filt_q1, \filter_ptr, \filt_q1, \in_q0, 3
    EE.VSMULAS.S8.QACC.LD.INCP \filt_q0, \filter_ptr, \filt_q0, \in_q0, 4
    EE.VSMULAS.S8.QACC.LD.INCP \filt_q1, \filter_ptr, \filt_q1, \in_q0, 5
    EE.VSMULAS.S8.QACC.LD.INCP \filt_q0, \filter_ptr, \filt_q0, \in_q0, 6
    EE.VSMULAS.S8.QACC.LD.INCP \filt_q1, \filter_ptr, \filt_q1, \in_q0, 7
    EE.VSMULAS.S8.QACC.LD.INCP \filt_q0, \filter_ptr, \filt_q0, \in_q0, 8
    EE.VSMULAS.S8.QACC  \filt_q1, \in_q0, 9
    EE.VSMULAS.S8.QACC  \filt_q0, \in_q0, 10
    j 16f

10:
    EE.VSMULAS.S8.QACC.LD.INCP \filt_q0, \filter_ptr, \filt_q0, \in_q0, 0
    EE.VSMULAS.S8.QACC.LD.INCP \filt_q1, \filter_ptr, \filt_q1, \in_q0, 1
    EE.VSMULAS.S8.QACC.LD.INCP \filt_q0, \filter_ptr, \filt_q0, \in_q0, 2
    EE.VSMULAS.S8.QACC.LD.INCP \filt_q1, \filter_ptr, \filt_q1, \in_q0, 3
    EE.VSMULAS.S8.QACC.LD.INCP \filt_q0, \filter_ptr, \filt_q0, \in_q0, 4
    EE.VSMULAS.S8.QACC.LD.INCP \filt_q1, \filter_ptr, \filt_q1, \in_q0, 5
    EE.VSMULAS.S8.QACC.LD.INCP \filt_q0, \filter_ptr, \filt_q0, \in_q0, 6
    EE.VSMULAS.S8.QACC.LD.INCP \filt_q1, \filter_ptr, \filt_q1, \in_q0, 7
    EE.VSMULAS.S8.QACC  \filt_q0, \in_q0, 8
    EE.VSMULAS.S8.QACC  \filt_q1, \in_q0, 9
    j 16f

9:
    bbsi \c_remainder, 0, 98f
    j 8f
98:
    EE.VSMULAS.S8.QACC.LD.INCP \filt_q0, \filter_ptr, \filt_q0, \in_q0, 0
    EE.VSMULAS.S8.QACC.LD.INCP \filt_q1, \filter_ptr, \filt_q1, \in_q0, 1
    EE.VSMULAS.S8.QACC.LD.INCP \filt_q0, \filter_ptr, \filt_q0, \in_q0, 2
    EE.VSMULAS.S8.QACC.LD.INCP \filt_q1, \filter_ptr, \filt_q1, \in_q0, 3
    EE.VSMULAS.S8.QACC.LD.INCP \filt_q0, \filter_ptr, \filt_q0, \in_q0, 4
    EE.VSMULAS.S8.QACC.LD.INCP \filt_q1, \filter_ptr, \filt_q1, \in_q0, 5
    EE.VSMULAS.S8.QACC.LD.INCP \filt_q0, \filter_ptr, \filt_q0, \in_q0, 6
    EE.VSMULAS.S8.QACC  \filt_q1, \in_q0, 7
    EE.VSMULAS.S8.QACC  \filt_q0, \in_q0, 8
    j 16f

8:
    EE.VSMULAS.S8.QACC.LD.INCP \filt_q0, \filter_ptr, \filt_q0, \in_q0, 0
    EE.VSMULAS.S8.QACC.LD.INCP \filt_q1, \filter_ptr, \filt_q1, \in_q0, 1
    EE.VSMULAS.S8.QACC.LD.INCP \filt_q0, \filter_ptr, \filt_q0, \in_q0, 2
    EE.VSMULAS.S8.QACC.LD.INCP \filt_q1, \filter_ptr, \filt_q1, \in_q0, 3
    EE.VSMULAS.S8.QACC.LD.INCP \filt_q0, \filter_ptr, \filt_q0, \in_q0, 4
    EE.VSMULAS.S8.QACC.LD.INCP \filt_q1, \filter_ptr, \filt_q1, \in_q0, 5
    EE.VSMULAS.S8.QACC  \filt_q0, \in_q0, 6
    EE.VSMULAS.S8.QACC  \filt_q1, \in_q0, 7
    j 16f

7:  # remainder < 8 cascade (local to unaligned_c)
    bbci \c_remainder, 2, 2f

    EE.VLD.128.IP \filt_q1, \filter_ptr, 16
    bbci \c_remainder, 1, 5f
    bbci \c_remainder, 0, 6f
    EE.VSMULAS.S8.QACC.LD.INCP \filt_q0, \filter_ptr, \filt_q0, \in_q0, 0
    EE.VSMULAS.S8.QACC.LD.INCP \filt_q1, \filter_ptr, \filt_q1, \in_q0, 1
    EE.VSMULAS.S8.QACC.LD.INCP \filt_q0, \filter_ptr, \filt_q0, \in_q0, 2
    EE.VSMULAS.S8.QACC.LD.INCP \filt_q1, \filter_ptr, \filt_q1, \in_q0, 3
    EE.VSMULAS.S8.QACC.LD.INCP \filt_q0, \filter_ptr, \filt_q0, \in_q0, 4
    EE.VSMULAS.S8.QACC  \filt_q1, \in_q0, 5
    EE.VSMULAS.S8.QACC  \filt_q0, \in_q0, 6
    j 16f

6:
    EE.VSMULAS.S8.QACC.LD.INCP \filt_q0, \filter_ptr, \filt_q0, \in_q0, 0
    EE.VSMULAS.S8.QACC.LD.INCP \filt_q1, \filter_ptr, \filt_q1, \in_q0, 1
    EE.VSMULAS.S8.QACC.LD.INCP \filt_q0, \filter_ptr, \filt_q0, \in_q0, 2
    EE.VSMULAS.S8.QACC.LD.INCP \filt_q1, \filter_ptr, \filt_q1, \in_q0, 3
    EE.VSMULAS.S8.QACC  \filt_q0, \in_q0, 4
    EE.VSMULAS.S8.QACC  \filt_q1, \in_q0, 5
    j 16f

5:
    bbci \c_remainder, 0, 4f
    # remainder == 5
    EE.VSMULAS.S8.QACC.LD.INCP \filt_q0, \filter_ptr, \filt_q0, \in_q0, 0
    EE.VSMULAS.S8.QACC.LD.INCP \filt_q1, \filter_ptr, \filt_q1, \in_q0, 1
    EE.VSMULAS.S8.QACC.LD.INCP \filt_q0, \filter_ptr, \filt_q0, \in_q0, 2
    EE.VSMULAS.S8.QACC  \filt_q1, \in_q0, 3
    EE.VSMULAS.S8.QACC  \filt_q0, \in_q0, 4
    j 16f

4:
    # remainder == 4
    EE.VSMULAS.S8.QACC.LD.INCP \filt_q0, \filter_ptr, \filt_q0, \in_q0, 0
    EE.VSMULAS.S8.QACC.LD.INCP \filt_q1, \filter_ptr, \filt_q1, \in_q0, 1
    EE.VSMULAS.S8.QACC  \filt_q0, \in_q0, 2
    EE.VSMULAS.S8.QACC  \filt_q1, \in_q0, 3
    j 16f

2:
    bbci \c_remainder, 1, 1f

    EE.VLD.128.IP \filt_q1, \filter_ptr, 16

    bbci \c_remainder, 0, 2f
    # remainder == 3
    EE.VSMULAS.S8.QACC.LD.INCP \filt_q0, \filter_ptr, \filt_q0, \in_q0, 0
    EE.VSMULAS.S8.QACC  \filt_q1, \in_q0, 1
    EE.VSMULAS.S8.QACC  \filt_q0, \in_q0, 2
    j 16f

2:
    # remainder == 2
    EE.VSMULAS.S8.QACC  \filt_q0, \in_q0, 0
    EE.VSMULAS.S8.QACC  \filt_q1, \in_q0, 1
    j 16f

1:
    # remainder == 1
    EE.VSMULAS.S8.QACC  \filt_q0, \in_q0, 0

16:
3:
    .endm

# OC remainder: ACCX-based accumulation for OC tail (1..15).
# This macro is an ACCX port of esp-dl's `tie728_s8_conv2d_1_1_c_n_remainder`
# used in the unaligned 11cn path. It:
#   - Uses unaligned loads (USAR.IP/USAR.XP) for both input and filter.
#   - Merges front/back aligned vectors with SRC.Q(.QUP) across IC blocks.
#   - Handles IC remainders via XP + SLCXXP, zeroing inactive lanes.
#   - Accumulates the full dot product into ACCX and finally extracts a
#     scalar i32 into `acc` via `EE.SRS.ACCX` with shift=0. The scalar
#     per-layer/per-channel epilogues then apply mac_shift and rounding,
#     which is a deliberate deviation from esp-dl (where SRS applies the
#     shift directly).
#
# Arguments:
#   acc         : GPR that will receive the accumulated dot product
#   input_ptr   : GPR pointer to start of this pixel's C dimension
#   filter_ptr  : GPR pointer to start of this OC's weights over C
#   c_div_x_1   : GPR, (ic/16) - 1, may be <0 when ic < 16
#   c_remainder : GPR, ic % 16 in 0..15 (esp-dl ArgsType::c_remainder)
#   tmp0/tmp1   : scratch GPRs (used for SAR save/restore and shifts)
#
.macro tie728_s8_conv2d_1_1_c_n_remainder acc input_ptr filter_ptr c_div_x_1 c_remainder tmp0 tmp1
    # Local vector naming:
    #   q0: input_front_aligned
    #   q1: input_back_aligned
    #   q2: filter_front_aligned
    #   q3: filter_back_aligned
    #   q4: merged input
    #   q5: merged filter

    EE.ZERO.ACCX

    # Large-IC path: at least one full 16-channel block.
    # Avoid blti immediate form; use bgez + long jump trampoline.
    bgez \c_div_x_1, 90f
    j 17f
90:

    # Prime front/back aligned streams for input and filter.
    EE.LD.128.USAR.IP q0, \input_ptr, 16   # input_front_aligned
    EE.LD.128.USAR.IP q2, \filter_ptr, 16  # filter_front_aligned

    # Safety check for aligned end-of-buffer read (mirroring unaligned_c)
    # If aligned (SAR=0) AND last block (c_div_x_1=0) AND no remainder (c_remainder=0),
    # then we don't need q1 and loading it might fault.
    rsr.sar \tmp0
    bnez \tmp0, 200f
    bnez \c_div_x_1, 200f
    bnez \c_remainder, 200f
    
    # Skip load, just advance pointer
    addi \input_ptr, \input_ptr, 16
    j 201f
    
200:
    EE.LD.128.USAR.IP q1, \input_ptr, 16   # input_back_aligned
201:

    # For every extra full 16-channel block, merge front/back and stream
    # filter via USAR, mirroring esp-dl's pattern but accumulating into ACCX.
    loopgtz \c_div_x_1, 0f
        EE.SRC.Q.QUP q4, q0, q1

        EE.LD.128.USAR.IP q3, \filter_ptr, 16
        EE.SRC.Q.QUP q5, q2, q3

        EE.LD.128.USAR.IP q1, \input_ptr, 16
        EE.VMULAS.S8.ACCX q5, q4
0:
    # Final full 16-channel block for input and filter.
    EE.SRC.Q.QUP q4, q0, q1

    # Land input_ptr at the end of C and align the filter stream using
    # USAR.XP over c_remainder. This mirrors esp-dl's use of XP to build
    # the last filter vector and then adjust input_ptr to the end of C.
    addi \input_ptr, \input_ptr, -16

    EE.LD.128.USAR.XP q3, \filter_ptr, \c_remainder
    add  \input_ptr, \input_ptr, \c_remainder
    EE.SRC.Q.QUP q5, q2, q3

    EE.VMULAS.S8.ACCX q5, q4

    # When there is no IC remainder, we are done: all channels have been
    # covered in ACCX at this point.
    beqz \c_remainder, 16f

    # Remainder in filter: esp-dl uses an additional USAR+SRC.Q path to
    # synthesize a vector whose low lanes contain the tail and whose high
    # lanes are zero after a left-shift. We reproduce that behavior using
    # a plain VLD+SRC.Q (as in their current code) and the SLCXXP shift.
    EE.VLD.128.IP q3, \filter_ptr, 0
    EE.SRC.Q q5, q2, q3

    # Remainder in input: reconstruct from the latched front/back pair using
    # a plain VLD (avoid SAR read/write under LLVM's assembler), then merge
    # with the front-aligned vector to produce the proper lane ordering.
    EE.VLD.128.IP q1, \input_ptr, 0
    EE.SRC.Q q4, q0, q1

    # Shift left by (15 - c_remainder) lanes so that only the active
    # remainder lanes contribute to ACCX, with the rest forced to zero.
    movi \tmp1, 15
    sub  \tmp1, \tmp1, \c_remainder
    movi \tmp0, 0
    # Use even-index destination registers for 2Q operations to satisfy
    # hardware pairing requirements. Mask into q0/q2, then consume q2*q0.
    EE.SLCXXP.2Q q0, q4, \tmp1, \tmp0
    EE.SLCXXP.2Q q2, q5, \tmp1, \tmp0

    EE.VMULAS.S8.ACCX q2, q0
    j 16f

17: # Small-IC path: ic < 16, only a remainder block.
    # Here we use USAR.XP directly for both filter and input, then apply
    # the same SLCXXP-based lane masking so that only c_remainder lanes
    # participate in the dot product.
    EE.LD.128.USAR.XP q2, \filter_ptr, \c_remainder
    EE.VLD.128.IP     q3, \filter_ptr, 0
    EE.SRC.Q          q5, q2, q3

    EE.LD.128.USAR.XP q0, \input_ptr, \c_remainder
    EE.VLD.128.IP     q1, \input_ptr, 0
    EE.SRC.Q          q4, q0, q1

    movi \tmp1, 15
    sub  \tmp1, \tmp1, \c_remainder
    movi \tmp0, 0
    EE.SLCXXP.2Q q0, q4, \tmp1, \tmp0
    EE.SLCXXP.2Q q2, q5, \tmp1, \tmp0

    EE.VMULAS.S8.ACCX q2, q0

    j 16f

7: # remainder < 8 cases (mirror esp-dl)
    # (intentionally removed from ACCX macro; see unaligned_c macro for this cascade)
16:
.endm

################################################################################################
# OC tail epilogue: ACCX + op_type (pl/pc × bias/no-bias × Linear/ReLU).
#
# This macro assumes ACCX already contains the accumulated dot product over C
# (from `tie728_s8_conv2d_1_1_c_n_remainder`). It:
#   - Extracts a scalar i32 via `EE.SRS.ACCX` with shift=0.
#   - Applies per-layer HALF_UP rounding and mac_shift, or per-channel
#     i16 bias/scale, using the same formulas as the scalar tail epilogues.
#   - Optionally applies ReLU based on `op_type` bit 2.
#
# Arguments:
#   op_type    : GPR, compact op-type bits (bit0=pc, bit1=bias, bit2=ReLU)
#   output_ptr : GPR, pointer to output byte (advanced by 1)
#   mac_shift  : GPR, per-layer mac_shift (ignored in per-channel path)
#   scale_base : GPR, base pointer for per-channel scales (pc_scales) for this tile
#   bias_base  : GPR, base pointer for per-channel bias16 (pc_bias_i16) for this tile (0 if none)
#   j          : GPR, byte offset into per-channel tables (2*j index)
#   acc        : GPR, receives scalar accumulator from ACCX
#   tmp0/tmp1  : GPRs, scratch
#
.macro tie728_s8_conv2d_1_1_tail_epilogue op_type output_ptr mac_shift scale_base bias_base j acc tmp0 tmp1
    # bit0 == 1 → per-channel; otherwise per-layer.
    bbci \op_type, 0, 0f

    ########################################################################
    # Per-channel path (PER_OC_W_EXP set)
    ########################################################################
    # Here `acc` is already a scalar accumulator in the accumulator exponent
    # domain (caller performed SRS.ACCX). We now apply the same per-channel
    # i16 bias/scale formula used in the scalar tail epilogue.
    # y = (((acc>>4) + bias16[j]) * scale16[j]) >> 11; saturate to i8
    srai \acc, \acc, 4
    beqz \bias_base, 4f
    add  \tmp0, \bias_base, \j
    l16si \tmp0, \tmp0, 0
    add  \acc, \acc, \tmp0
4:
    add  \tmp1, \scale_base, \j
    l16si \tmp1, \tmp1, 0
    mull \acc, \acc, \tmp1
    srai \acc, \acc, 11
    # Saturate to i8
    movi \tmp0, 127
    min  \acc, \acc, \tmp0
    movi \tmp0, -128
    max  \acc, \acc, \tmp0
    # Optional ReLU (bit2)
    bbci \op_type, 2, 5f
    movi \tmp0, 0
    blt  \acc, \tmp0, 6f
    j 5f
6:
    movi \acc, 0
5:
    s8i  \acc, \output_ptr, 0
    addi \output_ptr, \output_ptr, 1
    j 9f

0:  #######################################################################
    # Per-layer path (PER_OC_W_EXP clear)
    #######################################################################
    # Here `acc` is already a scalar accumulator in the accumulator exponent
    # domain (caller performed SRS.ACCX and, if needed, added per-layer
    # bias). We now apply HALF_UP rounding and optional ReLU.
    #
    # HALF_UP rounding: add +/- 2^(s-1) based on sign, then arithmetic >> s
    # Assumes mac_shift >= 0 (per-layer path selection).
    beqz \mac_shift, 1f
    addi \tmp0, \mac_shift, -1
    movi \tmp1, 1
    ssl  \tmp0
    sll  \tmp1, \tmp1            # tmp1 = 1<<(s-1)
    movi \tmp0, 0
    blt  \acc, \tmp0, 2f          # acc < 0 → subtract offset
    add  \acc, \acc, \tmp1
    j 3f
2:
    sub  \acc, \acc, \tmp1
3:
    ssr  \mac_shift
    sra  \acc, \acc
1:
    # Saturate to i8
    movi \tmp0, 127
    min  \acc, \acc, \tmp0
    movi \tmp0, -128
    max  \acc, \acc, \tmp0
    # Optional ReLU (bit2)
    bbci \op_type, 2, 7f
    movi \tmp0, 0
    blt  \acc, \tmp0, 8f
    j 7f
8:
    movi \acc, 0
7:
    s8i \acc, \output_ptr, 0
    addi \output_ptr, \output_ptr, 1
9:
.endm

# Small helpers for unaligned stores based on esp-dl macros
# dl_tie728_s8_unaligned_store0: spill 4 dwords from vector and store via s32i
.macro dl_tie728_s8_unaligned_store0 output_v, output_ptr, tmp32
    EE.MOVI.32.A \output_v, \tmp32, 0
    s32i \tmp32, \output_ptr, 0
    EE.MOVI.32.A \output_v, \tmp32, 1
    s32i \tmp32, \output_ptr, 4
    EE.MOVI.32.A \output_v, \tmp32, 2
    s32i \tmp32, \output_ptr, 8
    EE.MOVI.32.A \output_v, \tmp32, 3
    s32i \tmp32, \output_ptr, 12
    addi \output_ptr, \output_ptr, 16
.endm

# dl_tie728_s8_unaligned_store1: store low/high 64-bit halves directly
.macro dl_tie728_s8_unaligned_store1 output_v, output_ptr
    EE.VST.L.64.IP \output_v, \output_ptr, 8
    EE.VST.H.64.IP \output_v, \output_ptr, 8
.endm

# Per-channel result with bias (bias_ptr advances by 32B)
.macro tie728_s8_conv2d_per_channel_with_bias_result output_v scale_q scale_factor bias_ptr tmp tmp_q1
    movi \tmp, 4
    EE.SRCMB.S16.QACC \output_v, \tmp, 0
    mov \tmp, a1
    EE.ST.QACC_L.L.128.IP \tmp, 16
    EE.ST.QACC_L.H.32.IP \tmp, 16
    EE.ST.QACC_H.L.128.IP \tmp, 16
    EE.ST.QACC_H.H.32.IP \tmp, 4
    movi \tmp, 20
    EE.SRCMB.S16.QACC \tmp_q1, \tmp, 0
    addi \tmp, a1, 5
    l16si \tmp, \tmp, 0
    s16i \tmp, a1, 2
    addi \tmp, a1, 10
    l16si \tmp, \tmp, 0
    s16i \tmp, a1, 4
    addi \tmp, a1, 15
    l16si \tmp, \tmp, 0
    s16i \tmp, a1, 6
    addi \tmp, a1, 32
    l16si \tmp, \tmp, 0
    s16i \tmp, a1, 8
    addi \tmp, a1, 37
    l16si \tmp, \tmp, 0
    s16i \tmp, a1, 10
    addi \tmp, a1, 42
    l16si \tmp, \tmp, 0
    s16i \tmp, a1, 12
    addi \tmp, a1, 47
    l16si \tmp, \tmp, 0
    s16i \tmp, a1, 14
    EE.VLD.128.IP \output_v, a1, 0
    EE.VZIP.16 \output_v, \tmp_q1
    EE.VLD.128.IP \scale_q, \bias_ptr, 16
    movi \tmp, 11
    EE.VADDS.S16 \output_v, \output_v, \scale_q
    EE.VLD.128.IP \scale_q, \bias_ptr, 16
    ssr \tmp
    EE.VADDS.S16 \tmp_q1, \tmp_q1, \scale_q
    EE.VLD.128.IP \scale_q, \scale_factor, 16
    movi \tmp, 0
    EE.VMUL.S16 \output_v, \output_v, \scale_q
    EE.VLD.128.IP \scale_q, \scale_factor, 16
    EE.MOV.S16.QACC \output_v
    EE.VMUL.S16 \tmp_q1, \tmp_q1, \scale_q
    EE.SRCMB.S8.QACC \output_v, \tmp, 0
    EE.MOV.S16.QACC \tmp_q1
    EE.SRCMB.S8.QACC \tmp_q1, \tmp, 0
    EE.VUNZIP.8 \output_v, \tmp_q1
.endm

################################################################################################
# 3x3 Convolution Macro (tie728_s8_conv2d_33c16)
#
# Performs a 3x3 convolution by calling the 1x1 microkernel 9 times.
# It adjusts the input pointer using dilation offsets between calls.
#
# Arguments:
#   input_v0          : Vector register for input data (clobbered)
#   input_ptr         : GPR holding input pointer (modified)
#   filter_v0         : Vector register for filter data (clobbered)
#   filter_v1         : Vector register for filter data (clobbered)
#   filter_ptr        : GPR holding filter pointer (modified)
#   c_div_x_1         : GPR holding (ic / 16) - 1
#   dilation_x_offset : GPR holding offset for X dilation
#   dilation_y_offset : GPR holding offset for Y dilation
#
# Clobbers:
#   input_v0, filter_v0, filter_v1
#   input_ptr (advanced by dilation offsets)
#   filter_ptr (advanced by 9 * filter_size)
#   QACC (accumulates results)
################################################################################################
.macro tie728_s8_conv2d_33c16 input_v0 input_ptr filter_v0 filter_v1 filter_ptr c_div_x_1 dilation_x_offset dilation_y_offset
    # Row 0
    tie728_s8_conv2d_11c16 \input_v0, \input_ptr, \filter_v0, \filter_v1, \filter_ptr, \c_div_x_1
    add \input_ptr, \input_ptr, \dilation_x_offset

    tie728_s8_conv2d_11c16 \input_v0, \input_ptr, \filter_v0, \filter_v1, \filter_ptr, \c_div_x_1
    add \input_ptr, \input_ptr, \dilation_x_offset

    tie728_s8_conv2d_11c16 \input_v0, \input_ptr, \filter_v0, \filter_v1, \filter_ptr, \c_div_x_1
    add \input_ptr, \input_ptr, \dilation_y_offset

    # Row 1
    tie728_s8_conv2d_11c16 \input_v0, \input_ptr, \filter_v0, \filter_v1, \filter_ptr, \c_div_x_1
    add \input_ptr, \input_ptr, \dilation_x_offset

    tie728_s8_conv2d_11c16 \input_v0, \input_ptr, \filter_v0, \filter_v1, \filter_ptr, \c_div_x_1
    add \input_ptr, \input_ptr, \dilation_x_offset

    tie728_s8_conv2d_11c16 \input_v0, \input_ptr, \filter_v0, \filter_v1, \filter_ptr, \c_div_x_1
    add \input_ptr, \input_ptr, \dilation_y_offset

    # Row 2
    tie728_s8_conv2d_11c16 \input_v0, \input_ptr, \filter_v0, \filter_v1, \filter_ptr, \c_div_x_1
    add \input_ptr, \input_ptr, \dilation_x_offset

    tie728_s8_conv2d_11c16 \input_v0, \input_ptr, \filter_v0, \filter_v1, \filter_ptr, \c_div_x_1
    add \input_ptr, \input_ptr, \dilation_x_offset

    tie728_s8_conv2d_11c16 \input_v0, \input_ptr, \filter_v0, \filter_v1, \filter_ptr, \c_div_x_1
    # No final pointer adjustment needed for the last element in the block if we don't reuse input_ptr immediately
    # But for consistency with esp-dl, we leave it pointing to the next logical position if needed,
    # or just let the caller handle reset.
    # esp-dl does NOT add dilation_y_offset at the very end of the macro.
.endm

################################################################################################
# 3x3 Unaligned Convolution Macro (tie728_s8_conv2d_3_3_unaligned_c)
#
# Performs a 3x3 unaligned convolution by calling the 1x1 unaligned microkernel 9 times.
# It adjusts the input pointer using dilation offsets between calls.
#
# Arguments:
#   input_v0          : Vector register for input data (clobbered)
#   input_front       : Vector register for front aligned input (clobbered)
#   input_back        : Vector register for back aligned input (clobbered)
#   input_ptr         : GPR holding input pointer (modified)
#   filter_v0         : Vector register for filter data (clobbered)
#   filter_v1         : Vector register for filter data (clobbered)
#   filter_ptr        : GPR holding filter pointer (modified)
#   c_div_x_1         : GPR holding (ic / 16) - 1
#   dil_x             : GPR holding offset for X dilation
#   dil_y             : GPR holding offset for Y dilation
#   rem_c             : GPR holding ic % 16
#   sar               : GPR holding input SAR byte
#
################################################################################################
.macro tie728_s8_conv2d_3_3_unaligned_c input_v0 input_front input_back input_ptr filter_v0 filter_v1 filter_ptr c_div_x_1 dil_x dil_y rem_c sar args
    # 1
    tie728_s8_conv2d_1_1_unaligned_c \input_ptr, \filter_ptr, \c_div_x_1, \rem_c, \input_v0, \input_front, \filter_v0, \filter_v1, \sar, \args
    add \input_ptr, \input_ptr, \dil_x

    # 2
    tie728_s8_conv2d_1_1_unaligned_c \input_ptr, \filter_ptr, \c_div_x_1, \rem_c, \input_v0, \input_front, \filter_v0, \filter_v1, \sar, \args
    add \input_ptr, \input_ptr, \dil_x

    # 3
    tie728_s8_conv2d_1_1_unaligned_c \input_ptr, \filter_ptr, \c_div_x_1, \rem_c, \input_v0, \input_front, \filter_v0, \filter_v1, \sar, \args
    add \input_ptr, \input_ptr, \dil_y

    # 4
    tie728_s8_conv2d_1_1_unaligned_c \input_ptr, \filter_ptr, \c_div_x_1, \rem_c, \input_v0, \input_front, \filter_v0, \filter_v1, \sar, \args
    add \input_ptr, \input_ptr, \dil_x

    # 5
    tie728_s8_conv2d_1_1_unaligned_c \input_ptr, \filter_ptr, \c_div_x_1, \rem_c, \input_v0, \input_front, \filter_v0, \filter_v1, \sar, \args
    add \input_ptr, \input_ptr, \dil_x

    # 6
    tie728_s8_conv2d_1_1_unaligned_c \input_ptr, \filter_ptr, \c_div_x_1, \rem_c, \input_v0, \input_front, \filter_v0, \filter_v1, \sar, \args
    add \input_ptr, \input_ptr, \dil_y

    # 7
    tie728_s8_conv2d_1_1_unaligned_c \input_ptr, \filter_ptr, \c_div_x_1, \rem_c, \input_v0, \input_front, \filter_v0, \filter_v1, \sar, \args
    add \input_ptr, \input_ptr, \dil_x

    # 8
    tie728_s8_conv2d_1_1_unaligned_c \input_ptr, \filter_ptr, \c_div_x_1, \rem_c, \input_v0, \input_front, \filter_v0, \filter_v1, \sar, \args
    add \input_ptr, \input_ptr, \dil_x

    # 9
    tie728_s8_conv2d_1_1_unaligned_c \input_ptr, \filter_ptr, \c_div_x_1, \rem_c, \input_v0, \input_front, \filter_v0, \filter_v1, \sar, \args
    # No final add needed, input_ptr is scratch in the loop
.endm

################################################################################################
# 1x1 Unaligned Remainder Accumulate (tie728_s8_conv2d_1_1_c_n_remainder_accumulate)
#
# Same as tie728_s8_conv2d_1_1_c_n_remainder but DOES NOT zero ACCX at the start.
# Used for accumulating partial results in 3x3 kernels.
################################################################################################
.macro tie728_s8_conv2d_1_1_c_n_remainder_accumulate acc input_ptr filter_ptr c_div_x_1 c_remainder tmp0 tmp1
    # Local vector naming:
    #   q0: input_front_aligned
    #   q1: input_back_aligned
    #   q2: filter_front_aligned
    #   q3: filter_back_aligned
    #   q4: merged input
    #   q5: merged filter

    # EE.ZERO.ACCX  <-- REMOVED

    # Large-IC path: at least one full 16-channel block.
    bgez \c_div_x_1, 90f
    j 17f
90:
    EE.LD.128.USAR.IP q0, \input_ptr, 16   # input_front_aligned
    EE.LD.128.USAR.IP q2, \filter_ptr, 16  # filter_front_aligned
    EE.LD.128.USAR.IP q1, \input_ptr, 16   # input_back_aligned

    loopgtz \c_div_x_1, 0f
        EE.SRC.Q.QUP q4, q0, q1
        EE.LD.128.USAR.IP q3, \filter_ptr, 16
        EE.SRC.Q.QUP q5, q2, q3
        EE.LD.128.USAR.IP q1, \input_ptr, 16
        EE.VMULAS.S8.ACCX q5, q4
0:
    EE.SRC.Q.QUP q4, q0, q1
    addi \input_ptr, \input_ptr, -16
    EE.LD.128.USAR.XP q3, \filter_ptr, \c_remainder
    add  \input_ptr, \input_ptr, \c_remainder
    EE.SRC.Q.QUP q5, q2, q3
    EE.VMULAS.S8.ACCX q5, q4

    beqz \c_remainder, 16f

    EE.VLD.128.IP q3, \filter_ptr, 0
    EE.SRC.Q q5, q2, q3
    EE.VLD.128.IP q1, \input_ptr, 0
    EE.SRC.Q q4, q0, q1

    movi \tmp1, 15
    sub  \tmp1, \tmp1, \c_remainder
    movi \tmp0, 0
    EE.SLCXXP.2Q q0, q4, \tmp1, \tmp0
    EE.SLCXXP.2Q q2, q5, \tmp1, \tmp0
    EE.VMULAS.S8.ACCX q2, q0
    j 16f

17: # Small-IC path
    EE.LD.128.USAR.XP q2, \filter_ptr, \c_remainder
    EE.VLD.128.IP     q3, \filter_ptr, 0
    EE.SRC.Q          q5, q2, q3
    EE.LD.128.USAR.XP q0, \input_ptr, \c_remainder
    EE.VLD.128.IP     q1, \input_ptr, 0
    EE.SRC.Q          q4, q0, q1

    movi \tmp1, 15
    sub  \tmp1, \tmp1, \c_remainder
    movi \tmp0, 0
    EE.SLCXXP.2Q q0, q4, \tmp1, \tmp0
    EE.SLCXXP.2Q q2, q5, \tmp1, \tmp0
    EE.VMULAS.S8.ACCX q2, q0
16:
.endm

# HWC16 macro (shared for HWCN kernels)
#
# Iterates over filter height/width, performing 1x1 convolutions (11c16)
# and advancing pointers with dilation offsets.
#
# Arguments:
#   input_v0: vector register (scratch)
#   input_ptr: GPR (modified)
#   filter_v0: vector register (scratch)
#   filter_v1: vector register (scratch)
#   filter_ptr: GPR (modified)
#   c_div_x_1: GPR (input_channel / 16 - 1)
#   dilation_x_offset: GPR
#   dilation_y_offset: GPR
#   filter_h: GPR (scratch, modified)
#   filter_w: GPR (scratch, modified)
#   filter_offset_q: vector register (scratch)
#   args: GPR (pointer to KernelArgs)
#
# Clobbers:
#   QACC (via 11c16)
#   input_ptr, filter_ptr, filter_h, filter_w
#   input_v0, filter_v0, filter_v1, filter_offset_q
#
.macro tie728_s8_conv2d_hwc16 input_v0 input_ptr filter_v0 filter_v1 filter_ptr c_div_x_1 dilation_x_offset dilation_y_offset filter_h filter_w filter_offset_q, args
    # Load filter_width (KA_kw is at offset 44)
    l32i \filter_w,          \args,  44  # KA_kw
    # Fix: blti range is small, use trampoline
    bgei \filter_w, 2, .L_hwc16_width_ge_2
    j 3f
.L_hwc16_width_ge_2:

    # Load filter_height (KA_kh is at offset 40)
    l32i \filter_h,          \args,  40  # KA_kh
    1:
        l32i \filter_w,          \args,  44  # KA_kw
        2:
            tie728_s8_conv2d_11c16 \input_v0, \input_ptr, \filter_v0, \filter_v1, \filter_ptr, \c_div_x_1
            add \input_ptr, \input_ptr, \dilation_x_offset

            addi \filter_w, \filter_w, -1
        # Fix: bgei range is small, use trampoline
        blti \filter_w, 2, .L_hwc16_loop_next
        j 2b
.L_hwc16_loop_next:

        tie728_s8_conv2d_11c16 \input_v0, \input_ptr, \filter_v0, \filter_v1, \filter_ptr, \c_div_x_1

        EE.MOVI.32.A \filter_offset_q, \filter_w, 1
        add \filter_ptr, \filter_ptr, \filter_w
        add \input_ptr, \input_ptr, \dilation_y_offset

        addi \filter_h, \filter_h, -1
    bnez \filter_h, 1b
    j 5f

    # filter_w == 1
    3:
    l32i \filter_h,          \args,  40  # KA_kh
    EE.MOVI.32.A \filter_offset_q, \filter_w, 1
    4:
        tie728_s8_conv2d_11c16 \input_v0, \input_ptr, \filter_v0, \filter_v1, \filter_ptr, \c_div_x_1
        add \filter_ptr, \filter_ptr, \filter_w
        add \input_ptr, \input_ptr, \dilation_y_offset

        addi \filter_h, \filter_h, -1
    bnez \filter_h, 4b

    5:
    EE.MOVI.32.A \filter_offset_q, \filter_h, 2
    add \filter_ptr, \filter_ptr, \filter_h
.endm

