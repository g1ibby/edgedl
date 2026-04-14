/*
 * TIE728 SIMD Radix-2 Complex FFT for ESP32-S3
 *
 * Ported from esp-dl dl_fft2r_fc32_aes3.S
 * SPDX-License-Identifier: Apache-2.0
 *
 * This implements the Cooley-Tukey radix-2 DIT FFT algorithm using
 * ESP32-S3's TIE728 floating-point instructions for improved performance.
 */

    .section .iram1, "ax"
    .align  4
    .global dl_tie728_fft2r_fc32
    .type   dl_tie728_fft2r_fc32, @function

/*
 * void dl_tie728_fft2r_fc32(float *data, int N, float *twiddle_table)
 *
 * Parameters:
 *   a2: data - pointer to interleaved complex float array [re0,im0,re1,im1,...]
 *              Modified in-place
 *   a3: N - FFT size (number of complex samples)
 *   a4: twiddle_table - pointer to precomputed [cos,sin] pairs
 *
 * Register allocation:
 *   a2  - data pointer (preserved)
 *   a3  - N (preserved)
 *   a4  - twiddle_table pointer
 *   a5  - unused
 *   a6  - N2 (N/2, main loop counter)
 *   a7  - ie (butterfly span)
 *   a8  - j (twiddle index loop counter)
 *   a9  - unused
 *   a10 - twiddle pointer for current j
 *   a11 - ia (butterfly base index)
 *   a12 - m (ia + N2)
 *   a13 - ia pointer (data + ia*8)
 *   a14 - m pointer (data + m*8)
 *
 * FPU registers:
 *   f0  - cos (twiddle real)
 *   f1  - sin (twiddle imag)
 *   f2  - data[2*ia] (real part)
 *   f3  - data[2*ia+1] (imag part)
 *   f4  - data[2*m] (real part)
 *   f5  - data[2*m+1] (imag part)
 *   f6  - re_temp
 *   f7  - im_temp
 *   f8  - data[2*m] result (real)
 *   f9  - data[2*m+1] result (imag)
 *   f10 - data[2*ia] result (real)
 *   f11 - data[2*ia+1] result (imag)
 */

dl_tie728_fft2r_fc32:
    entry   a1, 16

    // a2: data pointer
    // a3: N
    // a4: twiddle_table

    // a6 = N2 = N/2
    srli    a6, a3, 1
    // a7 = ie = 1
    movi.n  a7, 1

.Lfft2r_stage_loop:
    // Stage loop: for (N2 = N/2; N2 > 0; N2 >>= 1)
    beqz    a6, .Lfft2r_done

    // a8 = j = 0
    movi.n  a8, 0
    // a11 = ia = 0
    movi.n  a11, 0

.Lfft2r_twiddle_loop:
    // Twiddle loop: for (j = 0; j < ie; j++)
    bge     a8, a7, .Lfft2r_twiddle_done

    // Load twiddle factors: c = w[2*j], s = w[2*j+1]
    // a10 = &twiddle_table[j*2] = twiddle_table + j*8
    addx8   a10, a8, a4
    ee.ldf.64.ip    f1, f0, a10, 0  // f0=cos, f1=sin

    // m = ia + N2
    add.n   a12, a11, a6
    // a14 = &data[2*m] = data + m*8
    addx8   a14, a12, a2

    // Butterfly loop: for (i = 0; i < N2; i++)
    loopnez a6, .Lfft2r_butterfly_end

        // Load data[2*m], data[2*m+1]
        ee.ldf.64.ip    f5, f4, a14, 0  // f4=re[m], f5=im[m]

        // re_temp = c * data[2*m]
        mul.s   f6, f0, f4
        // im_temp = c * data[2*m+1]
        mul.s   f7, f0, f5

        // a13 = &data[2*ia] = data + ia*8
        addx8   a13, a11, a2
        // Load data[2*ia], data[2*ia+1]
        ee.ldf.64.ip    f3, f2, a13, 0  // f2=re[ia], f3=im[ia]

        // re_temp += s * data[2*m+1]
        madd.s  f6, f1, f5
        // im_temp -= s * data[2*m]
        msub.s  f7, f1, f4

        // ia++
        addi    a11, a11, 1
        // m = ia + N2
        add.n   a12, a11, a6

        // data[2*m] = data[2*ia] - re_temp
        sub.s   f8, f2, f6
        // data[2*m+1] = data[2*ia+1] - im_temp
        sub.s   f9, f3, f7
        // data[2*ia] = data[2*ia] + re_temp
        add.s   f10, f2, f6
        // data[2*ia+1] = data[2*ia+1] + im_temp
        add.s   f11, f3, f7

        // Store results
        ee.stf.64.ip    f9, f8, a14, 0
        // Update m pointer for next iteration
        addx8   a14, a12, a2
        ee.stf.64.ip    f11, f10, a13, 0

.Lfft2r_butterfly_end:
    // ia += N2 (skip to next group)
    add.n   a11, a11, a6
    // j++
    addi.n  a8, a8, 1
    // Continue twiddle loop
    j       .Lfft2r_twiddle_loop

.Lfft2r_twiddle_done:
    // ie <<= 1
    slli    a7, a7, 1
    // N2 >>= 1
    srli    a6, a6, 1
    // Continue stage loop
    j       .Lfft2r_stage_loop

.Lfft2r_done:
    retw

    .size   dl_tie728_fft2r_fc32, . - dl_tie728_fft2r_fc32
