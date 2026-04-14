/*
 * TIE728 SIMD Mel Spectrogram Operations for ESP32-S3
 *
 * SPDX-License-Identifier: Apache-2.0
 */

    .section .iram1, "ax"
    .align  4
    .global dl_tie728_apply_window_i16_f32
    .type   dl_tie728_apply_window_i16_f32, @function
    .global dl_tie728_power_spectrum_f32
    .type   dl_tie728_power_spectrum_f32, @function
    .global dl_tie728_mel_accumulate
    .type   dl_tie728_mel_accumulate, @function

/*
 * void dl_tie728_apply_window_i16_f32(const int16_t *wave, const float *window, float *out, int n)
 */
dl_tie728_apply_window_i16_f32:
    entry   a1, 16

    // f0 = 1.0 / 32768.0
    movi    a6, 0x38000000  // 3.0517578125e-5
    wfr     f0, a6

    srli    a6, a5, 1
    loopnez a6, .Lwindow_end
        l16si   a7, a2, 0
        l16si   a8, a2, 2
        
        float.s f1, a7, 0
        float.s f2, a8, 0

        mul.s   f1, f1, f0
        mul.s   f2, f2, f0

        ee.ldf.64.ip f4, f3, a3, 8
        mul.s   f1, f1, f3
        mul.s   f2, f2, f4

        ee.stf.64.ip f2, f1, a4, 8
        addi    a2, a2, 4
.Lwindow_end:
    retw

/*
 * void dl_tie728_power_spectrum_f32(const float *complex_in, float *power_out, int n_bins)
 */
/*
 * void dl_tie728_power_spectrum_f32(const float *complex_in, float *power_out, int n_bins)
 */
dl_tie728_power_spectrum_f32:
    entry   a1, 16
    
    // Check if n_bins >= 4
    movi    a6, 4
    blt     a4, a6, .Lpower_remainder_loop

    // Main loop (4x unrolled)
    srli    a7, a4, 2      // a7 = n_bins / 4
    loopnez a7, .Lpower_loop_end
        // Load 4 complex samples (8 floats)
        // f0=re0, f1=im0
        ee.ldf.64.ip f1, f0, a2, 8
        // f2=re1, f3=im1
        ee.ldf.64.ip f3, f2, a2, 8
        // f4=re2, f5=im2
        ee.ldf.64.ip f5, f4, a2, 8
        // f6=re3, f7=im3
        ee.ldf.64.ip f7, f6, a2, 8

        // Compute squares (interleaved for pipeline efficiency)
        mul.s   f0, f0, f0      // re0^2
        mul.s   f2, f2, f2      // re1^2
        mul.s   f4, f4, f4      // re2^2
        mul.s   f6, f6, f6      // re3^2

        madd.s  f0, f1, f1      // + im0^2
        madd.s  f2, f3, f3      // + im1^2
        madd.s  f4, f5, f5      // + im2^2
        madd.s  f6, f7, f7      // + im3^2

        // Store 4 results (power_out)
        ssi     f0, a3, 0
        ssi     f2, a3, 4
        ssi     f4, a3, 8
        ssi     f6, a3, 12
        addi    a3, a3, 16
.Lpower_loop_end:

    // Handle remainder
    movi    a6, 3
    and     a4, a4, a6     // n % 4

.Lpower_remainder_loop:
    loopnez a4, .Lpower_end
        ee.ldf.64.ip f2, f1, a2, 8
        mul.s   f1, f1, f1
        madd.s  f1, f2, f2
        ssi     f1, a3, 0
        addi    a3, a3, 4
.Lpower_end:
    retw

/*
 * float dl_tie728_mel_accumulate(const float *power, int count, float start_w, float w_step)
 *
 * Compute sum( power[i] * (start_w + i * w_step) ) for i = 0..count-1
 * 2x unrolled for better instruction pipelining.
 *
 * Parameters (Xtensa ABI: floats passed as bits in a-registers):
 *   a2: power (float*)
 *   a3: count (int)
 *   a4: start_w (float bits)
 *   a5: w_step (float bits)
 * Return:
 *   a2: accumulator (float bits)
 */
dl_tie728_mel_accumulate:
    entry   a1, 16

    // Convert float parameters from a-registers to f-registers
    wfr     f0, a4      // f0 = w (current weight)
    wfr     f1, a5      // f1 = w_step

    // Initialize accumulator f2 = 0
    movi    a6, 0
    wfr     f2, a6

    // Check count < 4
    movi    a6, 4
    blt     a3, a6, .Lacc_remainder_loop

    // Initialize parallel accumulators
    wfr     f6, a6      // acc1 = 0
    wfr     f10, a6     // acc2 = 0
    wfr     f12, a6     // acc3 = 0

    // Prepare weights for 4x unrolling
    // w0 = f0 (already set)
    // w1 = w0 + w_step
    add.s   f5, f0, f1
    // w2 = w1 + w_step
    add.s   f9, f5, f1
    // w3 = w2 + w_step
    add.s   f11, f9, f1
    
    // w_step_4 = w_step * 4
    add.s   f13, f1, f1   // 2*d
    add.s   f13, f13, f13 // 4*d

    // Main loop (4x unrolled)
    srli    a7, a3, 2     // count / 4
    loopnez a7, .Lacc_loop_4x_end
        // Load p0, p1
        lsi     f3, a2, 0   // f3=p0
        lsi     f4, a2, 4   // f4=p1
        // Load p2, p3
        lsi     f7, a2, 8   // f7=p2
        lsi     f8, a2, 12  // f8=p3
        addi    a2, a2, 16

        madd.s  f2, f0, f3   // acc0 += w0 * p0
        madd.s  f6, f5, f4   // acc1 += w1 * p1
        madd.s  f10, f9, f7  // acc2 += w2 * p2
        madd.s  f12, f11, f8 // acc3 += w3 * p3

        add.s   f0, f0, f13  // w0 += 4d
        add.s   f5, f5, f13  // w1 += 4d
        add.s   f9, f9, f13  // w2 += 4d
        add.s   f11, f11, f13 // w3 += 4d
.Lacc_loop_4x_end:

    // Sum accumulators into f2
    add.s   f2, f2, f6
    add.s   f2, f2, f10
    add.s   f2, f2, f12

    // Handle remainder
    movi    a6, 3
    and     a3, a3, a6

.Lacc_remainder_loop:
    loopnez a3, .Lacc_done
        lsi     f3, a2, 0
        addi    a2, a2, 4
        madd.s  f2, f0, f3
        add.s   f0, f0, f1

.Lacc_done:
    // Return acc in a2 (Xtensa returns float as bits in a2)
    rfr     a2, f2
    retw

    .global dl_tie728_normalize_f32
    .type   dl_tie728_normalize_f32, @function
    .global dl_tie728_apply_window_i16_f32_x4
    .type   dl_tie728_apply_window_i16_f32_x4, @function

/*
 * void dl_tie728_normalize_f32(float *data, int count, float mean, float inv_std)
 * Computes: data[i] = (data[i] - mean) * inv_std
 *
 * Parameters (Xtensa ABI: floats passed as bits in a-registers):
 *   a2: data pointer
 *   a3: count
 *   a4: mean (float bits)
 *   a5: inv_std (float bits)
 */
dl_tie728_normalize_f32:
    entry   a1, 16

    // Convert float parameters from a-registers to f-registers
    wfr     f0, a4      // f0 = mean
    wfr     f1, a5      // f1 = inv_std

    // Process 2 elements per iteration
    srli    a4, a3, 1
    loopnez a4, .Lnorm_end
        // Load 2 floats using scalar loads (more reliable)
        lsi     f2, a2, 0            // f2 = data[0]
        lsi     f3, a2, 4            // f3 = data[1]
        sub.s   f2, f2, f0           // f2 = val0 - mean
        sub.s   f3, f3, f0           // f3 = val1 - mean
        mul.s   f2, f2, f1           // f2 *= inv_std
        mul.s   f3, f3, f1           // f3 *= inv_std
        ssi     f2, a2, 0            // Store data[0]
        ssi     f3, a2, 4            // Store data[1]
        addi    a2, a2, 8            // Increment pointer
.Lnorm_end:

    // Handle odd element
    bbci    a3, 0, .Lnorm_done
        lsi     f2, a2, 0
        sub.s   f2, f2, f0
        mul.s   f2, f2, f1
        ssi     f2, a2, 0
.Lnorm_done:
    retw

/*
 * void dl_tie728_apply_window_i16_f32_x4(const int16_t *wave, const float *window, float *out, int n)
 * 4x unrolled windowing: converts i16 to f32 and applies Hann window
 *
 * Parameters:
 *   a2: wave pointer (i16)
 *   a3: window pointer (f32)
 *   a4: out pointer (f32)
 *   a5: n (must be multiple of 4)
 */
dl_tie728_apply_window_i16_f32_x4:
    entry   a1, 16

    // f0 = 1.0 / 32768.0
    movi    a6, 0x38000000
    wfr     f0, a6

    // Process 4 samples per iteration (n/4 iterations)
    srli    a6, a5, 2
    loopnez a6, .Lwindow4_end
        // Load 4 i16 samples
        l16si   a7, a2, 0
        l16si   a8, a2, 2
        l16si   a9, a2, 4
        l16si   a10, a2, 6

        // Convert to float
        float.s f1, a7, 0
        float.s f2, a8, 0
        float.s f3, a9, 0
        float.s f4, a10, 0

        // Scale by 1/32768
        mul.s   f1, f1, f0
        mul.s   f2, f2, f0
        mul.s   f3, f3, f0
        mul.s   f4, f4, f0

        // Load 4 window coefficients
        ee.ldf.64.ip f6, f5, a3, 8
        ee.ldf.64.ip f8, f7, a3, 8

        // Apply window
        mul.s   f1, f1, f5
        mul.s   f2, f2, f6
        mul.s   f3, f3, f7
        mul.s   f4, f4, f8

        // Store results
        ee.stf.64.ip f2, f1, a4, 8
        ee.stf.64.ip f4, f3, a4, 8

        addi    a2, a2, 8
.Lwindow4_end:
    retw

    .global dl_tie728_log_f32
    .type   dl_tie728_log_f32, @function

/*
 * void dl_tie728_log_f32(float *data, int count, float eps)
 * Computes log(max(x, eps)) in-place.
 */
dl_tie728_log_f32:
    entry   a1, 48

    // a2: data
    // a3: count
    // a4: eps

    wfr     f0, a4      // f0 = eps

    // Constants
    movi    a6, 0x3F800000 // 1.0
    wfr     f1, a6
    movi    a6, 0x3F317218 // LN2
    wfr     f2, a6
    movi    a6, 0x3EAAAAAB // 0.3333333
    wfr     f3, a6
    movi    a6, 0x3E4CCCCD // 0.2
    wfr     f4, a6
    movi    a6, 0x3E124925 // 0.1428571
    wfr     f5, a6
    movi    a6, 0x40000000 // 2.0
    wfr     f6, a6
    
    // Reciprocal approx constants
    movi    a6, 0x3F555555 // 0.8333333
    wfr     f7, a6
    movi    a6, 0x3E2AAAAB // 0.1666667
    wfr     f8, a6

    // Loop
    loopnez a3, .Llog_only_end
        lsi     f9, a2, 0       // x
        
        // x = max(x, eps)
        olt.s   b0, f9, f0
        movt.s  f9, f0, b0

        // fast_logf(x)
        rfr     a6, f9
        extui   a7, a6, 23, 8
        addi    a7, a7, -127
        float.s f10, a7, 0      // e

        movi    a10, 0x007FFFFF
        and     a6, a6, a10
        movi    a10, 0x3F800000
        or      a6, a6, a10
        wfr     f11, a6         // m

        // y = (m-1)/(m+1)
        // d = m + 1
        add.s   f12, f11, f1    // d
        
        // inv_d approx: 0.8333 - 0.1667 * d
        // f13 = 0.1667 * d
        mul.s   f13, f8, f12
        // f13 = 0.8333 - f13
        sub.s   f13, f7, f13    // x0
        
        // Newton step: x1 = x0 * (2 - d*x0)
        // f14 = d * x0
        mul.s   f14, f12, f13
        // f14 = 2 - f14
        sub.s   f14, f6, f14
        // x1 = x0 * f14
        mul.s   f13, f13, f14   // inv_d
        
        // y = (m - 1) * inv_d
        sub.s   f14, f11, f1    // m - 1
        mul.s   f12, f14, f13   // y

        // Polynomial
        mul.s   f13, f12, f12   // y2
        
        // p = 0.2 + y2 * 0.1428571
        mov.s   f14, f4
        madd.s  f14, f13, f5
        
        // p = 0.33 + y2 * p
        mov.s   f15, f3
        madd.s  f15, f13, f14
        
        // p = 1.0 + y2 * p
        mov.s   f14, f1
        madd.s  f14, f13, f15
        
        // ln_m = 2 * y * p
        mul.s   f14, f14, f12
        mul.s   f14, f14, f6
        
        // res = e * LN2 + ln_m
        madd.s  f14, f10, f2
        
        // Store
        ssi     f14, a2, 0
        
        addi    a2, a2, 4
.Llog_only_end:

    retw
