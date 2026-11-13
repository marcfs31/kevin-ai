/* lm_quant.h - the q8 quantiser of docs/PLAN.md, bit for bit. Original code.
 *
 * For a row-major [rows, in] matrix and each group of 32 consecutive inputs in a row:
 *   absmax = max |x| over the group (the inputs are already f32; bf16 widens exactly)
 *   scale  = (float)((double)absmax / 127.0)    double division, ONE rounding to f32
 *   q      = clamp(round_half_even((double)x / (double)scale), -127, 127)
 *   scale == 0  ->  all 32 q are 0
 * `in` must be a multiple of 32 and every input finite; either violation is an error,
 * never a fallback. A -0.0 input quantises to q = 0 like +0.0. */
#ifndef LM_QUANT_H
#define LM_QUANT_H

#include <stdint.h>

#include "core/lm_kernels.h" /* LM_Q8_GROUP */

enum {
    LM_Q8_OK            = 0,
    LM_Q8_ERR_GROUP     = 1, /* in is not a positive multiple of 32 */
    LM_Q8_ERR_NONFINITE = 2  /* a NaN or Inf input; *bad_index is its flat index */
};

/* w: rows*in floats; q: rows*in int8; scales: rows*(in/32) floats. On an error nothing
 * about q/scales is guaranteed and, for LM_Q8_ERR_NONFINITE, *bad_index (if non-NULL)
 * holds the flat element index of the first offender. */
int lm_q8_quantize(const float *w, int64_t rows, int64_t in, int8_t *q, float *scales,
                   int64_t *bad_index);

/* Inverse for tests and tooling: out[r*in + i] = q[r*in + i] * scales[r*(in/32) + i/32]. */
void lm_q8_dequantize(const int8_t *q, const float *scales, int64_t rows, int64_t in, float *out);

#endif /* LM_QUANT_H */
