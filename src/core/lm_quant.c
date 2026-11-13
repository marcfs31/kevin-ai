/* lm_quant.c - see lm_quant.h. Original code. */
#include <math.h>

#include "core/lm_quant.h"

int lm_q8_quantize(const float *w, int64_t rows, int64_t in, int8_t *q, float *scales,
                   int64_t *bad_index)
{
    if (in <= 0 || in % LM_Q8_GROUP != 0) return LM_Q8_ERR_GROUP;
    const int64_t ng = in / LM_Q8_GROUP;
    for (int64_t r = 0; r < rows; r++) {
        for (int64_t g = 0; g < ng; g++) {
            const int64_t base = r * in + g * LM_Q8_GROUP;
            float amax = 0.0f;
            for (int i = 0; i < LM_Q8_GROUP; i++) {
                const float v = w[base + i];
                if (!isfinite(v)) {
                    if (bad_index) *bad_index = base + i;
                    return LM_Q8_ERR_NONFINITE;
                }
                const float a = fabsf(v);
                if (a > amax) amax = a;
            }
            const float scale = (float)((double)amax / 127.0);
            scales[r * ng + g] = scale;
            if (scale == 0.0f) {
                for (int i = 0; i < LM_Q8_GROUP; i++) q[base + i] = 0;
                continue;
            }
            const double ds = (double)scale;
            for (int i = 0; i < LM_Q8_GROUP; i++) {
                /* nearbyint honours the current rounding mode, nearest-even by default;
                 * lm-quant never changes it. */
                double v = nearbyint((double)w[base + i] / ds);
                if (v > 127.0) v = 127.0;
                if (v < -127.0) v = -127.0;
                q[base + i] = (int8_t)v;
            }
        }
    }
    return LM_Q8_OK;
}

void lm_q8_dequantize(const int8_t *q, const float *scales, int64_t rows, int64_t in, float *out)
{
    const int64_t ng = in / LM_Q8_GROUP;
    for (int64_t r = 0; r < rows; r++)
        for (int64_t i = 0; i < in; i++)
            out[r * in + i] = (float)q[r * in + i] * scales[r * ng + i / LM_Q8_GROUP];
}
