/* lm_kernels.h - matrix-vector kernels: fp32, bf16 weights, q8 weights.
 *
 * Ported from kimi-k3-in-c src/core/k3_ops.c (Apache-2.0): k3_matmul -> lm_matvec_f32,
 * k3_matmul_bf16 -> lm_matvec_bf16 (scalar, AVX2 and NEON paths), k3_matmul_q8 ->
 * lm_matvec_q8 (rewritten for this project's group-32 format, see below).
 *
 * All kernels compute y[o] = sum_i W[o][i] * x[i] for a row-major W of `out` rows and
 * `in` columns. They are pure functions over caller-owned buffers, OpenMP-parallel over
 * independent output rows only (no reduction crosses threads), so the result does not
 * depend on the thread count.
 *
 * DETERMINISM CONTRACT (build with no -ffast-math; every product that matters is an
 * explicit fused multiply-add and every reduction tree is written out, so no compiler
 * contraction choice can change a result):
 *   lm_matvec_f32 and lm_matvec_bf16: sixteen double accumulators, element i in lane
 *     i % 16, lanes reduced as ((a0+a4)+(a8+a12)) etc. then (b0+b1)+(b2+b3). The scalar,
 *     AVX2 and NEON paths are BIT-IDENTICAL.
 *   lm_matvec_q8: float arithmetic. Per group of 32 inputs, four fused steps over eight
 *     lanes (lane l covers inputs 8j+l), then acc[l] = fma(group[l], scale, acc[l]);
 *     the eight lanes reduce as ((a0+a4)+(a2+a6)) + ((a1+a5)+(a3+a7)). The scalar path
 *     emulates exactly that, so scalar, AVX2 and NEON are BIT-IDENTICAL here too.
 * The scalar functions are always compiled and are the reference; lm_matvec_* dispatch to
 * the best path this build has (lm_kernel_isa() names it). */
#ifndef LM_KERNELS_H
#define LM_KERNELS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* "avx2", "neon" or "scalar": the path lm_matvec_* take in this build. */
const char *lm_kernel_isa(void);

void lm_matvec_f32(float *y, const float *x, const float *W, int in, int out);
void lm_matvec_f32_scalar(float *y, const float *x, const float *W, int in, int out);

/* W is bf16 (raw uint16 bit patterns), widened to f32 on the fly. Any `in` >= 0. */
void lm_matvec_bf16(float *y, const float *x, const uint16_t *W, int in, int out);
void lm_matvec_bf16_scalar(float *y, const float *x, const uint16_t *W, int in, int out);

/* q8 format (docs/PLAN.md): q is int8 [out][in] row-major, scales is f32 [out][in/32],
 * w[o][i] = q[o][i] * scales[o][i/32]. `in` must be a multiple of 32: returns -1 and
 * writes nothing otherwise, 0 on success. */
#define LM_Q8_GROUP 32
int lm_matvec_q8(float *y, const float *x, const int8_t *q, const float *scales, int in, int out);
int lm_matvec_q8_scalar(float *y, const float *x, const int8_t *q, const float *scales,
                        int in, int out);

#ifdef __cplusplus
}
#endif
#endif /* LM_KERNELS_H */
