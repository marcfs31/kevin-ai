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

/* Batched form for prompt processing: X is [T][in] row-major, Y is [T][out] row-major, and
 *   Y[t][o] is BIT-IDENTICAL to lm_matvec_q8(.., X[t], ..)[o]
 * for every token t, every row o, any T and any thread count: each token sees exactly the
 * float operations of the single-token kernel, in the same order, and tokens never mix.
 * What changes is the cost: each weight is loaded and widened once per tile of tokens
 * instead of once per token. T = 0 writes nothing; `in` not a multiple of 32 or T < 0
 * returns -1 with nothing written, 0 on success. lm_matmul_q8_scalar is the reference. */
int lm_matmul_q8(float *Y, const float *X, int T, const int8_t *q, const float *scales,
                 int in, int out);
int lm_matmul_q8_scalar(float *Y, const float *X, int T, const int8_t *q, const float *scales,
                        int in, int out);

/* int8 activations (opt-in fast path; NOT bit-identical to lm_matvec_q8).
 *
 * lm_q8_quantize_acts rounds x to int8, one float scale per group of 32:
 *   amax = max |x| over the group; sx = amax / 127; qx = rint(x * (127 / amax)), |qx| <= 127.
 * An all-zero group gets sx = 0 and qx = 0. A group holding a NaN or an infinity gets
 * sx = NaN (qx = 0), so the result of any dot product that touches it is NaN, exactly as
 * with the float kernel. Rounding error per element is at most sx / 2 (+ a float ulp).
 * `in` must be a multiple of 32: returns -1 and writes nothing otherwise.
 *
 * lm_matvec_q8i computes y[o] = sum_g sr[g] * sx[g] * (sum_{i in g} q[o][i] * qx[i]) with
 * the inner sum in exact int32. Eight float lanes accumulate lane j = int sum of bytes
 * 4j..4j+3 of each group, acc[j] = fma((float)t_j, sr[g] * sx[g], acc[j]), then reduce as
 * ((a0+a4)+(a2+a6)) + ((a1+a5)+(a3+a7)). The scalar path emulates exactly that, so
 * scalar, AVX2 and NEON (dot-product extension) are BIT-IDENTICAL. Builds without AVX2 or
 * NEON dotprod run the scalar path. Returns -1 (nothing written) when `in` is not a
 * multiple of 32, 0 on success. Caller-owned buffers: qx[in], sx[in/32]. */
int lm_q8_quantize_acts(int8_t *qx, float *sx, const float *x, int in);
int lm_matvec_q8i(float *y, const int8_t *qx, const float *sx, const int8_t *q,
                  const float *scales, int in, int out);
int lm_matvec_q8i_scalar(float *y, const int8_t *qx, const float *sx, const int8_t *q,
                         const float *scales, int in, int out);

#ifdef __cplusplus
}
#endif
#endif /* LM_KERNELS_H */
