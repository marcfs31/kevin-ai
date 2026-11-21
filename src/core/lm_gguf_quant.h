/* lm_gguf_quant.h - matrix multiplies over ggml block-quantised weights (the types of
 * io/lm_gguf.h: Q4_K, Q5_K, Q6_K, Q8_0), activations in float. Original code; the block
 * layouts are ggml's (ggml-common.h), the maths is the dequantisation formula of each type.
 *
 * A weight matrix is [out][in] row-major with every row a whole number of blocks. Block
 * formulas, each evaluated in float exactly as written (fmaf where a product meets a sum,
 * so no compiler contraction choice can change a bit):
 *   Q4_K  256 elements = 8 sub-blocks of 32: w = fmaf(d*sc, q, -(dmin*m)), q in 0..15
 *   Q5_K  the same with q in 0..31 (a fifth bit from qh)
 *   Q6_K  256 elements = 16 sub-blocks of 16: w = (d*sc) * (q - 32), q in 0..63
 *   Q8_0  32 elements: w = d * q
 *   (d, dmin: f16 in the block; sc, m: 6-bit integers (Q4_K, Q5_K) or int8 (Q6_K).)
 *
 * DETERMINISM CONTRACT: Y[t][o] = sum_i W[o][i] * X[t][i] accumulated in 32 float lanes,
 * element i of a row in lane i % 32 by one fmaf per element in element order, the lanes
 * reduced as B[l] = (a[l] + a[l+16]) + (a[l+8] + a[l+24]) for l < 8, then
 * ((B0+B4)+(B2+B6)) + ((B1+B5)+(B3+B7)). The scalar path emulates that, so scalar and AVX2
 * are BIT-IDENTICAL, and a token's result never depends on T, on the other tokens or on the
 * thread count: lm_feed and lm_prefill agree bit for bit. */
#ifndef LM_GGUF_QUANT_H
#define LM_GGUF_QUANT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* "avx2" or "scalar": the path lm_gg_* take in this build. */
const char *lm_gg_isa(void);

/* Y[T][out] = X[T][in] * W^T for W of ggml type `type` ([out][in] row-major, `in` a multiple
 * of the type's block). 0, or -1 (nothing written) for an unsupported type, a bad size, T < 0.
 * T == 0 is a no-op. */
int lm_gg_matmul(float *Y, const float *X, int T, int type, const void *W, int in, int out);
int lm_gg_matmul_scalar(float *Y, const float *X, int T, int type, const void *W, int in, int out);

/* One block widened to floats (block_elems of them; see lm_gguf_type_block). */
void lm_gg_block(int type, const void *blk, float *w);
void lm_gg_block_scalar(int type, const void *blk, float *w);

/* Row `row` of W widened to float: in floats into dst. 0, or -1 for an unsupported type. */
int lm_gg_row(float *dst, int type, const void *W, int in, int row);

#ifdef __cplusplus
}
#endif
#endif /* LM_GGUF_QUANT_H */
