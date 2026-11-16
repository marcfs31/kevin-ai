/* lm_kernels.c - see lm_kernels.h for the contract.
 *
 * Ported from kimi-k3-in-c src/core/k3_ops.c (Apache-2.0). Changes: renamed; the scalar
 * reference of every kernel is a separate always-compiled function so tests can compare
 * it with the dispatched one; lm_matvec_f32 gains an AVX2 path (the original had a scalar
 * one); the bf16 kernel keeps the original's scalar/AVX2/NEON paths and its bit-identity
 * guarantee unchanged; lm_matvec_q8 is new, for the group-32 int8 + f32 scale format of
 * docs/PLAN.md (the original's per-row-scale draft kernel is not ported), with scalar,
 * AVX2 and NEON paths bound to the same bit-identity contract. */
#include <math.h>
#include <stdint.h>
#include <string.h>

#include "core/lm_bf16.h"
#include "core/lm_kernels.h"

/* Clang honours this; GCC never contracts across the explicit fma/add structure below
 * because every product that matters is already an fma() call. */
#if defined(__clang__)
#pragma STDC FP_CONTRACT OFF
#endif

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define LM_HAVE_AVX2 1
#elif defined(__ARM_NEON) && defined(__aarch64__)
#include <arm_neon.h>
#define LM_HAVE_NEON 1
#endif

/* The q8 kernels stream a memory-mapped file. The hardware prefetcher stops at every 4 KiB
 * page boundary, so one thread waits on memory far more than it computes (measured on the
 * i5-6500T: 2.2 GB/s per thread against a 17.5 GB/s memory limit). An explicit prefetch
 * a fixed distance ahead keeps loads in flight across pages and rows. It never faults and
 * never changes a result. */
#if defined(__GNUC__) || defined(__clang__)
#define LM_PREFETCH(p) __builtin_prefetch((const void *)(p), 0, 0)
#else
#define LM_PREFETCH(p) ((void)(p))
#endif
#define LM_Q8_PREFETCH_BYTES 1024 /* weight bytes ahead of the group being summed */

const char *lm_kernel_isa(void)
{
#if defined(LM_HAVE_AVX2)
    return "avx2";
#elif defined(LM_HAVE_NEON)
    return "neon";
#else
    return "scalar";
#endif
}

/* The fixed 16-lane reduction shared by the f32 and bf16 kernels. */
static inline double reduce16(const double a[16])
{
    const double b0 = (a[0] + a[4]) + (a[8]  + a[12]);
    const double b1 = (a[1] + a[5]) + (a[9]  + a[13]);
    const double b2 = (a[2] + a[6]) + (a[10] + a[14]);
    const double b3 = (a[3] + a[7]) + (a[11] + a[15]);
    return (b0 + b1) + (b2 + b3);
}

/* ------------------------------------------------------------------ fp32 */

static inline float dot_f32_scalar(const float *row, const float *x, int in)
{
    double a[16] = {0};
    int i = 0;
    for (; i + 15 < in; i += 16)
        for (int l = 0; l < 16; l++)
            a[l] = fma((double)row[i + l], (double)x[i + l], a[l]);
    double acc = reduce16(a);
    for (; i < in; i++) acc = fma((double)row[i], (double)x[i], acc);
    return (float)acc;
}

void lm_matvec_f32_scalar(float *y, const float *x, const float *W, int in, int out)
{
    for (int o = 0; o < out; o++) y[o] = dot_f32_scalar(W + (size_t)o * in, x, in);
}

void lm_matvec_f32(float *y, const float *x, const float *W, int in, int out)
{
#if defined(LM_HAVE_AVX2)
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (out > 64)
#endif
    for (int o = 0; o < out; o++) {
        const float *row = W + (size_t)o * in;
        __m256d v0 = _mm256_setzero_pd(), v1 = _mm256_setzero_pd();
        __m256d v2 = _mm256_setzero_pd(), v3 = _mm256_setzero_pd();
        int i = 0;
        for (; i + 15 < in; i += 16) {
            v0 = _mm256_fmadd_pd(_mm256_cvtps_pd(_mm_loadu_ps(row + i)),
                                 _mm256_cvtps_pd(_mm_loadu_ps(x + i)), v0);
            v1 = _mm256_fmadd_pd(_mm256_cvtps_pd(_mm_loadu_ps(row + i + 4)),
                                 _mm256_cvtps_pd(_mm_loadu_ps(x + i + 4)), v1);
            v2 = _mm256_fmadd_pd(_mm256_cvtps_pd(_mm_loadu_ps(row + i + 8)),
                                 _mm256_cvtps_pd(_mm_loadu_ps(x + i + 8)), v2);
            v3 = _mm256_fmadd_pd(_mm256_cvtps_pd(_mm_loadu_ps(row + i + 12)),
                                 _mm256_cvtps_pd(_mm_loadu_ps(x + i + 12)), v3);
        }
        /* lane j of vt is b_j of reduce16 */
        const __m256d vt = _mm256_add_pd(_mm256_add_pd(v0, v1), _mm256_add_pd(v2, v3));
        double b[4];
        _mm256_storeu_pd(b, vt);
        double acc = (b[0] + b[1]) + (b[2] + b[3]);
        for (; i < in; i++) acc = fma((double)row[i], (double)x[i], acc);
        y[o] = (float)acc;
    }
#else
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (out > 64)
#endif
    for (int o = 0; o < out; o++) y[o] = dot_f32_scalar(W + (size_t)o * in, x, in);
#endif
}

/* ------------------------------------------------------------------ bf16 */

static inline float dot_bf16_scalar(const uint16_t *row, const float *x, int in)
{
    double a[16] = {0};
    int i = 0;
    for (; i + 15 < in; i += 16)
        for (int l = 0; l < 16; l++)
            a[l] = fma((double)lm_bf16_to_f32(row[i + l]), (double)x[i + l], a[l]);
    double acc = reduce16(a);
    for (; i < in; i++) acc = fma((double)lm_bf16_to_f32(row[i]), (double)x[i], acc);
    return (float)acc;
}

void lm_matvec_bf16_scalar(float *y, const float *x, const uint16_t *W, int in, int out)
{
    for (int o = 0; o < out; o++) y[o] = dot_bf16_scalar(W + (size_t)o * in, x, in);
}

void lm_matvec_bf16(float *y, const float *x, const uint16_t *W, int in, int out)
{
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (out > 64)
#endif
    for (int o = 0; o < out; o++) {
        const uint16_t *row = W + (size_t)o * in;
#if defined(LM_HAVE_AVX2)
        /* Element i lands in lane i % 4 of accumulator (i / 4) % 4: scalar lane i % 16.
         * _mm256_fmadd_pd per lane is the same IEEE operation as scalar fma(). */
        __m256d v0 = _mm256_setzero_pd(), v1 = _mm256_setzero_pd();
        __m256d v2 = _mm256_setzero_pd(), v3 = _mm256_setzero_pd();
        int i = 0;
        for (; i + 15 < in; i += 16) {
            const __m128i h0 = _mm_loadl_epi64((const __m128i *)(row + i));
            const __m128i h1 = _mm_loadl_epi64((const __m128i *)(row + i + 4));
            const __m128i h2 = _mm_loadl_epi64((const __m128i *)(row + i + 8));
            const __m128i h3 = _mm_loadl_epi64((const __m128i *)(row + i + 12));
            v0 = _mm256_fmadd_pd(
                _mm256_cvtps_pd(_mm_castsi128_ps(_mm_slli_epi32(_mm_cvtepu16_epi32(h0), 16))),
                _mm256_cvtps_pd(_mm_loadu_ps(x + i)), v0);
            v1 = _mm256_fmadd_pd(
                _mm256_cvtps_pd(_mm_castsi128_ps(_mm_slli_epi32(_mm_cvtepu16_epi32(h1), 16))),
                _mm256_cvtps_pd(_mm_loadu_ps(x + i + 4)), v1);
            v2 = _mm256_fmadd_pd(
                _mm256_cvtps_pd(_mm_castsi128_ps(_mm_slli_epi32(_mm_cvtepu16_epi32(h2), 16))),
                _mm256_cvtps_pd(_mm_loadu_ps(x + i + 8)), v2);
            v3 = _mm256_fmadd_pd(
                _mm256_cvtps_pd(_mm_castsi128_ps(_mm_slli_epi32(_mm_cvtepu16_epi32(h3), 16))),
                _mm256_cvtps_pd(_mm_loadu_ps(x + i + 12)), v3);
        }
        const __m256d vt = _mm256_add_pd(_mm256_add_pd(v0, v1), _mm256_add_pd(v2, v3));
        double b[4];
        _mm256_storeu_pd(b, vt);
        double acc = (b[0] + b[1]) + (b[2] + b[3]);
        for (; i < in; i++) acc = fma((double)lm_bf16_to_f32(row[i]), (double)x[i], acc);
        y[o] = (float)acc;
#elif defined(LM_HAVE_NEON)
        /* Eight 2-lane double accumulators: wk holds the scalar lanes {2k, 2k+1}. */
        float64x2_t w0 = vdupq_n_f64(0.0), w1 = vdupq_n_f64(0.0);
        float64x2_t w2 = vdupq_n_f64(0.0), w3 = vdupq_n_f64(0.0);
        float64x2_t w4 = vdupq_n_f64(0.0), w5 = vdupq_n_f64(0.0);
        float64x2_t w6 = vdupq_n_f64(0.0), w7 = vdupq_n_f64(0.0);
        int i = 0;
        for (; i + 15 < in; i += 16) {
            const uint16x8_t h0 = vld1q_u16(row + i);
            const uint16x8_t h1 = vld1q_u16(row + i + 8);
            const float32x4_t f0 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(h0), 16));
            const float32x4_t f1 = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(h0), 16));
            const float32x4_t f2 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(h1), 16));
            const float32x4_t f3 = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(h1), 16));
            const float32x4_t x0 = vld1q_f32(x + i);
            const float32x4_t x1 = vld1q_f32(x + i + 4);
            const float32x4_t x2 = vld1q_f32(x + i + 8);
            const float32x4_t x3 = vld1q_f32(x + i + 12);
            w0 = vfmaq_f64(w0, vcvt_f64_f32(vget_low_f32(f0)), vcvt_f64_f32(vget_low_f32(x0)));
            w1 = vfmaq_f64(w1, vcvt_high_f64_f32(f0), vcvt_high_f64_f32(x0));
            w2 = vfmaq_f64(w2, vcvt_f64_f32(vget_low_f32(f1)), vcvt_f64_f32(vget_low_f32(x1)));
            w3 = vfmaq_f64(w3, vcvt_high_f64_f32(f1), vcvt_high_f64_f32(x1));
            w4 = vfmaq_f64(w4, vcvt_f64_f32(vget_low_f32(f2)), vcvt_f64_f32(vget_low_f32(x2)));
            w5 = vfmaq_f64(w5, vcvt_high_f64_f32(f2), vcvt_high_f64_f32(x2));
            w6 = vfmaq_f64(w6, vcvt_f64_f32(vget_low_f32(f3)), vcvt_f64_f32(vget_low_f32(x3)));
            w7 = vfmaq_f64(w7, vcvt_high_f64_f32(f3), vcvt_high_f64_f32(x3));
        }
        /* t0 = {b0,b1}, t1 = {b2,b3}: then (b0+b1)+(b2+b3) */
        const float64x2_t t0 = vaddq_f64(vaddq_f64(w0, w2), vaddq_f64(w4, w6));
        const float64x2_t t1 = vaddq_f64(vaddq_f64(w1, w3), vaddq_f64(w5, w7));
        double acc = vaddvq_f64(t0) + vaddvq_f64(t1);
        for (; i < in; i++) acc = fma((double)lm_bf16_to_f32(row[i]), (double)x[i], acc);
        y[o] = (float)acc;
#else
        y[o] = dot_bf16_scalar(row, x, in);
#endif
    }
}

/* ------------------------------------------------------------------ q8 */

/* Scalar emulation of the eight-lane vector kernels. */
static inline float dot_q8_scalar(const int8_t *q, const float *sc, const float *x, int in)
{
    float acc[8] = {0};
    for (int g = 0; g < in / LM_Q8_GROUP; g++) {
        float t[8] = {0};
        const int8_t *qg = q + (size_t)g * LM_Q8_GROUP;
        const float *xg = x + (size_t)g * LM_Q8_GROUP;
        for (int j = 0; j < 4; j++)
            for (int l = 0; l < 8; l++)
                t[l] = fmaf((float)qg[8 * j + l], xg[8 * j + l], t[l]);
        for (int l = 0; l < 8; l++) acc[l] = fmaf(t[l], sc[g], acc[l]);
    }
    const float m0 = (acc[0] + acc[4]) + (acc[2] + acc[6]);
    const float m1 = (acc[1] + acc[5]) + (acc[3] + acc[7]);
    return m0 + m1;
}

int lm_matvec_q8_scalar(float *y, const float *x, const int8_t *q, const float *scales,
                        int in, int out)
{
    if (in < 0 || in % LM_Q8_GROUP != 0) return -1;
    const int ng = in / LM_Q8_GROUP;
    for (int o = 0; o < out; o++)
        y[o] = dot_q8_scalar(q + (size_t)o * in, scales + (size_t)o * ng, x, in);
    return 0;
}

#if defined(LM_HAVE_AVX2)
#define LM_FORCE_INLINE static inline __attribute__((always_inline))
/* Eight int8 weights widened to f32. */
LM_FORCE_INLINE __m256 q8_w8_avx2(const int8_t *p)
{
    return _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)p)));
}

/* The single-token sum of one group of 32: t = fma(w3,x3, fma(w2,x2, fma(w1,x1, fma(w0,x0, 0)))).
 * Written without a loop on purpose: perf on the target machine measured about 41 instructions per
 * group (and 37 in the batched kernel) when gcc -O2 left the four-step loop rolled, with the widened
 * weights going through the stack; straight-line code keeps them in registers. Same operations in
 * the same order, so the results are bit-identical to the loop. */
LM_FORCE_INLINE __m256 q8_group_avx2(const int8_t *qg, const float *xg)
{
    __m256 t = _mm256_fmadd_ps(q8_w8_avx2(qg), _mm256_loadu_ps(xg), _mm256_setzero_ps());
    t = _mm256_fmadd_ps(q8_w8_avx2(qg + 8), _mm256_loadu_ps(xg + 8), t);
    t = _mm256_fmadd_ps(q8_w8_avx2(qg + 16), _mm256_loadu_ps(xg + 16), t);
    t = _mm256_fmadd_ps(q8_w8_avx2(qg + 24), _mm256_loadu_ps(xg + 24), t);
    return t;
}
#endif

int lm_matvec_q8(float *y, const float *x, const int8_t *q, const float *scales, int in, int out)
{
    if (in < 0 || in % LM_Q8_GROUP != 0) return -1;
    const int ng = in / LM_Q8_GROUP;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (out > 64)
#endif
    for (int o = 0; o < out; o++) {
        const int8_t *qr = q + (size_t)o * in;
        const float *sr = scales + (size_t)o * ng;
#if defined(LM_HAVE_AVX2)
        __m256 acc = _mm256_setzero_ps();
        for (int g = 0; g < ng; g++) {
            const int8_t *qg = qr + (size_t)g * LM_Q8_GROUP;
            const float *xg = x + (size_t)g * LM_Q8_GROUP;
            LM_PREFETCH(qg + LM_Q8_PREFETCH_BYTES);
            LM_PREFETCH(sr + g + 64);
            acc = _mm256_fmadd_ps(q8_group_avx2(qg, xg), _mm256_set1_ps(sr[g]), acc);
        }
        __m128 lo = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
        lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
        lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 1));
        y[o] = _mm_cvtss_f32(lo);
#elif defined(LM_HAVE_NEON)
        float32x4_t alo = vdupq_n_f32(0.0f), ahi = vdupq_n_f32(0.0f);
        for (int g = 0; g < ng; g++) {
            const int8_t *qg = qr + (size_t)g * LM_Q8_GROUP;
            const float *xg = x + (size_t)g * LM_Q8_GROUP;
            LM_PREFETCH(qg + LM_Q8_PREFETCH_BYTES);
            LM_PREFETCH(sr + g + 64);
            float32x4_t tlo = vdupq_n_f32(0.0f), thi = vdupq_n_f32(0.0f);
            for (int j = 0; j < 4; j++) {
                const int16x8_t s = vmovl_s8(vld1_s8(qg + 8 * j));
                tlo = vfmaq_f32(tlo, vcvtq_f32_s32(vmovl_s16(vget_low_s16(s))), vld1q_f32(xg + 8 * j));
                thi = vfmaq_f32(thi, vcvtq_f32_s32(vmovl_s16(vget_high_s16(s))), vld1q_f32(xg + 8 * j + 4));
            }
            const float32x4_t sv = vdupq_n_f32(sr[g]);
            alo = vfmaq_f32(alo, tlo, sv);
            ahi = vfmaq_f32(ahi, thi, sv);
        }
        /* (a0+a4, a1+a5, a2+a6, a3+a7) -> (l0+l2, l1+l3) -> (l0+l2)+(l1+l3) */
        const float32x4_t l4 = vaddq_f32(alo, ahi);
        const float32x2_t l2 = vadd_f32(vget_low_f32(l4), vget_high_f32(l4));
        y[o] = vget_lane_f32(vpadd_f32(l2, l2), 0);
#else
        y[o] = dot_q8_scalar(qr, sr, x, in);
#endif
    }
    return 0;
}

/* ------------------------------------------------------------------ q8, batched tokens */

#define LM_MATMUL_TB 4 /* tokens per tile: the weights of a group are widened once for all of them */
#define LM_MATMUL_RB 8 /* rows per block: a tile's inputs are reused by every row of the block */
#define LM_MATMUL_KB 32 /* groups per input slice: a tile's slice of inputs (TB * KB * 32 floats, 16 KiB)
                         * and the block's slice of weights (RB * KB * 32 bytes, 8 KiB) fit in a 32 KiB
                         * L1 together. Without slicing, a tile's inputs (40 KiB at in = 2560, 144 KiB
                         * at 9216) left L1 between rows and every row re-read them from L2 */

int lm_matmul_q8_scalar(float *Y, const float *X, int T, const int8_t *q, const float *scales,
                        int in, int out)
{
    if (in < 0 || in % LM_Q8_GROUP != 0 || T < 0) return -1;
    const int ng = in / LM_Q8_GROUP;
    for (int t = 0; t < T; t++)
        for (int o = 0; o < out; o++)
            Y[(size_t)t * out + o] =
                dot_q8_scalar(q + (size_t)o * in, scales + (size_t)o * ng, X + (size_t)t * in, in);
    return 0;
}

#if defined(LM_HAVE_AVX2)
static inline float hsum8_avx2(__m256 acc)
{
    __m128 lo = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
    lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 1));
    return _mm_cvtss_f32(lo);
}

/* The single-token row of lm_matvec_q8, for a token left over after the 4- and 2-token tiles. */
static inline float q8_row_avx2(const int8_t *qr, const float *sr, const float *x, int ng)
{
    __m256 acc = _mm256_setzero_ps();
    for (int g = 0; g < ng; g++) {
        const int8_t *qg = qr + (size_t)g * LM_Q8_GROUP;
        const float *xg = x + (size_t)g * LM_Q8_GROUP;
        acc = _mm256_fmadd_ps(q8_group_avx2(qg, xg), _mm256_set1_ps(sr[g]), acc);
    }
    return hsum8_avx2(acc);
}

/* One row against the nt (2 or 4) tokens at x0 (rows `in` apart), groups g0..g1-1, continuing
 * the accumulators in acc[0..nt-1]. Per token exactly the single-token sequence:
 * t = fma(w_j, x_j, t) for j = 0..3, then acc = fma(t, scale, acc). nt is a constant at every
 * call, so the unused half folds away. */
static inline void q8_tile_avx2(__m256 *acc, const float *x0, int in, const int8_t *qr, const float *sr,
                                int g0, int g1, int nt)
{
    const float *x1 = x0 + in, *x2 = nt == 4 ? x1 + in : x1, *x3 = nt == 4 ? x2 + in : x1;
    __m256 a0 = acc[0], a1 = acc[1], a2 = nt == 4 ? acc[2] : a0, a3 = nt == 4 ? acc[3] : a0;
    for (int g = g0; g < g1; g++) {
        const int8_t *qg = qr + (size_t)g * LM_Q8_GROUP;
        const size_t off = (size_t)g * LM_Q8_GROUP;
        LM_PREFETCH(qg + LM_Q8_PREFETCH_BYTES);
        LM_PREFETCH(sr + g + 64);
        const __m256 w0 = q8_w8_avx2(qg), w1 = q8_w8_avx2(qg + 8), w2 = q8_w8_avx2(qg + 16), w3 = q8_w8_avx2(qg + 24);
        const __m256 sv = _mm256_set1_ps(sr[g]);
#define LM_TOKEN_STEP(xt, acc)                                                               \
    do {                                                                                     \
        const float *xp_ = (xt) + off;                                                       \
        __m256 t_ = _mm256_fmadd_ps(w0, _mm256_loadu_ps(xp_), _mm256_setzero_ps());          \
        t_ = _mm256_fmadd_ps(w1, _mm256_loadu_ps(xp_ + 8), t_);                              \
        t_ = _mm256_fmadd_ps(w2, _mm256_loadu_ps(xp_ + 16), t_);                             \
        t_ = _mm256_fmadd_ps(w3, _mm256_loadu_ps(xp_ + 24), t_);                             \
        (acc) = _mm256_fmadd_ps(t_, sv, (acc));                                              \
    } while (0)
        LM_TOKEN_STEP(x0, a0);
        LM_TOKEN_STEP(x1, a1);
        if (nt == 4) {
            LM_TOKEN_STEP(x2, a2);
            LM_TOKEN_STEP(x3, a3);
        }
#undef LM_TOKEN_STEP
    }
    acc[0] = a0; acc[1] = a1;
    if (nt == 4) { acc[2] = a2; acc[3] = a3; }
}
#elif defined(LM_HAVE_NEON)
static inline float hsum8_neon(float32x4_t lo, float32x4_t hi)
{
    const float32x4_t l4 = vaddq_f32(lo, hi);
    const float32x2_t l2 = vadd_f32(vget_low_f32(l4), vget_high_f32(l4));
    return vget_lane_f32(vpadd_f32(l2, l2), 0);
}

static inline float q8_row_neon(const int8_t *qr, const float *sr, const float *x, int ng)
{
    float32x4_t alo = vdupq_n_f32(0.0f), ahi = vdupq_n_f32(0.0f);
    for (int g = 0; g < ng; g++) {
        const int8_t *qg = qr + (size_t)g * LM_Q8_GROUP;
        const float *xg = x + (size_t)g * LM_Q8_GROUP;
        float32x4_t tlo = vdupq_n_f32(0.0f), thi = vdupq_n_f32(0.0f);
        for (int j = 0; j < 4; j++) {
            const int16x8_t s = vmovl_s8(vld1_s8(qg + 8 * j));
            tlo = vfmaq_f32(tlo, vcvtq_f32_s32(vmovl_s16(vget_low_s16(s))), vld1q_f32(xg + 8 * j));
            thi = vfmaq_f32(thi, vcvtq_f32_s32(vmovl_s16(vget_high_s16(s))), vld1q_f32(xg + 8 * j + 4));
        }
        const float32x4_t sv = vdupq_n_f32(sr[g]);
        alo = vfmaq_f32(alo, tlo, sv);
        ahi = vfmaq_f32(ahi, thi, sv);
    }
    return hsum8_neon(alo, ahi);
}

/* acc holds, per token, the low and high halves: acc[2k], acc[2k+1]. */
static inline void q8_tile_neon(float32x4_t *acc, const float *x0, int in, const int8_t *qr, const float *sr,
                                int g0, int g1, int nt)
{
    const float *xs[LM_MATMUL_TB];
    float32x4_t alo[LM_MATMUL_TB], ahi[LM_MATMUL_TB];
    for (int k = 0; k < nt; k++) {
        xs[k] = x0 + (size_t)k * in;
        alo[k] = acc[2 * k];
        ahi[k] = acc[2 * k + 1];
    }
    for (int g = g0; g < g1; g++) {
        const int8_t *qg = qr + (size_t)g * LM_Q8_GROUP;
        const size_t off = (size_t)g * LM_Q8_GROUP;
        LM_PREFETCH(qg + LM_Q8_PREFETCH_BYTES);
        LM_PREFETCH(sr + g + 64);
        float32x4_t wl[4], wh[4];
        for (int j = 0; j < 4; j++) {
            const int16x8_t s = vmovl_s8(vld1_s8(qg + 8 * j));
            wl[j] = vcvtq_f32_s32(vmovl_s16(vget_low_s16(s)));
            wh[j] = vcvtq_f32_s32(vmovl_s16(vget_high_s16(s)));
        }
        const float32x4_t sv = vdupq_n_f32(sr[g]);
        for (int k = 0; k < nt; k++) {
            float32x4_t tlo = vdupq_n_f32(0.0f), thi = vdupq_n_f32(0.0f);
            for (int j = 0; j < 4; j++) {
                tlo = vfmaq_f32(tlo, wl[j], vld1q_f32(xs[k] + off + 8 * j));
                thi = vfmaq_f32(thi, wh[j], vld1q_f32(xs[k] + off + 8 * j + 4));
            }
            alo[k] = vfmaq_f32(alo[k], tlo, sv);
            ahi[k] = vfmaq_f32(ahi[k], thi, sv);
        }
    }
    for (int k = 0; k < nt; k++) {
        acc[2 * k] = alo[k];
        acc[2 * k + 1] = ahi[k];
    }
}
#endif

int lm_matmul_q8(float *Y, const float *X, int T, const int8_t *q, const float *scales,
                 int in, int out)
{
    if (in < 0 || in % LM_Q8_GROUP != 0 || T < 0) return -1;
    const int ng = in / LM_Q8_GROUP;
    const int Tfull = T - T % LM_MATMUL_TB;
    const int Tpair = Tfull + (T - Tfull) / 2 * 2; /* then one 2-token tile (an lm_verify of 2 is all of it) */
    const int nrb = (out + LM_MATMUL_RB - 1) / LM_MATMUL_RB;
    /* Rows in blocks, then token tiles, then input slices, then rows of the block. Each (row, token)
     * accumulator still sees its groups in order 0..ng-1 (it is parked in acc between slices,
     * which is exact), so only the order of independent computations differs from the one-token
     * kernel and every result is bit-identical. */
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (out > 64)
#endif
    for (int rb = 0; rb < nrb; rb++) {
        const int o0 = rb * LM_MATMUL_RB, o1 = o0 + LM_MATMUL_RB < out ? o0 + LM_MATMUL_RB : out;
#if defined(LM_HAVE_AVX2)
        int nt = LM_MATMUL_TB;
        for (int t0 = 0; t0 < Tpair; t0 += nt) {
            nt = t0 < Tfull ? LM_MATMUL_TB : 2;
            __m256 acc[LM_MATMUL_RB][LM_MATMUL_TB];
            for (int r = 0; r < o1 - o0; r++)
                for (int k = 0; k < LM_MATMUL_TB; k++) acc[r][k] = _mm256_setzero_ps();
            for (int g0 = 0; g0 < ng; g0 += LM_MATMUL_KB) {
                const int g1 = g0 + LM_MATMUL_KB < ng ? g0 + LM_MATMUL_KB : ng;
                for (int o = o0; o < o1; o++) {
                    const int8_t *qr = q + (size_t)o * in;
                    const float *sr = scales + (size_t)o * ng, *xt = X + (size_t)t0 * in;
                    if (nt == LM_MATMUL_TB) q8_tile_avx2(acc[o - o0], xt, in, qr, sr, g0, g1, LM_MATMUL_TB);
                    else q8_tile_avx2(acc[o - o0], xt, in, qr, sr, g0, g1, 2);
                }
            }
            for (int o = o0; o < o1; o++)
                for (int k = 0; k < nt; k++) Y[(size_t)(t0 + k) * out + o] = hsum8_avx2(acc[o - o0][k]);
        }
        for (int t = Tpair; t < T; t++)
            for (int o = o0; o < o1; o++)
                Y[(size_t)t * out + o] = q8_row_avx2(q + (size_t)o * in, scales + (size_t)o * ng, X + (size_t)t * in, ng);
#elif defined(LM_HAVE_NEON)
        int nt = LM_MATMUL_TB;
        for (int t0 = 0; t0 < Tpair; t0 += nt) {
            nt = t0 < Tfull ? LM_MATMUL_TB : 2;
            float32x4_t acc[LM_MATMUL_RB][2 * LM_MATMUL_TB];
            for (int r = 0; r < o1 - o0; r++)
                for (int k = 0; k < 2 * LM_MATMUL_TB; k++) acc[r][k] = vdupq_n_f32(0.0f);
            for (int g0 = 0; g0 < ng; g0 += LM_MATMUL_KB) {
                const int g1 = g0 + LM_MATMUL_KB < ng ? g0 + LM_MATMUL_KB : ng;
                for (int o = o0; o < o1; o++) {
                    const int8_t *qr = q + (size_t)o * in;
                    const float *sr = scales + (size_t)o * ng, *xt = X + (size_t)t0 * in;
                    if (nt == LM_MATMUL_TB) q8_tile_neon(acc[o - o0], xt, in, qr, sr, g0, g1, LM_MATMUL_TB);
                    else q8_tile_neon(acc[o - o0], xt, in, qr, sr, g0, g1, 2);
                }
            }
            for (int o = o0; o < o1; o++)
                for (int k = 0; k < nt; k++)
                    Y[(size_t)(t0 + k) * out + o] = hsum8_neon(acc[o - o0][2 * k], acc[o - o0][2 * k + 1]);
        }
        for (int t = Tpair; t < T; t++)
            for (int o = o0; o < o1; o++)
                Y[(size_t)t * out + o] = q8_row_neon(q + (size_t)o * in, scales + (size_t)o * ng, X + (size_t)t * in, ng);
#else
        for (int t = 0; t < T; t++)
            for (int o = o0; o < o1; o++)
                Y[(size_t)t * out + o] = dot_q8_scalar(q + (size_t)o * in, scales + (size_t)o * ng, X + (size_t)t * in, in);
#endif
    }
    return 0;
}

/* ------------------------------------------------------------------ q8, int8 activations */

/* AVX2 and NEON-with-dotprod have an int8 dot instruction; everything else runs the scalar
 * emulation, which is the reference the vector paths must match bit for bit. */

int lm_q8_quantize_acts(int8_t *qx, float *sx, const float *x, int in)
{
    if (in < 0 || in % LM_Q8_GROUP != 0) return -1;
    for (int g = 0; g < in / LM_Q8_GROUP; g++) {
        const float *xg = x + (size_t)g * LM_Q8_GROUP;
        int8_t *qg = qx + (size_t)g * LM_Q8_GROUP;
        float amax = 0.0f, probe = 0.0f;
        for (int i = 0; i < LM_Q8_GROUP; i++) {
            amax = fmaxf(amax, fabsf(xg[i]));
            probe += xg[i] * 0.0f; /* stays 0 for finite x; NaN for a NaN or an infinity */
        }
        if (probe != probe) { /* poison the group: every dot product touching it is NaN */
            sx[g] = NAN;
            memset(qg, 0, LM_Q8_GROUP);
            continue;
        }
        if (amax == 0.0f) {
            sx[g] = 0.0f;
            memset(qg, 0, LM_Q8_GROUP);
            continue;
        }
        const double inv = 127.0 / (double)amax; /* double: no overflow for denormal groups */
        for (int i = 0; i < LM_Q8_GROUP; i++) {
            long v = lrint((double)xg[i] * inv);
            qg[i] = (int8_t)(v > 127 ? 127 : v < -127 ? -127 : v);
        }
        sx[g] = amax / 127.0f;
    }
    return 0;
}

/* Scalar emulation of the eight-lane vector kernels: lane j sums bytes 4j..4j+3 of a group
 * in exact int32, then one fused step into float lane j. */
static inline float dot_q8i_scalar(const int8_t *q, const float *sc, const int8_t *qx,
                                   const float *sx, int in)
{
    float acc[8] = {0};
    for (int g = 0; g < in / LM_Q8_GROUP; g++) {
        const int8_t *qg = q + (size_t)g * LM_Q8_GROUP;
        const int8_t *xg = qx + (size_t)g * LM_Q8_GROUP;
        const float s = sc[g] * sx[g];
        for (int j = 0; j < 8; j++) {
            int t = 0;
            for (int k = 0; k < 4; k++) t += (int)qg[4 * j + k] * (int)xg[4 * j + k];
            acc[j] = fmaf((float)t, s, acc[j]);
        }
    }
    const float m0 = (acc[0] + acc[4]) + (acc[2] + acc[6]);
    const float m1 = (acc[1] + acc[5]) + (acc[3] + acc[7]);
    return m0 + m1;
}

int lm_matvec_q8i_scalar(float *y, const int8_t *qx, const float *sx, const int8_t *q,
                         const float *scales, int in, int out)
{
    if (in < 0 || in % LM_Q8_GROUP != 0) return -1;
    const int ng = in / LM_Q8_GROUP;
    for (int o = 0; o < out; o++)
        y[o] = dot_q8i_scalar(q + (size_t)o * in, scales + (size_t)o * ng, qx, sx, in);
    return 0;
}

int lm_matvec_q8i(float *y, const int8_t *qx, const float *sx, const int8_t *q,
                  const float *scales, int in, int out)
{
    if (in < 0 || in % LM_Q8_GROUP != 0) return -1;
    const int ng = in / LM_Q8_GROUP;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (out > 64)
#endif
    for (int o = 0; o < out; o++) {
        const int8_t *qr = q + (size_t)o * in;
        const float *sr = scales + (size_t)o * ng;
#if defined(LM_HAVE_AVX2)
        const __m256i ones = _mm256_set1_epi16(1);
        __m256 acc = _mm256_setzero_ps();
        for (int g = 0; g < ng; g++) {
            const int8_t *qg = qr + (size_t)g * LM_Q8_GROUP;
            LM_PREFETCH(qg + LM_Q8_PREFETCH_BYTES);
            LM_PREFETCH(sr + g + 64);
            const __m256i w = _mm256_loadu_si256((const __m256i *)qg);
            const __m256i a = _mm256_loadu_si256((const __m256i *)(qx + (size_t)g * LM_Q8_GROUP));
            /* |w| as unsigned (128 fits) times a with w's sign: w*a, and the int16 pair sums
             * cannot saturate (2 * 128 * 127 = 32512) because |a| <= 127. */
            const __m256i p = _mm256_maddubs_epi16(_mm256_abs_epi8(w), _mm256_sign_epi8(a, w));
            const __m256i t = _mm256_madd_epi16(p, ones); /* lane j: bytes 4j..4j+3 */
            acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(t), _mm256_set1_ps(sr[g] * sx[g]), acc);
        }
        __m128 lo = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
        lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
        lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 1));
        y[o] = _mm_cvtss_f32(lo);
#elif defined(LM_HAVE_NEON) && defined(__ARM_FEATURE_DOTPROD)
        float32x4_t alo = vdupq_n_f32(0.0f), ahi = vdupq_n_f32(0.0f);
        for (int g = 0; g < ng; g++) {
            const int8_t *qg = qr + (size_t)g * LM_Q8_GROUP;
            const int8_t *xg = qx + (size_t)g * LM_Q8_GROUP;
            LM_PREFETCH(qg + LM_Q8_PREFETCH_BYTES);
            LM_PREFETCH(sr + g + 64);
            const int32x4_t t0 = vdotq_s32(vdupq_n_s32(0), vld1q_s8(qg), vld1q_s8(xg));
            const int32x4_t t1 = vdotq_s32(vdupq_n_s32(0), vld1q_s8(qg + 16), vld1q_s8(xg + 16));
            const float32x4_t sv = vdupq_n_f32(sr[g] * sx[g]);
            alo = vfmaq_f32(alo, vcvtq_f32_s32(t0), sv);
            ahi = vfmaq_f32(ahi, vcvtq_f32_s32(t1), sv);
        }
        const float32x4_t l4 = vaddq_f32(alo, ahi);
        const float32x2_t l2 = vadd_f32(vget_low_f32(l4), vget_high_f32(l4));
        y[o] = vget_lane_f32(vpadd_f32(l2, l2), 0);
#else
        y[o] = dot_q8i_scalar(qr, sr, qx, sx, in);
#endif
    }
    return 0;
}
