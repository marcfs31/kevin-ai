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
            __m256 t = _mm256_setzero_ps();
            for (int j = 0; j < 4; j++) {
                const __m128i b = _mm_loadl_epi64((const __m128i *)(qg + 8 * j));
                t = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b)),
                                    _mm256_loadu_ps(xg + 8 * j), t);
            }
            acc = _mm256_fmadd_ps(t, _mm256_set1_ps(sr[g]), acc);
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
