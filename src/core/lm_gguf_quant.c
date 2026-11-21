/* lm_gguf_quant.c - see lm_gguf_quant.h. Original code. */
#include "core/lm_gguf_quant.h"

#include <math.h>
#include <stddef.h>
#include <string.h>

#include "core/lm_bf16.h"
#include "io/lm_gguf.h"

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define LM_GG_AVX2 1
#endif

const char *lm_gg_isa(void)
{
#ifdef LM_GG_AVX2
    return "avx2";
#else
    return "scalar";
#endif
}

static inline float f16at(const uint8_t *p)
{
    uint16_t h;
    memcpy(&h, p, 2);
    return lm_f16_to_f32(h);
}

/* ggml get_scale_min_k4: the 6-bit scale and min of sub-block j from the 12 packed bytes. */
static inline void scale_min(int j, const uint8_t *q, int *sc, int *m)
{
    if (j < 4) {
        *sc = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

/* ------------------------------------------------------------------ scalar block widening */

static void blk_q4k(const uint8_t *b, float *w)
{
    const float d = f16at(b), dmin = f16at(b + 2);
    const uint8_t *sc12 = b + 4, *qs = b + 16;
    for (int j = 0; j < 4; j++) {
        int s1, m1, s2, m2;
        scale_min(2 * j, sc12, &s1, &m1);
        scale_min(2 * j + 1, sc12, &s2, &m2);
        const float d1 = d * (float)s1, n1 = dmin * (float)m1;
        const float d2 = d * (float)s2, n2 = dmin * (float)m2;
        const uint8_t *q = qs + 32 * j;
        for (int l = 0; l < 32; l++) {
            w[64 * j + l] = fmaf(d1, (float)(q[l] & 0xF), -n1);
            w[64 * j + 32 + l] = fmaf(d2, (float)(q[l] >> 4), -n2);
        }
    }
}

static void blk_q5k(const uint8_t *b, float *w)
{
    const float d = f16at(b), dmin = f16at(b + 2);
    const uint8_t *sc12 = b + 4, *qh = b + 16, *qs = b + 48;
    for (int j = 0; j < 4; j++) {
        int s1, m1, s2, m2;
        scale_min(2 * j, sc12, &s1, &m1);
        scale_min(2 * j + 1, sc12, &s2, &m2);
        const float d1 = d * (float)s1, n1 = dmin * (float)m1;
        const float d2 = d * (float)s2, n2 = dmin * (float)m2;
        const uint8_t *q = qs + 32 * j;
        const unsigned u1 = 1u << (2 * j), u2 = 2u << (2 * j);
        for (int l = 0; l < 32; l++) {
            w[64 * j + l] = fmaf(d1, (float)((q[l] & 0xF) + ((qh[l] & u1) ? 16 : 0)), -n1);
            w[64 * j + 32 + l] = fmaf(d2, (float)((q[l] >> 4) + ((qh[l] & u2) ? 16 : 0)), -n2);
        }
    }
}

static void blk_q6k(const uint8_t *b, float *w)
{
    const uint8_t *ql = b, *qh = b + 128;
    const int8_t *sc = (const int8_t *)(b + 192);
    const float d = f16at(b + 208);
    for (int h = 0; h < 2; h++) {
        for (int l = 0; l < 32; l++) {
            const int is = l / 16;
            const int q1 = ((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
            const int q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            const int q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            const int q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            w[l] = (d * (float)sc[is]) * (float)q1;
            w[l + 32] = (d * (float)sc[is + 2]) * (float)q2;
            w[l + 64] = (d * (float)sc[is + 4]) * (float)q3;
            w[l + 96] = (d * (float)sc[is + 6]) * (float)q4;
        }
        w += 128; ql += 64; qh += 32; sc += 8;
    }
}

static void blk_q80(const uint8_t *b, float *w)
{
    const float d = f16at(b);
    const int8_t *q = (const int8_t *)(b + 2);
    for (int l = 0; l < 32; l++) w[l] = d * (float)q[l];
}

void lm_gg_block_scalar(int type, const void *blk, float *w)
{
    switch (type) {
    case LM_GG_Q4_K: blk_q4k((const uint8_t *)blk, w); break;
    case LM_GG_Q5_K: blk_q5k((const uint8_t *)blk, w); break;
    case LM_GG_Q6_K: blk_q6k((const uint8_t *)blk, w); break;
    case LM_GG_Q8_0: blk_q80((const uint8_t *)blk, w); break;
    default: break;
    }
}

/* ------------------------------------------------------------------ AVX2 block widening */
#ifdef LM_GG_AVX2

static inline __m256 u8x8_ps(__m128i bytes) { return _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(bytes)); }

static void blk_q4k_avx2(const uint8_t *b, float *w)
{
    const float d = f16at(b), dmin = f16at(b + 2);
    const uint8_t *sc12 = b + 4, *qs = b + 16;
    const __m256i m0f = _mm256_set1_epi8(0x0F);
    for (int j = 0; j < 4; j++) {
        int s1, m1, s2, m2;
        scale_min(2 * j, sc12, &s1, &m1);
        scale_min(2 * j + 1, sc12, &s2, &m2);
        const __m256 d1 = _mm256_set1_ps(d * (float)s1), n1 = _mm256_set1_ps(-(dmin * (float)m1));
        const __m256 d2 = _mm256_set1_ps(d * (float)s2), n2 = _mm256_set1_ps(-(dmin * (float)m2));
        const __m256i v = _mm256_loadu_si256((const __m256i *)(qs + 32 * j));
        const __m256i lo = _mm256_and_si256(v, m0f);
        const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), m0f);
        const __m128i l0 = _mm256_castsi256_si128(lo), l1 = _mm256_extracti128_si256(lo, 1);
        const __m128i h0 = _mm256_castsi256_si128(hi), h1 = _mm256_extracti128_si256(hi, 1);
        float *o = w + 64 * j;
        _mm256_storeu_ps(o, _mm256_fmadd_ps(d1, u8x8_ps(l0), n1));
        _mm256_storeu_ps(o + 8, _mm256_fmadd_ps(d1, u8x8_ps(_mm_srli_si128(l0, 8)), n1));
        _mm256_storeu_ps(o + 16, _mm256_fmadd_ps(d1, u8x8_ps(l1), n1));
        _mm256_storeu_ps(o + 24, _mm256_fmadd_ps(d1, u8x8_ps(_mm_srli_si128(l1, 8)), n1));
        _mm256_storeu_ps(o + 32, _mm256_fmadd_ps(d2, u8x8_ps(h0), n2));
        _mm256_storeu_ps(o + 40, _mm256_fmadd_ps(d2, u8x8_ps(_mm_srli_si128(h0, 8)), n2));
        _mm256_storeu_ps(o + 48, _mm256_fmadd_ps(d2, u8x8_ps(h1), n2));
        _mm256_storeu_ps(o + 56, _mm256_fmadd_ps(d2, u8x8_ps(_mm_srli_si128(h1, 8)), n2));
    }
}

static void blk_q6k_avx2(const uint8_t *b, float *wout)
{
    const uint8_t *ql = b, *qh = b + 128;
    const int8_t *sc = (const int8_t *)(b + 192);
    const float d = f16at(b + 208);
    const __m256i m0f = _mm256_set1_epi32(0xF), m3 = _mm256_set1_epi32(3), c32 = _mm256_set1_epi32(32);
    for (int h = 0; h < 2; h++) {
        for (int l = 0; l < 32; l += 8) {
            const int is = l / 16;
            const __m256i a = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(ql + l)));
            const __m256i a2 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(ql + l + 32)));
            const __m256i hh = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(qh + l)));
            __m256i q1 = _mm256_or_si256(_mm256_and_si256(a, m0f), _mm256_slli_epi32(_mm256_and_si256(hh, m3), 4));
            __m256i q2 = _mm256_or_si256(_mm256_and_si256(a2, m0f), _mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(hh, 2), m3), 4));
            __m256i q3 = _mm256_or_si256(_mm256_srli_epi32(a, 4), _mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(hh, 4), m3), 4));
            __m256i q4 = _mm256_or_si256(_mm256_srli_epi32(a2, 4), _mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(hh, 6), m3), 4));
            q1 = _mm256_sub_epi32(q1, c32); q2 = _mm256_sub_epi32(q2, c32);
            q3 = _mm256_sub_epi32(q3, c32); q4 = _mm256_sub_epi32(q4, c32);
            _mm256_storeu_ps(wout + l, _mm256_mul_ps(_mm256_set1_ps(d * (float)sc[is]), _mm256_cvtepi32_ps(q1)));
            _mm256_storeu_ps(wout + l + 32, _mm256_mul_ps(_mm256_set1_ps(d * (float)sc[is + 2]), _mm256_cvtepi32_ps(q2)));
            _mm256_storeu_ps(wout + l + 64, _mm256_mul_ps(_mm256_set1_ps(d * (float)sc[is + 4]), _mm256_cvtepi32_ps(q3)));
            _mm256_storeu_ps(wout + l + 96, _mm256_mul_ps(_mm256_set1_ps(d * (float)sc[is + 6]), _mm256_cvtepi32_ps(q4)));
        }
        wout += 128; ql += 64; qh += 32; sc += 8;
    }
}

static void blk_q80_avx2(const uint8_t *b, float *w)
{
    const __m256 d = _mm256_set1_ps(f16at(b));
    const int8_t *q = (const int8_t *)(b + 2);
    for (int k = 0; k < 4; k++) {
        const __m256i v = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(q + 8 * k)));
        _mm256_storeu_ps(w + 8 * k, _mm256_mul_ps(d, _mm256_cvtepi32_ps(v)));
    }
}

#endif /* LM_GG_AVX2 */

void lm_gg_block(int type, const void *blk, float *w)
{
#ifdef LM_GG_AVX2
    switch (type) {
    case LM_GG_Q4_K: blk_q4k_avx2((const uint8_t *)blk, w); return;
    case LM_GG_Q6_K: blk_q6k_avx2((const uint8_t *)blk, w); return;
    case LM_GG_Q8_0: blk_q80_avx2((const uint8_t *)blk, w); return;
    default: break;
    }
#endif
    lm_gg_block_scalar(type, blk, w);
}

/* ------------------------------------------------------------------ the multiply */

#define TILE 8     /* tokens that share one widening of a block */
#define BLK_MAX 256

/* One block's contribution to the 32-lane accumulator of one token. */
static inline void dot_scalar(float *acc, const float *w, const float *x, int n)
{
    for (int i = 0; i < n; i++) acc[i & 31] = fmaf(w[i], x[i], acc[i & 31]);
}

static inline float reduce_scalar(const float *a)
{
    float B[8];
    for (int l = 0; l < 8; l++) B[l] = (a[l] + a[l + 16]) + (a[l + 8] + a[l + 24]);
    return ((B[0] + B[4]) + (B[2] + B[6])) + ((B[1] + B[5]) + (B[3] + B[7]));
}

#ifdef LM_GG_AVX2
static inline void dot_avx2(float *acc, const float *w, const float *x, int n)
{
    __m256 a0 = _mm256_loadu_ps(acc), a1 = _mm256_loadu_ps(acc + 8);
    __m256 a2 = _mm256_loadu_ps(acc + 16), a3 = _mm256_loadu_ps(acc + 24);
    for (int i = 0; i < n; i += 32) {
        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(w + i), _mm256_loadu_ps(x + i), a0);
        a1 = _mm256_fmadd_ps(_mm256_loadu_ps(w + i + 8), _mm256_loadu_ps(x + i + 8), a1);
        a2 = _mm256_fmadd_ps(_mm256_loadu_ps(w + i + 16), _mm256_loadu_ps(x + i + 16), a2);
        a3 = _mm256_fmadd_ps(_mm256_loadu_ps(w + i + 24), _mm256_loadu_ps(x + i + 24), a3);
    }
    _mm256_storeu_ps(acc, a0); _mm256_storeu_ps(acc + 8, a1);
    _mm256_storeu_ps(acc + 16, a2); _mm256_storeu_ps(acc + 24, a3);
}

static inline float reduce_avx2(const float *a)
{
    const __m256 B = _mm256_add_ps(_mm256_add_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(a + 16)),
                                   _mm256_add_ps(_mm256_loadu_ps(a + 8), _mm256_loadu_ps(a + 24)));
    const __m128 s = _mm_add_ps(_mm256_castps256_ps128(B), _mm256_extractf128_ps(B, 1));   /* b[l]+b[l+4] */
    const __m128 s2 = _mm_add_ps(s, _mm_movehl_ps(s, s));                                  /* (b0+b4)+(b2+b6), (b1+b5)+(b3+b7) */
    return _mm_cvtss_f32(_mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 1)));
}
#endif

static int check(int type, int in, int *be, size_t *bb)
{
    if (type == LM_GG_F32 || type == LM_GG_F16) return -1;
    if (lm_gguf_type_block(type, be, bb) != 0) return -1;
    if (in <= 0 || in % *be != 0 || (*be > BLK_MAX)) return -1;
    return 0;
}

static int matmul(float *Y, const float *X, int T, int type, const void *W, int in, int out, int simd)
{
    int be;
    size_t bb;
    if (T < 0 || out < 0 || check(type, in, &be, &bb) != 0) return -1;
    if (T == 0 || out == 0) return 0;
    const int nb = in / be;
    const uint8_t *base = (const uint8_t *)W;
    const size_t rowb = (size_t)nb * bb;
#pragma omp parallel for schedule(static) if (out > 64)
    for (int o = 0; o < out; o++) {
        const uint8_t *row = base + (size_t)o * rowb;
        float w[BLK_MAX] __attribute__((aligned(32)));
        float acc[TILE][32];
        for (int t0 = 0; t0 < T; t0 += TILE) {
            const int nt = T - t0 < TILE ? T - t0 : TILE;
            memset(acc, 0, sizeof acc);
            for (int b = 0; b < nb; b++) {
#ifdef LM_GG_AVX2
                if (simd) lm_gg_block(type, row + (size_t)b * bb, w);
                else
#endif
                    lm_gg_block_scalar(type, row + (size_t)b * bb, w);
                for (int t = 0; t < nt; t++) {
                    const float *x = X + (size_t)(t0 + t) * (size_t)in + (size_t)b * (size_t)be;
#ifdef LM_GG_AVX2
                    if (simd && be % 32 == 0) dot_avx2(acc[t], w, x, be);
                    else
#endif
                        dot_scalar(acc[t], w, x, be);
                }
            }
            for (int t = 0; t < nt; t++) {
#ifdef LM_GG_AVX2
                Y[(size_t)(t0 + t) * (size_t)out + (size_t)o] = simd ? reduce_avx2(acc[t]) : reduce_scalar(acc[t]);
#else
                Y[(size_t)(t0 + t) * (size_t)out + (size_t)o] = reduce_scalar(acc[t]);
#endif
            }
        }
    }
    (void)simd;
    return 0;
}

int lm_gg_matmul(float *Y, const float *X, int T, int type, const void *W, int in, int out)
{
    return matmul(Y, X, T, type, W, in, out, 1);
}

int lm_gg_matmul_scalar(float *Y, const float *X, int T, int type, const void *W, int in, int out)
{
    return matmul(Y, X, T, type, W, in, out, 0);
}

int lm_gg_row(float *dst, int type, const void *W, int in, int row)
{
    int be;
    size_t bb;
    if (check(type, in, &be, &bb) != 0 || row < 0) return -1;
    const int nb = in / be;
    const uint8_t *r = (const uint8_t *)W + (size_t)row * (size_t)nb * bb;
    for (int b = 0; b < nb; b++) lm_gg_block(type, r + (size_t)b * bb, dst + (size_t)b * (size_t)be);
    return 0;
}
