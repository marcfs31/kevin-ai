/* test_kernels.c - matvec kernels (fp32, bf16, q8) against plain double-precision
 * reference loops, on real-sized rows (in = 2560 and 9216, the model's hidden and
 * intermediate widths) and on small and awkward ones; scalar path against the dispatched
 * (AVX2 / NEON / scalar) path, which the kernel contract says agree bit for bit. Style
 * follows kimi-k3-in-c tests/unit/test_ops.c (Apache-2.0). */
#include "lm_test.h"

#include "core/lm_bf16.h"
#include "core/lm_kernels.h"

static int bits_equal(const float *a, const float *b, int n)
{
    return memcmp(a, b, (size_t)n * sizeof(float)) == 0;
}

/* sum_i |w_i x_i| per row: the scale a rounding-error bound is relative to */
static double abs_dot(const double *w, const float *x, int in)
{
    double s = 0;
    for (int i = 0; i < in; i++) s += fabs(w[i] * (double)x[i]);
    return s;
}

static void fill_x(float *x, int n)
{
    for (int i = 0; i < n; i++) x[i] = lt_rand() * 3.0f;
}

/* ---- fp32 ---- */

static void check_f32(int in, int out)
{
    float *W = (float *)malloc((size_t)in * out * sizeof *W), *x = (float *)malloc((size_t)in * sizeof *x);
    float *y = (float *)malloc((size_t)out * sizeof *y), *ys = (float *)malloc((size_t)out * sizeof *ys);
    fill_x(x, in);
    for (size_t i = 0; i < (size_t)in * out; i++) W[i] = lt_rand();
    lm_matvec_f32(y, x, W, in, out);
    lm_matvec_f32_scalar(ys, x, W, in, out);
    CHECK_MSG(bits_equal(y, ys, out), "f32 in=%d out=%d: %s path differs from scalar", in, out, lm_kernel_isa());
    for (int o = 0; o < out; o++) {
        double ref = 0, w[9216];
        for (int i = 0; i < in; i++) { w[i] = (double)W[(size_t)o * in + i]; ref += w[i] * (double)x[i]; }
        const double tol = 1e-6 * abs_dot(w, x, in) + 1e-30;
        if (fabs((double)y[o] - ref) > tol) { CHECK_NEAR(y[o], ref, tol); break; }
    }
    free(W); free(x); free(y); free(ys);
}

/* ---- bf16 ---- */

static void check_bf16(int in, int out)
{
    uint16_t *W = (uint16_t *)malloc((size_t)in * out * sizeof *W);
    float *x = (float *)malloc((size_t)in * sizeof *x);
    float *y = (float *)malloc((size_t)out * sizeof *y), *ys = (float *)malloc((size_t)out * sizeof *ys);
    fill_x(x, in);
    for (size_t i = 0; i < (size_t)in * out; i++) W[i] = lt_bf16(lt_rand());
    lm_matvec_bf16(y, x, W, in, out);
    lm_matvec_bf16_scalar(ys, x, W, in, out);
    CHECK_MSG(bits_equal(y, ys, out), "bf16 in=%d out=%d: %s path differs from scalar", in, out, lm_kernel_isa());
    int bad = 0;
    for (int o = 0; o < out && !bad; o++) {
        double ref = 0, w[9216];
        for (int i = 0; i < in; i++) { w[i] = (double)lm_bf16_to_f32(W[(size_t)o * in + i]); ref += w[i] * (double)x[i]; }
        const double tol = 1e-6 * abs_dot(w, x, in) + 1e-30;
        if (fabs((double)y[o] - ref) > tol) { CHECK_NEAR(y[o], ref, tol); bad = 1; }
    }
    /* the bf16 kernel must equal the fp32 kernel run on the widened weights, bit for bit */
    float *Wf = (float *)malloc((size_t)in * out * sizeof *Wf), *yf = (float *)malloc((size_t)out * sizeof *yf);
    for (size_t i = 0; i < (size_t)in * out; i++) Wf[i] = lm_bf16_to_f32(W[i]);
    lm_matvec_f32_scalar(yf, x, Wf, in, out);
    CHECK_MSG(bits_equal(y, yf, out), "bf16 in=%d out=%d differs from fp32 on widened weights", in, out);
    free(W); free(x); free(y); free(ys); free(Wf); free(yf);
}

/* ---- q8 ---- */

static void check_q8(int in, int out)
{
    const int ng = in / 32;
    int8_t *q = (int8_t *)malloc((size_t)in * out);
    float *sc = (float *)malloc((size_t)ng * out * sizeof *sc);
    float *x = (float *)malloc((size_t)in * sizeof *x);
    float *y = (float *)malloc((size_t)out * sizeof *y), *ys = (float *)malloc((size_t)out * sizeof *ys);
    fill_x(x, in);
    for (size_t i = 0; i < (size_t)in * out; i++) q[i] = (int8_t)(lt_rand() * 127.0f);
    for (size_t i = 0; i < (size_t)ng * out; i++) sc[i] = (lt_rand() * 0.5f + 0.5f) * 0.02f;
    CHECK(lm_matvec_q8(y, x, q, sc, in, out) == 0);
    CHECK(lm_matvec_q8_scalar(ys, x, q, sc, in, out) == 0);
    CHECK_MSG(bits_equal(y, ys, out), "q8 in=%d out=%d: %s path differs from scalar", in, out, lm_kernel_isa());
    int bad = 0;
    for (int o = 0; o < out && !bad; o++) {
        double ref = 0, w[9216];
        for (int i = 0; i < in; i++) {
            w[i] = (double)q[(size_t)o * in + i] * (double)sc[(size_t)o * ng + i / 32];
            ref += w[i] * (double)x[i];
        }
        const double tol = 1e-5 * abs_dot(w, x, in) + 1e-30;   /* float accumulation within groups */
        if (fabs((double)y[o] - ref) > tol) { CHECK_NEAR(y[o], ref, tol); bad = 1; }
    }
    free(q); free(sc); free(x); free(y); free(ys);
}

static void test_known_values(void)
{
    /* hand-computed: row 0 = 1..32 times x = ones -> 528 * scale 0.5 = 264;
     * row 1 = all -1 (q) times x = i -> -(0+...+31) * 2 = -992 */
    int8_t q[64];
    float sc[2] = { 0.5f, 2.0f }, x[32], y[2] = { 0, 0 };
    for (int i = 0; i < 32; i++) { q[i] = (int8_t)(i + 1); q[32 + i] = -1; x[i] = 1.0f; }
    CHECK(lm_matvec_q8(y, x, q, sc, 32, 1) == 0 && y[0] == 264.0f);
    for (int i = 0; i < 32; i++) x[i] = (float)i;
    CHECK(lm_matvec_q8(y, x, q, sc, 32, 2) == 0);
    {
        double r0 = 0;
        for (int i = 0; i < 32; i++) r0 += (double)(i + 1) * i;
        CHECK_NEAR(y[0], 0.5 * r0, 1e-3);
        CHECK(y[1] == -992.0f);
    }
    /* two groups with different scales: per-group scaling must apply to its own group only */
    int8_t q2[64];
    float sc2[2] = { 1.0f, 100.0f }, x2[64], y2 = 0;
    for (int i = 0; i < 64; i++) { q2[i] = 1; x2[i] = 1.0f; }
    CHECK(lm_matvec_q8(&y2, x2, q2, sc2, 64, 1) == 0 && y2 == 32.0f * 1.0f + 32.0f * 100.0f);
    /* NEGATIVE: a transposed scale layout ([in/32][out] instead of [out][in/32]) must NOT
     * reproduce the right answer, so the layout in docs/PLAN.md is actually exercised */
    int8_t q3[2 * 64];
    float sc3[4] = { 1.0f, 2.0f, 3.0f, 4.0f }, sc3t[4] = { 1.0f, 3.0f, 2.0f, 4.0f }, x3[64], ya[2], yb[2];
    for (int i = 0; i < 128; i++) q3[i] = 1;
    for (int i = 0; i < 64; i++) x3[i] = 1.0f;
    lm_matvec_q8(ya, x3, q3, sc3, 64, 2);        /* row0: 32*1+32*2 = 96, row1: 32*3+32*4 = 224 */
    lm_matvec_q8(yb, x3, q3, sc3t, 64, 2);
    CHECK(ya[0] == 96.0f && ya[1] == 224.0f);
    CHECK(yb[0] != ya[0] && yb[1] != ya[1] && yb[0] == 128.0f);
    /* a sign flip on one weight is visible (the int8 sign is not dropped by the widening) */
    q3[5] = -1;
    lm_matvec_q8(yb, x3, q3, sc3, 64, 2);
    CHECK(yb[0] == ya[0] - 2.0f);
    /* -128 is representable in int8 and must widen as -128, not wrap */
    int8_t q4[32];
    float sc4 = 1.0f, x4[32], y4;
    for (int i = 0; i < 32; i++) { q4[i] = 0; x4[i] = 1.0f; }
    q4[0] = -128;
    CHECK(lm_matvec_q8(&y4, x4, q4, &sc4, 32, 1) == 0 && y4 == -128.0f);
}

static void test_q8_bad_in(void)
{
    int8_t q[64] = { 0 };
    float sc[2] = { 1, 1 }, x[64] = { 0 }, y[1] = { 123.0f };
    CHECK(lm_matvec_q8(y, x, q, sc, 33, 1) == -1 && y[0] == 123.0f);
    CHECK(lm_matvec_q8(y, x, q, sc, 16, 1) == -1 && y[0] == 123.0f);
    CHECK(lm_matvec_q8_scalar(y, x, q, sc, 48, 1) == -1 && y[0] == 123.0f);
    CHECK(lm_matvec_q8(y, x, q, sc, -32, 1) == -1);
}

static void test_known_bf16_f32(void)
{
    /* hand-computed small cases, odd lengths so the scalar tail is exercised */
    const float x[5] = { 1, 2, 3, 4, 5 };
    const uint16_t wb[10] = { 0x3f80, 0x4000, 0x4040, 0x4080, 0x40a0,      /* 1 2 3 4 5 */
                              0xbf80, 0xbf80, 0xbf80, 0xbf80, 0xbf80 };    /* -1 x5 */
    float y[2];
    lm_matvec_bf16(y, x, wb, 5, 2);
    CHECK(y[0] == 55.0f && y[1] == -15.0f);
    const float wf[10] = { 1, 2, 3, 4, 5, -1, -1, -1, -1, -1 };
    lm_matvec_f32(y, x, wf, 5, 2);
    CHECK(y[0] == 55.0f && y[1] == -15.0f);
    /* in = 0 gives zeros */
    y[0] = 9.0f;
    lm_matvec_f32(y, x, wf, 0, 1);
    CHECK(y[0] == 0.0f);
    lm_matvec_bf16(y, x, wb, 0, 1);
    CHECK(y[0] == 0.0f);
    /* a bf16 weight and its fp32 widening differ from the fp32 source of lower precision:
     * 1.00390625 is not representable in bf16 (rounds to 1.0 by nearest-even) */
    const uint16_t one = lt_bf16(1.00390625f);
    const float x1 = 1.0f;
    lm_matvec_bf16(y, &x1, &one, 1, 1);
    CHECK(y[0] == 1.0f);
}

static void test_unaligned(void)
{
    /* nothing may assume 32-byte alignment of x or W */
    enum { IN = 96, OUT = 3 };
    float xb[IN + 1], Wb[IN * OUT + 1], y1[OUT], y2[OUT];
    uint16_t hb[IN * OUT + 1];
    int8_t qb[IN * OUT + 1];
    float sb[(IN / 32) * OUT + 1];
    for (int i = 0; i < IN + 1; i++) xb[i] = lt_rand();
    for (int i = 0; i < IN * OUT + 1; i++) { Wb[i] = lt_rand(); hb[i] = lt_bf16(Wb[i]); qb[i] = (int8_t)(Wb[i] * 100); }
    for (int i = 0; i < (IN / 32) * OUT + 1; i++) sb[i] = 0.01f * (float)(i + 1);
    lm_matvec_f32(y1, xb + 1, Wb + 1, IN, OUT);
    lm_matvec_f32_scalar(y2, xb + 1, Wb + 1, IN, OUT);
    CHECK(bits_equal(y1, y2, OUT));
    lm_matvec_bf16(y1, xb + 1, hb + 1, IN, OUT);
    lm_matvec_bf16_scalar(y2, xb + 1, hb + 1, IN, OUT);
    CHECK(bits_equal(y1, y2, OUT));
    lm_matvec_q8(y1, xb + 1, qb + 1, sb + 1, IN, OUT);
    lm_matvec_q8_scalar(y2, xb + 1, qb + 1, sb + 1, IN, OUT);
    CHECK(bits_equal(y1, y2, OUT));
}

int main(void)
{
    printf("kernel path: %s\n", lm_kernel_isa());
    static const int small_in[] = { 1, 2, 3, 4, 5, 7, 15, 16, 17, 31, 32, 33, 63, 100 };
    for (size_t i = 0; i < sizeof small_in / sizeof *small_in; i++) {
        check_f32(small_in[i], 5);
        check_bf16(small_in[i], 5);
    }
    static const int q_in[] = { 32, 64, 96, 128, 1024 };
    for (size_t i = 0; i < sizeof q_in / sizeof *q_in; i++) check_q8(q_in[i], 5);
    /* real sizes; out = 70 crosses the OpenMP threshold (out > 64) on builds that have it */
    check_f32(2560, 70); check_f32(9216, 70);
    check_bf16(2560, 70); check_bf16(9216, 70);
    check_q8(2560, 70); check_q8(9216, 70);
    check_f32(2560, 1); check_bf16(2560, 1); check_q8(2560, 1);
    test_known_values();
    test_q8_bad_in();
    test_known_bf16_f32();
    test_unaligned();
    return lt_finish("test_kernels");
}
