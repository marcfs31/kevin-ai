/* test_kernels_q8i.c - the int8-activation q8 kernel: the quantiser's error bound and edge
 * cases (zero group, NaN, infinity), exact results on whole-number inputs, the int16
 * saturation corner (-128 weights against 127 activations), the error against the float
 * kernel's exact value, and bit identity of the dispatched (AVX2 / NEON dotprod / scalar)
 * path with the scalar reference. */
#include "lm_test.h"

#include "core/lm_kernels.h"

#define G LM_Q8_GROUP

static int bits_equal(const float *a, const float *b, int n)
{
    return memcmp(a, b, (size_t)n * sizeof(float)) == 0;
}

static void fill_x(float *x, int n)
{
    for (int i = 0; i < n; i++) x[i] = lt_rand() * 3.0f;
}

static int8_t rand_w(void)
{
    int v = (int)(lt_rand() * 128.0f);
    return (int8_t)(v > 127 ? 127 : v < -128 ? -128 : v);
}

/* ---- quantiser ---- */

static void test_quantiser(void)
{
    enum { IN = 96 };
    float x[IN], sx[IN / G];
    int8_t qx[IN];
    fill_x(x, IN);
    for (int i = 0; i < G; i++) x[G + i] = 0.0f; /* group 1: all zero */
    CHECK(lm_q8_quantize_acts(qx, sx, x, IN) == 0);
    for (int g = 0; g < IN / G; g++) {
        float amax = 0;
        for (int i = 0; i < G; i++) amax = fmaxf(amax, fabsf(x[g * G + i]));
        CHECK_MSG(sx[g] == amax / 127.0f, "group %d scale %.9g, expected %.9g", g, sx[g], amax / 127.0f);
        int hit127 = 0;
        for (int i = 0; i < G; i++) {
            const int q = qx[g * G + i];
            CHECK(q >= -127 && q <= 127);
            if (q == 127 || q == -127) hit127 = 1;
            CHECK_MSG(fabs((double)x[g * G + i] - (double)q * (double)sx[g]) <= 0.5001 * (double)sx[g],
                      "group %d element %d: rounding error above sx/2", g, i);
        }
        CHECK_MSG(hit127 == (amax != 0.0f), "group %d: the largest element must map to +-127 (zero group: none)", g);
    }
    CHECK(sx[1] == 0.0f);
    for (int i = 0; i < G; i++) CHECK(qx[G + i] == 0);

    /* a denormal group must neither overflow nor trap: the scale underflows to 0 at worst */
    float xd[G];
    int8_t qd[G];
    float sd;
    for (int i = 0; i < G; i++) xd[i] = (i & 1) ? 1e-44f : -1e-44f;
    CHECK(lm_q8_quantize_acts(qd, &sd, xd, G) == 0);
    for (int i = 0; i < G; i++) CHECK(qd[i] >= -127 && qd[i] <= 127);
    CHECK(sd >= 0.0f);
}

static void test_bad_in(void)
{
    int8_t q[64] = { 0 }, qx[64] = { 0 };
    float sc[2] = { 1, 1 }, sx[2] = { 1, 1 }, x[64] = { 0 }, y[1] = { 123.0f };
    CHECK(lm_q8_quantize_acts(qx, sx, x, 33) == -1);
    CHECK(lm_q8_quantize_acts(qx, sx, x, -32) == -1);
    CHECK(lm_matvec_q8i(y, qx, sx, q, sc, 33, 1) == -1 && y[0] == 123.0f);
    CHECK(lm_matvec_q8i(y, qx, sx, q, sc, 16, 1) == -1 && y[0] == 123.0f);
    CHECK(lm_matvec_q8i_scalar(y, qx, sx, q, sc, 48, 1) == -1 && y[0] == 123.0f);
    CHECK(lm_matvec_q8i(y, qx, sx, q, sc, -32, 1) == -1);
}

/* NEGATIVE cases: a NaN or an infinity must poison the result, never be rounded away */
static void test_non_finite(void)
{
    enum { IN = 96, OUT = 4 };
    const float bad[3] = { NAN, INFINITY, -INFINITY };
    int8_t q[IN * OUT], qx[IN], qx_ok[IN];
    float sc[(IN / G) * OUT], x[IN], x_ok[IN], sx[IN / G], sx_ok[IN / G], y[OUT];
    for (int i = 0; i < IN * OUT; i++) q[i] = rand_w();
    for (int i = 0; i < (IN / G) * OUT; i++) sc[i] = 0.01f;
    fill_x(x_ok, IN);
    CHECK(lm_q8_quantize_acts(qx_ok, sx_ok, x_ok, IN) == 0);
    CHECK(lm_matvec_q8i(y, qx_ok, sx_ok, q, sc, IN, OUT) == 0);
    for (int o = 0; o < OUT; o++) CHECK(isfinite(y[o]));
    for (int b = 0; b < 3; b++) {
        memcpy(x, x_ok, sizeof x);
        x[G + 7] = bad[b];
        CHECK(lm_q8_quantize_acts(qx, sx, x, IN) == 0);
        CHECK(isnan(sx[1]));
        for (int i = 0; i < G; i++) CHECK(qx[G + i] == 0);
        CHECK(sx[0] == sx_ok[0] && sx[2] == sx_ok[2]);                  /* neighbours untouched */
        CHECK(memcmp(qx, qx_ok, G) == 0 && memcmp(qx + 2 * G, qx_ok + 2 * G, G) == 0);
        CHECK(lm_matvec_q8i(y, qx, sx, q, sc, IN, OUT) == 0);
        for (int o = 0; o < OUT; o++) CHECK_MSG(isnan(y[o]), "bad value %d: row %d is %g, expected NaN", b, o, y[o]);
    }
}

/* ---- exact results ---- */

static void test_exact_whole_numbers(void)
{
    /* activations are whole numbers with a 127 in every group, so sx = 1 and qx = x exactly;
     * scales are powers of two, so every product and sum below 2^24 is exact in float */
    enum { IN = 64, OUT = 5 };
    static const float pow2[4] = { 0.25f, 0.5f, 1.0f, 2.0f };
    float x[IN], sx[IN / G], sc[(IN / G) * OUT], y[OUT];
    int8_t qx[IN], q[IN * OUT];
    for (int i = 0; i < IN; i++) x[i] = (float)(int)(lt_rand() * 127.0f);
    x[0] = 127.0f; x[G] = -127.0f;
    for (int i = 0; i < IN * OUT; i++) q[i] = rand_w();
    for (int i = 0; i < (IN / G) * OUT; i++) sc[i] = pow2[i & 3];
    CHECK(lm_q8_quantize_acts(qx, sx, x, IN) == 0);
    CHECK(sx[0] == 1.0f && sx[1] == 1.0f);
    for (int i = 0; i < IN; i++) CHECK(qx[i] == (int8_t)x[i]);
    CHECK(lm_matvec_q8i(y, qx, sx, q, sc, IN, OUT) == 0);
    for (int o = 0; o < OUT; o++) {
        double ref = 0;
        for (int i = 0; i < IN; i++) ref += (double)q[o * IN + i] * (double)sc[o * (IN / G) + i / G] * (double)x[i];
        CHECK_MSG((double)y[o] == ref, "row %d: %.9g, exact %.9g", o, y[o], ref);
    }
}

static void test_saturation_corner(void)
{
    /* -128 weights against 127 activations is the largest int16 pair sum (2 * 128 * 127 =
     * 32512): it must not saturate, in either sign combination */
    enum { IN = 64, OUT = 4 };
    int8_t q[IN * OUT], qx[IN];
    float x[IN], sx[IN / G], sc[(IN / G) * OUT], y[OUT];
    for (int o = 0; o < OUT; o++)
        for (int i = 0; i < IN; i++) q[o * IN + i] = (o & 1) ? 127 : -128;
    for (int i = 0; i < IN; i++) x[i] = (i & 1) ? 127.0f : -127.0f;
    for (int i = 0; i < (IN / G) * OUT; i++) sc[i] = (i & 1) ? 0.5f : 1.0f;
    CHECK(lm_q8_quantize_acts(qx, sx, x, IN) == 0);
    CHECK(lm_matvec_q8i(y, qx, sx, q, sc, IN, OUT) == 0);
    for (int o = 0; o < OUT; o++) {
        double ref = 0;
        for (int i = 0; i < IN; i++) ref += (double)q[o * IN + i] * (double)sc[o * (IN / G) + i / G] * (double)x[i];
        CHECK_MSG((double)y[o] == ref, "saturation row %d: %.9g, exact %.9g", o, y[o], ref);
    }
}

/* ---- error against the exact value, and bit identity with scalar ---- */

static void check_shape(int in, int out)
{
    const int ng = in / G;
    int8_t *q = (int8_t *)malloc((size_t)in * out), *qx = (int8_t *)malloc((size_t)in);
    float *sc = (float *)malloc((size_t)ng * out * sizeof *sc), *x = (float *)malloc((size_t)in * sizeof *x);
    float *sx = (float *)malloc((size_t)ng * sizeof *sx);
    float *y = (float *)malloc((size_t)out * sizeof *y), *ys = (float *)malloc((size_t)out * sizeof *ys);
    fill_x(x, in);
    for (size_t i = 0; i < (size_t)in * out; i++) q[i] = rand_w();
    for (size_t i = 0; i < (size_t)ng * out; i++) sc[i] = (lt_rand() * 0.5f + 0.5f) * 0.02f;
    CHECK(lm_q8_quantize_acts(qx, sx, x, in) == 0);
    CHECK(lm_matvec_q8i(y, qx, sx, q, sc, in, out) == 0);
    CHECK(lm_matvec_q8i_scalar(ys, qx, sx, q, sc, in, out) == 0);
    CHECK_MSG(bits_equal(y, ys, out), "q8i in=%d out=%d: %s path differs from scalar", in, out, lm_kernel_isa());
    int bad = 0;
    for (int o = 0; o < out && !bad; o++) {
        double ref = 0, quant = 0, absdot = 0;
        for (int g = 0; g < ng; g++) {
            double wabs = 0;
            for (int i = 0; i < G; i++) {
                const double w = (double)q[(size_t)o * in + g * G + i] * (double)sc[(size_t)o * ng + g];
                ref += w * (double)x[g * G + i];
                absdot += fabs(w * (double)x[g * G + i]);
                wabs += fabs(w);
            }
            quant += 0.5001 * (double)sx[g] * wabs;     /* sum_i |w_i| * (rounding error <= sx/2) */
        }
        const double tol = quant + 1e-5 * absdot + 1e-30; /* + float accumulation */
        if (fabs((double)y[o] - ref) > tol) { CHECK_NEAR(y[o], ref, tol); bad = 1; }
    }
    free(q); free(qx); free(sc); free(x); free(sx); free(y); free(ys);
}

static void test_unaligned(void)
{
    /* nothing may assume 32-byte alignment of any buffer */
    enum { IN = 96, OUT = 3 };
    float xb[IN], sxb[IN / G + 1], sb[(IN / G) * OUT + 1], y1[OUT], y2[OUT];
    int8_t qb[IN * OUT + 1], qxb[IN + 1];
    fill_x(xb, IN);
    for (int i = 0; i < IN * OUT + 1; i++) qb[i] = rand_w();
    for (int i = 0; i < (IN / G) * OUT + 1; i++) sb[i] = 0.01f * (float)(i + 1);
    CHECK(lm_q8_quantize_acts(qxb + 1, sxb + 1, xb, IN) == 0);
    CHECK(lm_matvec_q8i(y1, qxb + 1, sxb + 1, qb + 1, sb + 1, IN, OUT) == 0);
    CHECK(lm_matvec_q8i_scalar(y2, qxb + 1, sxb + 1, qb + 1, sb + 1, IN, OUT) == 0);
    CHECK(bits_equal(y1, y2, OUT));
}

int main(void)
{
    printf("kernel path: %s\n", lm_kernel_isa());
    test_quantiser();
    test_bad_in();
    test_non_finite();
    test_exact_whole_numbers();
    test_saturation_corner();
    test_unaligned();
    static const int ins[] = { 32, 64, 96, 128, 1024, 2560, 9216 };
    static const int outs[] = { 1, 5, 70 }; /* 70 crosses the OpenMP threshold (out > 64) */
    for (size_t i = 0; i < sizeof ins / sizeof *ins; i++)
        for (size_t j = 0; j < sizeof outs / sizeof *outs; j++) check_shape(ins[i], outs[j]);
    return lt_finish("test_kernels_q8i");
}
