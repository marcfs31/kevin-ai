/* test_matmul_q8.c - the batched q8 kernel lm_matmul_q8: for every token and row it must be
 * BIT-IDENTICAL to lm_matvec_q8 (and to its scalar reference), whatever the number of
 * tokens (below, at and past the 4-token tile), the shape, the thread count and the
 * pointer alignment; tokens must not leak into each other; bad arguments write nothing. */
#include "lm_test.h"

#include "core/lm_kernels.h"

#ifdef _OPENMP
#include <omp.h>
#endif

static int bits_equal(const float *a, const float *b, size_t n)
{
    return memcmp(a, b, n * sizeof(float)) == 0;
}

static void fill_weights(int8_t *q, size_t n)
{
    for (size_t i = 0; i < n; i++) q[i] = (i % 97 == 0) ? (int8_t)-128 : (int8_t)(lt_rand() * 127.0f);
}

static void fill_scales(float *sc, size_t n)
{
    for (size_t i = 0; i < n; i++) sc[i] = (lt_rand() * 0.5f + 0.5f) * 0.02f;
}

static void fill_x(float *x, size_t n)
{
    for (size_t i = 0; i < n; i++) x[i] = lt_rand() * 3.0f;
}

static void check_shape(int in, int out, int T)
{
    const int ng = in / LM_Q8_GROUP;
    const size_t ny = (size_t)(T > 0 ? T : 1) * out;
    int8_t *q = (int8_t *)malloc((size_t)in * out);
    float *sc = (float *)malloc((size_t)ng * out * sizeof *sc);
    float *X = (float *)malloc((size_t)(T > 0 ? T : 1) * in * sizeof *X);
    float *Y = (float *)malloc(ny * sizeof *Y), *Yref = (float *)malloc(ny * sizeof *Yref);
    float *Ys = (float *)malloc(ny * sizeof *Ys);
    fill_weights(q, (size_t)in * out);
    fill_scales(sc, (size_t)ng * out);
    fill_x(X, (size_t)T * in);
    CHECK(lm_matmul_q8(Y, X, T, q, sc, in, out) == 0);
    for (int t = 0; t < T; t++) CHECK(lm_matvec_q8(Yref + (size_t)t * out, X + (size_t)t * in, q, sc, in, out) == 0);
    CHECK_MSG(bits_equal(Y, Yref, (size_t)T * out), "matmul in=%d out=%d T=%d: %s path differs from T matvec calls",
              in, out, T, lm_kernel_isa());
    CHECK(lm_matmul_q8_scalar(Ys, X, T, q, sc, in, out) == 0);
    CHECK_MSG(bits_equal(Y, Ys, (size_t)T * out), "matmul in=%d out=%d T=%d: %s path differs from scalar", in, out, T,
              lm_kernel_isa());
    free(q); free(sc); free(X); free(Y); free(Yref); free(Ys);
}

static void test_known_values(void)
{
    /* row 0 = 1..32, row 1 = all -1; token t is the constant (t + 1); scales 0.5 and 2:
     * y[t][0] = 528 * (t+1) * 0.5, y[t][1] = -32 * (t+1) * 2 -- exact in float. 6 tokens
     * cross the 4-token tile into the single-token tail. */
    enum { T = 6, IN = 32, OUT = 2 };
    int8_t q[IN * OUT];
    float sc[OUT] = { 0.5f, 2.0f }, X[T * IN], Y[T * OUT];
    for (int i = 0; i < IN; i++) { q[i] = (int8_t)(i + 1); q[IN + i] = -1; }
    for (int t = 0; t < T; t++) for (int i = 0; i < IN; i++) X[t * IN + i] = (float)(t + 1);
    CHECK(lm_matmul_q8(Y, X, T, q, sc, IN, OUT) == 0);
    for (int t = 0; t < T; t++) {
        CHECK_MSG(Y[t * OUT] == 264.0f * (float)(t + 1), "token %d row 0: %g", t, Y[t * OUT]);
        CHECK_MSG(Y[t * OUT + 1] == -64.0f * (float)(t + 1), "token %d row 1: %g", t, Y[t * OUT + 1]);
    }
}

/* NEGATIVE: one poisoned token must only poison its own outputs; its neighbours in the same
 * tile and in the tail keep the bits they have without it */
static void test_tokens_do_not_leak(void)
{
    enum { T = 7, IN = 64, OUT = 5 };
    int8_t q[IN * OUT];
    float sc[(IN / 32) * OUT], X[T * IN], clean[T * OUT], dirty[T * OUT];
    fill_weights(q, sizeof q);
    fill_scales(sc, sizeof sc / sizeof *sc);
    fill_x(X, sizeof X / sizeof *X);
    CHECK(lm_matmul_q8(clean, X, T, q, sc, IN, OUT) == 0);
    for (int poisoned = 0; poisoned < T; poisoned++) {
        float Xp[T * IN];
        memcpy(Xp, X, sizeof X);
        Xp[poisoned * IN + 40] = NAN;
        CHECK(lm_matmul_q8(dirty, Xp, T, q, sc, IN, OUT) == 0);
        for (int t = 0; t < T; t++)
            for (int o = 0; o < OUT; o++) {
                const float v = dirty[t * OUT + o];
                if (t == poisoned) CHECK_MSG(isnan(v), "poisoned token %d row %d is %g, expected NaN", t, o, v);
                else CHECK_MSG(!memcmp(&v, &clean[t * OUT + o], sizeof v), "token %d row %d changed when token %d was poisoned",
                               t, o, poisoned);
            }
    }
}

static void test_unaligned(void)
{
    /* nothing may assume 32-byte alignment of X, q, scales or Y */
    enum { T = 9, IN = 96, OUT = 3 };
    float xb[T * IN + 1], sb[(IN / 32) * OUT + 1], yb[T * OUT + 1], ys[T * OUT + 1];
    int8_t qb[IN * OUT + 1];
    fill_x(xb, sizeof xb / sizeof *xb);
    fill_weights(qb, sizeof qb);
    for (size_t i = 0; i < sizeof sb / sizeof *sb; i++) sb[i] = 0.01f * (float)(i + 1);
    CHECK(lm_matmul_q8(yb + 1, xb + 1, T, qb + 1, sb + 1, IN, OUT) == 0);
    CHECK(lm_matmul_q8_scalar(ys + 1, xb + 1, T, qb + 1, sb + 1, IN, OUT) == 0);
    CHECK(bits_equal(yb + 1, ys + 1, (size_t)T * OUT));
}

static void test_thread_count(void)
{
#ifdef _OPENMP
    enum { T = 11, IN = 256, OUT = 70 };   /* out > 64: the parallel loop runs */
    static int8_t q[IN * OUT];
    static float sc[(IN / 32) * OUT], X[T * IN], y1[T * OUT], y4[T * OUT];
    fill_weights(q, sizeof q);
    fill_scales(sc, sizeof sc / sizeof *sc);
    fill_x(X, sizeof X / sizeof *X);
    const int saved = omp_get_max_threads();
    omp_set_num_threads(1);
    CHECK(lm_matmul_q8(y1, X, T, q, sc, IN, OUT) == 0);
    omp_set_num_threads(4);
    CHECK(lm_matmul_q8(y4, X, T, q, sc, IN, OUT) == 0);
    omp_set_num_threads(saved);
    CHECK(bits_equal(y1, y4, (size_t)T * OUT));
#endif
}

static void test_bad_arguments(void)
{
    int8_t q[64] = { 0 };
    float sc[2] = { 1, 1 }, X[64] = { 0 }, Y[4] = { 123, 123, 123, 123 };
    CHECK(lm_matmul_q8(Y, X, 2, q, sc, 33, 1) == -1 && Y[0] == 123.0f);
    CHECK(lm_matmul_q8(Y, X, 2, q, sc, -32, 1) == -1 && Y[0] == 123.0f);
    CHECK(lm_matmul_q8_scalar(Y, X, 2, q, sc, 48, 1) == -1 && Y[0] == 123.0f);
    CHECK(lm_matmul_q8(Y, X, -1, q, sc, 32, 1) == -1 && Y[0] == 123.0f);
    CHECK(lm_matmul_q8(Y, X, 0, q, sc, 32, 1) == 0 && Y[0] == 123.0f);   /* no tokens: nothing written */
}

int main(void)
{
    printf("kernel path: %s\n", lm_kernel_isa());
    /* 1056 and 2080 are one group past one and two 32-group input slices (the last slice holds a
     * single group); 2560 ends on a half slice; 9216 is exactly nine slices */
    static const int shapes[][2] = { { 32, 1 }, { 64, 5 }, { 96, 70 }, { 1056, 9 }, { 2080, 13 }, { 2560, 70 }, { 9216, 5 } };
    static const int ts[] = { 0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33 };
    for (size_t s = 0; s < sizeof shapes / sizeof *shapes; s++)
        for (size_t i = 0; i < sizeof ts / sizeof *ts; i++) check_shape(shapes[s][0], shapes[s][1], ts[i]);
    test_known_values();
    test_tokens_do_not_leak();
    test_unaligned();
    test_thread_count();
    test_bad_arguments();
    return lt_finish("test_matmul_q8");
}
