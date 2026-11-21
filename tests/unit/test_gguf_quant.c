/* test_gguf_quant.c - the block-quantised matrix multiply: block widening against the formula,
 * scalar and AVX2 paths bit-identical, a token's result independent of T and of its batch
 * mates, and the refusals. Blocks are random bytes with a finite f16 scale. */
#include "lm_test.h"

#include "core/lm_bf16.h"
#include "core/lm_gguf_quant.h"
#include "io/lm_gguf.h"

static unsigned long long rs = 88172645463325252ULL;
static unsigned rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (unsigned)(rs >> 16); }

static void random_blocks(unsigned char *b, int type, int nblocks)
{
    int be; size_t bb;
    lm_gguf_type_block(type, &be, &bb);
    for (int i = 0; i < nblocks; i++) {
        unsigned char *p = b + (size_t)i * bb;
        for (size_t k = 0; k < bb; k++) p[k] = (unsigned char)rnd();
        /* finite, modest f16 scales: exponent 5..14 */
        const uint16_t h1 = (uint16_t)((rnd() & 0x8000) | ((5 + rnd() % 10) << 10) | (rnd() & 0x3ff));
        const uint16_t h2 = (uint16_t)((rnd() & 0x8000) | ((5 + rnd() % 10) << 10) | (rnd() & 0x3ff));
        size_t d_at = type == LM_GG_Q6_K ? 208 : 0;
        memcpy(p + d_at, &h1, 2);
        if (type == LM_GG_Q4_K || type == LM_GG_Q5_K) memcpy(p + 2, &h2, 2);
    }
}

static const int TYPES[] = { LM_GG_Q4_K, LM_GG_Q5_K, LM_GG_Q6_K, LM_GG_Q8_0 };

static void test_block_formula(void)
{
    /* Q8_0: w = d * q, exact */
    unsigned char b[34] = { 0 };
    const uint16_t d = 0x3800;   /* 0.5 */
    memcpy(b, &d, 2);
    for (int i = 0; i < 32; i++) b[2 + i] = (unsigned char)(int8_t)(i - 16);
    float w[32];
    lm_gg_block(LM_GG_Q8_0, b, w);
    for (int i = 0; i < 32; i++) CHECK(w[i] == 0.5f * (float)(i - 16));

    /* Q4_K: scales 1..8 (sub-block j has sc = j + 1, m = 0), d = 1, qs nibbles all 3 / 5 */
    unsigned char k[144] = { 0 };
    const uint16_t one = 0x3c00;
    memcpy(k, &one, 2);
    for (int j = 0; j < 4; j++) { k[4 + j] = (unsigned char)(j + 1); k[8 + j] = 0; }
    for (int j = 4; j < 8; j++) k[8 + j] = (unsigned char)(j + 1);   /* sc[j] low nibble, m = 0; sc < 16 */
    for (int i = 0; i < 128; i++) k[16 + i] = 0x53;   /* low nibble 3, high nibble 5 */
    float x[256];
    lm_gg_block(LM_GG_Q4_K, k, x);
    for (int j = 0; j < 4; j++)
        for (int l = 0; l < 32; l++) {
            CHECK(x[64 * j + l] == (float)((2 * j + 1) * 3));
            CHECK(x[64 * j + 32 + l] == (float)((2 * j + 2) * 5));
        }
}

static void test_paths_agree(void)
{
    for (size_t ti = 0; ti < sizeof TYPES / sizeof *TYPES; ti++) {
        const int type = TYPES[ti];
        int be; size_t bb;
        lm_gguf_type_block(type, &be, &bb);
        unsigned char blk[256];
        float a[256], c[256];
        for (int r = 0; r < 50; r++) {
            random_blocks(blk, type, 1);
            lm_gg_block(type, blk, a);
            lm_gg_block_scalar(type, blk, c);
            CHECK_MSG(!memcmp(a, c, (size_t)be * sizeof(float)), "%s block %d", lm_gguf_type_name(type), r);
        }
        const int in = 3 * 256, out = 70, T = 19;
        unsigned char *W = (unsigned char *)malloc((size_t)out * (in / be) * bb);
        random_blocks(W, type, out * (in / be));
        float *X = (float *)malloc((size_t)T * in * sizeof(float));
        for (int i = 0; i < T * in; i++) X[i] = (float)((int)(rnd() % 2001) - 1000) / 500.0f;
        float *Y = (float *)malloc((size_t)T * out * sizeof(float)), *Ys = (float *)malloc((size_t)T * out * sizeof(float));
        float *one = (float *)malloc((size_t)out * sizeof(float));
        CHECK(lm_gg_matmul(Y, X, T, type, W, in, out) == 0);
        CHECK(lm_gg_matmul_scalar(Ys, X, T, type, W, in, out) == 0);
        CHECK_MSG(!memcmp(Y, Ys, (size_t)T * out * sizeof(float)), "%s: %s and scalar differ", lm_gguf_type_name(type), lm_gg_isa());
        /* a token alone gives exactly the row it has inside a batch (any T, any tile) */
        for (int t = 0; t < T; t += 3) {
            CHECK(lm_gg_matmul(one, X + (size_t)t * in, 1, type, W, in, out) == 0);
            CHECK_MSG(!memcmp(one, Y + (size_t)t * out, (size_t)out * sizeof(float)), "%s token %d of %d", lm_gguf_type_name(type), t, T);
        }
        /* against a double reference over the widened rows */
        float *row = (float *)malloc((size_t)in * sizeof(float));
        for (int o = 0; o < out; o += 17) {
            CHECK(lm_gg_row(row, type, W, in, o) == 0);
            for (int t = 0; t < T; t += 6) {
                double s = 0, mag = 0;
                for (int i = 0; i < in; i++) { s += (double)row[i] * X[(size_t)t * in + i]; mag += fabs((double)row[i] * X[(size_t)t * in + i]); }
                CHECK_NEAR(Y[(size_t)t * out + o], s, 1e-5 * mag + 1e-30);
            }
        }
        free(row); free(W); free(X); free(Y); free(Ys); free(one);
    }
}

static void test_refusals(void)
{
    float y[4] = { 7, 7, 7, 7 }, x[256] = { 0 };
    unsigned char w[210 * 4] = { 0 };
    CHECK(lm_gg_matmul(y, x, 1, LM_GG_F32, w, 256, 1) == -1);   /* F32/F16 matrices are not this kernel's */
    CHECK(lm_gg_matmul(y, x, 1, 99, w, 256, 1) == -1);
    CHECK(lm_gg_matmul(y, x, 1, LM_GG_Q4_K, w, 100, 1) == -1);  /* not a whole number of blocks */
    CHECK(lm_gg_matmul(y, x, -1, LM_GG_Q4_K, w, 256, 1) == -1);
    CHECK(lm_gg_matmul(y, x, 0, LM_GG_Q4_K, w, 256, 1) == 0 && y[0] == 7);   /* T == 0 writes nothing */
    CHECK(y[0] == 7);
    CHECK(lm_gg_row(x, 99, w, 256, 0) == -1);
}

int main(void)
{
    test_block_formula();
    test_paths_agree();
    test_refusals();
    printf("kernel path: %s\n", lm_gg_isa());
    return lt_finish("test_gguf_quant");
}
