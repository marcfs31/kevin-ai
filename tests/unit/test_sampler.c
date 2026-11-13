/* test_sampler.c -- lm_sampler: PCG32 reference vector, determinism, greedy ties, top-k,
 * top-p, error cases. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "chat/lm_sampler.h"

static int g_fail = 0, g_checks = 0;
#define CHECK(c) do { g_checks++; if (!(c)) { g_fail++; fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #c); } } while (0)

/* Reference PCG32 (pcg-c basic): seeded with (42, 54) it yields a15c02b7 7b47f409 ba1d3330
 * 83d2f293 bfa4784b cbed606e. */
static uint64_t r_state, r_inc;
static uint32_t ref_pcg(void)
{
    uint64_t old = r_state;
    r_state = old * UINT64_C(6364136223846793005) + r_inc;
    uint32_t x = (uint32_t)(((old >> 18) ^ old) >> 27), rot = (uint32_t)(old >> 59);
    return (x >> rot) | (x << ((-rot) & 31));
}
static void ref_seed(uint64_t init, uint64_t seq)
{
    r_state = 0; r_inc = (seq << 1) | 1; ref_pcg(); r_state += init; ref_pcg();
}

static void test_pcg_stream(void)
{
    ref_seed(42, 54);
    CHECK(ref_pcg() == 0xa15c02b7u && ref_pcg() == 0x7b47f409u && ref_pcg() == 0xba1d3330u);

    /* Drive the sampler's own stream from that state: with p = (0.7, 0.3) and the draws
     * u = (hi<<21 ^ lo&0x1fffff)/2^53 from consecutive output pairs, the picks follow
     * from the reference outputs. */
    LmSampler s;
    lm_sampler_init(&s, 1.0, 1.0, 0, 1);
    s.state = 0; s.inc = (UINT64_C(54) << 1) | 1;
    ref_seed(42, 54);
    uint64_t st = r_state; /* state after seeding; the sampler continues from here */
    s.state = st;
    float logits[2] = { logf(0.7f), logf(0.3f) };
    ref_seed(42, 54);
    r_state = st;
    for (int i = 0; i < 6; i++) {
        uint64_t hi = ref_pcg(), lo = ref_pcg();
        double u = (double)((hi << 21) ^ (lo & 0x1fffffu)) / 9007199254740992.0;
        int want = u < 0.7 ? 0 : 1, got = -1;
        CHECK(lm_sampler_next(&s, logits, 2, &got) == 0 && got == want);
    }
    lm_sampler_free(&s);
}

static int draw_seq(uint64_t seed, int *out, int n)
{
    LmSampler s;
    lm_sampler_init(&s, 0.8, 0.9, 5, seed);
    float l[16];
    for (int i = 0; i < 16; i++) l[i] = (float)(i % 5) * 0.3f;
    for (int i = 0; i < n; i++) if (lm_sampler_next(&s, l, 16, &out[i])) { lm_sampler_free(&s); return -1; }
    lm_sampler_free(&s);
    return 0;
}

static void test_determinism(void)
{
    int a[64], b[64], c[64];
    CHECK(draw_seq(7, a, 64) == 0 && draw_seq(7, b, 64) == 0 && draw_seq(8, c, 64) == 0);
    CHECK(!memcmp(a, b, sizeof a));
    CHECK(memcmp(a, c, sizeof a) != 0);
}

static void test_greedy(void)
{
    LmSampler s;
    lm_sampler_init(&s, 0.0, 0.5, 3, 1);
    float tie[6] = { 1, 5, 2, 5, 5, 0 };
    int got = -1;
    CHECK(lm_sampler_next(&s, tie, 6, &got) == 0 && got == 1); /* lowest id wins a tie */
    uint64_t before = s.state;
    float one[1] = { -3.0f };
    CHECK(lm_sampler_next(&s, one, 1, &got) == 0 && got == 0);
    CHECK(s.state == before);                                  /* greedy never draws */
    float inf[3] = { 0, INFINITY, INFINITY };
    CHECK(lm_sampler_next(&s, inf, 3, &got) == 0 && got == 1);
    lm_sampler_free(&s);

    /* top_k 1 and a tiny top_p are greedy too, with ties to the lowest id */
    lm_sampler_init(&s, 1.0, 1.0, 1, 3);
    for (int i = 0; i < 20; i++) CHECK(lm_sampler_next(&s, tie, 6, &got) == 0 && got == 1);
    lm_sampler_free(&s);
    lm_sampler_init(&s, 1.0, 1e-9, 0, 3);
    for (int i = 0; i < 20; i++) CHECK(lm_sampler_next(&s, tie, 6, &got) == 0 && got == 1);
    lm_sampler_free(&s);
}

static void test_distribution(void)
{
    float l[4] = { logf(0.5f), logf(0.3f), logf(0.15f), logf(0.05f) };
    int count[4] = { 0 };
    LmSampler s;
    lm_sampler_init(&s, 1.0, 1.0, 0, 11);
    for (int i = 0; i < 40000; i++) { int g; CHECK(lm_sampler_next(&s, l, 4, &g) == 0); count[g]++; }
    lm_sampler_free(&s);
    CHECK(fabs(count[0] / 40000.0 - 0.5) < 0.01 && fabs(count[1] / 40000.0 - 0.3) < 0.01 && fabs(count[3] / 40000.0 - 0.05) < 0.01);

    /* top-k 2: only ids 0 and 1, renormalised to 0.625 / 0.375 */
    memset(count, 0, sizeof count);
    lm_sampler_init(&s, 1.0, 1.0, 2, 11);
    for (int i = 0; i < 40000; i++) { int g; lm_sampler_next(&s, l, 4, &g); count[g]++; }
    lm_sampler_free(&s);
    CHECK(count[2] == 0 && count[3] == 0 && fabs(count[0] / 40000.0 - 0.625) < 0.01);

    /* top-p 0.7: the prefix {0.5, 0.3} (mass 0.8 >= 0.7) survives, 0.625 / 0.375 */
    memset(count, 0, sizeof count);
    lm_sampler_init(&s, 1.0, 0.7, 0, 11);
    for (int i = 0; i < 40000; i++) { int g; lm_sampler_next(&s, l, 4, &g); count[g]++; }
    lm_sampler_free(&s);
    CHECK(count[2] == 0 && count[3] == 0 && fabs(count[0] / 40000.0 - 0.625) < 0.01);

    /* top-k selection over a large vocabulary agrees with the full sort: k = n is the
     * sort path, k < n the selection path; with top_p tiny both must give the argmax */
    enum { N = 5000 };
    float *big = (float *)malloc(N * sizeof *big);
    for (int i = 0; i < N; i++) big[i] = (float)((i * 2654435761u) % 1000) / 100.0f;
    int a, b;
    lm_sampler_init(&s, 1.0, 1e-9, 0, 5);
    CHECK(lm_sampler_next(&s, big, N, &a) == 0);
    lm_sampler_free(&s);
    lm_sampler_init(&s, 1.0, 1e-9, 20, 5);
    CHECK(lm_sampler_next(&s, big, N, &b) == 0);
    lm_sampler_free(&s);
    int best = 0;
    for (int i = 0; i < N; i++) if (big[i] > big[best]) best = i;
    CHECK(a == best && b == best);
    free(big);
}

static void test_errors(void)
{
    LmSampler s;
    int got = 0;
    float l[3] = { 1, 2, 3 }, nan_l[3] = { 1, NAN, 3 };
    lm_sampler_init(&s, 0.7, 0.8, 20, 1);
    CHECK(lm_sampler_next(&s, nan_l, 3, &got) == -1);
    CHECK(lm_sampler_next(&s, l, 0, &got) == -1);
    CHECK(lm_sampler_next(&s, NULL, 3, &got) == -1);
    lm_sampler_free(&s);
    lm_sampler_init(&s, -1.0, 0.8, 0, 1);
    CHECK(lm_sampler_next(&s, l, 3, &got) == -1);
    lm_sampler_free(&s);
    lm_sampler_init(&s, 0.7, 0.0, 0, 1);
    CHECK(lm_sampler_next(&s, l, 3, &got) == -1);
    lm_sampler_free(&s);
    lm_sampler_init(&s, 0.7, 1.5, 0, 1);
    CHECK(lm_sampler_next(&s, l, 3, &got) == -1);
    lm_sampler_free(&s);
    lm_sampler_init(&s, 0.7, 1.0, -2, 1);
    CHECK(lm_sampler_next(&s, l, 3, &got) == -1);
    lm_sampler_free(&s);
}

int main(void)
{
    test_pcg_stream();
    test_determinism();
    test_greedy();
    test_distribution();
    test_errors();
    printf("test_sampler: %d checks, %d failed\n", g_checks, g_fail);
    return g_fail ? 1 : 0;
}
