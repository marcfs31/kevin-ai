/* lm_sampler.c -- adapted from oss_sampler.c / k3_sampler.c (Apache-2.0); see lm_sampler.h
 * for the changes. */
#include "chat/lm_sampler.h"

#include <math.h>
#include <stdlib.h>

static uint64_t splitmix64(uint64_t *x)
{
    *x += UINT64_C(0x9e3779b97f4a7c15);
    uint64_t z = *x;
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}

static uint32_t pcg32(LmSampler *s)
{
    uint64_t old = s->state;
    s->state = old * UINT64_C(6364136223846793005) + s->inc;
    uint32_t x = (uint32_t)(((old >> 18) ^ old) >> 27);
    uint32_t rot = (uint32_t)(old >> 59);
    return (x >> rot) | (x << ((-rot) & 31));
}

static double unit(LmSampler *s)
{
    /* 53 bits, so no platform PRNG or libc rand() leaks into the output. The two draws
     * are sequenced explicitly: C leaves the order of operands unspecified. */
    uint64_t hi = pcg32(s);
    uint64_t lo = pcg32(s);
    uint64_t x = (hi << 21) ^ (lo & UINT32_C(0x1fffff));
    return (double)x * (1.0 / 9007199254740992.0);
}

void lm_sampler_init(LmSampler *s, double temperature, double top_p, int top_k, uint64_t seed)
{
    uint64_t x = seed;
    uint64_t state = splitmix64(&x), seq = splitmix64(&x);
    s->temperature = temperature;
    s->top_p = top_p;
    s->top_k = top_k;
    s->state = 0;
    s->inc = (seq << 1u) | 1u;
    (void)pcg32(s);
    s->state += state;
    (void)pcg32(s);
    s->cand = NULL;
    s->prob = NULL;
    s->cap = 0;
}

void lm_sampler_free(LmSampler *s)
{
    free(s->cand);
    free(s->prob);
    s->cand = NULL;
    s->prob = NULL;
    s->cap = 0;
}

/* Descending logit, ties to the lower id: a total order, so the sort is the same across
 * libc implementations. */
static int cmp_desc(const void *aa, const void *bb)
{
    const LmCand *a = (const LmCand *)aa, *b = (const LmCand *)bb;
    if (a->l > b->l) return -1;
    if (a->l < b->l) return 1;
    return a->id < b->id ? -1 : a->id > b->id;
}

int lm_sampler_next(LmSampler *s, const float *logits, int n, int *out)
{
    if (!logits || !out || n <= 0 || !(s->temperature >= 0.0) || !(s->top_p > 0.0) || s->top_p > 1.0 || s->top_k < 0) return -1;
    int best = 0;
    for (int i = 0; i < n; i++) {
        if (logits[i] != logits[i]) return -1; /* NaN */
        if (logits[i] > logits[best]) best = i;
    }
    if (s->temperature == 0.0) { *out = best; return 0; }

    int k = (s->top_k > 0 && s->top_k < n) ? s->top_k : n;
    if (k > s->cap) {
        LmCand *c = (LmCand *)malloc((size_t)k * sizeof *c);
        double *p = (double *)malloc((size_t)k * sizeof *p);
        if (!c || !p) { free(c); free(p); return -1; }
        free(s->cand);
        free(s->prob);
        s->cand = c;
        s->prob = p;
        s->cap = k;
    }
    int nc = 0;
    if (k == n) {
        for (int i = 0; i < n; i++) { s->cand[i].l = (double)logits[i]; s->cand[i].id = i; }
        nc = n;
        qsort(s->cand, (size_t)n, sizeof *s->cand, cmp_desc);
    } else {
        /* Selection of the k best, kept sorted. Scanning ids upward and inserting after any
         * equal logit gives the same lowest-id-first order as cmp_desc. */
        for (int i = 0; i < n; i++) {
            double l = (double)logits[i];
            if (nc == k && l <= s->cand[k - 1].l) continue;
            int j = nc < k ? nc++ : k - 1;
            while (j > 0 && s->cand[j - 1].l < l) { s->cand[j] = s->cand[j - 1]; j--; }
            s->cand[j].l = l;
            s->cand[j].id = i;
        }
    }

    const double max = s->cand[0].l / s->temperature;
    if (!isfinite(max)) { *out = s->cand[0].id; return 0; }
    double sum = 0.0;
    for (int i = 0; i < nc; i++) {
        s->prob[i] = exp(s->cand[i].l / s->temperature - max);
        sum += s->prob[i];
    }
    if (!(sum > 0.0) || !isfinite(sum)) { *out = s->cand[0].id; return 0; }
    /* Nucleus over the renormalised survivors: the shortest prefix whose mass reaches top_p. */
    const double target = s->top_p * sum;
    double keep = 0.0;
    int nk = 0;
    while (nk < nc && keep < target) keep += s->prob[nk++];
    if (!nk) nk = 1;
    double r = unit(s) * keep, acc = 0.0;
    for (int i = 0; i < nk; i++) {
        acc += s->prob[i];
        if (r < acc || i + 1 == nk) { *out = s->cand[i].id; return 0; }
    }
    return -1;
}
