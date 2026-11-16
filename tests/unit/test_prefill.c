/* test_prefill.c - lm_prefill against lm_feed on the tiny Qwen3.5 and Gemma 4 fixtures
 * (run from the repository root; a missing fixture is a FAILURE). The contract is that n
 * tokens through lm_prefill leave the model in a state BIT-IDENTICAL to n calls of lm_feed:
 * the last logits are equal, and so is everything that follows (which exposes the conv,
 * delta-rule and KV state of every layer). Also: chunk boundaries (32 tokens), prompts
 * split across calls, bf16 / q8 / q8 with the int8 fast path, and the error contracts
 * (nothing is fed and the model is unchanged on LM_ERR_TOKEN and LM_ERR_CTX). */
#include "lm_test.h"

#include "lm/lm.h"

#define TAIL 5 /* tokens fed after the prefill to expose the state */

typedef struct { const char *dir, *weights; int fast; const char *label; } Case;

static const Case CASES[] = {
    { "tests/fixtures/qwen35-tiny", NULL, 0, "qwen bf16" },
    { "tests/fixtures/qwen35-tiny", "tests/fixtures/qwen35-tiny/model.q8.safetensors", 0, "qwen q8" },
    { "tests/fixtures/qwen35-tiny", "tests/fixtures/qwen35-tiny/model.q8.safetensors", 1, "qwen q8 fast" },
    { "tests/fixtures/gemma4-tiny", NULL, 0, "gemma bf16" },
    { "tests/fixtures/gemma4-tiny", "tests/fixtures/gemma4-tiny/model.q8.safetensors", 0, "gemma q8" },
};

static LmModel *open_model(const Case *c, int ctx)
{
    char err[512] = "";
    LmOpts o = { c->weights, ctx, 0, c->fast, 0 };
    LmModel *m = lm_open(c->dir, &o, err, sizeof err);
    CHECK_MSG(m != NULL, "%s: lm_open: %s", c->label, err);
    return m;
}

static void make_tokens(int *tok, int n, int vocab)
{
    unsigned s = 12345u;
    for (int i = 0; i < n; i++) { s = s * 1103515245u + 12345u; tok[i] = (int)((s >> 8) % (unsigned)vocab); }
}

static int same(const float *a, const float *b, int n) { return !memcmp(a, b, (size_t)n * sizeof(float)); }

/* feed the same TAIL tokens to both models with logits each step; every row must match */
static void tails_match(const Case *c, LmModel *a, LmModel *b, int V, const char *what)
{
    float *la = (float *)malloc((size_t)V * 4), *lb = (float *)malloc((size_t)V * 4);
    for (int k = 0; k < TAIL; k++) {
        const int t = (k * 37 + 11) % V;
        CHECK(lm_feed(a, t, la) == LM_OK && lm_feed(b, t, lb) == LM_OK);
        CHECK_MSG(same(la, lb, V), "%s %s: logits differ %d tokens after the prefill", c->label, what, k + 1);
    }
    free(la); free(lb);
}

static void check_n(const Case *c, int n)
{
    LmModel *a = open_model(c, 0), *b = open_model(c, 0);
    if (!a || !b) { lm_close(a); lm_close(b); return; }
    const int V = lm_n_vocab(a);
    int tok[80];
    make_tokens(tok, n, V);
    float *la = (float *)malloc((size_t)V * 4), *lb = (float *)malloc((size_t)V * 4);
    for (int i = 0; i < n; i++) CHECK(lm_feed(a, tok[i], i == n - 1 ? la : NULL) == LM_OK);
    CHECK(lm_prefill(b, tok, n, lb) == LM_OK);
    CHECK(lm_pos(a) == n && lm_pos(b) == n);
    CHECK_MSG(same(la, lb, V), "%s n=%d: last logits differ from n lm_feed calls", c->label, n);
    tails_match(c, a, b, V, "after prefill");
    free(la); free(lb);
    lm_close(a); lm_close(b);
}

static void check_split(const Case *c)
{
    /* a prompt prefilled in three calls (one of them 1 token, one crossing a chunk) = one call */
    LmModel *a = open_model(c, 0), *b = open_model(c, 0);
    if (!a || !b) { lm_close(a); lm_close(b); return; }
    const int V = lm_n_vocab(a), n = 70;
    int tok[80];
    make_tokens(tok, n, V);
    CHECK(lm_prefill(a, tok, n, NULL) == LM_OK);
    CHECK(lm_prefill(b, tok, 20, NULL) == LM_OK);
    CHECK(lm_prefill(b, tok + 20, 1, NULL) == LM_OK);
    CHECK(lm_prefill(b, tok + 21, n - 21, NULL) == LM_OK);
    CHECK(lm_pos(a) == n && lm_pos(b) == n);
    tails_match(c, a, b, V, "split prefill");
    lm_close(a); lm_close(b);
}

/* NEGATIVE: a rejected call must leave the model exactly as it was */
static void check_errors(const Case *c)
{
    LmModel *m = open_model(c, 8), *fresh = open_model(c, 8);
    if (!m || !fresh) { lm_close(m); lm_close(fresh); return; }
    const int V = lm_n_vocab(m);
    int tok[10];
    make_tokens(tok, 10, V);
    float *l = (float *)malloc((size_t)V * 4);
    l[0] = 123.0f;

    int bad[4] = { tok[0], tok[1], V, tok[3] };        /* the third id is out of range */
    CHECK(lm_prefill(m, bad, 4, l) == LM_ERR_TOKEN && lm_pos(m) == 0 && l[0] == 123.0f);
    bad[2] = -1;
    CHECK(lm_prefill(m, bad, 4, l) == LM_ERR_TOKEN && lm_pos(m) == 0);
    CHECK(lm_prefill(m, tok, -1, l) == LM_ERR_TOKEN && lm_pos(m) == 0);
    CHECK(lm_prefill(m, NULL, 3, l) == LM_ERR_TOKEN && lm_pos(m) == 0);
    CHECK(lm_prefill(m, tok, 9, l) == LM_ERR_CTX && lm_pos(m) == 0 && l[0] == 123.0f);   /* ctx is 8 */
    CHECK(lm_prefill(m, NULL, 0, l) == LM_OK && lm_pos(m) == 0 && l[0] == 123.0f);       /* n == 0 is a no-op */
    tails_match(c, m, fresh, V, "after rejected prefills");   /* m behaves like a model never touched */

    LmModel *full = open_model(c, 8);
    if (full) {
        CHECK(lm_prefill(full, tok, 8, NULL) == LM_OK && lm_pos(full) == 8);               /* exactly fits */
        CHECK(lm_prefill(full, tok, 1, l) == LM_ERR_CTX && lm_pos(full) == 8);             /* and no room for more */
        lm_close(full);
    }
    free(l);
    lm_close(m); lm_close(fresh);
}

int main(void)
{
    static const int NS[] = { 1, 2, 3, 4, 5, 7, 8, 31, 32, 33, 40, 64, 65 };
    for (size_t i = 0; i < sizeof CASES / sizeof *CASES; i++) {
        char probe[600];
        snprintf(probe, sizeof probe, "%s/config.json", CASES[i].dir);
        if (access(probe, R_OK) != 0) {
            fprintf(stderr, "  FAIL fixture %s not found (run from the repository root)\n", probe);
            return 1;
        }
        for (size_t k = 0; k < sizeof NS / sizeof *NS; k++) check_n(&CASES[i], NS[k]);
        check_split(&CASES[i]);
        check_errors(&CASES[i]);
    }
    return lt_finish("test_prefill");
}
