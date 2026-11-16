/* test_snapshot.c - lm_snapshot / lm_restore on the tiny Qwen3.5 and Gemma 4 fixtures (run from
 * the repository root; a missing fixture is a FAILURE). A snapshot taken at position n and
 * restored after the model went elsewhere (more tokens, a reset) must leave it BIT-IDENTICAL
 * to the moment it was taken: the logits of every token fed afterwards are equal, which
 * exposes the conv, delta-rule and KV state of every layer, the Gemma sliding-window rings
 * included (the fixture window is 8, so positions past 8 wrap). Negative cases: without the
 * restore the logits differ, and a snapshot does not restore into another model. */
#include "lm_test.h"

#include "lm/lm.h"

#define CTX 64
#define TAIL 6
#define JUNK 12

typedef struct { const char *dir, *weights; int fast; const char *label; } Case;

static const Case CASES[] = {
    { "tests/fixtures/qwen35-tiny", NULL, 0, "qwen bf16" },
    { "tests/fixtures/qwen35-tiny", "tests/fixtures/qwen35-tiny/model.q8.safetensors", 0, "qwen q8" },
    { "tests/fixtures/qwen35-tiny", "tests/fixtures/qwen35-tiny/model.q8.safetensors", 1, "qwen q8 fast" },
    { "tests/fixtures/gemma4-tiny", NULL, 0, "gemma bf16" },
    { "tests/fixtures/gemma4-tiny", "tests/fixtures/gemma4-tiny/model.q8.safetensors", 0, "gemma q8" },
};

static LmModel *open_model(const Case *c)
{
    char err[512] = "";
    LmOpts o = { c->weights, CTX, 0, c->fast, 0 };
    LmModel *m = lm_open(c->dir, &o, err, sizeof err);
    CHECK_MSG(m != NULL, "%s: lm_open: %s", c->label, err);
    return m;
}

static void make_tokens(int *tok, int n, int vocab, unsigned seed)
{
    for (int i = 0; i < n; i++) { seed = seed * 1103515245u + 12345u; tok[i] = (int)((seed >> 8) % (unsigned)vocab); }
}

/* feed the tail token by token, keeping every row of logits */
static int run_tail(LmModel *m, const int *tail, float *rows, int V)
{
    for (int t = 0; t < TAIL; t++)
        if (lm_feed(m, tail[t], rows + (size_t)t * V) != LM_OK) return -1;
    return 0;
}

static void check_case(const Case *c, int n)
{
    LmModel *m = open_model(c);
    if (!m) return;
    const int V = lm_n_vocab(m);
    const size_t bytes = (size_t)TAIL * V * sizeof(float);
    int prefix[48], tail[TAIL], junk[JUNK];
    make_tokens(prefix, n, V, 7u);
    make_tokens(tail, TAIL, V, 99u);
    make_tokens(junk, JUNK, V, 31u);
    float *ref = (float *)malloc(bytes), *got = (float *)malloc(bytes);
    LmSnapshot *s = NULL;
    if (!ref || !got) { CHECK_MSG(0, "%s: out of memory", c->label); goto out; }

    CHECK_MSG(lm_prefill(m, prefix, n, NULL) == LM_OK, "%s n=%d: prefill", c->label, n);
    s = lm_snapshot(m);
    CHECK_MSG(s != NULL, "%s n=%d: lm_snapshot returned NULL", c->label, n);
    if (!s) goto out;
    CHECK_MSG(lm_snapshot_pos(s) == n, "%s n=%d: snapshot pos %d", c->label, n, lm_snapshot_pos(s));
    CHECK_MSG(run_tail(m, tail, ref, V) == 0, "%s n=%d: reference tail", c->label, n);

    /* go elsewhere, restore, run the same tail: bit-identical */
    CHECK(lm_prefill(m, junk, JUNK, NULL) == LM_OK);
    CHECK_MSG(lm_restore(m, s) == LM_OK && lm_pos(m) == n, "%s n=%d: restore after more tokens", c->label, n);
    CHECK_MSG(run_tail(m, tail, got, V) == 0 && !memcmp(ref, got, bytes),
              "%s n=%d: logits after a restore differ from those at the snapshot", c->label, n);

    /* after a reset as well; the same snapshot restores more than once */
    lm_reset(m);
    CHECK_MSG(lm_restore(m, s) == LM_OK && lm_pos(m) == n, "%s n=%d: restore after a reset", c->label, n);
    CHECK_MSG(run_tail(m, tail, got, V) == 0 && !memcmp(ref, got, bytes),
              "%s n=%d: logits after reset + restore differ", c->label, n);

    /* NEGATIVE: other tokens in front change what follows (so the checks above can fail) */
    if (n > 0) {
        lm_reset(m);
        CHECK(lm_prefill(m, junk, n < JUNK ? n : JUNK, NULL) == LM_OK);
        CHECK_MSG(run_tail(m, tail, got, V) == 0 && memcmp(ref, got, bytes) != 0,
                  "%s n=%d: other tokens gave the same logits; the test cannot see the state", c->label, n);
    }

    /* NEGATIVE: a snapshot belongs to the model it was taken from */
    LmModel *other = open_model(c);
    if (other) {
        CHECK_MSG(lm_restore(other, s) != LM_OK && lm_pos(other) == 0, "%s: a snapshot restored into another model", c->label);
        lm_close(other);
    }
    /* Gemma at position 0 holds no state at all; Qwen always has its conv and delta state */
    if (n > 0) CHECK_MSG(lm_snapshot_bytes(s) > 0, "%s n=%d: empty snapshot", c->label, n);
out:
    lm_snapshot_free(s);
    free(ref);
    free(got);
    lm_close(m);
}

int main(void)
{
    static const int NS[] = { 0, 1, 5, 7, 8, 9, 17, 33, 40 };
    for (size_t i = 0; i < sizeof CASES / sizeof *CASES; i++) {
        char probe[600];
        snprintf(probe, sizeof probe, "%s/config.json", CASES[i].dir);
        if (access(probe, R_OK) != 0) {
            fprintf(stderr, "  FAIL fixture %s not found (run from the repository root)\n", probe);
            return 1;
        }
        for (size_t k = 0; k < sizeof NS / sizeof *NS; k++) check_case(&CASES[i], NS[k]);
    }
    lm_snapshot_free(NULL);
    return lt_finish("test_snapshot");
}
