/* test_mark.c - lm_mark / lm_restore of a mark on the tiny Qwen3.5, Gemma 4 and Qwen2 fixtures
 * (run from the repository root; a missing fixture is a FAILURE). A mark taken at position n
 * and restored after more tokens were fed must leave the model BIT-IDENTICAL to the moment it
 * was taken: the logits of every token fed afterwards are equal. The junk fed in between is
 * longer than the Gemma fixture's window of 8, so its rings wrap and the mark must bring them
 * back. Negative cases: a mark does not rewind after a reset, after a full restore, into
 * another model, or forwards; and a mark is smaller than a snapshot once there are KV rows. */
#include "lm_test.h"

#include "lm/lm.h"

#define CTX 64
#define TAIL 6
#define JUNK 12

typedef struct { const char *dir, *weights; const char *label; int kv_free; } Case;

static const Case CASES[] = {
    { "tests/fixtures/qwen35-tiny", NULL, "qwen bf16", 0 },
    { "tests/fixtures/qwen35-tiny", "tests/fixtures/qwen35-tiny/model.q8.safetensors", "qwen q8", 0 },
    { "tests/fixtures/gemma4-tiny", NULL, "gemma bf16", 0 },
    { "tests/fixtures/gemma4-tiny", "tests/fixtures/gemma4-tiny/model.q8.safetensors", "gemma q8", 0 },
    { "tests/fixtures/qwen2-tiny", NULL, "qwen2", 1 },
};

static LmModel *open_model(const Case *c)
{
    char err[512] = "";
    LmOpts o = { c->weights, CTX, 0, 0, 0, 0 };
    LmModel *m = lm_open(c->dir, &o, err, sizeof err);
    CHECK_MSG(m != NULL, "%s: lm_open: %s", c->label, err);
    return m;
}

static void make_tokens(int *tok, int n, int vocab, unsigned seed)
{
    for (int i = 0; i < n; i++) { seed = seed * 1103515245u + 12345u; tok[i] = (int)((seed >> 8) % (unsigned)vocab); }
}

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
    int prefix[40], tail[TAIL], junk[JUNK];
    make_tokens(prefix, n, V, 7u);
    make_tokens(tail, TAIL, V, 99u);
    make_tokens(junk, JUNK, V, 31u);
    float *ref = (float *)malloc(bytes), *got = (float *)malloc(bytes);
    LmSnapshot *k = NULL, *later = NULL, *full = NULL;
    if (!ref || !got) { CHECK_MSG(0, "%s: out of memory", c->label); goto out; }

    CHECK_MSG(lm_prefill(m, prefix, n, NULL) == LM_OK, "%s n=%d: prefill", c->label, n);
    k = lm_mark(m);
    CHECK_MSG(k != NULL && lm_snapshot_pos(k) == n, "%s n=%d: lm_mark", c->label, n);
    if (!k) goto out;
    CHECK_MSG(run_tail(m, tail, ref, V) == 0, "%s n=%d: reference tail", c->label, n);

    /* go further, rewind, run the same tail: bit-identical, as often as needed */
    for (int rep = 0; rep < 2; rep++) {
        CHECK(lm_prefill(m, junk, JUNK, NULL) == LM_OK);
        CHECK_MSG(lm_restore(m, k) == LM_OK && lm_pos(m) == n, "%s n=%d: rewind %d", c->label, n, rep);
        CHECK_MSG(run_tail(m, tail, got, V) == 0 && !memcmp(ref, got, bytes),
                  "%s n=%d: logits after rewind %d differ from those at the mark", c->label, n, rep);
    }

    /* a mark taken later still rewinds over the earlier one's tail; then the earlier one again */
    later = lm_mark(m);
    full = lm_snapshot(m);
    CHECK(later && lm_snapshot_pos(later) == n + TAIL && full);
    /* the size does not grow with the position (Gemma's rings do until they wrap), and the KV
     * rows are not in it */
    if (later && full) {
        CHECK_MSG(lm_snapshot_bytes(later) == lm_snapshot_bytes(k) || !strncmp(c->label, "gemma", 5),
                  "%s n=%d: mark size depends on the position", c->label, n);
        if (c->kv_free) CHECK_MSG(lm_snapshot_bytes(later) == 0, "%s: a Qwen2 mark holds state", c->label);
        CHECK_MSG(lm_snapshot_bytes(later) < lm_snapshot_bytes(full), "%s n=%d: mark %zu bytes, snapshot %zu",
                  c->label, n, lm_snapshot_bytes(later), lm_snapshot_bytes(full));
    }
    lm_snapshot_free(full);
    CHECK(lm_prefill(m, junk, JUNK, NULL) == LM_OK);
    CHECK_MSG(later && lm_restore(m, later) == LM_OK && lm_pos(m) == n + TAIL, "%s n=%d: later mark", c->label, n);
    CHECK_MSG(lm_restore(m, k) == LM_OK && lm_pos(m) == n, "%s n=%d: back to the first mark", c->label, n);
    CHECK_MSG(run_tail(m, tail, got, V) == 0 && !memcmp(ref, got, bytes), "%s n=%d: after two rewinds", c->label, n);

    /* NEGATIVE: a mark never rewinds forwards */
    lm_restore(m, k);
    CHECK_MSG(later && lm_restore(m, later) != LM_OK && lm_pos(m) == n, "%s n=%d: a mark ahead of the model restored", c->label, n);

    /* NEGATIVE: after a reset or a full restore the rows below the mark may be anything */
    full = lm_snapshot(m);
    CHECK(full != NULL);
    lm_reset(m);
    CHECK(lm_prefill(m, junk, JUNK, NULL) == LM_OK);
    CHECK_MSG(lm_restore(m, k) != LM_OK && lm_pos(m) == JUNK, "%s n=%d: a mark rewound after a reset", c->label, n);
    k = (lm_snapshot_free(k), NULL);
    k = lm_mark(m);
    CHECK(full && lm_restore(m, full) == LM_OK);
    CHECK(lm_prefill(m, junk, JUNK, NULL) == LM_OK);
    CHECK_MSG(k && lm_restore(m, k) != LM_OK, "%s n=%d: a mark rewound after a full restore", c->label, n);

    /* NEGATIVE: other tokens in front change what follows (so the checks above can fail) */
    if (n > 0) {
        lm_reset(m);
        CHECK(lm_prefill(m, junk, n < JUNK ? n : JUNK, NULL) == LM_OK);
        CHECK_MSG(run_tail(m, tail, got, V) == 0 && memcmp(ref, got, bytes) != 0,
                  "%s n=%d: other tokens gave the same logits; the test cannot see the state", c->label, n);
    }

    /* NEGATIVE: a mark belongs to the model it was taken from */
    LmModel *other = open_model(c);
    if (other && k) {
        CHECK(lm_prefill(other, junk, JUNK, NULL) == LM_OK);
        CHECK_MSG(lm_restore(other, k) != LM_OK && lm_pos(other) == JUNK, "%s: a mark restored into another model", c->label);
    }
    lm_close(other);

out:
    lm_snapshot_free(k);
    lm_snapshot_free(later);
    lm_snapshot_free(full);
    free(ref);
    free(got);
    lm_close(m);
}

int main(void)
{
    static const int NS[] = { 0, 1, 7, 8, 9, 17, 33 };
    for (size_t i = 0; i < sizeof CASES / sizeof *CASES; i++) {
        if (access(CASES[i].dir, R_OK) != 0) {
            fprintf(stderr, "  FAIL fixture %s not found (run from the repository root)\n", CASES[i].dir);
            return 1;
        }
        for (size_t k = 0; k < sizeof NS / sizeof *NS; k++) check_case(&CASES[i], NS[k]);
    }
    return lt_finish("test_mark");
}
