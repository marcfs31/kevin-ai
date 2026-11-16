/* test_mtp.c - Qwen3.5's MTP head and speculative decoding on the tiny fixture (run from the
 * repository root; a missing fixture is a FAILURE). Everything is checked BIT-EXACT against
 * plain lm_feed decoding, the deterministic oracle (the head's own maths is checked against
 * the float64 reference by tools/check_golden.py):
 *   - the head's draft logits do not depend on how the tokens were fed (one by one, one
 *     prefill, chunks of any size, across the 32-token chunk boundary);
 *   - lm_verify's rows are the logits lm_feed gives for the same tokens;
 *   - after lm_accept(1) (a rejected draft) or lm_accept(2), every later logit and draft equals
 *     a model that was only ever fed the kept tokens;
 *   - a snapshot carries the head's state.
 * Negative cases: calls out of order return LM_ERR_STATE and change nothing; a model opened
 * without the head and a Gemma model refuse it, and a q8 file without the head is refused with
 * it (and opens without it). */
#include "lm_test.h"

#include "lm/lm.h"
#include "io/lm_st.h"
#include "io/lm_stw.h"

#define DIR "tests/fixtures/qwen35-tiny"
#define Q8 DIR "/model.q8.safetensors"
#define CTX 96
#define N 44

typedef struct { const char *weights; int fast; const char *label; } Case;
static const Case CASES[] = { { NULL, 0, "bf16" }, { Q8, 0, "q8" }, { Q8, 1, "q8 fast" } };

static LmModel *open_m(const Case *c, int mtp)
{
    char err[512] = "";
    LmOpts o = { c->weights, CTX, 0, c->fast, mtp };
    LmModel *m = lm_open(DIR, &o, err, sizeof err);
    CHECK_MSG(m != NULL, "%s: lm_open: %s", c->label, err);
    return m;
}

static void make_tokens(int *tok, int n, int vocab, unsigned seed)
{
    for (int i = 0; i < n; i++) { seed = seed * 1103515245u + 12345u; tok[i] = (int)((seed >> 8) % (unsigned)vocab); }
}

static int argmax(const float *r, int n) { int b = 0; for (int i = 1; i < n; i++) if (r[i] > r[b]) b = i; return b; }

/* feed tok[0..n) in chunks of `chunk` (0 = token by token with lm_feed) */
static int feed_split(LmModel *m, const int *tok, int n, int chunk)
{
    for (int i = 0; i < n;) {
        if (chunk == 0) { if (lm_feed(m, tok[i], NULL) != LM_OK) return -1; i++; continue; }
        const int k = n - i < chunk ? n - i : chunk;
        if (lm_prefill(m, tok + i, k, NULL) != LM_OK) return -1;
        i += k;
    }
    return 0;
}

static void test_draft_paths(const Case *c)
{
    LmModel *m = open_m(c, 1);
    if (!m) return;
    const int V = lm_n_vocab(m);
    const size_t row = (size_t)V * sizeof(float);
    int tok[N + 1];
    make_tokens(tok, N + 1, V, 5u);
    float *ref = (float *)malloc(row), *got = (float *)malloc(row);
    static const int SPLITS[] = { 0, 1, 3, 7, 32, 40, N };
    static const int AT[] = { 1, 2, 9, 31, 32, 33, 40, N };
    for (size_t a = 0; a < sizeof AT / sizeof *AT; a++) {
        const int n = AT[a];
        lm_reset(m);
        CHECK(feed_split(m, tok, n, 0) == 0);
        CHECK(lm_draft(m, tok[n], ref) == LM_OK);
        for (size_t s = 1; s < sizeof SPLITS / sizeof *SPLITS; s++) {
            lm_reset(m);
            CHECK(feed_split(m, tok, n, SPLITS[s]) == 0);
            CHECK_MSG(lm_draft(m, tok[n], got) == LM_OK && !memcmp(ref, got, row),
                      "%s: the draft after %d tokens fed in chunks of %d differs from token by token", c->label, n, SPLITS[s]);
        }
        /* a draft changes nothing: another token first, then the same one, gives the same row */
        CHECK(lm_draft(m, (tok[n] + 1) % V, got) == LM_OK);
        CHECK(lm_draft(m, tok[n], got) == LM_OK && !memcmp(ref, got, row));
        CHECK(lm_pos(m) == n);
    }
    /* NEGATIVE: the draft depends on the token it is given and on the history */
    lm_reset(m);
    CHECK(feed_split(m, tok, 9, 0) == 0);
    CHECK(lm_draft(m, tok[9], ref) == LM_OK && lm_draft(m, (tok[9] + 1) % V, got) == LM_OK);
    CHECK(memcmp(ref, got, row) != 0);
    lm_reset(m);
    CHECK(feed_split(m, tok + 1, 9, 0) == 0);
    CHECK(lm_draft(m, tok[9], got) == LM_OK && memcmp(ref, got, row) != 0);
    free(ref);
    free(got);
    lm_close(m);
}

/* verify [a, b] at position n, accept k, then draft and feed on: everything equals a model
 * that was only fed the kept tokens */
static void test_verify_accept(const Case *c)
{
    LmModel *m = open_m(c, 1), *r = open_m(c, 1);
    if (!m || !r) { lm_close(m); lm_close(r); return; }
    const int V = lm_n_vocab(m);
    const size_t row = (size_t)V * sizeof(float);
    int tok[N];
    make_tokens(tok, N, V, 11u);
    float *rows = (float *)malloc(2 * row), *want = (float *)malloc(row), *got = (float *)malloc(row);
    static const int AT[] = { 1, 5, 30, 31, 32, 40 };
    for (size_t a = 0; a < sizeof AT / sizeof *AT; a++) {
        const int n = AT[a];
        for (int k = 1; k <= 2; k++) {
            const int pair[2] = { tok[n], k == 2 ? tok[n + 1] : (tok[n + 1] + 7) % V };
            lm_reset(m);
            lm_reset(r);
            CHECK(lm_prefill(m, tok, n, NULL) == LM_OK && lm_prefill(r, tok, n, NULL) == LM_OK);
            CHECK(lm_verify(m, pair, 2, rows) == LM_OK && lm_pos(m) == n + 2);
            CHECK(lm_feed(r, pair[0], want) == LM_OK);
            CHECK_MSG(!memcmp(rows, want, row), "%s n=%d: verify row 0 differs from lm_feed", c->label, n);
            if (k == 2) {
                CHECK(lm_feed(r, pair[1], want) == LM_OK);
                CHECK_MSG(!memcmp(rows + V, want, row), "%s n=%d: verify row 1 differs from lm_feed", c->label, n);
            }
            CHECK(lm_accept(m, k) == LM_OK && lm_pos(m) == n + k && lm_pos(r) == n + k);
            CHECK(lm_draft(m, tok[n + k], got) == LM_OK && lm_draft(r, tok[n + k], want) == LM_OK);
            CHECK_MSG(!memcmp(got, want, row), "%s n=%d k=%d: the draft after accept differs", c->label, n, k);
            for (int t = n + k; t < n + k + 3 && t < N; t++) {
                CHECK(lm_feed(m, tok[t], got) == LM_OK && lm_feed(r, tok[t], want) == LM_OK);
                CHECK_MSG(!memcmp(got, want, row), "%s n=%d k=%d: logits at %d differ after accept", c->label, n, k, t);
            }
        }
    }
    /* NEGATIVE: keeping a wrong token changes what follows (so the checks above can fail) */
    lm_reset(m);
    lm_reset(r);
    const int bad[2] = { tok[5], (tok[6] + 7) % V };
    CHECK(lm_prefill(m, tok, 5, NULL) == LM_OK && lm_prefill(r, tok, 5, NULL) == LM_OK);
    CHECK(lm_prefill(m, bad, 2, NULL) == LM_OK && lm_feed(r, tok[5], NULL) == LM_OK);
    CHECK(lm_feed(m, tok[6], got) == LM_OK && lm_feed(r, tok[6], want) == LM_OK && memcmp(got, want, row) != 0);
    free(rows);
    free(want);
    free(got);
    lm_close(m);
    lm_close(r);
}

/* greedy decoding through lm_greedy_check (what lm-run --mtp runs) gives the plain greedy ids,
 * with drafts both accepted and rejected */
static void test_greedy_equal(const Case *c)
{
    LmModel *m = open_m(c, 1);
    if (!m) return;
    const int V = lm_n_vocab(m);
    const size_t row = (size_t)V * sizeof(float);
    int tok[8], plain[24], spec[24];
    make_tokens(tok, 8, V, 3u);
    float *lg = (float *)malloc(row), *rows = (float *)malloc(2 * row);
    CHECK(lm_prefill(m, tok, 8, lg) == LM_OK);
    for (int i = 0; i < 24; i++) { plain[i] = argmax(lg, V); CHECK(lm_feed(m, plain[i], lg) == LM_OK); }
    /* drafts from the plain run itself (a perfect head), every third one made wrong */
    lm_reset(m);
    CHECK(lm_prefill(m, tok, 8, lg) == LM_OK);
    int n = 0, accepted = 0, rejected = 0;
    while (n < 23) {
        spec[n] = argmax(lg, V);
        const int d = (n % 3 == 2) ? (plain[n + 1] + 1) % V : plain[n + 1];
        int kept = 0;
        CHECK(lm_greedy_check(m, spec[n], d, rows, lg, &kept) == LM_OK && lm_pos(m) == 8 + n + kept);
        n++;
        if (kept == 2) { spec[n++] = d; accepted++; }
        else { CHECK(kept == 1); rejected++; }
    }
    if (n < 24) spec[n] = argmax(lg, V), n++;
    CHECK_MSG(accepted > 3 && rejected > 3, "%s: accepted %d rejected %d", c->label, accepted, rejected);
    CHECK_MSG(!memcmp(plain, spec, (size_t)n * sizeof(int)), "%s: speculative greedy ids differ from plain greedy", c->label);
    free(lg);
    free(rows);
    lm_close(m);
}

static void test_snapshot_carries_head(const Case *c)
{
    LmModel *m = open_m(c, 1);
    if (!m) return;
    const int V = lm_n_vocab(m);
    const size_t row = (size_t)V * sizeof(float);
    int tok[20];
    make_tokens(tok, 20, V, 17u);
    float *a = (float *)malloc(row), *b = (float *)malloc(row);
    CHECK(lm_prefill(m, tok, 12, NULL) == LM_OK);
    LmSnapshot *s = lm_snapshot(m);
    CHECK(s != NULL && lm_draft(m, tok[12], a) == LM_OK);
    CHECK(lm_prefill(m, tok + 12, 6, NULL) == LM_OK);
    CHECK(lm_restore(m, s) == LM_OK && lm_draft(m, tok[12], b) == LM_OK && !memcmp(a, b, row));
    lm_reset(m);
    CHECK(lm_restore(m, s) == LM_OK && lm_draft(m, tok[12], b) == LM_OK && !memcmp(a, b, row));
    lm_snapshot_free(s);
    free(a);
    free(b);
    lm_close(m);
}

static void test_refusals(void)
{
    LmModel *m = open_m(&CASES[0], 1);
    if (!m) return;
    const int V = lm_n_vocab(m);
    float *rows = (float *)malloc(3 * (size_t)V * sizeof(float));
    const int t3[3] = { 1, 2, 3 };
    CHECK(lm_draft(m, 1, rows) == LM_ERR_STATE);            /* nothing fed */
    CHECK(lm_accept(m, 1) == LM_ERR_STATE);                 /* no verify */
    CHECK(lm_verify(m, t3, 3, rows) == LM_ERR_STATE && lm_pos(m) == 0);
    CHECK(lm_verify(m, t3, 0, rows) == LM_ERR_STATE);
    CHECK(lm_feed(m, 1, NULL) == LM_OK);
    CHECK(lm_draft(m, V, rows) == LM_ERR_TOKEN && lm_draft(m, -1, rows) == LM_ERR_TOKEN);
    const int badtok[2] = { 1, V };
    CHECK(lm_verify(m, badtok, 2, rows) == LM_ERR_TOKEN && lm_pos(m) == 1);
    CHECK(lm_verify(m, t3, 2, rows) == LM_OK);
    CHECK(lm_accept(m, 3) == LM_ERR_STATE && lm_accept(m, 0) == LM_ERR_STATE);
    CHECK(lm_feed(m, 4, NULL) == LM_OK);
    CHECK(lm_accept(m, 1) == LM_ERR_STATE && lm_pos(m) == 4); /* a feed after the verify ends it */
    CHECK(lm_verify(m, t3, 2, rows) == LM_OK && lm_accept(m, 2) == LM_OK);
    CHECK(lm_accept(m, 1) == LM_ERR_STATE && lm_pos(m) == 6);  /* once only */
    int kept = 5;
    CHECK(lm_greedy_check(m, V, 1, rows, rows, &kept) == LM_ERR_TOKEN && kept == 0 && lm_pos(m) == 6);
    lm_close(m);

    /* NEGATIVE: the context window must hold both verify tokens */
    char err[512] = "";
    LmOpts small = { NULL, 4, 0, 0, 1 };
    m = lm_open(DIR, &small, err, sizeof err);
    CHECK_MSG(m != NULL, "%s", err);
    if (m) {
        CHECK(lm_prefill(m, t3, 3, NULL) == LM_OK);
        CHECK(lm_verify(m, t3, 2, rows) == LM_ERR_CTX && lm_pos(m) == 3);
        lm_close(m);
    }
    /* without the head */
    m = open_m(&CASES[0], 0);
    if (m) {
        CHECK(lm_feed(m, 1, NULL) == LM_OK);
        CHECK(lm_draft(m, 1, rows) == LM_ERR_STATE && lm_verify(m, t3, 2, rows) == LM_ERR_STATE);
        lm_close(m);
    }
    /* a Gemma model has no head */
    LmOpts g = { NULL, 64, 0, 0, 1 };
    err[0] = 0;
    CHECK(lm_open("tests/fixtures/gemma4-tiny", &g, err, sizeof err) == NULL && strstr(err, "MTP"));
    free(rows);
}

/* a q8 file written before the head was kept (no mtp.* tensors) opens without --mtp and is
 * refused with it */
static void test_q8_without_head(void)
{
    char dir[512], path[600], err[512] = "";
    lt_mkdtemp(dir);
    snprintf(path, sizeof path, "%s/nohead.q8.safetensors", dir);
    LmSt st;
    memset(&st, 0, sizeof st);
    CHECK_MSG(lm_st_open(&st, Q8, 0, err, sizeof err) == 0, "%s", err);
    LmStWriter *w = lm_stw_create(path, err, sizeof err);
    int ok = w != NULL;
    for (int f = 0; ok && f < st.nfiles; f++)
        for (int i = 0; ok && i < st.files[f].nmeta; i++)
            ok = lm_stw_meta(w, st.files[f].meta_key[i], st.files[f].meta_val[i], err, sizeof err) == 0;
    for (int i = 0; ok && i < st.nt; i++)
        if (strncmp(st.t[i].name, "mtp.", 4) != 0)
            ok = lm_stw_declare(w, st.t[i].name, st.t[i].dtype, st.t[i].ndim, st.t[i].shape, err, sizeof err) == 0;
    ok = ok && lm_stw_begin(w, err, sizeof err) == 0;
    for (const char *name; ok && (name = lm_stw_next_name(w));) {
        const LmTensor *t = lm_st_find(&st, name);
        void *buf = t ? malloc((size_t)t->nbytes + 1) : NULL;
        ok = buf && lm_st_read(&st, t, buf, err, sizeof err) == 0 && lm_stw_tensor_begin(w, name, err, sizeof err) == 0 &&
             lm_stw_write(w, buf, t->nbytes, err, sizeof err) == 0 && lm_stw_tensor_end(w, err, sizeof err) == 0;
        free(buf);
    }
    if (ok) ok = lm_stw_finish(w, err, sizeof err) == 0;
    else lm_stw_abort(w);
    lm_st_close(&st);
    CHECK_MSG(ok, "writing %s: %s", path, err);
    if (ok) {
        const Case nohead = { path, 0, "q8 without head" };
        LmModel *m = open_m(&nohead, 0);
        CHECK(m != NULL);
        lm_close(m);
        LmOpts o = { path, CTX, 0, 0, 1 };
        err[0] = 0;
        CHECK_MSG(lm_open(DIR, &o, err, sizeof err) == NULL && strstr(err, "no MTP head"), "err: %s", err);
    }
    lt_rmtree(dir);
}

int main(void)
{
    if (access(DIR "/config.json", R_OK) != 0 || access(Q8, R_OK) != 0) {
        fprintf(stderr, "  FAIL fixture %s not found (run from the repository root)\n", DIR);
        return 1;
    }
    for (size_t i = 0; i < sizeof CASES / sizeof *CASES; i++) {
        test_draft_paths(&CASES[i]);
        test_verify_accept(&CASES[i]);
        test_greedy_equal(&CASES[i]);
        test_snapshot_carries_head(&CASES[i]);
    }
    test_refusals();
    test_q8_without_head();
    return lt_finish("test_mtp");
}
