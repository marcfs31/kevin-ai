/* test_qwen2.c - the Qwen2 GGUF family behind lm_open on tests/fixtures/qwen2-tiny (run from the
 * repository root; absence is a FAILURE). The logits against the float64 oracle are the job of
 * `make gate`; this file checks the engine's own invariants, all BITWISE: lm_prefill equals
 * token-by-token lm_feed across chunk boundaries, a snapshot restores exactly, a reset returns
 * to the first logits, and the refusals. */
#include "lm_test.h"

#include "lm/lm.h"

#define FIX "tests/fixtures/qwen2-tiny"
#define V 320
#define CTX 128

static LmModel *open_ctx(const char *path, int ctx, char *err, size_t n)
{
    LmOpts o = { NULL, ctx, 0, 0, 0 };
    return lm_open(path, &o, err, n);
}

static void tokens(int *t, int n, unsigned seed)
{
    for (int i = 0; i < n; i++) { seed = seed * 1103515245u + 12345u; t[i] = (int)((seed >> 8) % 296u); }
}

static void test_feed_vs_prefill(void)
{
    char err[512] = "";
    LmModel *a = open_ctx(FIX, CTX, err, sizeof err), *b = open_ctx(FIX "/model.gguf", CTX, err, sizeof err);
    CHECK_MSG(a && b, "open: %s", err);
    if (!a || !b) return;
    CHECK(lm_n_vocab(a) == V && lm_n_ctx(a) == CTX && lm_pos(a) == 0);
    int tok[100];
    tokens(tok, 100, 7);   /* 100 tokens: three full prefill chunks of 32 and a tail of 4 */
    static float la[V], lb[V];
    for (int i = 0; i < 100; i++) CHECK(lm_feed(a, tok[i], i == 99 ? la : NULL) == LM_OK);
    CHECK(lm_prefill(b, tok, 100, lb) == LM_OK);
    CHECK(lm_pos(a) == 100 && lm_pos(b) == 100);
    CHECK_MSG(!memcmp(la, lb, sizeof la), "prefill logits differ from feed");
    for (int i = 0; i < V; i++) CHECK(isfinite(la[i]));
    /* continuing after the prefill stays identical */
    static float na[V], nb[V];
    CHECK(lm_feed(a, 5, na) == LM_OK && lm_feed(b, 5, nb) == LM_OK);
    CHECK(!memcmp(na, nb, sizeof na));
    /* uneven splits give the same state */
    lm_reset(b);
    CHECK(lm_pos(b) == 0);
    CHECK(lm_prefill(b, tok, 37, NULL) == LM_OK && lm_prefill(b, tok + 37, 63, lb) == LM_OK);
    CHECK(!memcmp(la, lb, sizeof la));
    /* a reset returns to the very first logits */
    static float f1[V], f2[V];
    lm_reset(a);
    CHECK(lm_feed(a, tok[0], f1) == LM_OK);
    lm_reset(a);
    CHECK(lm_feed(a, tok[0], f2) == LM_OK && !memcmp(f1, f2, sizeof f1));
    lm_close(a);
    lm_close(b);
}

static void test_snapshot(void)
{
    char err[512] = "";
    LmModel *m = open_ctx(FIX, CTX, err, sizeof err);
    CHECK_MSG(m != NULL, "open: %s", err);
    if (!m) return;
    int tok[40], junk[20];
    tokens(tok, 40, 11);
    tokens(junk, 20, 99);
    CHECK(lm_prefill(m, tok, 40, NULL) == LM_OK);
    LmSnapshot *s = lm_snapshot(m);
    CHECK(s && lm_snapshot_pos(s) == 40);
    static float want[V], got[V], other[V];
    CHECK(lm_feed(m, 17, want) == LM_OK);
    CHECK(lm_prefill(m, junk, 20, NULL) == LM_OK);
    CHECK(lm_feed(m, 17, other) == LM_OK);
    CHECK_MSG(memcmp(want, other, sizeof want) != 0, "negative control: different history must change the logits");
    CHECK(lm_restore(m, s) == LM_OK && lm_pos(m) == 40);
    CHECK(lm_feed(m, 17, got) == LM_OK);
    CHECK_MSG(!memcmp(want, got, sizeof want), "restored logits differ");
    lm_reset(m);
    CHECK(lm_restore(m, s) == LM_OK && lm_pos(m) == 40);
    CHECK(lm_feed(m, 17, got) == LM_OK && !memcmp(want, got, sizeof want));
    lm_snapshot_free(s);
    lm_close(m);
}

static void test_limits(void)
{
    char err[512] = "";
    LmModel *m = open_ctx(FIX, 16, err, sizeof err);
    CHECK_MSG(m != NULL, "open: %s", err);
    if (!m) return;
    static float l[V];
    CHECK(lm_feed(m, -1, l) == LM_ERR_TOKEN && lm_feed(m, V, l) == LM_ERR_TOKEN);
    int tok[20];
    tokens(tok, 20, 3);
    CHECK(lm_prefill(m, tok, 20, l) == LM_ERR_CTX && lm_pos(m) == 0);   /* nothing fed */
    tok[3] = V;
    CHECK(lm_prefill(m, tok, 10, l) == LM_ERR_TOKEN && lm_pos(m) == 0);
    tok[3] = 1;
    CHECK(lm_prefill(m, tok, 16, l) == LM_OK && lm_feed(m, 1, l) == LM_ERR_CTX);
    static float rows[4 * 256];
    CHECK(lm_prefill_image(m, rows, 4, 256, 2, 2, NULL, NULL) == LM_ERR_STATE);   /* text only */
    CHECK(lm_draft(m, 1, l) == LM_ERR_STATE);
    lm_close(m);

    LmOpts o = { "x.safetensors", 0, 0, 0, 0 };
    CHECK(lm_open(FIX, &o, err, sizeof err) == NULL && strstr(err, "its own weights"));
    LmOpts mtp = { NULL, 0, 0, 0, 1 };
    CHECK(lm_open(FIX, &mtp, err, sizeof err) == NULL && strstr(err, "MTP"));
    LmOpts neg = { NULL, -1, 0, 0, 0 };
    CHECK(lm_open(FIX, &neg, err, sizeof err) == NULL);
    /* a context the memory cannot hold is refused, never silently shrunk */
    LmOpts big = { NULL, 1 << 30, 0, 0, 0 };
    CHECK(lm_open(FIX, &big, err, sizeof err) == NULL);
}

int main(void)
{
    test_feed_vs_prefill();
    test_snapshot();
    test_limits();
    return lt_finish("test_qwen2");
}
