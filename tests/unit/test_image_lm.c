/* test_image_lm.c - Qwen image rows into the language model: M-RoPE positions, the cache index,
 * snapshots and lm_reset. Logits against the oracle are the gate's job. Run from the repo root. */
#include "lm_test.h"

#include <string.h>

#include "core/lm_ops.h"
#include "io/lm_image.h"
#include "lm/lm.h"
#include "model/lm_vision.h"

#define FIX "tests/fixtures/qwen35-vl-tiny"
#define IMG FIX "/img_64x96.ppm"

static void fill(float *x, int n)
{
    for (int i = 0; i < n; i++) x[i] = (float)((i % 7) - 3) * 0.25f;
}

static void test_mrope_matches_rope(void)
{
    float a[64], b[64], orig[64];
    fill(orig, 64);
    memcpy(a, orig, sizeof a);
    memcpy(b, orig, sizeof b);
    CHECK(lm_rope(a, 2, 32, 8, 5, 1.0e7f) == 0);
    CHECK(lm_mrope(b, 2, 32, 8, 5, 5, 5, 1.0e7f, 3, 3) == 0);
    CHECK(memcmp(a, b, sizeof a) == 0);

    memcpy(b, orig, sizeof b);
    CHECK(lm_mrope(b, 2, 32, 8, 0, 1, 2, 1.0e7f, 3, 3) == 0);
    CHECK(memcmp(orig, b, sizeof b) != 0);
    CHECK(lm_mrope(b, 2, 32, 8, -1, 0, 0, 1.0e7f, 3, 3) == -1);
}

/* 64x96 is a 2x3 merged grid. With no prefix the block is start, six rows, end. */
static void expect_64x96(const int *thw)
{
    static const int want[] = {
        0, 0, 0,
        1, 1, 1,  1, 1, 2,  1, 1, 3,
        1, 2, 1,  1, 2, 2,  1, 2, 3,
        4, 4, 4
    };
    CHECK(memcmp(thw, want, sizeof want) == 0);
}

static int encode(LmVision *v, const char *path, float **rows, int *n, int *gh, int *gw)
{
    char err[512];
    uint8_t *rgb = NULL;
    int w, h;
    if (lm_ppm_read(path, &rgb, &w, &h, err, sizeof err) != 0) return -1;
    *n = lm_vision_tokens(v, h, w);
    *gh = h / lm_vision_info(v)->unit;
    *gw = w / lm_vision_info(v)->unit;
    *rows = (float *)malloc((size_t)(*n) * lm_vision_info(v)->out_dim * sizeof(float));
    const int rc = *rows ? lm_vision_encode(v, rgb, h, w, *rows, err, sizeof err) : -1;
    free(rgb);
    return rc;
}

static void test_block(void)
{
    char err[512];
    LmOpts o = { NULL, 64, 0, 0, 0 };
    LmModel *m = lm_open(FIX, &o, err, sizeof err);
    LmVision *v = lm_vision_open(FIX, NULL, err, sizeof err);
    CHECK_MSG(m && v, "%s", err);
    if (!m || !v) { lm_close(m); lm_vision_close(v); return; }
    float *rows = NULL;
    int n, gh, gw;
    CHECK(encode(v, IMG, &rows, &n, &gh, &gw) == 0);
    CHECK(n == 6 && gh == 2 && gw == 3);
    int thw[8 * 3];
    float *logits = (float *)malloc((size_t)lm_n_vocab(m) * sizeof(float));
    CHECK(lm_prefill_image(m, rows, n, gh, gw, logits, thw) == LM_OK);
    expect_64x96(thw);
    CHECK(lm_pos(m) == n + 2);
    CHECK(lm_prefill_image(m, rows, n, 2, 2, NULL, NULL) == LM_ERR_TOKEN);
    CHECK(lm_pos(m) == n + 2);

    LmModel *m2 = lm_open(FIX, &o, err, sizeof err);
    CHECK(m2 != NULL);
    const int prefix[] = { 1, 2 };
    CHECK(lm_prefill(m2, prefix, 2, NULL) == LM_OK);
    int thw2[8 * 3];
    CHECK(lm_prefill_image(m2, rows, n, gh, gw, NULL, thw2) == LM_OK);
    CHECK(thw2[0] == 2 && thw2[1] == 2 && thw2[2] == 2);
    CHECK(thw2[3] == 3 && thw2[4] == 3 && thw2[5] == 3);
    CHECK(thw2[7 * 3] == 6 && thw2[7 * 3 + 1] == 6 && thw2[7 * 3 + 2] == 6);
    CHECK(lm_pos(m2) == 2 + n + 2);

    LmSnapshot *s = lm_snapshot(m2);
    CHECK(s != NULL);
    float detour[384];
    CHECK(lm_feed(m2, 9, NULL) == LM_OK);
    CHECK(lm_feed(m2, 7, detour) == LM_OK);
    CHECK(lm_restore(m2, s) == LM_OK);
    float back[384];
    CHECK(lm_feed(m2, 7, back) == LM_OK);
    LmModel *m3 = lm_open(FIX, &o, err, sizeof err);
    CHECK(lm_prefill(m3, prefix, 2, NULL) == LM_OK);
    CHECK(lm_prefill_image(m3, rows, n, gh, gw, NULL, NULL) == LM_OK);
    float straight[384];
    CHECK(lm_feed(m3, 7, straight) == LM_OK);
    CHECK(memcmp(back, straight, sizeof back) == 0);
    CHECK(memcmp(detour, straight, sizeof detour) != 0);

    lm_reset(m2);
    float reset_row[384];
    CHECK(lm_feed(m2, 7, reset_row) == LM_OK);
    LmModel *fresh = lm_open(FIX, &o, err, sizeof err);
    float fresh_row[384];
    CHECK(lm_feed(fresh, 7, fresh_row) == LM_OK);
    CHECK(memcmp(reset_row, fresh_row, sizeof reset_row) == 0);

    lm_close(fresh); lm_close(m3); lm_snapshot_free(s); lm_close(m2);
    free(logits); free(rows);
    lm_vision_close(v);
    lm_close(m);
}

static void test_refusals(void)
{
    char err[512];
    LmOpts o = { NULL, 32, 0, 0, 0 };
    float row[64] = { 0 };
    LmModel *g = lm_open("tests/fixtures/gemma4-tiny", &o, err, sizeof err);
    CHECK_MSG(g != NULL, "%s", err);
    if (g) {
        CHECK(lm_prefill_image(g, row, 1, 1, 1, NULL, NULL) == LM_ERR_STATE);
        CHECK(lm_pos(g) == 0);
        lm_close(g);
    }
    LmModel *t = lm_open("tests/fixtures/qwen35-tiny", &o, err, sizeof err);
    CHECK_MSG(t != NULL, "%s", err);
    if (t) {
        CHECK(lm_prefill_image(t, row, 1, 1, 1, NULL, NULL) == LM_ERR_TOKEN);
        CHECK(lm_pos(t) == 0);
        lm_close(t);
    }
}

int main(void)
{
    test_mrope_matches_rope();
    test_block();
    test_refusals();
    return lt_finish("test_image_lm");
}
