/* test_image.c - lm_prefill_image (include/lm/lm.h) on the tiny Qwen3.5 and Gemma 4 vision fixtures
 * (run from the repository root; a missing fixture is a FAILURE). After an image the state must be
 * exactly what feeding the same sequence gives: tokens fed one by one and as a batch afterwards
 * give BIT-IDENTICAL logits (the RoPE position, not the cache slot, drives Qwen's attention), a
 * snapshot taken after the image restores exactly (the RoPE position is part of it), and the
 * refusals change nothing: LM_ERR_STATE without an encoder, LM_ERR_IMAGE for a size or a null
 * image, LM_ERR_CTX when the image and its two marker tokens do not fit. The goldens of
 * tools/check_golden.py (check 6) own the numbers; this file owns the contracts. */
#include "lm_test.h"

#include "io/lm_image.h"
#include "lm/lm.h"

#define CTX 64
#define TAIL 5

typedef struct { const char *dir, *image, *label; } Case;

static const Case CASES[] = {
    { "tests/fixtures/qwen35-vl-tiny", "img_64x96.ppm", "qwen" },
    { "tests/fixtures/gemma4-vl-tiny", "img_96x144.ppm", "gemma" },
};

static LmModel *open_model(const Case *c, const char *weights, int vision, int ctx)
{
    char err[512] = "";
    LmOpts o = { weights, ctx, 0, 0, 0, vision };
    LmModel *m = lm_open(c->dir, &o, err, sizeof err);
    CHECK_MSG(m != NULL, "%s: lm_open: %s", c->label, err);
    return m;
}

static uint8_t *load(const Case *c, const char *leaf, int *w, int *h)
{
    char path[1024], err[256];
    uint8_t *rgb = NULL;
    snprintf(path, sizeof path, "%s/%s", c->dir, leaf);
    CHECK_MSG(lm_ppm_read(path, &rgb, w, h, err, sizeof err) == 0, "%s: %s", c->label, err);
    return rgb;
}

static void check_case(const Case *c, const char *weights)
{
    char q8[1024];
    if (weights) snprintf(q8, sizeof q8, "%s/%s", c->dir, weights);
    LmModel *m = open_model(c, weights ? q8 : NULL, 1, CTX);
    LmModel *b = open_model(c, weights ? q8 : NULL, 1, CTX);
    int w = 0, h = 0;
    uint8_t *rgb = load(c, c->image, &w, &h);
    if (!m || !b || !rgb) { lm_close(m); lm_close(b); free(rgb); return; }
    const int V = lm_n_vocab(m), N = lm_image_tokens(m, h, w);
    CHECK_MSG(lm_vision_status(m) == NULL, "%s: %s", c->label, lm_vision_status(m));
    CHECK(N == 6);
    int pre[3] = { 11, 22, 33 }, tail[TAIL] = { 7, 100, 3, 250, 42 };
    float *one = (float *)malloc((size_t)TAIL * V * sizeof(float)), *batch = (float *)malloc((size_t)V * sizeof(float));
    float *again = (float *)malloc((size_t)TAIL * V * sizeof(float));
    CHECK(one && batch && again);
    if (!one || !batch || !again) goto out;

    /* text, image, then the tail: token by token on m, as one batch on b */
    CHECK(lm_prefill(m, pre, 3, NULL) == LM_OK && lm_prefill(b, pre, 3, NULL) == LM_OK);
    CHECK(lm_prefill_image(m, rgb, h, w, NULL) == LM_OK);
    CHECK(lm_prefill_image(b, rgb, h, w, NULL) == LM_OK);
    CHECK(lm_pos(m) == 3 + N + 2 && lm_pos(b) == lm_pos(m));
    LmSnapshot *snap = lm_snapshot(m);
    CHECK(snap != NULL && lm_snapshot_pos(snap) == lm_pos(m));
    for (int t = 0; t < TAIL; t++) CHECK(lm_feed(m, tail[t], one + (size_t)t * V) == LM_OK);
    CHECK(lm_prefill(b, tail, TAIL, batch) == LM_OK);
    CHECK_MSG(!memcmp(batch, one + (size_t)(TAIL - 1) * V, (size_t)V * sizeof(float)),
              "%s: prefill after an image differs from feeding token by token", c->label);

    /* a snapshot after the image restores exactly, however far the model went on or was reset */
    lm_reset(m);
    CHECK(lm_prefill(m, tail, TAIL, NULL) == LM_OK);
    CHECK(snap && lm_restore(m, snap) == LM_OK && lm_pos(m) == 3 + N + 2);
    for (int t = 0; t < TAIL; t++) CHECK(lm_feed(m, tail[t], again + (size_t)t * V) == LM_OK);
    CHECK_MSG(!memcmp(again, one, (size_t)TAIL * V * sizeof(float)), "%s: restore after an image is not exact", c->label);
    /* NEGATIVE: the image matters (the same tail after text only gives other logits) */
    lm_reset(m);
    CHECK(lm_prefill(m, pre, 3, NULL) == LM_OK && lm_prefill(m, tail, TAIL, batch) == LM_OK);
    CHECK(memcmp(batch, one + (size_t)(TAIL - 1) * V, (size_t)V * sizeof(float)) != 0);
    lm_snapshot_free(snap);

    /* refusals leave the model untouched */
    lm_reset(m);
    CHECK(lm_prefill_image(m, NULL, h, w, NULL) == LM_ERR_IMAGE && lm_pos(m) == 0);
    CHECK(lm_prefill_image(m, rgb, h + 1, w, NULL) == LM_ERR_IMAGE && lm_pos(m) == 0);
    CHECK(lm_prefill_image(m, rgb, 0, w, NULL) == LM_ERR_IMAGE && lm_pos(m) == 0);
    CHECK(lm_image_tokens(m, h + 1, w) == -1 && lm_image_tokens(m, 0, 0) == -1);
out:
    free(one); free(batch); free(again); free(rgb);
    lm_close(m);
    lm_close(b);
}

static void check_refusals(const Case *c)
{
    int w = 0, h = 0;
    uint8_t *rgb = load(c, c->image, &w, &h);
    /* no encoder asked for: LM_ERR_STATE, with the reason, nothing fed */
    LmModel *t = open_model(c, NULL, 0, CTX);
    if (t && rgb) {
        CHECK(lm_vision_status(t) != NULL && strstr(lm_vision_status(t), "not requested"));
        CHECK(lm_prefill_image(t, rgb, h, w, NULL) == LM_ERR_STATE && lm_pos(t) == 0);
        CHECK(lm_image_tokens(t, h, w) == -1);
    }
    lm_close(t);
    /* a text-only checkpoint opens with images asked for; the reason names what is missing */
    char err[512] = "";
    const char *tiny = !strcmp(c->label, "qwen") ? "tests/fixtures/qwen35-tiny" : "tests/fixtures/gemma4-tiny";
    LmOpts o = { NULL, CTX, 0, 0, 0, 1 };
    LmModel *x = lm_open(tiny, &o, err, sizeof err);
    CHECK_MSG(x != NULL, "%s: text-only model with vision=1: %s", c->label, err);
    if (x && rgb) {
        CHECK(lm_vision_status(x) != NULL && strstr(lm_vision_status(x), "no image encoder"));
        CHECK(lm_prefill_image(x, rgb, h, w, NULL) == LM_ERR_STATE && lm_pos(x) == 0);
    }
    lm_close(x);
    /* the image (6 rows) and its two markers need 8 positions: a window of 7 refuses, nothing fed,
     * and a window of 8 takes it exactly */
    LmModel *s = open_model(c, NULL, 1, 7);
    if (s && rgb) CHECK(lm_prefill_image(s, rgb, h, w, NULL) == LM_ERR_CTX && lm_pos(s) == 0);
    lm_close(s);
    s = open_model(c, NULL, 1, 8);
    if (s && rgb) {
        CHECK(lm_prefill_image(s, rgb, h, w, NULL) == LM_OK && lm_pos(s) == 8);
        CHECK(lm_feed(s, 1, NULL) == LM_ERR_CTX);
    }
    lm_close(s);
    free(rgb);
}

int main(void)
{
    for (size_t i = 0; i < sizeof CASES / sizeof CASES[0]; i++) {
        check_case(&CASES[i], NULL);
        check_case(&CASES[i], "model.q8.safetensors");
        check_refusals(&CASES[i]);
    }
    return lt_finish("test_image");
}
