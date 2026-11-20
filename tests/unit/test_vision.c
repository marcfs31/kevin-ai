/* test_vision.c - the PPM reader and the image encoder (src/model/lm_vision.h) against the
 * oracle goldens in tests/fixtures/qwen35-vl-tiny, plus the refusals. The fixture is read in
 * place; its absence is a FAILURE, not a skip. */
#include "lm_test.h"

#include <math.h>

#include "io/lm_image.h"
#include "json.h"
#include "model/lm_vision.h"

#define FIX "tests/fixtures/qwen35-vl-tiny"
#define GFIX "tests/fixtures/gemma4-vl-tiny"

static char g_dir[512];

static const char *P(const char *leaf)
{
    static char buf[4][1024];
    static int k;
    k = (k + 1) % 4;
    lt_path(buf[k], g_dir, leaf);
    return buf[k];
}

static void test_ppm(void)
{
    char err[256];
    uint8_t *rgb;
    int w, h;
    const unsigned char good[] = "P6\n2 1\n255\n\x01\x02\x03\x04\x05\x06";
    lt_write_file(P("good.ppm"), good, sizeof good - 1);
    CHECK_MSG(lm_ppm_read(P("good.ppm"), &rgb, &w, &h, err, sizeof err) == 0, "%s", err);
    CHECK(w == 2 && h == 1 && rgb[0] == 1 && rgb[5] == 6);
    free(rgb);

    lt_write_file(P("short.ppm"), good, sizeof good - 2);
    CHECK_ERR(lm_ppm_read(P("short.ppm"), &rgb, &w, &h, err, sizeof err), err, "truncated");
    CHECK(rgb == NULL);
    lt_write_text(P("p3.ppm"), "P3\n1 1\n255\n1 2 3\n");
    CHECK_ERR(lm_ppm_read(P("p3.ppm"), &rgb, &w, &h, err, sizeof err), err, "P6");
    lt_write_text(P("max.ppm"), "P6\n1 1\n65535\n\x01\x02\x03\x04\x05\x06");
    CHECK_ERR(lm_ppm_read(P("max.ppm"), &rgb, &w, &h, err, sizeof err), err, "maxval");
    lt_write_text(P("zero.ppm"), "P6\n0 1\n255\n");
    CHECK_ERR(lm_ppm_read(P("zero.ppm"), &rgb, &w, &h, err, sizeof err), err, "outside");
    lt_write_text(P("huge.ppm"), "P6\n100000 1\n255\n");
    CHECK_ERR(lm_ppm_read(P("huge.ppm"), &rgb, &w, &h, err, sizeof err), err, "outside");
    lt_write_text(P("comment.ppm"), "P6\n# hi\n1 1\n255\n\x01\x02\x03");
    CHECK_ERR(lm_ppm_read(P("comment.ppm"), &rgb, &w, &h, err, sizeof err), err, "P6");
    CHECK_ERR(lm_ppm_read(P("missing.ppm"), &rgb, &w, &h, err, sizeof err), err, "cannot open");
}

static jval *read_json(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    const long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *s = (char *)malloc((size_t)n + 1);
    const size_t got = s ? fread(s, 1, (size_t)n, f) : 0;
    fclose(f);
    if (!s || got != (size_t)n) { free(s); return NULL; }
    s[n] = '\0';
    jval *v = json_parse(s, NULL);
    free(s);
    return v;
}

/* Every golden image within 2e-4 * max|golden| (docs/PLAN.md gate), and a second encode
 * bit-identical to the first. */
static void check_variant(const char *fix, const char *weights, const char *golden)
{
    char err[512];
    LmVision *v = lm_vision_open(fix, weights, err, sizeof err);
    CHECK_MSG(v != NULL, "%s", err);
    jval *g = read_json(golden);
    CHECK_MSG(g != NULL, "cannot read %s", golden);
    if (!v || !g) { lm_vision_close(v); json_free_tree(g); return; }
    const LmVisionInfo *in = lm_vision_info(v);
    if (!strcmp(fix, FIX)) {
        CHECK(in->family == LM_VISION_QWEN && in->unit == 32 && in->out_dim == 64);
        /* in-vocab stand-ins: the tiny vocab is 384, so the real ids do not index it */
        CHECK(in->image_token == 5 && in->start_token == 4 && in->end_token == 6);
    } else {
        CHECK(in->family == LM_VISION_GEMMA && in->unit == 48 && in->out_dim == 64);
        /* In-range stand-ins: the tiny vocab is 384, so the real ids do not fit the text model.
         * The parser is what this checks; a real checkpoint still carries 258880 / 255999 / 258882. */
        CHECK(in->image_token == 12 && in->start_token == 11 && in->end_token == 13);
    }
    jval *imgs = json_get(g, "images");
    CHECK(imgs && imgs->t == J_ARR && imgs->len == 3);
    for (int i = 0; imgs && i < imgs->len; i++) {
        jval *im = imgs->kids[i], *rows = json_get(im, "rows");
        char path[1024];
        snprintf(path, sizeof path, "%s/%s", fix, json_get(im, "image")->str);
        uint8_t *rgb;
        int w, h;
        CHECK_MSG(lm_ppm_read(path, &rgb, &w, &h, err, sizeof err) == 0, "%s", err);
        const int nt = lm_vision_tokens(v, h, w);
        CHECK(nt == rows->len);
        float *a = (float *)malloc((size_t)nt * in->out_dim * sizeof(float));
        float *b = (float *)malloc((size_t)nt * in->out_dim * sizeof(float));
        CHECK_MSG(lm_vision_encode(v, rgb, h, w, a, err, sizeof err) == 0, "%s", err);
        CHECK_MSG(lm_vision_encode(v, rgb, h, w, b, err, sizeof err) == 0, "%s", err);
        CHECK(memcmp(a, b, (size_t)nt * in->out_dim * sizeof(float)) == 0);
        double top = 0, worst = 0;
        for (int r = 0; r < nt; r++)
            for (int c = 0; c < in->out_dim; c++) top = fmax(top, fabs(rows->kids[r]->kids[c]->num));
        for (int r = 0; r < nt; r++)
            for (int c = 0; c < in->out_dim; c++)
                worst = fmax(worst, fabs(a[(size_t)r * in->out_dim + c] - rows->kids[r]->kids[c]->num));
        CHECK_MSG(worst <= 2e-4 * top, "%s %s: error %g > %g", golden, path, worst, 2e-4 * top);
        free(a); free(b); free(rgb);
    }
    json_free_tree(g);
    lm_vision_close(v);
}

static void test_refusals(void)
{
    char err[512];
    LmVision *v = lm_vision_open(FIX, NULL, err, sizeof err);
    CHECK_MSG(v != NULL, "%s", err);
    if (!v) return;
    CHECK(lm_vision_tokens(v, 64, 96) == 6);
    CHECK(lm_vision_tokens(v, 32, 32) == 1);
    CHECK(lm_vision_tokens(v, 48, 64) == -1);       /* not a multiple of patch * merge */
    CHECK(lm_vision_tokens(v, 0, 32) == -1);
    CHECK(lm_vision_tokens(v, 32 * 64, 32 * 64) == 4096);  /* exactly LM_VISION_MAX_PATCHES */
    CHECK(lm_vision_tokens(v, 32 * 65, 32 * 64) == -1);    /* one block row more */
    static uint8_t px[48 * 64 * 3];
    float out[64];
    CHECK_ERR(lm_vision_encode(v, px, 48, 64, out, err, sizeof err), err, "multiples of 32");
    lm_vision_close(v);

    /* the text-only fixture's visual stubs are not an encoder: refused, naming a tensor */
    CHECK(lm_vision_open("tests/fixtures/qwen35-tiny", NULL, err, sizeof err) == NULL);
    CHECK_MSG(strstr(err, "model.visual.") != NULL, "%s", err);
    /* a family without an encoder */
    lt_write_text(P("config.json"), "{\"model_type\":\"llama\"}");
    CHECK_ERR(lm_vision_open(g_dir, NULL, err, sizeof err) == NULL ? -1 : 0, err, "no supported vision encoder");

    /* Gemma: sides in multiples of patch * pooling, and inside the position table */
    v = lm_vision_open(GFIX, NULL, err, sizeof err);
    CHECK_MSG(v != NULL, "%s", err);
    if (!v) return;
    CHECK(lm_vision_tokens(v, 96, 144) == 6);
    CHECK(lm_vision_tokens(v, 64, 96) == -1);
    CHECK(lm_vision_tokens(v, 48 * 5, 48 * 5) == 25);    /* 15 patches a side: inside the 16-entry table */
    CHECK(lm_vision_tokens(v, 48 * 6, 48) == -1);        /* 18 patches: past it */
    lm_vision_close(v);
    /* the text-only Gemma fixture's vision stubs are refused, naming a tensor */
    CHECK(lm_vision_open("tests/fixtures/gemma4-tiny", NULL, err, sizeof err) == NULL);
    CHECK_MSG(strstr(err, "model.vision_tower.") || strstr(err, "model.embed_vision."), "%s", err);
}

int main(void)
{
    lt_mkdtemp(g_dir);
    test_ppm();
    check_variant(FIX, NULL, FIX "/golden_vision_bf16.json");
    check_variant(FIX, FIX "/model.q8.safetensors", FIX "/golden_vision_q8.json");
    check_variant(GFIX, NULL, GFIX "/golden_vision_bf16.json");
    check_variant(GFIX, GFIX "/model.q8.safetensors", GFIX "/golden_vision_q8.json");
    test_refusals();
    lt_rmtree(g_dir);
    return lt_finish("test_vision");
}
