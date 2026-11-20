/* test_quant.c - the q8 quantiser (docs/PLAN.md rules, edge cases) and lm-quant's
 * pipeline (lm_quantize_checkpoint, the library behind bin/lm-quant) on tiny generated
 * checkpoints built in a mkdtemp directory. */
#include "lm_test.h"

#include "core/lm_bf16.h"
#include "core/lm_kernels.h"
#include "core/lm_quant.h"
#include "io/lm_quantfile.h"
#include "io/lm_sha256.h"
#include "io/lm_st.h"

static char g_dir[512];

static char *P(const char *leaf)
{
    static char b[8][1024];
    static int i;
    i = (i + 1) & 7;
    lt_path(b[i], g_dir, leaf);
    return b[i];
}

/* ------------------------------------------------------------------ quantiser */

static void quant_group(const float *g, int8_t *q, float *scale)
{
    int64_t bad;
    CHECK(lm_q8_quantize(g, 1, 32, q, scale, &bad) == LM_Q8_OK);
}

static void test_edge_cases(void)
{
    float g[32];
    int8_t q[32];
    float sc;

    /* zero group: scale 0 and all-zero q, not NaN from 0/0 */
    memset(g, 0, sizeof g);
    memset(q, 0x55, sizeof q);
    quant_group(g, q, &sc);
    CHECK(sc == 0.0f);
    for (int i = 0; i < 32; i++) CHECK(q[i] == 0);

    /* all -0.0: also a zero group */
    for (int i = 0; i < 32; i++) g[i] = -0.0f;
    memset(q, 0x55, sizeof q);
    quant_group(g, q, &sc);
    CHECK(sc == 0.0f);
    for (int i = 0; i < 32; i++) CHECK(q[i] == 0);

    /* absmax 127 gives scale exactly 1.0, so x/scale = x and the rounding is visible:
     * half-way cases go to the EVEN neighbour, for both signs */
    memset(g, 0, sizeof g);
    g[0] = 127.0f;
    g[1] = 0.5f;  g[2] = 1.5f;  g[3] = 2.5f;  g[4] = 3.5f;  g[5] = 126.5f;
    g[6] = -0.5f; g[7] = -1.5f; g[8] = -2.5f; g[9] = -3.5f; g[10] = -126.5f;
    g[11] = 0.49999997f; g[12] = 0.50000006f; g[13] = -0.0f;
    quant_group(g, q, &sc);
    CHECK(sc == 1.0f);
    CHECK(q[0] == 127);
    CHECK(q[1] == 0 && q[2] == 2 && q[3] == 2 && q[4] == 4 && q[5] == 126);        /* ties to even */
    CHECK(q[6] == 0 && q[7] == -2 && q[8] == -2 && q[9] == -4 && q[10] == -126);   /* sign symmetric */
    CHECK(q[11] == 0 && q[12] == 1);                                               /* just below / above the tie */
    CHECK(q[13] == 0);                                                             /* -0.0 -> 0, not -0 garbage */
    /* NEGATIVE: round-half-up / half-away-from-zero would give 1, 2, 3, 4 and 127: not what we do */
    CHECK(q[1] != 1 && q[3] != 3 && q[5] != 127);

    /* the max element maps to exactly +-127; the sign of the max is kept */
    for (int i = 0; i < 32; i++) g[i] = (float)(i - 16) * 0.01f;
    g[7] = -9.0f;
    quant_group(g, q, &sc);
    CHECK(q[7] == -127);
    CHECK(sc == (float)(9.0 / 127.0));

    /* the scale is one double division rounded once to f32 */
    for (int i = 0; i < 32; i++) g[i] = 0.0f;
    g[3] = 1.0f;
    quant_group(g, q, &sc);
    CHECK(sc == (float)(1.0 / 127.0));
    CHECK(q[3] == 127);

    /* a lone tiny value (bf16 subnormal range) is not flushed: scale > 0, q[3] = 127 */
    g[3] = 9.18e-41f;     /* subnormal float */
    quant_group(g, q, &sc);
    CHECK(sc > 0.0f && q[3] == 127);

    /* non-finite input is an error naming the element, not a clamp or a silent zero */
    int64_t bad = -1;
    float w[64];
    int8_t qq[64];
    float ss[2];
    for (int i = 0; i < 64; i++) w[i] = 1.0f;
    w[40] = NAN;
    CHECK(lm_q8_quantize(w, 1, 64, qq, ss, &bad) == LM_Q8_ERR_NONFINITE && bad == 40);
    w[40] = INFINITY;
    bad = -1;
    CHECK(lm_q8_quantize(w, 1, 64, qq, ss, &bad) == LM_Q8_ERR_NONFINITE && bad == 40);
    w[40] = -INFINITY;
    CHECK(lm_q8_quantize(w, 1, 64, qq, ss, NULL) == LM_Q8_ERR_NONFINITE);
    w[40] = 1.0f;
    w[63] = NAN;           /* second row of a 2-row call: flat index reported */
    bad = -1;
    CHECK(lm_q8_quantize(w, 2, 32, qq, ss, &bad) == LM_Q8_ERR_NONFINITE && bad == 63);

    /* in must be a positive multiple of 32 */
    w[63] = 1.0f;
    CHECK(lm_q8_quantize(w, 1, 33, qq, ss, NULL) == LM_Q8_ERR_GROUP);
    CHECK(lm_q8_quantize(w, 1, 31, qq, ss, NULL) == LM_Q8_ERR_GROUP);
    CHECK(lm_q8_quantize(w, 1, 0, qq, ss, NULL) == LM_Q8_ERR_GROUP);
    CHECK(lm_q8_quantize(w, 2, 32, qq, ss, NULL) == LM_Q8_OK);
}

static void test_properties(void)
{
    /* random matrix with groups of wildly different magnitude: nearest-int property, bound
     * |x - q*scale| <= scale/2, |q| <= 127, and q = +-127 at each group's max */
    enum { ROWS = 20, IN = 128 };
    static float w[ROWS * IN], back[ROWS * IN];
    static int8_t q[ROWS * IN], q2[ROWS * IN];
    static float sc[ROWS * IN / 32], sc2[ROWS * IN / 32];
    for (int r = 0; r < ROWS; r++)
        for (int i = 0; i < IN; i++) {
            const float mag = powf(10.0f, (float)((i / 32) - 2));   /* 1e-2 .. 1e1 per group */
            w[r * IN + i] = lt_rand() * mag;
        }
    CHECK(lm_q8_quantize(w, ROWS, IN, q, sc, NULL) == LM_Q8_OK);
    lm_q8_dequantize(q, sc, ROWS, IN, back);
    for (int r = 0; r < ROWS; r++)
        for (int g = 0; g < IN / 32; g++) {
            const float scale = sc[r * (IN / 32) + g];
            float amax = 0;
            int imax = 0;
            for (int i = 0; i < 32; i++) {
                const int k = r * IN + g * 32 + i;
                if (fabsf(w[k]) > amax) { amax = fabsf(w[k]); imax = k; }
                CHECK(q[k] >= -127 && q[k] <= 127);
                /* the nearest-integer property, in double so it is exact */
                const double ratio = (double)w[k] / (double)scale;
                CHECK_NEAR(ratio, (double)q[k], 0.5 + 1e-9);
                if (fabs(ratio - (double)q[k]) > 0.5 - 1e-9 && fabs(ratio - (double)q[k]) < 0.5 + 1e-9)
                    CHECK(q[k] % 2 == 0);   /* a tie went to the even neighbour */
                CHECK(fabs((double)w[k] - (double)back[k]) <= 0.5 * (double)scale * (1.0 + 1e-6) + 1e-30);
            }
            CHECK(abs(q[imax]) == 127);
            CHECK((q[imax] > 0) == (w[imax] > 0));
            CHECK(scale == (float)((double)amax / 127.0));
        }
    /* deterministic, and row-blocking does not change a bit: quantise row by row */
    CHECK(lm_q8_quantize(w, ROWS, IN, q2, sc2, NULL) == LM_Q8_OK && !memcmp(q, q2, sizeof q) && !memcmp(sc, sc2, sizeof sc));
    for (int r = 0; r < ROWS; r++) CHECK(lm_q8_quantize(w + r * IN, 1, IN, q2 + r * IN, sc2 + r * (IN / 32), NULL) == LM_Q8_OK);
    CHECK(!memcmp(q, q2, sizeof q) && !memcmp(sc, sc2, sizeof sc));

    /* end to end: the q8 matvec of the quantised matrix is within sum|x| * scale/2 of the
     * true matvec of the source */
    float x[IN], y[ROWS], ref[ROWS];
    for (int i = 0; i < IN; i++) x[i] = lt_rand();
    CHECK(lm_matvec_q8(y, x, q, sc, IN, ROWS) == 0);
    for (int r = 0; r < ROWS; r++) {
        double d = 0, bound = 0;
        for (int i = 0; i < IN; i++) {
            d += (double)w[r * IN + i] * (double)x[i];
            bound += 0.5 * (double)sc[r * (IN / 32) + i / 32] * fabs((double)x[i]);
        }
        ref[r] = (float)d;
        CHECK_NEAR(y[r], ref[r], bound * 1.001 + 1e-5);
    }
}

/* ------------------------------------------------------------------ checkpoints */

typedef struct { const char *name, *dtype; int ndim; int64_t shape[4]; const void *data; size_t nbytes; } Src;

/* Write the tensors as one safetensors file (header padded to 8). */
static void write_src(const char *path, const Src *t, int n)
{
    char *hdr = (char *)malloc(1 << 16);
    size_t hn = 0, off = 0;
    hn += (size_t)snprintf(hdr + hn, (1 << 16) - hn, "{");
    for (int i = 0; i < n; i++) {
        hn += (size_t)snprintf(hdr + hn, (1 << 16) - hn, "%s\"%s\":{\"dtype\":\"%s\",\"shape\":[", i ? "," : "", t[i].name, t[i].dtype);
        for (int d = 0; d < t[i].ndim; d++) hn += (size_t)snprintf(hdr + hn, (1 << 16) - hn, "%s%lld", d ? "," : "", (long long)t[i].shape[d]);
        hn += (size_t)snprintf(hdr + hn, (1 << 16) - hn, "],\"data_offsets\":[%zu,%zu]}", off, off + t[i].nbytes);
        off += t[i].nbytes;
    }
    hn += (size_t)snprintf(hdr + hn, (1 << 16) - hn, "}");
    unsigned char *data = (unsigned char *)malloc(off ? off : 1);
    off = 0;
    for (int i = 0; i < n; i++) { memcpy(data + off, t[i].data, t[i].nbytes); off += t[i].nbytes; }
    lt_write_st_raw8(path, hdr, data, off);
    free(hdr);
    free(data);
}

static uint16_t *rand_bf16(size_t n, float mag)
{
    uint16_t *p = (uint16_t *)malloc(n * 2);
    for (size_t i = 0; i < n; i++) p[i] = lt_bf16(lt_rand() * mag);
    return p;
}

static int names_ok(const LmSt *s, const char *const *want, int n)
{
    if (s->nt != n) return 0;
    for (int i = 0; i < n; i++) if (!lm_st_find(s, want[i])) return 0;
    return 1;
}

static void test_checkpoint(void)
{
    char err[512];
    mkdir(P("ck"), 0755);

    /* tiny sharded checkpoint in HF layout */
    uint16_t *embed = rand_bf16(40 * 64, 1.0f);                  /* [40,64] 2-D weight */
    uint16_t *qproj = rand_bf16(32 * 96, 0.3f);                  /* [32,96] */
    uint16_t *zrow = (uint16_t *)calloc(8 * 32, 2);              /* [8,32]: all-zero rows -> scale 0 */
    memset(zrow, 0, 8 * 32 * 2);
    for (int i = 0; i < 32; i++) zrow[3 * 32 + i] = lt_bf16(lt_rand());  /* only row 3 is non-zero */
    uint16_t *norm = rand_bf16(64, 1.0f);                        /* 1-D ".weight": copied, NOT quantised */
    uint16_t *conv = rand_bf16(12 * 1 * 4, 1.0f);                /* 3-D conv weight: copied */
    float alog[4] = { 0.5f, -1.0f, 2.0f, 0.25f };                /* F32, copied */
    uint16_t *vis = rand_bf16(16 * 32, 1.0f);                    /* visual: quantised like text */
    uint16_t *mtp = rand_bf16(16 * 32, 1.0f);                    /* mtp: kept like the text weights */
    uint16_t *visbad = rand_bf16(8 * 30, 1.0f);                  /* visual with in=30: copied, not an error */
    float f32w[8 * 32];                                          /* F32 2-D weight: quantised from f32 */
    for (int i = 0; i < 8 * 32; i++) f32w[i] = lt_rand();

    const Src s1[] = {
        { "model.language_model.embed_tokens.weight", "BF16", 2, { 40, 64 }, embed, 40 * 64 * 2 },
        { "model.language_model.layers.0.linear_attn.A_log", "F32", 1, { 4 }, alog, 16 },
        { "model.visual.blocks.0.attn.qkv.weight", "BF16", 2, { 16, 32 }, vis, 16 * 32 * 2 },
        { "model.visual.merger.weight", "BF16", 2, { 8, 30 }, visbad, 8 * 30 * 2 },
    };
    const Src s2[] = {
        { "model.language_model.layers.0.self_attn.q_proj.weight", "BF16", 2, { 32, 96 }, qproj, 32 * 96 * 2 },
        { "model.language_model.layers.0.input_layernorm.weight", "BF16", 1, { 64 }, norm, 64 * 2 },
        { "model.language_model.layers.0.linear_attn.conv1d.weight", "BF16", 3, { 12, 1, 4 }, conv, 12 * 4 * 2 },
        { "model.language_model.layers.0.zrows.weight", "BF16", 2, { 8, 32 }, zrow, 8 * 32 * 2 },
        { "model.language_model.layers.0.f32mat.weight", "F32", 2, { 8, 32 }, f32w, sizeof f32w },
        { "mtp.fc.weight", "BF16", 2, { 16, 32 }, mtp, 16 * 32 * 2 },
    };
    write_src(P("ck/model-00001-of-00002.safetensors"), s1, 4);
    write_src(P("ck/model-00002-of-00002.safetensors"), s2, 6);
    {
        char idx[4096];
        int n = snprintf(idx, sizeof idx, "{\"metadata\":{\"total_size\":1},\"weight_map\":{");
        for (int i = 0; i < 4; i++) n += snprintf(idx + n, sizeof idx - (size_t)n, "%s\"%s\":\"model-00001-of-00002.safetensors\"", i ? "," : "", s1[i].name);
        for (int i = 0; i < 6; i++) n += snprintf(idx + n, sizeof idx - (size_t)n, ",\"%s\":\"model-00002-of-00002.safetensors\"", s2[i].name);
        snprintf(idx + n, sizeof idx - (size_t)n, "}}");
        lt_write_text(P("ck/model.safetensors.index.json"), idx);
    }

    LmQuantStats stt;
    int rc = lm_quantize_checkpoint(P("ck"), P("q8.safetensors"), &stt, err, sizeof err);
    CHECK_MSG(rc == 0, "%s", err);
    CHECK(stt.n_quantised == 6 && stt.n_copied == 4 && stt.n_ignored == 0);   /* embed, q_proj, zrows, f32mat, mtp.fc, visual qkv | norm, conv, A_log, visual merger */
    CHECK(access(P("q8.safetensors.tmp"), F_OK) != 0);

    LmSt o, src;
    CHECK_MSG(lm_st_open(&o, P("q8.safetensors"), 1, err, sizeof err) == 0, "%s", err);
    CHECK_MSG(lm_st_open(&src, P("ck"), 0, err, sizeof err) == 0, "%s", err);
    static const char *const want[] = {
        "model.language_model.embed_tokens.weight", "model.language_model.embed_tokens.weight.q8scale",
        "model.language_model.layers.0.self_attn.q_proj.weight", "model.language_model.layers.0.self_attn.q_proj.weight.q8scale",
        "model.language_model.layers.0.zrows.weight", "model.language_model.layers.0.zrows.weight.q8scale",
        "model.language_model.layers.0.f32mat.weight", "model.language_model.layers.0.f32mat.weight.q8scale",
        "model.language_model.layers.0.input_layernorm.weight",
        "model.language_model.layers.0.linear_attn.conv1d.weight",
        "model.language_model.layers.0.linear_attn.A_log",
        "mtp.fc.weight", "mtp.fc.weight.q8scale",            /* the MTP head is kept */
        "model.visual.blocks.0.attn.qkv.weight", "model.visual.blocks.0.attn.qkv.weight.q8scale",
        "model.visual.merger.weight" };                      /* vision is kept too */
    CHECK(names_ok(&o, want, 16));
    {
        const LmTensor *vb = lm_st_find(&o, "model.visual.merger.weight");
        CHECK(vb && vb->dtype == LM_DT_BF16 && vb->ndim == 2 && vb->shape[0] == 8 && vb->shape[1] == 30);
        const LmTensor *vq = lm_st_find(&o, "model.visual.blocks.0.attn.qkv.weight");
        CHECK(vq && vq->dtype == LM_DT_I8);
    }

    /* metadata */
    char want_src[65];
    CHECK(lm_sha256_file(P("ck/model.safetensors.index.json"), want_src, err, sizeof err) == 0);
    CHECK(lm_st_meta(&o, "format") && !strcmp(lm_st_meta(&o, "format"), "lm-q8"));
    CHECK(lm_st_meta(&o, "group") && !strcmp(lm_st_meta(&o, "group"), "32"));
    CHECK(lm_st_meta(&o, "source") && !strcmp(lm_st_meta(&o, "source"), want_src));

    /* every quantised tensor: shape/dtype, dequantises within scale/2 of the widened source,
     * and equals the library quantiser applied to the whole matrix (so row blocking is
     * invisible). The zero-row matrix must have scale 0 on rows other than 3. */
    static const char *const qn[] = {
        "model.language_model.embed_tokens.weight", "model.language_model.layers.0.self_attn.q_proj.weight",
        "model.language_model.layers.0.zrows.weight", "model.language_model.layers.0.f32mat.weight" };
    for (int k = 0; k < 4; k++) {
        const LmTensor *a = lm_st_find(&src, qn[k]);
        const LmTensor *tq = lm_st_find(&o, qn[k]);
        char sn[256];
        snprintf(sn, sizeof sn, "%s.q8scale", qn[k]);
        const LmTensor *ts = lm_st_find(&o, sn);
        CHECK(a && tq && ts);
        if (!a || !tq || !ts) continue;
        const int64_t rows = a->shape[0], in = a->shape[1], ng = in / 32;
        CHECK(tq->dtype == LM_DT_I8 && tq->ndim == 2 && tq->shape[0] == rows && tq->shape[1] == in);
        CHECK(ts->dtype == LM_DT_F32 && ts->ndim == 2 && ts->shape[0] == rows && ts->shape[1] == ng);
        float *w = (float *)malloc((size_t)(rows * in) * 4), *back = (float *)malloc((size_t)(rows * in) * 4);
        int8_t *q = (int8_t *)malloc((size_t)(rows * in)), *qr = (int8_t *)malloc((size_t)(rows * in));
        float *sc = (float *)malloc((size_t)(rows * ng) * 4), *scr = (float *)malloc((size_t)(rows * ng) * 4);
        CHECK(lm_st_read_f32(&src, a, w, err, sizeof err) == 0);
        CHECK(lm_st_read(&o, tq, q, err, sizeof err) == 0 && lm_st_read(&o, ts, sc, err, sizeof err) == 0);
        CHECK(lm_q8_quantize(w, rows, in, qr, scr, NULL) == LM_Q8_OK);
        CHECK(!memcmp(q, qr, (size_t)(rows * in)) && !memcmp(sc, scr, (size_t)(rows * ng) * 4));
        lm_q8_dequantize(q, sc, rows, in, back);
        int worst = 0;
        for (int64_t i = 0; i < rows * in; i++)
            if (fabs((double)w[i] - (double)back[i]) > 0.5 * (double)sc[i / 32 % ng + (i / in) * ng] * (1.0 + 1e-6) + 2e-7 * fabs((double)w[i]) + 1e-30) worst++;
        CHECK_MSG(worst == 0, "%s: %d elements further than scale/2 from the source", qn[k], worst);
        if (k == 2)
            for (int r = 0; r < 8; r++) CHECK((sc[r] == 0.0f) == (r != 3));
        free(w); free(back); free(q); free(qr); free(sc); free(scr);
    }

    /* non-2-D and 1-D ".weight" tensors: copied byte for byte, same dtype and shape */
    static const char *const cn[] = {
        "model.language_model.layers.0.input_layernorm.weight", "model.language_model.layers.0.linear_attn.conv1d.weight",
        "model.language_model.layers.0.linear_attn.A_log" };
    for (int k = 0; k < 3; k++) {
        const LmTensor *a = lm_st_find(&src, cn[k]), *b = lm_st_find(&o, cn[k]);
        CHECK(a && b && a->dtype == b->dtype && a->ndim == b->ndim && a->nbytes == b->nbytes);
        if (!a || !b) continue;
        for (int d = 0; d < a->ndim; d++) CHECK(a->shape[d] == b->shape[d]);
        unsigned char *x = (unsigned char *)malloc((size_t)a->nbytes), *y = (unsigned char *)malloc((size_t)a->nbytes);
        CHECK(lm_st_read(&src, a, x, err, sizeof err) == 0 && lm_st_read(&o, b, y, err, sizeof err) == 0);
        CHECK(!memcmp(x, y, (size_t)a->nbytes));
        free(x); free(y);
    }

    /* byte layout of the file: compact JSON, metadata first, tensors sorted by name,
     * data packed back to back, header padded with spaces to a multiple of 8 */
    {
        FILE *f = fopen(P("q8.safetensors"), "rb");
        unsigned char l8[8];
        CHECK(f && fread(l8, 1, 8, f) == 8);
        uint64_t N = 0;
        for (int i = 7; i >= 0; i--) N = (N << 8) | l8[i];
        char *h = (char *)malloc(N + 1);
        CHECK(fread(h, 1, N, f) == N);
        h[N] = '\0';
        fclose(f);
        CHECK(N % 8 == 0);
        size_t e = N;
        while (e > 0 && h[e - 1] == ' ') e--;                    /* strip the padding */
        CHECK(h[0] == '{' && h[e - 1] == '}');
        CHECK(strncmp(h, "{\"__metadata__\":{\"format\":\"lm-q8\",\"group\":\"32\",\"source\":\"", 57) == 0);
        int spaces = 0;
        for (size_t i = 0; i < e; i++) spaces += h[i] == ' ';
        CHECK(spaces == 0);                                      /* no whitespace inside the compact JSON */
        CHECK(e < N || N % 8 == 0);
        /* names appear in ascending byte order, and data_offsets chain from 0 */
        char prev[256] = "";
        int64_t cursor = 0;
        const char *p = strstr(h, "},\"model.");                /* first tensor entry follows the metadata object */
        CHECK(p != NULL);
        int count = 0;
        while (p && (p = strstr(p, "\":{\"dtype\"")) != NULL) { /* every tensor entry: "<name>":{"dtype"... */
            const char *qe = p, *q = p;
            while (q > h && q[-1] != '"') q--;
            char nm[256];
            snprintf(nm, sizeof nm, "%.*s", (int)(qe - q), q);
            CHECK(strcmp(prev, nm) < 0);
            snprintf(prev, sizeof prev, "%s", nm);
            const char *d = strstr(qe, "\"data_offsets\":[");
            long long s0 = 0, s1 = 0;
            CHECK(d && sscanf(d, "\"data_offsets\":[%lld,%lld]", &s0, &s1) == 2);
            CHECK(s0 == cursor);
            cursor = s1;
            count++;
            p = d + 10;
        }
        CHECK(count == 16);
        struct stat sb;
        CHECK(stat(P("q8.safetensors"), &sb) == 0 && (int64_t)sb.st_size == 8 + (int64_t)N + cursor);
        free(h);
    }
    lm_st_close(&o);
    lm_st_close(&src);

    /* running it again over the same input is byte-identical (no timestamps, no ordering luck) */
    rc = lm_quantize_checkpoint(P("ck"), P("q8b.safetensors"), NULL, err, sizeof err);
    CHECK_MSG(rc == 0, "%s", err);
    {
        FILE *a = fopen(P("q8.safetensors"), "rb"), *b = fopen(P("q8b.safetensors"), "rb");
        int same = a && b, ca, cb;
        while (same && (ca = fgetc(a)) != EOF) { cb = fgetc(b); if (ca != cb) same = 0; }
        if (same && fgetc(b) != EOF) same = 0;
        CHECK(same);
        if (a) fclose(a);
        if (b) fclose(b);
    }

    /* a single-file checkpoint: source hash is the file's own */
    mkdir(P("one"), 0755);
    {
        const Src s[] = { s2[0], s2[1] };
        write_src(P("one/model.safetensors"), s, 2);
        rc = lm_quantize_checkpoint(P("one"), P("one.q8"), &stt, err, sizeof err);
        CHECK_MSG(rc == 0, "%s", err);
        CHECK(stt.n_quantised == 1 && stt.n_copied == 1 && stt.n_ignored == 0);
        LmSt t;
        char h[65];
        CHECK(lm_st_open(&t, P("one.q8"), 1, err, sizeof err) == 0);
        CHECK(lm_sha256_file(P("one/model.safetensors"), h, err, sizeof err) == 0);
        CHECK(lm_st_meta(&t, "source") && !strcmp(lm_st_meta(&t, "source"), h));
        lm_st_close(&t);
    }

    free(embed); free(qproj); free(zrow); free(norm); free(conv); free(vis); free(mtp); free(visbad);
}

static void expect_fail(const char *dir, const char *needle, const char *label)
{
    char err[512];
    const int rc = lm_quantize_checkpoint(dir, P("fail.q8"), NULL, err, sizeof err);
    CHECK_MSG(rc != 0 && strstr(err, needle), "%s: rc=%d err=\"%s\" (wanted \"%s\")", label, rc, err, needle);
    CHECK_MSG(access(P("fail.q8"), F_OK) != 0 && access(P("fail.q8.tmp"), F_OK) != 0,
              "%s: a failed run left a file behind", label);
}

static void test_failures(void)
{
    uint16_t *ok = rand_bf16(8 * 64, 1.0f);

    /* in not a multiple of 32: loud error naming the tensor, nothing written */
    mkdir(P("bad_in"), 0755);
    {
        uint16_t *w = rand_bf16(8 * 40, 1.0f);
        const Src s[] = { { "model.language_model.good.weight", "BF16", 2, { 8, 64 }, ok, 8 * 64 * 2 },
                          { "model.language_model.odd.weight", "BF16", 2, { 8, 40 }, w, 8 * 40 * 2 } };
        write_src(P("bad_in/model.safetensors"), s, 2);
        expect_fail(P("bad_in"), "model.language_model.odd.weight has in = 40", "in % 32 != 0");
        free(w);
    }

    /* NaN and Inf anywhere in a quantised tensor */
    for (int k = 0; k < 2; k++) {
        char d[64];
        snprintf(d, sizeof d, "nan%d", k);
        mkdir(P(d), 0755);
        uint16_t *w = rand_bf16(8 * 64, 1.0f);
        w[5 * 64 + 33] = k == 0 ? 0x7fc0 /* NaN */ : 0x7f80 /* +Inf */;
        const Src s[] = { { "model.language_model.a.weight", "BF16", 2, { 8, 64 }, ok, 8 * 64 * 2 }, { "model.language_model.b.weight", "BF16", 2, { 8, 64 }, w, 8 * 64 * 2 } };
        char f[96];
        snprintf(f, sizeof f, "%s/model.safetensors", d);
        write_src(P(f), s, 2);
        expect_fail(P(d), "model.language_model.b.weight has a non-finite value at element 353", k == 0 ? "NaN" : "Inf");
        free(w);
    }

    /* a NaN in a tensor that is merely COPIED (a norm, 1-D) is not our business */
    mkdir(P("nan_copy"), 0755);
    {
        uint16_t nn[4] = { 0x7fc0, 0, 0, 0 };
        const Src s[] = { { "model.language_model.a.weight", "BF16", 2, { 8, 64 }, ok, 8 * 64 * 2 }, { "model.language_model.n.weight", "BF16", 1, { 4 }, nn, 8 } };
        write_src(P("nan_copy/model.safetensors"), s, 2);
        char err[256];
        CHECK_MSG(lm_quantize_checkpoint(P("nan_copy"), P("nan_copy.q8"), NULL, err, sizeof err) == 0, "%s", err);
    }

    /* an integer 2-D .weight cannot be quantised as if it were a float matrix */
    mkdir(P("i8w"), 0755);
    {
        int8_t z[8 * 32] = { 0 };
        const Src s[] = { { "model.language_model.a.weight", "BF16", 2, { 8, 64 }, ok, 8 * 64 * 2 }, { "model.language_model.b.weight", "I8", 2, { 8, 32 }, z, sizeof z } };
        write_src(P("i8w/model.safetensors"), s, 2);
        expect_fail(P("i8w"), "model.language_model.b.weight has dtype I8", "I8 weight");
    }

    /* no 2-D weights: not a model */
    mkdir(P("no2d"), 0755);
    {
        const Src s[] = { { "model.language_model.a.bias", "BF16", 1, { 8 }, ok, 16 } };
        write_src(P("no2d/model.safetensors"), s, 1);
        expect_fail(P("no2d"), "no 2-D .weight tensors", "no weights");
    }

    /* a source name that would collide with the derived scale table */
    mkdir(P("coll"), 0755);
    {
        const Src s[] = { { "model.language_model.a.weight", "BF16", 2, { 8, 64 }, ok, 8 * 64 * 2 }, { "model.language_model.a.weight.q8scale", "F32", 1, { 1 }, ok, 4 } };
        write_src(P("coll/model.safetensors"), s, 2);
        expect_fail(P("coll"), "collides with the .q8scale suffix", "name collision");
    }

    /* missing and truncated sources */
    expect_fail(P("does_not_exist"), "cannot open", "missing dir");
    {
        char err[256];
        CHECK(lm_quantize_checkpoint(P("does_not_exist"), P("x.q8"), NULL, err, sizeof err) != 0);
    }
    mkdir(P("cut"), 0755);
    {
        uint16_t *w = rand_bf16(8 * 64, 1.0f);
        const Src s[] = { { "model.language_model.a.weight", "BF16", 2, { 8, 64 }, w, 8 * 64 * 2 } };
        write_src(P("cut/full.safetensors"), s, 1);
        FILE *f = fopen(P("cut/full.safetensors"), "rb");
        unsigned char buf[4096];
        const size_t n = fread(buf, 1, sizeof buf, f);
        fclose(f);
        lt_write_file(P("cut/model.safetensors"), buf, n - 10);
        expect_fail(P("cut"), "truncated", "truncated source");
        free(w);
    }
    free(ok);
}

/* A tensor bigger than one 4M-element block, so the row-blocked path runs more than once,
 * with a ragged last block; compared against the quantiser applied to the whole matrix. */
static void test_multiblock(void)
{
    enum { ROWS = 2200, IN = 2048 };           /* 4.5M elements: 2 blocks of 2048 rows, then 152 */
    char err[512];
    uint16_t *w = rand_bf16((size_t)ROWS * IN, 2.0f);
    mkdir(P("big"), 0755);
    const Src s[] = { { "model.language_model.big.weight", "BF16", 2, { ROWS, IN }, w, (size_t)ROWS * IN * 2 } };
    write_src(P("big/model.safetensors"), s, 1);
    int rc = lm_quantize_checkpoint(P("big"), P("big.q8"), NULL, err, sizeof err);
    CHECK_MSG(rc == 0, "%s", err);
    LmSt o;
    CHECK_MSG(lm_st_open(&o, P("big.q8"), 1, err, sizeof err) == 0, "%s", err);
    const LmTensor *tq = lm_st_find(&o, "model.language_model.big.weight"), *ts = lm_st_find(&o, "model.language_model.big.weight.q8scale");
    CHECK(tq && ts);
    if (tq && ts) {
        float *f = (float *)malloc((size_t)ROWS * IN * 4);
        int8_t *q = (int8_t *)malloc((size_t)ROWS * IN), *qr = (int8_t *)malloc((size_t)ROWS * IN);
        float *sc = (float *)malloc((size_t)ROWS * (IN / 32) * 4), *scr = (float *)malloc((size_t)ROWS * (IN / 32) * 4);
        for (size_t i = 0; i < (size_t)ROWS * IN; i++) f[i] = lm_bf16_to_f32(w[i]);
        CHECK(lm_q8_quantize(f, ROWS, IN, qr, scr, NULL) == LM_Q8_OK);
        CHECK(lm_st_read(&o, tq, q, err, sizeof err) == 0 && lm_st_read(&o, ts, sc, err, sizeof err) == 0);
        CHECK(!memcmp(q, qr, (size_t)ROWS * IN));
        CHECK(!memcmp(sc, scr, (size_t)ROWS * (IN / 32) * 4));
        free(f); free(q); free(qr); free(sc); free(scr);
    }
    lm_st_close(&o);
    free(w);
}

/* Byte-identical to the Python reference (tools/quant_ref.py) on its tiny fixture. The
 * fixture is read in place, never copied; absence is a FAILURE, not a skip. */
/* Regression: a checkpoint's vision tower has rank-5 tensors (Conv3d). Before the first fix
 * the reader rejected them at open; the quantiser now keeps the patch embedding, flattened to
 * the 2-D matrix the encoder uses. */
static void test_rank_above_four_vision_is_flattened(void)
{
    char err[256];
    mkdir(P("r5"), 0755);
    unsigned char data[88];
    memset(data, 0, sizeof data);
    const uint16_t one = 0x3f80;                       /* bf16 1.0 */
    for (int i = 0; i < 32; i++) memcpy(data + 24 + 2 * i, &one, 2);
    lt_write_st_raw8(P("r5/model.safetensors"),
        "{\"model.visual.patch_embed.proj.weight\":{\"dtype\":\"F32\",\"shape\":[1,1,1,2,2],\"data_offsets\":[0,16]},"
        "\"model.language_model.norm.weight\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[16,24]},"
        "\"model.language_model.layers.0.mlp.weight\":{\"dtype\":\"BF16\",\"shape\":[1,32],\"data_offsets\":[24,88]}}",
        data, sizeof data);
    LmQuantStats st;
    int rc = lm_quantize_checkpoint(P("r5"), P("r5.q8.safetensors"), &st, err, sizeof err);
    CHECK_MSG(rc == 0, "%s", err);
    CHECK(st.n_quantised == 1 && st.n_copied == 2 && st.n_ignored == 0);
    LmSt o;
    CHECK_MSG(lm_st_open(&o, P("r5.q8.safetensors"), 1, err, sizeof err) == 0, "%s", err);
    const LmTensor *pe = lm_st_find(&o, "model.visual.patch_embed.proj.weight");
    CHECK(pe && pe->dtype == LM_DT_F32 && pe->ndim == 2 && pe->shape[0] == 1 && pe->shape[1] == 4);
    lm_st_close(&o);
}

/* Negative case: the in % 32 leniency is for vision only; a text or MTP matrix still fails. */
static void test_text_width_rule_unchanged(void)
{
    char err[256];
    mkdir(P("w30"), 0755);
    unsigned char data[60 + 64];
    memset(data, 0, sizeof data);
    lt_write_st_raw8(P("w30/model.safetensors"),
        "{\"model.language_model.layers.0.mlp.weight\":{\"dtype\":\"BF16\",\"shape\":[1,32],\"data_offsets\":[0,64]},"
        "\"mtp.fc.weight\":{\"dtype\":\"BF16\",\"shape\":[1,30],\"data_offsets\":[64,124]}}",
        data, sizeof data);
    LmQuantStats st;
    CHECK(lm_quantize_checkpoint(P("w30"), P("w30.q8.safetensors"), &st, err, sizeof err) != 0);
    CHECK(strstr(err, "mtp.fc.weight") != NULL);
}

static void test_reference_fixture(void)
{
    static const char *const cands[] = { "tests/fixtures/qwen35-tiny" };
    const char *dir = NULL;
    char probe[1200];
    for (int i = 0; i < 1 && !dir; i++) {
        snprintf(probe, sizeof probe, "%s/model.q8.safetensors", cands[i]);
        if (access(probe, R_OK) == 0) dir = cands[i];
    }
    CHECK_MSG(dir != NULL, "reference fixture not found in tests/fixtures/qwen35-tiny nor the agent-a0aba2250d37ff760 worktree");
    if (!dir) return;
    char err[512];
    int rc = lm_quantize_checkpoint(dir, P("ref.q8"), NULL, err, sizeof err);
    CHECK_MSG(rc == 0, "%s", err);
    FILE *a = fopen(P("ref.q8"), "rb"), *b = fopen(probe, "rb");
    CHECK(a && b);
    if (!a || !b) return;
    long off = 0;
    int ca = 0, cb = 0, same = 1;
    for (;;) {
        ca = fgetc(a); cb = fgetc(b);
        if (ca != cb) { same = 0; break; }
        if (ca == EOF) break;
        off++;
    }
    fclose(a); fclose(b);
    CHECK_MSG(same, "output differs from %s at byte %ld (%d vs %d)", probe, off, ca, cb);
}

int main(void)
{
    lt_mkdtemp(g_dir);
    test_edge_cases();
    test_properties();
    test_checkpoint();
    test_failures();
    test_multiblock();
    test_rank_above_four_vision_is_flattened();
    test_text_width_rule_unchanged();
    test_reference_fixture();
    lt_rmtree(g_dir);
    return lt_finish("test_quant");
}
