/* test_gemma.c - lm_open / lm_feed / lm_reset on the Gemma 4 tiny fixture
 * (tests/fixtures/gemma4-tiny, run from the repository root; absence is a FAILURE), bf16 and
 * q8: logits against the float64 oracle's goldens (tolerance 2e-4 * max|logit| of the row, as
 * the gate), argmax, greedy ids, reset determinism, token/context errors, the KV-sharing and
 * ring-cache behaviour, config rejections on mutated copies of config.json, and the state
 * memory of the REAL E4B shape at ctx 4096. Written to a temp dir, never to the fixtures. */
#include "lm_test.h"

#include "lm/lm.h"
#include "json.h"
#include "model/lm_gemma.h"
#include "model/lm_gemma_config.h"
#include "model/lm_layers.h"

#define FIX "tests/fixtures/gemma4-tiny"
#define Q8 FIX "/model.q8.safetensors"
#define PATH_MAX_ 4096

static char g_dir[512];

static char *P(const char *leaf)
{
    static char b[4][1024];
    static int i;
    i = (i + 1) & 3;
    lt_path(b[i], g_dir, leaf);
    return b[i];
}

static char *read_all(const char *path, size_t *len)
{
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    const long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *b = (char *)malloc((size_t)n + 1);
    const size_t got = fread(b, 1, (size_t)n, f);
    fclose(f);
    b[got] = '\0';
    if (len) *len = got;
    return b;
}

static LmModel *open_model(const char *weights, int ctx)
{
    char err[512];
    LmOpts o = { weights, ctx, 0, 0 };
    LmModel *m = lm_open(FIX, &o, err, sizeof err);
    CHECK_MSG(m != NULL, "lm_open: %s", err);
    return m;
}

static int same(const float *a, const float *b, int n) { return !memcmp(a, b, (size_t)n * sizeof(float)); }

static int argmax(const float *x, int n)
{
    int b = 0;
    for (int i = 1; i < n; i++) if (x[i] > x[b]) b = i;
    return b;
}

typedef struct { int n, V; int *tok; float *logits; int np, ng; int *prompt, *out; } Golden;

static int ints(jval *a, int **out)
{
    *out = (int *)malloc((size_t)(a->len ? a->len : 1) * sizeof(int));
    for (int i = 0; i < a->len; i++) (*out)[i] = (int)a->kids[i]->num;
    return a->len;
}

static int load_golden(const char *file, Golden *g)
{
    char path[256];
    snprintf(path, sizeof path, FIX "/%s", file);
    char *txt = read_all(path, NULL);
    if (!txt) { fprintf(stderr, "  FAIL cannot read %s\n", path); lt_fails++; return -1; }
    jval *r = json_parse(txt, NULL);
    free(txt);
    if (!r) { fprintf(stderr, "  FAIL %s does not parse\n", path); lt_fails++; return -1; }
    g->n = ints(json_get(r, "tokens"), &g->tok);
    jval *lg = json_get(r, "logits");
    g->V = lg->kids[0]->len;
    g->logits = (float *)malloc((size_t)g->n * g->V * sizeof(float));
    for (int t = 0; t < g->n; t++)
        for (int i = 0; i < g->V; i++) g->logits[(size_t)t * g->V + i] = (float)lg->kids[t]->kids[i]->num;
    jval *gr = json_get(r, "greedy");
    g->np = ints(json_get(gr, "prompt"), &g->prompt);
    g->ng = ints(json_get(gr, "out"), &g->out);
    json_free_tree(r);
    return 0;
}

static void free_golden(Golden *g) { free(g->tok); free(g->logits); free(g->prompt); free(g->out); }

static void test_family(const char *weights, const char *golden, const char *label)
{
    Golden G;
    if (load_golden(golden, &G) != 0) return;
    LmModel *m = open_model(weights, 0);
    if (!m) { free_golden(&G); return; }
    const int V = lm_n_vocab(m), n = G.n;
    CHECK(V == 384 && G.V == V && lm_n_ctx(m) == 4096 && lm_pos(m) == 0);
    float *all = (float *)malloc((size_t)n * V * 4), *row = (float *)malloc((size_t)V * 4);

    /* the whole 40-token golden (5x the sliding window): tolerance, argmax, finiteness */
    double worst = 0;
    for (int t = 0; t < n; t++) {
        CHECK(lm_feed(m, G.tok[t], all + (size_t)t * V) == LM_OK && lm_pos(m) == t + 1);
        const float *got = all + (size_t)t * V, *want = G.logits + (size_t)t * V;
        double mx = 0;
        for (int i = 0; i < V; i++) if (fabs((double)want[i]) > mx) mx = fabs((double)want[i]);
        int bad = 0;
        for (int i = 0; i < V; i++) {
            const double e = fabs((double)got[i] - (double)want[i]);
            if (e / (2e-4 * mx) > worst) worst = e / (2e-4 * mx);
            if (!(e <= 2e-4 * mx)) bad++;
        }
        CHECK_MSG(bad == 0, "%s: position %d: %d logits beyond 2e-4*max|logit|", label, t, bad);
        CHECK_MSG(argmax(got, V) == argmax(want, V), "%s: position %d argmax %d, golden %d", label, t, argmax(got, V), argmax(want, V));
    }
    printf("  %s: worst logit error = %.1f%% of the tolerance\n", label, worst * 100.0);
    {   /* the softcap is live: every logit is strictly inside +-cap, and the cap bites */
        double mx = 0;
        for (int i = 0; i < n * V; i++) { CHECK(isfinite(all[i]) && fabs(all[i]) < 16.0); if (fabs(all[i]) > mx) mx = fabs(all[i]); }
        CHECK(mx > 8.0);
    }

    /* reset, then re-feed: bitwise identical logits at every position */
    lm_reset(m);
    CHECK(lm_pos(m) == 0);
    for (int t = 0; t < n; t++) {
        CHECK(lm_feed(m, G.tok[t], row) == LM_OK);
        CHECK_MSG(same(row, all + (size_t)t * V, V), "%s: after reset position %d differs", label, t);
    }
    /* NEGATIVE: feeding on without a reset changes the logits (so the check above can fail) */
    CHECK(lm_feed(m, G.tok[0], row) == LM_OK && !same(row, all, V));
    lm_reset(m);

    /* NULL-logits prefill then one logits call equals the full run */
    for (int t = 0; t < n - 1; t++) CHECK(lm_feed(m, G.tok[t], NULL) == LM_OK);
    CHECK(lm_feed(m, G.tok[n - 1], row) == LM_OK);
    CHECK_MSG(same(row, all + (size_t)(n - 1) * V, V), "%s: NULL-logits prefill changes the result", label);

    /* a fresh model fed a prefix equals the incremental run at that position */
    for (int p = 3; p < n; p += 11) {
        LmModel *f = open_model(weights, 0);
        for (int t = 0; t <= p; t++) CHECK(lm_feed(f, G.tok[t], row) == LM_OK);
        CHECK_MSG(same(row, all + (size_t)p * V, V), "%s: fresh run to position %d differs", label, p);
        lm_close(f);
    }

    /* greedy continuation of the 20-token prompt equals the golden ids */
    lm_reset(m);
    for (int t = 0; t < G.np; t++) CHECK(lm_feed(m, G.prompt[t], row) == LM_OK);
    int greedy_ok = 1;
    for (int k = 0; k < G.ng; k++) {
        const int a = argmax(row, V);
        if (a != G.out[k]) greedy_ok = 0;
        CHECK(lm_feed(m, a, row) == LM_OK);
    }
    CHECK_MSG(greedy_ok, "%s: greedy ids differ from the golden", label);

    /* invalid tokens are refused with nothing fed */
    lm_reset(m);
    CHECK(lm_feed(m, G.tok[0], row) == LM_OK);
    CHECK(lm_feed(m, -1, row) == LM_ERR_TOKEN && lm_pos(m) == 1);
    CHECK(lm_feed(m, V, row) == LM_ERR_TOKEN && lm_pos(m) == 1);
    CHECK(lm_feed(m, 2147483647, row) == LM_ERR_TOKEN && lm_pos(m) == 1);
    CHECK(lm_feed(m, G.tok[1], row) == LM_OK && same(row, all + V, V));
    lm_close(m);

    /* a full context window: LM_ERR_CTX, nothing fed, the cached state is undisturbed. ctx 5 is
     * below the sliding window (ring of 5), ctx 12 is above it (ring of 8, wrapped), ctx 40 holds
     * the whole golden: every prefix is bitwise equal to the ctx 4096 run, so the ring and the
     * mask are independent of how big the cache is */
    static const int ctxs[3] = { 5, 12, 40 };
    for (int ci = 0; ci < 3; ci++) {
        const int ctx = ctxs[ci];
        m = open_model(weights, ctx);
        CHECK(m && lm_n_ctx(m) == ctx);
        if (!m) continue;
        for (int t = 0; t < ctx; t++) {
            CHECK(lm_feed(m, G.tok[t], row) == LM_OK);
            CHECK_MSG(same(row, all + (size_t)t * V, V), "%s: ctx %d position %d differs from the ctx 4096 run", label, ctx, t);
        }
        CHECK(lm_feed(m, G.tok[ctx % n], row) == LM_ERR_CTX && lm_pos(m) == ctx);
        CHECK(lm_feed(m, G.tok[0], NULL) == LM_ERR_CTX && lm_pos(m) == ctx);
        lm_reset(m);
        CHECK(lm_pos(m) == 0);
        for (int t = 0; t < ctx; t++) CHECK(lm_feed(m, G.tok[t], row) == LM_OK);
        CHECK(same(row, all + (size_t)(ctx - 1) * V, V));
        lm_close(m);
    }
    free(all); free(row);
    free_golden(&G);
}

/* ------------------------------------------------------------------ KV sharing: reuse layers never bind k/v */

/* copy a safetensors file to dst, renaming (same length) every tensor whose name contains
 * `from` by replacing that text with `to`; returns the number of replacements */
static int patch_names(const char *src, const char *dst, const char *from, const char *to)
{
    size_t len;
    char *b = read_all(src, &len);
    CHECK(b != NULL);
    if (!b) return 0;
    uint64_t hl = 0;
    for (int i = 7; i >= 0; i--) hl = hl << 8 | (unsigned char)b[i];
    CHECK(strlen(from) == strlen(to));
    int n = 0;
    char *hdr = b + 8, *end = b + 8 + hl, *at;
    while ((at = (char *)memmem(hdr, (size_t)(end - hdr), from, strlen(from))) != NULL) {
        memcpy(at, to, strlen(to));
        n++;
    }
    lt_write_file(dst, b, len);
    free(b);
    return n;
}

static void test_reuse_layers_never_bind(const char *weights, const char *file, const char *label)
{
    Golden G;
    if (load_golden(weights ? "golden_q8.json" : "golden_bf16.json", &G) != 0) return;
    char err[512];
    mkdir(P("kv"), 0755);
    char *cfg = read_all(FIX "/config.json", NULL);
    lt_write_text(P("kv/config.json"), cfg);
    free(cfg);
    /* layers 8..11 reuse (first_shared = 8): hide their k_proj, v_proj and k_norm (and q8 scales) */
    char src[1024];
    snprintf(src, sizeof src, FIX "/%s", file);
    int renamed = 0;
    static const char *const kinds[3] = { "k_proj", "v_proj", "k_norm" };
    char cur[1024];
    snprintf(cur, sizeof cur, "%s", src);
    for (int l = 8; l < 12; l++)
        for (int k = 0; k < 3; k++) {
            char from[128], to[128];
            snprintf(from, sizeof from, "layers.%d.self_attn.%s.weight", l, kinds[k]);
            snprintf(to, sizeof to, "layers.%d.self_attn.%s.weigh_", l, kinds[k]);
            char next[1024];
            snprintf(next, sizeof next, "%s", P((l * 3 + k) & 1 ? "kv/a.safetensors" : "kv/b.safetensors"));
            renamed += patch_names(cur, next, from, to);
            snprintf(cur, sizeof cur, "%s", next);
        }
    CHECK_MSG(renamed >= 12, "%s: only %d tensor names patched", label, renamed);
    LmOpts o = { NULL, 0, 0, 0 };
    char wpath[1024];
    if (weights) { snprintf(wpath, sizeof wpath, "%s", cur); o.weights = wpath; }
    else {
        size_t len;
        char *b = read_all(cur, &len);
        lt_write_file(P("kv/model.safetensors"), b, len);
        free(b);
    }
    LmModel *m = lm_open(P("kv"), &o, err, sizeof err);
    CHECK_MSG(m != NULL, "%s: a model without the reuse layers' k_proj/v_proj/k_norm must open: %s", label, err);
    if (m) {
        float *row = (float *)malloc((size_t)lm_n_vocab(m) * 4);
        for (int t = 0; t < G.n; t++) {
            CHECK(lm_feed(m, G.tok[t], row) == LM_OK);
            double mx = 0, worst = 0;
            const float *want = G.logits + (size_t)t * G.V;
            for (int i = 0; i < G.V; i++) if (fabs((double)want[i]) > mx) mx = fabs((double)want[i]);
            for (int i = 0; i < G.V; i++) worst = fmax(worst, fabs((double)row[i] - (double)want[i]));
            CHECK_MSG(worst <= 2e-4 * mx, "%s: position %d off by %g without the reuse layers' k/v tensors", label, t, worst);
        }
        free(row);
        lm_close(m);
    }
    /* NEGATIVE: hide layer 7's k_proj (a producer): the open must fail and name it */
    {
        char from[128], to[128];
        snprintf(from, sizeof from, "layers.7.self_attn.k_proj.weight");
        snprintf(to, sizeof to, "layers.7.self_attn.k_proj.weigh_");
        patch_names(src, P("kv/bad.safetensors"), from, to);
        mkdir(P("kvbad"), 0755);
        cfg = read_all(FIX "/config.json", NULL);
        lt_write_text(P("kvbad/config.json"), cfg);
        free(cfg);
        LmOpts ob = { NULL, 0, 0, 0 };
        char bpath[1024];
        if (weights) { snprintf(bpath, sizeof bpath, "%s", P("kv/bad.safetensors")); ob.weights = bpath; }
        else {
            size_t len;
            char *b = read_all(P("kv/bad.safetensors"), &len);
            lt_write_file(P("kvbad/model.safetensors"), b, len);
            free(b);
        }
        LmModel *bm = lm_open(P("kvbad"), &ob, err, sizeof err);
        CHECK_MSG(bm == NULL && strstr(err, "layers.7.self_attn.k_proj.weight"), "%s: producer k_proj missing: \"%s\"", label, err);
        if (bm) lm_close(bm);
    }
    free_golden(&G);
}

/* ------------------------------------------------------------------ rejections */

/* replace `from` by `to` in config.json, searching only text_config (which comes after the
 * audio_config that repeats several keys); `from` must occur. Writes g_dir/cfg/config.json. */
static void mutate_t(const char *from, const char *to)
{
    char *txt = read_all(FIX "/config.json", NULL);
    CHECK(txt != NULL);
    if (!txt) return;
    char *tc = strstr(txt, "\"text_config\"");
    char *at = tc ? strstr(tc, from) : NULL;
    CHECK_MSG(at != NULL, "mutation anchor \"%s\" not in text_config", from);
    char *out = (char *)malloc(strlen(txt) + strlen(to) + 1);
    if (at) snprintf(out, strlen(txt) + strlen(to) + 1, "%.*s%s%s", (int)(at - txt), txt, to, at + strlen(from));
    else snprintf(out, strlen(txt) + 1, "%s", txt);
    mkdir(P("cfg"), 0755);
    lt_write_text(P("cfg/config.json"), out);
    free(txt);
    free(out);
}

static void mutate_top(const char *from, const char *to)
{
    char *txt = read_all(FIX "/config.json", NULL);
    char *at = strstr(txt, from);
    CHECK_MSG(at != NULL, "mutation anchor \"%s\" not in config.json", from);
    char *out = (char *)malloc(strlen(txt) + strlen(to) + 1);
    if (at) snprintf(out, strlen(txt) + strlen(to) + 1, "%.*s%s%s", (int)(at - txt), txt, to, at + strlen(from));
    else snprintf(out, strlen(txt) + 1, "%s", txt);
    mkdir(P("cfg"), 0755);
    lt_write_text(P("cfg/config.json"), out);
    free(txt);
    free(out);
}

static void expect_open_fail(const char *dir, const LmOpts *o, const char *needle, const char *label)
{
    char err[512] = "";
    LmModel *m = lm_open(dir, o, err, sizeof err);
    CHECK_MSG(m == NULL && strstr(err, needle), "%s: m=%p err=\"%s\" (wanted \"%s\")", label, (void *)m, err, needle);
    if (m) lm_close(m);
}

static void expect_cfg_fail(const char *needle, const char *label) { expect_open_fail(P("cfg"), NULL, needle, label); }

static void test_config_rejections(void)
{
    /* a config-only directory: every failure below comes from the config, before any weights */
    mutate_top("\"model_type\": \"gemma4\",", "\"model_type\": \"llama\",");
    expect_cfg_fail("model_type \"llama\" is not supported (supported: qwen3_5, gemma4)", "unknown family names the supported ones");
    mutate_t("\"model_type\": \"gemma4_text\"", "\"model_type\": \"gemma4_vl\"");
    expect_cfg_fail("text_config.model_type \"gemma4_vl\"", "wrong text model_type");

    static const char *const fields[] = { "hidden_size", "intermediate_size", "num_hidden_layers", "vocab_size",
        "vocab_size_per_layer_input", "rms_norm_eps", "num_attention_heads", "num_key_value_heads", "head_dim",
        "global_head_dim", "hidden_activation", "sliding_window", "num_kv_shared_layers", "hidden_size_per_layer_input",
        "final_logit_softcapping", "tie_word_embeddings", "layer_types", "rope_parameters", "enable_moe_block",
        "attention_k_eq_v", "use_double_wide_mlp", "attention_bias", "use_bidirectional_attention",
        "num_global_key_value_heads" };
    for (size_t i = 0; i < sizeof fields / sizeof *fields; i++) {
        char from[64], to[64];
        snprintf(from, sizeof from, "\"%s\"", fields[i]);
        snprintf(to, sizeof to, "\"x_%s\"", fields[i]);
        mutate_t(from, to);
        char want[96];
        snprintf(want, sizeof want, "missing \"%s\"", fields[i]);
        if (!strcmp(fields[i], "tie_word_embeddings")) snprintf(want, sizeof want, "tie_word_embeddings");
        if (!strcmp(fields[i], "layer_types")) snprintf(want, sizeof want, "layer_types");
        expect_cfg_fail(want, fields[i]);
    }
    /* the unsupported flags, each with its own message */
    mutate_t("\"enable_moe_block\": false", "\"enable_moe_block\": true");
    expect_cfg_fail("enable_moe_block=true is not supported", "moe");
    mutate_t("\"attention_k_eq_v\": false", "\"attention_k_eq_v\": true");
    expect_cfg_fail("attention_k_eq_v=true is not supported", "k_eq_v");
    mutate_t("\"use_double_wide_mlp\": false", "\"use_double_wide_mlp\": true");
    expect_cfg_fail("use_double_wide_mlp=true is not supported", "double wide mlp");
    mutate_t("\"attention_bias\": false", "\"attention_bias\": true");
    expect_cfg_fail("attention_bias=true is not supported", "attention bias");
    mutate_t("\"use_bidirectional_attention\": null", "\"use_bidirectional_attention\": true");
    expect_cfg_fail("use_bidirectional_attention is not supported", "bidirectional");
    mutate_t("\"num_global_key_value_heads\": null", "\"num_global_key_value_heads\": 2");
    expect_cfg_fail("num_global_key_value_heads is not supported", "global kv heads");
    mutate_t("\"pad_token_id\": 0,", "\"pad_token_id\": 0, \"per_layer_config\": {\"0\": {}},");
    expect_cfg_fail("per_layer_config is not supported", "per_layer_config");
    mutate_t("\"hidden_activation\": \"gelu_pytorch_tanh\"", "\"hidden_activation\": \"gelu\"");
    expect_cfg_fail("hidden_activation \"gelu\" is not supported", "activation");
    mutate_t("\"tie_word_embeddings\": true", "\"tie_word_embeddings\": false");
    expect_cfg_fail("tie_word_embeddings=false is not supported", "untied embeddings");
    mutate_top("\"tie_word_embeddings\": true,\n    \"transformers_version\"", "\"tie_word_embeddings\": false,\n    \"transformers_version\"");
    expect_cfg_fail("tie_word_embeddings=false is not supported", "untied embeddings (top level)");
    mutate_t("\"final_logit_softcapping\": 16.0", "\"final_logit_softcapping\": 0.0");
    expect_cfg_fail("final_logit_softcapping must be a positive number", "softcap zero");
    mutate_t("\"final_logit_softcapping\": 16.0", "\"final_logit_softcapping\": -3.0");
    expect_cfg_fail("final_logit_softcapping must be a positive number", "softcap negative");
    mutate_t("\"final_logit_softcapping\": 16.0", "\"final_logit_softcapping\": null");
    expect_cfg_fail("final_logit_softcapping must be a positive number", "softcap null");
    mutate_t("\"hidden_size_per_layer_input\": 32", "\"hidden_size_per_layer_input\": 0");
    expect_cfg_fail("hidden_size_per_layer_input=0 is not supported", "no per-layer input");
    mutate_t("\"vocab_size_per_layer_input\": 384", "\"vocab_size_per_layer_input\": 400");
    expect_cfg_fail("vocab_size_per_layer_input (400) differs", "ple vocab");
    mutate_t("\"rope_type\": \"proportional\"", "\"rope_type\": \"yarn\"");
    expect_cfg_fail("rope_type \"yarn\" is not supported", "rope type");
    mutate_t("\"rope_type\": \"default\"", "\"rope_type\": \"default\", \"partial_rotary_factor\": 0.5");
    expect_cfg_fail("default rope with partial_rotary_factor != 1", "default rope partial");
    mutate_t("\"full_attention\"\n        ],", "\"sliding_attention\"\n        ],");
    expect_cfg_fail("the last layer must be full_attention", "last layer");
    mutate_t("\"sliding_attention\",\n            \"sliding_attention\",\n            \"sliding_attention\",\n            \"sliding_attention\",\n            \"sliding_attention\",\n            \"full_attention\",",
             "\"sliding_attention\",\n            \"sliding_attention\",\n            \"sliding_attention\",\n            \"sliding_attention\",\n            \"sliding_attention\",\n            \"sliding_attention\",");
    expect_cfg_fail("no earlier layer of its type produces it", "full layer without a producer");
    mutate_t("\"layer_types\": [\n            \"sliding_attention\"", "\"layer_types\": [\n            \"linear_attention\"");
    expect_cfg_fail("unsupported layer type", "layer type");
    mutate_t("\"num_hidden_layers\": 12", "\"num_hidden_layers\": 13");
    expect_cfg_fail("layer_types", "layer count mismatch");
    mutate_t("\"num_key_value_heads\": 2", "\"num_key_value_heads\": 3");
    expect_cfg_fail("not divisible", "head divisibility");
    mutate_t("\"head_dim\": 16", "\"head_dim\": 15");
    expect_cfg_fail("must be even", "odd head dim");
    mutate_t("\"num_kv_shared_layers\": 4", "\"num_kv_shared_layers\": 12");
    expect_cfg_fail("num_kv_shared_layers", "all layers shared");
    mutate_t("\"sliding_window\": 8", "\"sliding_window\": 0");
    expect_cfg_fail("sliding_window", "zero window");
    mutate_t("\"hidden_size\": 64", "\"hidden_size\": 6.5");
    expect_cfg_fail("hidden_size", "fractional dimension");
    mutate_t("\"rms_norm_eps\": 1e-06", "\"rms_norm_eps\": 0");
    expect_cfg_fail("rms_norm_eps", "zero eps");
    lt_write_text(P("cfg/config.json"), "{ not json");
    expect_cfg_fail("not a valid JSON", "malformed config");
    lt_write_text(P("cfg/config.json"), "{\"model_type\":\"gemma4\"}");
    expect_cfg_fail("text_config", "no text_config");
}

static void link_fixture_weights(const char *dir)
{
    char real[PATH_MAX_], src[PATH_MAX_ + 128], dst[PATH_MAX_ + 128];
    CHECK(realpath(FIX, real) != NULL);
    snprintf(src, sizeof src, "%s/model.safetensors", real);
    snprintf(dst, sizeof dst, "%s/model.safetensors", dir);
    CHECK(symlink(src, dst) == 0);
}

static void test_binding_rejections(void)
{
    mkdir(P("nw"), 0755);
    char *txt = read_all(FIX "/config.json", NULL);
    lt_write_text(P("nw/config.json"), txt);
    free(txt);
    expect_open_fail(P("nw"), NULL, "neither model.safetensors.index.json nor model.safetensors", "no weights");

    /* a config that disagrees with the weights: the tensor and its shape are named */
    mutate_t("\"hidden_size\": 64", "\"hidden_size\": 32");
    mkdir(P("shape"), 0755);
    char *cfg = read_all(P("cfg/config.json"), NULL);
    lt_write_text(P("shape/config.json"), cfg);
    free(cfg);
    link_fixture_weights(P("shape"));
    expect_open_fail(P("shape"), NULL, "embed_tokens.weight has shape", "config/weights shape mismatch");

    /* one more layer than the checkpoint has */
    mutate_t("\"num_hidden_layers\": 12", "\"num_hidden_layers\": 13");
    {
        char *c = read_all(P("cfg/config.json"), NULL);
        const char *anchor = "\"full_attention\"\n        ],";
        char *tc = strstr(c, "\"text_config\"");
        char *at = tc ? strstr(tc, anchor) : NULL;
        CHECK(at != NULL);
        if (at) {
            char *out = (char *)malloc(strlen(c) + 64);
            snprintf(out, strlen(c) + 64, "%.*s\"full_attention\",\n            \"full_attention\"\n        ],%s", (int)(at - c), c, at + strlen(anchor));
            mkdir(P("nolayer"), 0755);
            lt_write_text(P("nolayer/config.json"), out);
            link_fixture_weights(P("nolayer"));
            free(out);
            expect_open_fail(P("nolayer"), NULL, "tensor", "missing layer tensor");
        }
        free(c);
    }

    /* the bf16 file passed as q8 weights; the q8 file used as a bf16 checkpoint */
    LmOpts o = { FIX "/model.safetensors", 0, 0, 0 };
    expect_open_fail(FIX, &o, "not an lm-q8 group-32 file", "bf16 file as q8");
    {
        char err[512] = "";
        mkdir(P("q8dir"), 0755);
        char *c = read_all(FIX "/config.json", NULL);
        lt_write_text(P("q8dir/config.json"), c);
        free(c);
        size_t n;
        char *q = read_all(Q8, &n);
        lt_write_file(P("q8dir/model.safetensors"), q, n);
        free(q);
        LmModel *m = lm_open(P("q8dir"), NULL, err, sizeof err);
        CHECK_MSG(m == NULL && strstr(err, "q8 file"), "q8 as bf16: \"%s\"", err);
        if (m) lm_close(m);
    }
    LmOpts bad = { NULL, -1, 0, 0 };
    expect_open_fail(FIX, &bad, "ctx", "negative ctx");
    LmOpts miss = { "/nonexistent/model.q8.safetensors", 0, 0, 0 };
    expect_open_fail(FIX, &miss, "cannot open", "missing weights file");
}

/* ------------------------------------------------------------------ the real E4B shape */

static void test_e4b_memory(void)
{
    /* the real E4B text_config numbers (docs/PLAN.md), written to a temp dir */
    char layer_types[2048] = "";
    for (int i = 0; i < 42; i++)
        snprintf(layer_types + strlen(layer_types), sizeof layer_types - strlen(layer_types), "%s\"%s\"", i ? "," : "",
                 i % 6 == 5 ? "full_attention" : "sliding_attention");
    char cfg[8192];
    snprintf(cfg, sizeof cfg,
        "{\"model_type\":\"gemma4\",\"tie_word_embeddings\":true,\"text_config\":{\"model_type\":\"gemma4_text\","
        "\"hidden_size\":2560,\"intermediate_size\":10240,\"num_hidden_layers\":42,\"vocab_size\":262144,"
        "\"vocab_size_per_layer_input\":262144,\"rms_norm_eps\":1e-06,\"num_attention_heads\":8,\"num_key_value_heads\":2,"
        "\"head_dim\":256,\"global_head_dim\":512,\"hidden_activation\":\"gelu_pytorch_tanh\",\"sliding_window\":512,"
        "\"num_kv_shared_layers\":18,\"hidden_size_per_layer_input\":256,\"final_logit_softcapping\":30.0,"
        "\"tie_word_embeddings\":true,\"enable_moe_block\":false,\"attention_k_eq_v\":false,\"use_double_wide_mlp\":false,"
        "\"attention_bias\":false,\"use_bidirectional_attention\":null,\"num_global_key_value_heads\":null,"
        "\"layer_types\":[%s],\"rope_parameters\":{\"sliding_attention\":{\"rope_type\":\"default\",\"rope_theta\":10000.0},"
        "\"full_attention\":{\"rope_type\":\"proportional\",\"partial_rotary_factor\":0.25,\"rope_theta\":1000000.0}}}}",
        layer_types);
    mkdir(P("e4b"), 0755);
    lt_write_text(P("e4b/config.json"), cfg);
    LmGemmaCfg c;
    char err[512] = "";
    CHECK_MSG(lm_gemma_config_load(P("e4b"), &c, err, sizeof err) == 0, "real E4B config: %s", err);
    CHECK(c.first_shared == 24 && c.kv_shared == 18 && c.ple_dim == 256);
    CHECK(c.producer[24] == 22 && c.producer[29] == 23 && c.producer[30] == 22 && c.producer[41] == 23 && c.producer[0] == 0);
    CHECK(c.embed_scale == 50.5f && c.ple_scale == 16.0f);

    const size_t F = sizeof(float);
    const size_t sliding_kv = 20 * (size_t)512 * 2 * 256 * 2;     /* floats: 20 sliding producers, ring of 512, K and V */
    const size_t full_kv = 4 * (size_t)4096 * 2 * 512 * 2;        /* 4 full producers, ctx 4096 */
    const size_t LP = 42 * 256;
    const size_t scratch = 4 * 2560 + 2 * 10240                   /* h n mix mlp_out, gate+up */
                         + 8 * 512 + 2 * (2 * 512) + 8 * 512       /* q k v attn at the largest head size */
                         + 8 * 4096                                /* scores */
                         + 2 * LP + LP / 32 + 1 + 256 + 2560       /* per-layer inputs, scales, ple_g, ple_d */
                         + (size_t)LM_PREFILL_CHUNK * (4 * 2560 + 2 * 10240 + 2 * (8 * 512) + 2 * (2 * 512) + 2 * LP + 256);  /* prefill chunk buffers */
    const size_t want = (sliding_kv + full_kv + scratch) * F + LP * 2;
    const size_t got = lm_gemma_state_bytes(&c, 4096);
    CHECK_MSG(got == want, "state bytes %zu, formula %zu", got, want);
    printf("  real E4B at ctx 4096: KV caches %.1f MiB (sliding %.1f + full %.1f), scratch %.2f MiB, state total %.1f MiB\n",
           (double)(sliding_kv + full_kv) * F / 1048576.0, (double)sliding_kv * F / 1048576.0, (double)full_kv * F / 1048576.0,
           ((double)scratch * F + LP * 2) / 1048576.0, (double)got / 1048576.0);
    /* the ring makes the sliding caches independent of ctx */
    CHECK(lm_gemma_state_bytes(&c, 8192) - lm_gemma_state_bytes(&c, 4096) == (full_kv + 8 * (size_t)4096) * F);
    lm_gemma_config_free(&c);
}

int main(void)
{
    char probe[256];
    snprintf(probe, sizeof probe, "%s/config.json", FIX);
    if (access(probe, R_OK) != 0) {
        fprintf(stderr, "  FAIL fixture %s not found (run from the repository root)\n", FIX);
        return 1;
    }
    lt_mkdtemp(g_dir);
    test_family(NULL, "golden_bf16.json", "bf16");
    test_family(Q8, "golden_q8.json", "q8");
    test_reuse_layers_never_bind(NULL, "model.safetensors", "bf16");
    test_reuse_layers_never_bind(Q8, "model.q8.safetensors", "q8");
    test_config_rejections();
    test_binding_rejections();
    test_e4b_memory();
    lt_rmtree(g_dir);
    return lt_finish("test_gemma");
}
