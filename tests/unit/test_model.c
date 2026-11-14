/* test_model.c - lm_open / lm_feed / lm_reset behaviour on the tiny fixture checkpoint
 * (tests/fixtures/qwen35-tiny, run from the repository root; absence is a FAILURE), for
 * both the bf16 shards and the q8 file, plus config and binding rejections on mutated
 * copies of config.json. Numerical agreement with the oracle is the job of `make gate`
 * and test_ops_fixtures; this file checks the engine's own invariants, all BITWISE. */
#include "lm_test.h"

#include "lm/lm.h"

#define FIX "tests/fixtures/qwen35-tiny"
#include <limits.h>
#define PATH_MAX_ 4096
#define Q8 FIX "/model.q8.safetensors"

static char g_dir[512];

static char *P(const char *leaf)
{
    static char b[4][1024];
    static int i;
    i = (i + 1) & 3;
    lt_path(b[i], g_dir, leaf);
    return b[i];
}

static const int TOK[12] = { 5, 17, 383, 0, 42, 99, 7, 250, 31, 2, 311, 64 };

static LmModel *open_model(const char *weights, int ctx)
{
    char err[512];
    LmOpts o = { weights, ctx, 0 };
    LmModel *m = lm_open(FIX, &o, err, sizeof err);
    CHECK_MSG(m != NULL, "lm_open: %s", err);
    return m;
}

static int same(const float *a, const float *b, int n) { return !memcmp(a, b, (size_t)n * sizeof(float)); }

static void test_invariants(const char *weights, const char *label)
{
    const int n = 12;
    LmModel *m = open_model(weights, 0);
    if (!m) return;
    const int V = lm_n_vocab(m);
    CHECK(V == 384 && lm_n_ctx(m) == 4096 && lm_pos(m) == 0);
    float *all = (float *)malloc((size_t)n * V * 4), *row = (float *)malloc((size_t)V * 4);

    /* the incremental run: logits after each token */
    for (int t = 0; t < n; t++) {
        CHECK(lm_feed(m, TOK[t], all + (size_t)t * V) == LM_OK);
        CHECK(lm_pos(m) == t + 1);
    }
    for (int i = 0; i < n * V; i++) CHECK(isfinite(all[i]));

    /* position-by-position feed equals a fresh model fed the prefix */
    for (int p = 0; p < n; p += 3) {
        LmModel *f = open_model(weights, 0);
        for (int t = 0; t <= p; t++) CHECK(lm_feed(f, TOK[t], row) == LM_OK);
        CHECK_MSG(same(row, all + (size_t)p * V, V), "%s: fresh run to position %d differs", label, p);
        lm_close(f);
    }

    /* reset, then re-feed: identical logits at every position, and state really was cleared */
    lm_reset(m);
    CHECK(lm_pos(m) == 0);
    for (int t = 0; t < n; t++) {
        CHECK(lm_feed(m, TOK[t], row) == LM_OK);
        CHECK_MSG(same(row, all + (size_t)t * V, V), "%s: after reset position %d differs", label, t);
    }
    /* NEGATIVE: without the reset the state would differ (guards against a no-op reset check) */
    CHECK(lm_feed(m, TOK[0], row) == LM_OK);
    lm_reset(m);
    CHECK(lm_feed(m, TOK[0], row) == LM_OK && same(row, all, V));
    lm_reset(m);

    /* logits == NULL prefill then one logits call equals feeding everything with logits */
    for (int t = 0; t < n - 1; t++) CHECK(lm_feed(m, TOK[t], NULL) == LM_OK);
    CHECK(lm_feed(m, TOK[n - 1], row) == LM_OK);
    CHECK_MSG(same(row, all + (size_t)(n - 1) * V, V), "%s: NULL-logits prefill changes the result", label);

    /* an invalid token is refused with nothing fed and the state untouched */
    lm_reset(m);
    CHECK(lm_feed(m, TOK[0], row) == LM_OK);
    CHECK(lm_feed(m, -1, row) == LM_ERR_TOKEN && lm_pos(m) == 1);
    CHECK(lm_feed(m, V, row) == LM_ERR_TOKEN && lm_pos(m) == 1);
    CHECK(lm_feed(m, 2147483647, row) == LM_ERR_TOKEN && lm_pos(m) == 1);
    CHECK(lm_feed(m, TOK[1], row) == LM_OK && same(row, all + V, V));
    lm_close(m);

    /* a full context window: LM_ERR_CTX, nothing fed, the cached state is not disturbed */
    m = open_model(weights, 5);
    CHECK(m && lm_n_ctx(m) == 5);
    if (m) {
        for (int t = 0; t < 5; t++) CHECK(lm_feed(m, TOK[t], row) == LM_OK && same(row, all + (size_t)t * V, V));
        CHECK(lm_feed(m, TOK[5], row) == LM_ERR_CTX && lm_pos(m) == 5);
        CHECK(lm_feed(m, TOK[5], NULL) == LM_ERR_CTX && lm_pos(m) == 5);
        lm_reset(m);
        CHECK(lm_pos(m) == 0);
        for (int t = 0; t < 5; t++) CHECK(lm_feed(m, TOK[t], row) == LM_OK);
        CHECK(same(row, all + 4 * (size_t)V, V));
        lm_close(m);
    }
    free(all); free(row);
}

/* ------------------------------------------------------------------ rejections */

static char *read_all(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    char *b = (char *)malloc(1 << 16);
    const size_t n = fread(b, 1, (1 << 16) - 1, f);
    fclose(f);
    b[n] = '\0';
    return b;
}

/* replace `from` by `to` in the fixture's config.json (from must occur), write to g_dir/cfg/ */
static void mutate(const char *from, const char *to)
{
    char *txt = read_all(FIX "/config.json");
    CHECK(txt != NULL);
    if (!txt) return;
    char *at = strstr(txt, from);
    CHECK_MSG(at != NULL, "mutation anchor \"%s\" not in config.json", from);
    char out[1 << 16];
    if (at) snprintf(out, sizeof out, "%.*s%s%s", (int)(at - txt), txt, to, at + strlen(from));
    else snprintf(out, sizeof out, "%s", txt);
    mkdir(P("cfg"), 0755);
    lt_write_text(P("cfg/config.json"), out);
    free(txt);
}

static void expect_open_fail(const char *dir, const LmOpts *o, const char *needle, const char *label)
{
    char err[512] = "";
    LmModel *m = lm_open(dir, o, err, sizeof err);
    CHECK_MSG(m == NULL && strstr(err, needle), "%s: m=%p err=\"%s\" (wanted \"%s\")", label, (void *)m, err, needle);
    if (m) lm_close(m);
}

static void test_config_rejections(void)
{
    /* a config-only directory: every failure below must come from the config, before any weights */
    mutate("\"model_type\": \"qwen3_5\",\n    \"text_config\"", "\"model_type\": \"llama\",\n    \"text_config\"");
    expect_open_fail(P("cfg"), NULL, "model_type \"llama\" is not supported (supported: qwen3_5, gemma4)", "wrong model_type");
    mutate("\"model_type\": \"qwen3_5\",\n    \"text_config\"", "\"text_config\"");
    expect_open_fail(P("cfg"), NULL, "\"model_type\"", "missing model_type");
    static const char *const fields[] = { "hidden_size", "intermediate_size", "num_hidden_layers", "vocab_size",
        "rms_norm_eps", "num_attention_heads", "num_key_value_heads", "head_dim", "linear_num_key_heads",
        "linear_num_value_heads", "linear_key_head_dim", "linear_value_head_dim", "linear_conv_kernel_dim",
        "hidden_act", "attn_output_gate", "layer_types", "rope_parameters", "tie_word_embeddings" };
    for (size_t i = 0; i < sizeof fields / sizeof *fields; i++) {
        if (!strcmp(fields[i], "tie_word_embeddings")) continue;   /* present twice: handled below */
        char from[64], to[64];
        snprintf(from, sizeof from, "\"%s\"", fields[i]);
        snprintf(to, sizeof to, "\"x_%s\"", fields[i]);
        mutate(from, to);
        expect_open_fail(P("cfg"), NULL, fields[i], fields[i]);
    }
    /* tie_word_embeddings appears at top level and in text_config; drop both */
    {
        char *t = read_all(FIX "/config.json");
        char *a;
        while ((a = strstr(t, "\"tie_word_embeddings\"")) != NULL) a[1] = 'x';
        mkdir(P("cfg"), 0755);
        lt_write_text(P("cfg/config.json"), t);
        free(t);
        expect_open_fail(P("cfg"), NULL, "tie_word_embeddings", "missing tie_word_embeddings");
    }
    mutate("\"rope_type\": \"default\"", "\"rope_type\": \"yarn\"");
    expect_open_fail(P("cfg"), NULL, "rope_type \"yarn\"", "rope_type");
    mutate("\"hidden_act\": \"silu\"", "\"hidden_act\": \"gelu\"");
    expect_open_fail(P("cfg"), NULL, "hidden_act \"gelu\"", "hidden_act");
    mutate("\"attn_output_gate\": true", "\"attn_output_gate\": false");
    expect_open_fail(P("cfg"), NULL, "attn_output_gate", "attn_output_gate false");
    mutate("\"full_attention\"\n        ],", "\"sliding_attention\"\n        ],");
    expect_open_fail(P("cfg"), NULL, "unsupported layer type", "layer type");
    mutate("\"num_hidden_layers\": 8", "\"num_hidden_layers\": 9");
    expect_open_fail(P("cfg"), NULL, "layer_types", "layer count mismatch");
    mutate("\"num_key_value_heads\": 2", "\"num_key_value_heads\": 3");
    expect_open_fail(P("cfg"), NULL, "not divisible", "head divisibility");
    mutate("\"partial_rotary_factor\": 0.25", "\"partial_rotary_factor\": 0.1");
    expect_open_fail(P("cfg"), NULL, "rotary dimension", "odd rotary dim");
    mutate("\"hidden_size\": 64", "\"hidden_size\": 0");
    expect_open_fail(P("cfg"), NULL, "hidden_size", "zero dimension");
    mutate("\"hidden_size\": 64", "\"hidden_size\": 6.5");
    expect_open_fail(P("cfg"), NULL, "hidden_size", "fractional dimension");
    lt_write_text(P("cfg/config.json"), "{ not json");
    expect_open_fail(P("cfg"), NULL, "not a valid JSON", "malformed config");
    lt_write_text(P("cfg/config.json"), "{\"model_type\":\"qwen3_5\"}");
    expect_open_fail(P("cfg"), NULL, "text_config", "no text_config");
    mkdir(P("empty"), 0755);
    expect_open_fail(P("empty"), NULL, "cannot open", "no config.json");
    expect_open_fail(P("nonexistent"), NULL, "cannot open", "no directory");
}

static void link_fixture_weights(const char *dir)
{
    char real[PATH_MAX_], src[2048], dst[2048];
    static const char *const files[] = { "model.safetensors.index.json",
        "model.safetensors-00001-of-00002.safetensors", "model.safetensors-00002-of-00002.safetensors" };
    CHECK(realpath(FIX, real) != NULL);
    for (int i = 0; i < 3; i++) {
        snprintf(src, sizeof src, "%s/%s", real, files[i]);
        snprintf(dst, sizeof dst, "%s/%s", dir, files[i]);
        CHECK(symlink(src, dst) == 0);
    }
}

static void test_binding_rejections(void)
{
    /* valid config, no weights next to it */
    mkdir(P("nw"), 0755);
    char *txt = read_all(FIX "/config.json");
    lt_write_text(P("nw/config.json"), txt);
    free(txt);
    expect_open_fail(P("nw"), NULL, "neither model.safetensors.index.json nor model.safetensors", "no weights");

    /* a config that disagrees with the weights: the tensor and its shape are named */
    mutate("\"hidden_size\": 64", "\"hidden_size\": 32");
    mkdir(P("shape"), 0755);
    char *cfg = read_all(P("cfg/config.json"));
    lt_write_text(P("shape/config.json"), cfg);
    free(cfg);
    link_fixture_weights(P("shape"));
    expect_open_fail(P("shape"), NULL, "embed_tokens.weight has shape", "config/weights shape mismatch");

    /* a tensor the config needs but the weights lack: one more layer than the checkpoint has */
    mutate("\"num_hidden_layers\": 8", "\"num_hidden_layers\": 9");
    {
        char *c = read_all(P("cfg/config.json"));
        char *at = strstr(c, "\"full_attention\"\n        ],");
        CHECK(at != NULL);
        if (at) {
            char out[1 << 16];
            snprintf(out, sizeof out, "%.*s\"full_attention\",\n            \"linear_attention\"\n        ],", (int)(at - c), c);
            snprintf(out + strlen(out), sizeof out - strlen(out), "%s", at + strlen("\"full_attention\"\n        ],"));
            mkdir(P("nolayer"), 0755);
            lt_write_text(P("nolayer/config.json"), out);
            link_fixture_weights(P("nolayer"));
            expect_open_fail(P("nolayer"), NULL, "layers.8.input_layernorm.weight is missing", "missing layer tensor");
        }
        free(c);
    }

    /* the bf16 shard passed as the q8 weights file */
    LmOpts o = { FIX "/model.safetensors-00001-of-00002.safetensors", 0, 0 };
    expect_open_fail(FIX, &o, "not an lm-q8 group-32 file", "bf16 file as q8");
    /* the q8 file is not a bf16 checkpoint */
    {
        char err[512] = "";
        mkdir(P("q8dir"), 0755);
        char *c = read_all(FIX "/config.json");
        lt_write_text(P("q8dir/config.json"), c);
        free(c);
        /* a directory holding only the q8 file as model.safetensors: binding must name the I8 tensor */
        char *q = (char *)malloc(1 << 22);
        FILE *f = fopen(Q8, "rb");
        const size_t n = f ? fread(q, 1, 1 << 22, f) : 0;
        if (f) fclose(f);
        CHECK(n > 0 && n < (1 << 22));
        lt_write_file(P("q8dir/model.safetensors"), q, n);
        free(q);
        LmModel *m = lm_open(P("q8dir"), NULL, err, sizeof err);
        CHECK_MSG(m == NULL && strstr(err, "q8 file"), "q8 as bf16: \"%s\"", err);
        if (m) lm_close(m);
    }
    /* bad ctx */
    LmOpts bad = { NULL, -1, 0 };
    expect_open_fail(FIX, &bad, "ctx", "negative ctx");
    /* a missing weights file */
    LmOpts miss = { "/nonexistent/model.q8.safetensors", 0, 0 };
    expect_open_fail(FIX, &miss, "cannot open", "missing weights file");
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
    test_invariants(NULL, "bf16");
    test_invariants(Q8, "q8");
    test_config_rejections();
    test_binding_rejections();
    lt_rmtree(g_dir);
    return lt_finish("test_model");
}
