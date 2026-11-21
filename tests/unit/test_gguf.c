/* test_gguf.c - the GGUF reader on the tiny Qwen2 fixture (tests/fixtures/qwen2-tiny, run from
 * the repository root; absence is a FAILURE) and on corrupted copies of it: every refusal
 * names its cause. */
#include "lm_test.h"

#include "io/lm_gguf.h"

#define FIX "tests/fixtures/qwen2-tiny/model.gguf"

static unsigned char *slurp(const char *path, size_t *n)
{
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    fseek(f, 0, SEEK_END);
    *n = (size_t)ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char *b = (unsigned char *)malloc(*n);
    if (fread(b, 1, *n, f) != *n) exit(1);
    fclose(f);
    return b;
}

static int try_open(const char *path, char *err, size_t n)
{
    LmGguf g;
    err[0] = 0;
    const int rc = lm_gguf_open(&g, path, err, n);
    if (rc == 0) lm_gguf_close(&g);
    return rc;
}

static void test_fixture(void)
{
    char err[512] = "";
    LmGguf g;
    CHECK_MSG(lm_gguf_open(&g, FIX, err, sizeof err) == 0, "open: %s", err);
    CHECK(g.version == 3 && g.alignment == 32 && g.nt == 1 + 2 * 12 + 2);
    LmGgufStr s;
    CHECK(lm_gguf_get_str(&g, "general.architecture", &s) == 0 && s.n == 5 && !memcmp(s.s, "qwen2", 5));
    uint64_t u = 0;
    double f = 0;
    CHECK(lm_gguf_get_u64(&g, "qwen2.block_count", &u) == 0 && u == 2);
    CHECK(lm_gguf_get_f64(&g, "qwen2.rope.freq_base", &f) == 0 && f == 1000000.0);
    CHECK(lm_gguf_get_u64(&g, "tokenizer.ggml.add_bos_token", &u) == 0 && u == 0);   /* bool */
    CHECK(lm_gguf_get_u64(&g, "general.architecture", &u) != 0);                      /* a string is no integer */
    CHECK(lm_gguf_get_u64(&g, "no.such.key", &u) != 0);
    int n = 0;
    LmGgufStr *toks = lm_gguf_str_array(&g, "tokenizer.ggml.tokens", &n, err, sizeof err);
    CHECK(toks && n == 320 && toks[0].n >= 1);
    free(toks);
    int64_t *types = lm_gguf_int_array(&g, "tokenizer.ggml.token_type", &n, err, sizeof err);
    CHECK(types && n == 320 && types[0] == 1 && types[298] == 3 && types[319] == 5);
    free(types);
    CHECK(lm_gguf_str_array(&g, "qwen2.block_count", &n, err, sizeof err) == NULL);
    CHECK(strstr(err, "not a string array"));
    const LmGgufTensor *t = lm_gguf_find(&g, "blk.0.attn_q.weight");
    CHECK(t && t->type == LM_GG_Q4_K && t->ndim == 2 && t->ne[0] == 256 && t->ne[1] == 256 && t->nbytes == 256 * 256 / 256 * 144);
    CHECK(t && t->off % 32 == 0 && lm_gguf_data(&g, t) == g.map + t->off);
    t = lm_gguf_find(&g, "blk.0.attn_q.bias");
    CHECK(t && t->type == LM_GG_F32 && t->ndim == 1 && t->nbytes == 256 * 4);
    CHECK(lm_gguf_find(&g, "nope") == NULL);
    lm_gguf_close(&g);
    lm_gguf_close(&g);   /* closing twice is safe */
}

static void test_corrupt(void)
{
    size_t n;
    unsigned char *orig = slurp(FIX, &n);
    char dir[512], p[1024], err[512];
    lt_mkdtemp(dir);
    lt_path(p, dir, "bad.gguf");

    unsigned char *b = (unsigned char *)malloc(n);
    int rc;

    memcpy(b, orig, n); b[0] = 'X';
    lt_write_file(p, b, n); rc = try_open(p, err, sizeof err); CHECK_ERR(rc, err, "bad magic");

    memcpy(b, orig, n); b[4] = 9;
    lt_write_file(p, b, n); rc = try_open(p, err, sizeof err); CHECK_ERR(rc, err, "version 9");

    lt_write_file(p, orig, 10); rc = try_open(p, err, sizeof err); CHECK_ERR(rc, err, "too short");

    lt_write_file(p, orig, 200); rc = try_open(p, err, sizeof err); CHECK_ERR(rc, err, "");   /* truncated in the metadata */
    CHECK(strstr(err, "past the end") || strstr(err, "truncated"));

    lt_write_file(p, orig, n - 4096); rc = try_open(p, err, sizeof err); CHECK_ERR(rc, err, "truncated");   /* tensor data cut */

    memcpy(b, orig, n);
    uint64_t huge = (uint64_t)1 << 40;
    memcpy(b + 8, &huge, 8);   /* tensor count */
    lt_write_file(p, b, n); rc = try_open(p, err, sizeof err); CHECK_ERR(rc, err, "exceed the limits");

    memcpy(b, orig, n);
    memcpy(b + 16, &huge, 8);   /* metadata count */
    lt_write_file(p, b, n); rc = try_open(p, err, sizeof err); CHECK_ERR(rc, err, "exceed the limits");

    rc = try_open("/no/such/file.gguf", err, sizeof err); CHECK_ERR(rc, err, "cannot open");

    free(b);
    free(orig);
    lt_rmtree(dir);
}

static void test_locate(void)
{
    char dir[512], p[1024], out[4096];
    lt_mkdtemp(dir);
    CHECK(!lm_gguf_locate(dir, out, sizeof out));            /* empty directory */
    lt_path(p, dir, "a.gguf"); lt_write_text(p, "x");
    CHECK(lm_gguf_locate(dir, out, sizeof out) && strstr(out, "a.gguf"));
    CHECK(lm_gguf_locate(p, out, sizeof out));
    lt_path(p, dir, "b.gguf"); lt_write_text(p, "x");
    CHECK(!lm_gguf_locate(dir, out, sizeof out));            /* two files: ambiguous */
    unlink(p);
    lt_path(p, dir, "config.json"); lt_write_text(p, "{}");
    CHECK(!lm_gguf_locate(dir, out, sizeof out));            /* a config.json means a safetensors model */
    CHECK(!lm_gguf_locate("tests/fixtures/qwen35-tiny", out, sizeof out));
    CHECK(lm_gguf_locate("tests/fixtures/qwen2-tiny", out, sizeof out));
    lt_rmtree(dir);
}

int main(void)
{
    test_fixture();
    test_corrupt();
    test_locate();
    return lt_finish("test_gguf");
}
