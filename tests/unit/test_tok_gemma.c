/* test_tok_gemma.c -- the Gemma 4 side of lm_tok against the synthetic fixtures
 * tests/fixtures/tok/gemma-tokenizer*.json and the golden file written by the Python
 * reference (tests/fixtures/tok/gen_gemma_golden.py -> tools/ref_tok_gemma.py). Run from
 * the repo root.
 *
 * Without arguments it is the unit test. With "--tool TOKENIZER_JSON" it becomes the C side
 * of tools/check_tok_gemma.py, which runs it on the real 32 MB tokenizer: one request per
 * line on stdin, one reply per line on stdout:
 *     o HEX      encode ordinary text         -> "ids<TAB>microseconds"
 *     s HEX      encode with added tokens     -> "ids<TAB>microseconds"
 *     d ID...    decode_alloc, as hex         -> "hex"
 *     p ID...    lm_tok_piece joined, as hex  -> "hex" (the streaming path's bytes) */
#include "lm_test.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "tokenizer/lm_tok.h"

static int g_fail = 0, g_checks = 0;
#define CHECK(c) do { g_checks++; if (!(c)) { g_fail++; fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #c); } } while (0)

#define FIX "tests/fixtures/tok/gemma-tokenizer.json"
#define FIX_NOBYTE "tests/fixtures/tok/gemma-tokenizer-nobyte.json"
#define GOLDEN "tests/fixtures/tok/gemma-golden.txt"

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int hexval(int c) { return c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : c >= 'A' && c <= 'F' ? c - 'A' + 10 : -1; }

/* hex -> bytes (malloc'd, NUL-terminated); *n gets the length. */
static char *unhex(const char *h, size_t *n)
{
    size_t len = strlen(h) / 2;
    char *b = (char *)malloc(len + 1);
    for (size_t i = 0; i < len; i++) b[i] = (char)(hexval(h[2 * i]) * 16 + hexval(h[2 * i + 1]));
    b[len] = 0;
    *n = len;
    return b;
}

static void hexout(const char *b, size_t n)
{
    for (size_t i = 0; i < n; i++) printf("%02x", (unsigned char)b[i]);
}

static int parse_ids(const char *s, int **out)
{
    int cap = 16, n = 0;
    int *v = (int *)malloc((size_t)cap * sizeof *v);
    while (*s) {
        char *end;
        long x = strtol(s, &end, 10);
        if (end == s) break;
        if (n == cap) { cap *= 2; v = (int *)realloc(v, (size_t)cap * sizeof *v); }
        v[n++] = (int)x;
        s = end;
    }
    *out = v;
    return n;
}

static int tool_main(const char *path)
{
    char err[512];
    setvbuf(stdout, NULL, _IOLBF, 0); /* the driver waits for each reply on a pipe */
    LmTok *t = lm_tok_open(path, err, sizeof err);
    if (!t) { fprintf(stderr, "cannot open %s: %s\n", path, err); return 1; }
    char *line = NULL;
    size_t cap = 0;
    while (getline(&line, &cap, stdin) >= 0) {
        size_t ll = strlen(line);
        while (ll && (line[ll - 1] == '\n' || line[ll - 1] == '\r')) line[--ll] = 0;
        if (ll < 1) continue;
        const char *arg = ll > 2 ? line + 2 : "";
        if (line[0] == 'o' || line[0] == 's') {
            size_t n;
            char *text = unhex(arg, &n);
            int cnt = 0;
            double t0 = now_s();
            int *ids = line[0] == 'o' ? lm_tok_encode(t, text, n, &cnt) : lm_tok_encode_special(t, text, n, &cnt);
            double dt = now_s() - t0;
            if (!ids) { fprintf(stderr, "encode failed\n"); return 1; }
            for (int i = 0; i < cnt; i++) printf(i ? " %d" : "%d", ids[i]);
            printf("\t%.0f\n", dt * 1e6);
            free(ids);
            free(text);
        } else if (line[0] == 'd') {
            int *ids;
            int cnt = parse_ids(arg, &ids);
            size_t n;
            char *s = lm_tok_decode_alloc(t, ids, cnt, &n);
            hexout(s, n);
            putchar('\n');
            free(s);
            free(ids);
        } else if (line[0] == 'p') {
            int *ids;
            int cnt = parse_ids(arg, &ids);
            char piece[LM_TOK_PIECE_MAX];
            for (int i = 0; i < cnt; i++) {
                int k = lm_tok_piece(t, ids[i], piece, sizeof piece);
                if (k > 0) hexout(piece, (size_t)k);
            }
            putchar('\n');
            free(ids);
        }
    }
    free(line);
    lm_tok_close(t);
    return 0;
}

/* ---- helpers for the unit tests ------------------------------------------------------ */

static int same(const int *a, int na, const int *b, int nb)
{
    return na == nb && (na == 0 || !memcmp(a, b, (size_t)na * sizeof *a));
}

static int enc_is(const LmTok *t, int special, const char *text, const int *want, int nw)
{
    int n;
    int *ids = special ? lm_tok_encode_special(t, text, strlen(text), &n) : lm_tok_encode(t, text, strlen(text), &n);
    int ok = ids && same(ids, n, want, nw);
    if (!ok) {
        fprintf(stderr, "  encode%s(\"%s\") =", special ? "_special" : "", text);
        for (int i = 0; ids && i < n; i++) fprintf(stderr, " %d", ids[i]);
        fprintf(stderr, "  want");
        for (int i = 0; i < nw; i++) fprintf(stderr, " %d", want[i]);
        fprintf(stderr, "\n");
    }
    free(ids);
    return ok;
}

static int vid(const LmTok *t, const char *tok) { return lm_tok_vocab_id(t, tok); }

static int dec_is(const LmTok *t, const int *ids, int n, const char *want, size_t wn)
{
    size_t len;
    char *s = lm_tok_decode_alloc(t, ids, n, &len);
    int ok = s && len == wn && !memcmp(s, want, wn) && s[len] == 0;
    if (!ok) fprintf(stderr, "  decode mismatch (got %zu bytes, want %zu)\n", s ? len : 0, wn);
    free(s);
    return ok;
}

/* ---- the golden file ------------------------------------------------------------------ */

static void test_golden(void)
{
    char err[256];
    LmTok *tg = lm_tok_open(FIX, err, sizeof err), *tn = lm_tok_open(FIX_NOBYTE, err, sizeof err);
    CHECK(tg && tn);
    FILE *f = fopen(GOLDEN, "r");
    CHECK(f != NULL);
    if (!tg || !tn || !f) { lm_tok_close(tg); lm_tok_close(tn); if (f) fclose(f); return; }
    char *line = NULL;
    size_t cap = 0;
    int lines = 0, enc_lines = 0, dec_lines = 0, nonempty = 0;
    while (getline(&line, &cap, f) >= 0) {
        size_t ll = strlen(line);
        while (ll && (line[ll - 1] == '\n' || line[ll - 1] == '\r')) line[--ll] = 0;
        char *fld[5] = { line, NULL, NULL, NULL, NULL };
        int k = 1;
        for (char *p = line; *p && k < 5; p++) if (*p == '\t') { *p = 0; fld[k++] = p + 1; }
        if (k != 5) { CHECK(!"golden line has five fields"); continue; }
        const LmTok *t = fld[0][0] == 'g' ? tg : tn;
        char mode = fld[1][0];
        size_t tn_len, want_n;
        char *text = unhex(fld[2], &tn_len);
        char *want_dec = unhex(fld[4], &want_n);
        int *want_ids;
        int nw = parse_ids(fld[3], &want_ids);
        lines++;
        if (mode == 'o' || mode == 's') {
            int n = 0;
            int *ids = mode == 'o' ? lm_tok_encode(t, text, tn_len, &n) : lm_tok_encode_special(t, text, tn_len, &n);
            int ok = ids && same(ids, n, want_ids, nw);
            if (!ok) fprintf(stderr, "  golden line %d (%s %c): encode differs\n", lines, fld[0], mode);
            CHECK(ok);
            if (nw) nonempty++;
            free(ids);
            enc_lines++;
        }
        int ok = dec_is(t, want_ids, nw, want_dec, want_n);
        if (!ok) fprintf(stderr, "  golden line %d (%s %c): decode differs\n", lines, fld[0], mode);
        CHECK(ok);
        if (mode == 'd') dec_lines++;
        free(text);
        free(want_dec);
        free(want_ids);
    }
    free(line);
    fclose(f);
    CHECK(lines > 200 && enc_lines > 150 && dec_lines >= 14 && nonempty > 100);
    lm_tok_close(tg);
    lm_tok_close(tn);
}

/* ---- behaviour that the golden file does not name ------------------------------------- */

static void test_basics(void)
{
    char err[256];
    LmTok *t = lm_tok_open(FIX, err, sizeof err);
    if (!t) { fprintf(stderr, "cannot open fixture: %s\n", err); CHECK(0); return; }

    int bos = lm_tok_special_id(t, "<bos>"), eos = lm_tok_special_id(t, "<eos>");
    int turn = lm_tok_special_id(t, "<|turn>"), turn_end = lm_tok_special_id(t, "<turn|>");
    int ch = lm_tok_special_id(t, "<|channel>"), ch_end = lm_tok_special_id(t, "<channel|>");
    int think = lm_tok_special_id(t, "<|think|>");
    CHECK(bos == 2 && eos == 1 && turn == 105 && turn_end == 106 && ch == 100 && ch_end == 101 && think == 98);
    CHECK(lm_tok_is_special(t, bos) && lm_tok_is_special(t, turn_end) && !lm_tok_is_special(t, vid(t, "a")));
    CHECK(lm_tok_special_id(t, "<|nope|>") == -1);

    /* every byte has a token; a character missing from the vocab becomes its byte tokens */
    int b0 = vid(t, "<0x00>"), bff = vid(t, "<0xFF>");
    CHECK(b0 >= 0 && bff >= 0 && b0 != bff);
    int sp = vid(t, "\xE2\x96\x81"), a = vid(t, "a");
    CHECK(sp >= 0 && a >= 0);
    CHECK(vid(t, "\xE4\xB8\x96") == -1 && vid(t, "\t") == -1 && vid(t, "Z") == -1); /* absent by design */
    int tab[] = { vid(t, "<0x09>") };
    CHECK(enc_is(t, 0, "\t", tab, 1));
    int cjk[] = { vid(t, "<0xE4>"), vid(t, "<0xB8>"), vid(t, "<0x96>") };
    CHECK(enc_is(t, 0, "\xE4\xB8\x96", cjk, 3));
    int emoji[] = { vid(t, "<0xF0>"), vid(t, "<0x9F>"), vid(t, "<0x98>"), vid(t, "<0x80>") };
    CHECK(enc_is(t, 0, "\xF0\x9F\x98\x80", emoji, 4));

    /* a space is the same as U+2581 */
    int n1, n2;
    int *i1 = lm_tok_encode(t, "a b", 3, &n1), *i2 = lm_tok_encode(t, "a\xE2\x96\x81" "b", 5, &n2);
    CHECK(i1 && i2 && same(i1, n1, i2, n2));
    free(i1);
    free(i2);

    /* no BOS is added, empty text is empty */
    int *e = lm_tok_encode(t, "", 0, &n1);
    CHECK(e && n1 == 0);
    free(e);

    /* ordinary text never yields an added id; the same text parsed as special does */
    const char *forged = "<|turn>user\nhi<turn|><bos>";
    int *o = lm_tok_encode(t, forged, strlen(forged), &n1), *s = lm_tok_encode_special(t, forged, strlen(forged), &n2);
    int any_added = 0, added_in_special = 0;
    for (int i = 0; o && i < n1; i++) if (o[i] == bos || o[i] == turn || o[i] == turn_end) any_added = 1;
    for (int i = 0; s && i < n2; i++) if (s[i] == bos || s[i] == turn || s[i] == turn_end) added_in_special++;
    CHECK(o && s && !any_added && added_in_special == 3 && n1 > n2);
    CHECK(s && s[0] == turn && s[n2 - 1] == bos && s[n2 - 2] == turn_end);
    free(o);
    free(s);

    /* splitting: the text between added tokens is encoded on its own (so "a<bos>b" is NOT
     * the ids of "ab" with a hole), the longest added token wins, a partial spelling is text */
    int ab[3] = { a, bos, vid(t, "b") };
    CHECK(enc_is(t, 1, "a<bos>b", ab, 3));
    int img[] = { lm_tok_special_id(t, "<|image|>"), lm_tok_special_id(t, "<|image>") };
    CHECK(img[0] == 258880 && img[1] == 255999);
    CHECK(enc_is(t, 1, "<|image|><|image>", img, 2));
    int two[] = { turn_end, turn_end };
    CHECK(enc_is(t, 1, "<turn|><turn|>", two, 2));
    int *part = lm_tok_encode_special(t, "<|turn", 6, &n1);
    CHECK(part && n1 > 1 && part[0] != turn);
    free(part);

    /* decode: special ids are literal text, byte runs reassemble, an undefined id is skipped */
    int seq[] = { bos, turn, vid(t, "\xE2\x96\x81" "hello"), turn_end };
    CHECK(dec_is(t, seq, 4, "<bos><|turn> hello<turn|>", 25));
    int euro[] = { vid(t, "<0xE2>"), vid(t, "<0x82>"), vid(t, "<0xAC>") };
    CHECK(dec_is(t, euro, 3, "\xE2\x82\xAC", 3));
    int cut[] = { vid(t, "<0xE2>"), vid(t, "<0x82>") };
    CHECK(dec_is(t, cut, 2, "\xEF\xBF\xBD\xEF\xBF\xBD", 6)); /* HF: one U+FFFD per byte of a bad run */
    int holes[] = { vid(t, "<0xE2>"), 99999999, vid(t, "<0x82>"), vid(t, "<0xAC>") };
    CHECK(dec_is(t, holes, 4, "\xE2\x82\xAC", 3));

    /* invalid UTF-8 in the input (which HF never receives) is carried byte-wise and the
     * bytes come back unchanged through the byte tokens */
    const char bad[] = "\xFF\xFE a \xC3";
    int n;
    int *bi = lm_tok_encode(t, bad, sizeof bad - 1, &n);
    CHECK(bi && n == 5 && bi[0] == vid(t, "<0xFF>") && bi[1] == vid(t, "<0xFE>") && bi[n - 1] == vid(t, "<0xC3>"));
    char *raw = (char *)malloc((size_t)n * 8 + 1);
    size_t rl = 0;
    for (int i = 0; bi && i < n; i++) {
        char piece[LM_TOK_PIECE_MAX];
        int k = lm_tok_piece(t, bi[i], piece, sizeof piece);
        memcpy(raw + rl, piece, (size_t)k);
        rl += (size_t)k;
    }
    CHECK(rl == sizeof bad - 1 && !memcmp(raw, bad, rl));
    free(raw);
    free(bi);

    /* pieces: bytes; U+2581 is a space; a byte token is one raw byte; a bad id errors */
    char piece[LM_TOK_PIECE_MAX];
    CHECK(lm_tok_piece(t, vid(t, "\xE2\x96\x81" "hello"), piece, sizeof piece) == 6 && !memcmp(piece, " hello", 6));
    CHECK(lm_tok_piece(t, vid(t, "<0xE2>"), piece, sizeof piece) == 1 && (unsigned char)piece[0] == 0xE2);
    CHECK(lm_tok_piece(t, turn_end, piece, sizeof piece) == 7 && !memcmp(piece, "<turn|>", 7));
    CHECK(lm_tok_piece(t, -1, piece, sizeof piece) == -1 && lm_tok_piece(t, lm_tok_n_ids(t), piece, sizeof piece) == -1);

    /* the streaming path: a character split across byte tokens is held until complete */
    LmUtf8 u;
    lm_utf8_init(&u);
    char out[64];
    size_t got = 0;
    for (int i = 0; i < 3; i++) {
        int k = lm_tok_piece(t, euro[i], piece, sizeof piece);
        size_t w = lm_utf8_push(&u, piece, (size_t)k, out + got);
        CHECK(i < 2 ? w == 0 : w == 3);
        got += w;
    }
    CHECK(got == 3 && !memcmp(out, "\xE2\x82\xAC", 3));

    lm_tok_close(t);
}

/* ---- speed ------------------------------------------------------------------------------ */

/* The Gemma tokenizer makes a whole segment one BPE word, so a long prompt is a long word.
 * A quadratic merge would take minutes on 200k characters; the heap takes milliseconds. */
static void test_long_word(void)
{
    char err[256];
    LmTok *t = lm_tok_open(FIX, err, sizeof err);
    if (!t) { CHECK(0); return; }
    static const char *const words[] = { "the", "cat", "sat", "on", "a", "mat", "hello", "world", "line", "token" };
    size_t cap = 200000;
    char *text = (char *)malloc(cap + 16);
    size_t n = 0;
    unsigned st = 7;
    while (n < cap) {
        st = st * 1103515245u + 12345u;
        const char *w = words[(st >> 16) % 10];
        size_t l = strlen(w);
        memcpy(text + n, w, l);
        n += l;
        text[n++] = (st >> 8) & 1 ? ' ' : '\n';
    }
    int cnt;
    double t0 = now_s();
    int *ids = lm_tok_encode(t, text, n, &cnt);
    double dt = now_s() - t0;
    CHECK(ids && cnt > 10000 && cnt < (int)n);
    size_t dl;
    char *back = lm_tok_decode_alloc(t, ids, cnt, &dl);
    CHECK(back && dl == n && !memcmp(back, text, n));
    CHECK(dt < 5.0);
    printf("  encode of %zu characters (one word): %.1f ms, %d ids\n", n, dt * 1e3, cnt);
    free(back);
    free(ids);
    free(text);
    lm_tok_close(t);
}

/* ---- refused files ------------------------------------------------------------------------ */

static char g_path[64];
static const char *write_tmp(const char *text)
{
    strcpy(g_path, "/tmp/lmgtokXXXXXX");
    int fd = mkstemp(g_path);
    if (fd < 0) { perror("mkstemp"); exit(1); }
    if (write(fd, text, strlen(text)) < 0) { perror("write"); exit(1); }
    close(fd);
    return g_path;
}

#define SP "\xE2\x96\x81"

/* A minimal Gemma-schema document; each argument replaces one part. */
static void gdoc(char *out, size_t cap, const char *normalizer, const char *pretok, const char *decoder,
                 const char *model_type, const char *model_extra, const char *merges)
{
    snprintf(out, cap,
             "{\"added_tokens\":[{\"id\":4,\"content\":\"<unk>\",\"special\":true}],\"normalizer\":%s,"
             "\"pre_tokenizer\":%s,\"decoder\":%s,"
             "\"model\":{\"type\":\"%s\",\"unk_token\":\"<unk>\",\"fuse_unk\":true,%s"
             "\"vocab\":{\"" SP "\":0,\"a\":1,\"b\":2,\"ab\":3,\"<unk>\":4},\"merges\":%s}}",
             normalizer, pretok, decoder, model_type, model_extra, merges);
}

#define NZ "{\"type\":\"Replace\",\"pattern\":{\"String\":\" \"},\"content\":\"" SP "\"}"
#define PT "{\"type\":\"Split\",\"pattern\":{\"String\":\" \"},\"behavior\":\"MergedWithPrevious\",\"invert\":false}"
#define DC "{\"type\":\"Sequence\",\"decoders\":[{\"type\":\"Replace\",\"pattern\":{\"String\":\"" SP "\"},\"content\":\" \"},{\"type\":\"ByteFallback\"},{\"type\":\"Fuse\"}]}"

static void test_errors(void)
{
    char err[256], d[4096];
    LmTok *t;

    /* the baseline document opens, and merges as "a b" strings work too */
    gdoc(d, sizeof d, NZ, PT, DC, "BPE", "\"byte_fallback\":false,", "[[\"a\",\"b\"]]");
    t = lm_tok_open(write_tmp(d), err, sizeof err);
    CHECK(t != NULL);
    int want_ab[] = { 3 };
    if (t) { CHECK(enc_is(t, 0, "ab", want_ab, 1)); lm_tok_close(t); }
    unlink(g_path);
    gdoc(d, sizeof d, NZ, PT, DC, "BPE", "\"byte_fallback\":false,", "[\"a b\"]");
    t = lm_tok_open(write_tmp(d), err, sizeof err);
    CHECK(t != NULL);
    if (t) { CHECK(enc_is(t, 0, "ab", want_ab, 1)); lm_tok_close(t); }
    unlink(g_path);

    /* an unsupported model type */
    gdoc(d, sizeof d, NZ, PT, DC, "Unigram", "", "[[\"a\",\"b\"]]");
    t = lm_tok_open(write_tmp(d), err, sizeof err);
    CHECK(!t && strstr(err, "model.type") && strstr(err, "Unigram"));
    unlink(g_path);

    /* a Replace that is not " " -> U+2581 */
    gdoc(d, sizeof d, "{\"type\":\"Replace\",\"pattern\":{\"String\":\"x\"},\"content\":\"" SP "\"}", PT, DC, "BPE", "", "[[\"a\",\"b\"]]");
    t = lm_tok_open(write_tmp(d), err, sizeof err);
    CHECK(!t && strstr(err, "normalizer"));
    unlink(g_path);

    /* the wrong pre-tokenizer, decoder, ignore_merges */
    gdoc(d, sizeof d, NZ, "{\"type\":\"Whitespace\"}", DC, "BPE", "", "[[\"a\",\"b\"]]");
    t = lm_tok_open(write_tmp(d), err, sizeof err);
    CHECK(!t && strstr(err, "pre_tokenizer"));
    unlink(g_path);
    gdoc(d, sizeof d, NZ, PT, "{\"type\":\"ByteLevel\"}", "BPE", "", "[[\"a\",\"b\"]]");
    t = lm_tok_open(write_tmp(d), err, sizeof err);
    CHECK(!t && strstr(err, "decoder"));
    unlink(g_path);
    gdoc(d, sizeof d, NZ, PT, DC, "BPE", "\"ignore_merges\":true,", "[[\"a\",\"b\"]]");
    t = lm_tok_open(write_tmp(d), err, sizeof err);
    CHECK(!t && strstr(err, "ignore_merges"));
    unlink(g_path);
    gdoc(d, sizeof d, NZ, PT, DC, "BPE", "\"dropout\":0.1,", "[[\"a\",\"b\"]]");
    t = lm_tok_open(write_tmp(d), err, sizeof err);
    CHECK(!t && strstr(err, "dropout"));
    unlink(g_path);

    /* a merge naming a token that is not in the vocab, or whose result is not */
    gdoc(d, sizeof d, NZ, PT, DC, "BPE", "", "[[\"a\",\"z\"]]");
    t = lm_tok_open(write_tmp(d), err, sizeof err);
    CHECK(!t && strstr(err, "missing from model.vocab"));
    unlink(g_path);
    gdoc(d, sizeof d, NZ, PT, DC, "BPE", "", "[[\"b\",\"a\"]]");
    t = lm_tok_open(write_tmp(d), err, sizeof err);
    CHECK(!t && strstr(err, "missing from model.vocab"));
    unlink(g_path);

    /* Qwen files stay refused for byte_fallback; the Gemma family is chosen by the normalizer only */
    t = lm_tok_open("tests/fixtures/tok/qwen-tokenizer.json", err, sizeof err);
    CHECK(t != NULL);
    lm_tok_close(t);
}

int main(int argc, char **argv)
{
    if (argc == 3 && !strcmp(argv[1], "--tool")) return tool_main(argv[2]);
    if (argc != 1) { fprintf(stderr, "usage: test_tok_gemma [--tool TOKENIZER_JSON]\n"); return 2; }
    test_golden();
    test_basics();
    test_long_word();
    test_errors();
    printf("test_tok_gemma: %d checks, %d failed\n", g_checks, g_fail);
    return g_fail ? 1 : 0;
}
