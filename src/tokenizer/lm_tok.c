/* lm_tok.c -- see lm_tok.h. Adapted from oss_tok.c (unreviewed salvage); wraps
 * third_party/tok.h. */
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* tok.h is a header-only library whose functions all have file scope: a translation unit
 * that uses part of the API draws -Wunused-function for the rest. The suppression is
 * scoped to this include so first-party dead code still warns. tok.h must come first
 * (it defines _GNU_SOURCE). */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#include "tok.h"
#pragma GCC diagnostic pop

#include "tokenizer/lm_nfc.h"
#include "tokenizer/lm_tok.h"
#include "io/lm_gguf.h"

enum { FAM_QWEN = 0, FAM_GEMMA = 1 };

/* (left id, right id) -> (rank, merged id): the Gemma merge table, keyed by ids like the
 * Hugging Face implementation so that a merge costs one probe and no string work. */
typedef struct { uint64_t key; int rank, id; } PairEnt; /* key 0 = empty (see pair_key) */

struct LmTok {
    Tok t;
    int nfc;    /* normalizer is NFC */
    int family; /* FAM_QWEN | FAM_GEMMA */
    /* Gemma only */
    int byte_fb, fuse_unk, unk_id;
    int byte_id[256];            /* id of "<0xXX>", -1 when the vocab has none */
    unsigned char sp_first[256]; /* some added token starts with this byte */
    PairEnt *pair;
    size_t pair_cap;             /* power of two */
};

static void seterr(char *err, size_t n, const char *fmt, const char *a, const char *b)
{
    if (err && n) snprintf(err, n, fmt, a ? a : "", b ? b : "");
}

/* Read the whole file into a NUL-terminated buffer. NULL with err set on failure. */
static char *read_file(const char *path, char *err, size_t errlen)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        seterr(err, errlen, "cannot open tokenizer file %s: %s", path, strerror(errno));
        return NULL;
    }
    char *buf = NULL;
    long n = -1;
    if (fseek(f, 0, SEEK_END) == 0) n = ftell(f);
    if (n < 0 || fseek(f, 0, SEEK_SET) != 0) {
        seterr(err, errlen, "cannot size tokenizer file %s: %s", path, strerror(errno));
        fclose(f);
        return NULL;
    }
    if (n == 0) {
        seterr(err, errlen, "tokenizer file %s is empty", path, NULL);
        fclose(f);
        return NULL;
    }
    if (n > (1L << 30)) {
        seterr(err, errlen, "tokenizer file %s is implausibly large", path, NULL);
        fclose(f);
        return NULL;
    }
    buf = (char *)malloc((size_t)n + 1);
    if (!buf || fread(buf, 1, (size_t)n, f) != (size_t)n) {
        seterr(err, errlen, "cannot read tokenizer file %s", path, NULL);
        free(buf);
        fclose(f);
        return NULL;
    }
    fclose(f);
    buf[n] = 0;
    return buf;
}

static int is_str(const jval *v) { return v && v->t == J_STR && v->str; }

/* Normalizer: absent / null = none, {"type":"NFC"} = NFC, a Sequence of only those = NFC if
 * any is NFC. Anything else (lowercase, NFKC, strip accents...) changes the ids and is
 * refused rather than ignored. Returns 0 and sets *nfc, or -1 with err. */
static int check_normalizer(jval *nz, int *nfc, char *err, size_t errlen)
{
    *nfc = 0;
    if (!nz || nz->t == J_NULL) return 0;
    jval *type = json_get(nz, "type");
    if (!is_str(type)) { seterr(err, errlen, "tokenizer.json: normalizer has no \"type\"", NULL, NULL); return -1; }
    if (!strcmp(type->str, "NFC")) { *nfc = 1; return 0; }
    if (!strcmp(type->str, "Sequence")) {
        jval *list = json_get(nz, "normalizers");
        if (!list || list->t != J_ARR) { seterr(err, errlen, "tokenizer.json: normalizer Sequence has no \"normalizers\" array", NULL, NULL); return -1; }
        for (int i = 0; i < list->len; i++) {
            int sub = 0;
            if (check_normalizer(list->kids[i], &sub, err, errlen)) return -1;
            if (sub) *nfc = 1;
        }
        return 0;
    }
    seterr(err, errlen, "tokenizer.json: normalizer type %s is not supported (only NFC)", type->str, NULL);
    return -1;
}

/* Every condition under which tok_load() would exit(), plus the assumptions tok.h bakes
 * in without checking. Returns 0 when the document is usable. */
static int validate_gemma(jval *root, jval *model, char *err, size_t errlen);

static int validate(jval *root, int *nfc, int *family, char *err, size_t errlen)
{
    if (!root || root->t != J_OBJ) { seterr(err, errlen, "tokenizer.json: top level is not a JSON object", NULL, NULL); return -1; }
    jval *nz = json_get(root, "normalizer");
    jval *nzt = nz && nz->t == J_OBJ ? json_get(nz, "type") : NULL;
    *family = (is_str(nzt) && !strcmp(nzt->str, "Replace")) ? FAM_GEMMA : FAM_QWEN;
    if (*family == FAM_QWEN && check_normalizer(nz, nfc, err, errlen)) return -1;
    jval *model = json_get(root, "model");
    if (!model || model->t != J_OBJ) { seterr(err, errlen, "tokenizer.json: missing \"model\" object", NULL, NULL); return -1; }
    jval *type = json_get(model, "type");
    if (!is_str(type) || strcmp(type->str, "BPE") != 0) {
        seterr(err, errlen, "tokenizer.json: model.type is %s, only BPE is supported", is_str(type) ? type->str : "missing", NULL);
        return -1;
    }
    jval *bf = json_get(model, "byte_fallback");
    if (*family == FAM_QWEN && bf && bf->t == J_BOOL && bf->boolean) {
        seterr(err, errlen, "tokenizer.json: model.byte_fallback true is not supported", NULL, NULL);
        return -1;
    }
    jval *vocab = json_get(model, "vocab");
    if (!vocab || vocab->t != J_OBJ || vocab->len == 0) { seterr(err, errlen, "tokenizer.json: missing or empty model.vocab", NULL, NULL); return -1; }
    for (int i = 0; i < vocab->len; i++) {
        jval *v = vocab->kids[i];
        if (!v || v->t != J_NUM || v->num < 0 || v->num > (double)(1 << 21)) {
            seterr(err, errlen, "tokenizer.json: model.vocab entry %s has a non-numeric or out-of-range id", vocab->keys[i], NULL);
            return -1;
        }
    }
    jval *merges = json_get(model, "merges");
    if (!merges || merges->t != J_ARR || merges->len == 0) {
        seterr(err, errlen, "tokenizer.json: model.merges is missing or empty (a BPE file needs its merge list)", NULL, NULL);
        return -1;
    }
    for (int i = 0; i < merges->len; i++) {
        jval *m = merges->kids[i];
        int ok;
        if (m && m->t == J_STR && m->str) {
            const char *sp = strchr(m->str, ' ');
            ok = sp && sp != m->str && sp[1] && !strchr(sp + 1, ' ');
        } else {
            ok = m && m->t == J_ARR && m->len == 2 && is_str(m->kids[0]) && is_str(m->kids[1]);
        }
        if (!ok) {
            seterr(err, errlen, "tokenizer.json: model.merges entries must be \"left right\" strings or [left, right] pairs", NULL, NULL);
            return -1;
        }
    }
    jval *added = json_get(root, "added_tokens");
    if (!added || added->t != J_ARR) { seterr(err, errlen, "tokenizer.json: missing \"added_tokens\" array", NULL, NULL); return -1; }
    for (int i = 0; i < added->len; i++) {
        jval *a = added->kids[i];
        jval *ji = a && a->t == J_OBJ ? json_get(a, "id") : NULL;
        jval *jc = a && a->t == J_OBJ ? json_get(a, "content") : NULL;
        if (!ji || ji->t != J_NUM || ji->num < 0 || ji->num > (double)(1 << 21) || !is_str(jc)) {
            seterr(err, errlen, "tokenizer.json: added_tokens entry %s is malformed (needs numeric id and string content)", is_str(jc) ? jc->str : "?", NULL);
            return -1;
        }
    }
    if (*family == FAM_GEMMA) return validate_gemma(root, model, err, errlen);
    int qwen = 0;
    jval *pt = json_get(root, "pre_tokenizer");
    jval *ps = pt ? json_get(pt, "pretokenizers") : NULL;
    if (ps && ps->t == J_ARR) {
        for (int i = 0; i < ps->len; i++) {
            jval *pat = json_get(ps->kids[i], "pattern");
            jval *rx = pat ? json_get(pat, "Regex") : NULL;
            if (is_str(rx) && strstr(rx->str, "[\\p{L}\\p{M}]+") && !strstr(rx->str, "\\p{Lu}")) qwen = 1;
            if (is_str(rx) && strstr(rx->str, "?\\p{L}+|\\p{N}|") && !strstr(rx->str, "\\p{M}") && !strstr(rx->str, "\\p{Lu}")) qwen = 1;
        }
    }
    if (!qwen) {
        seterr(err, errlen, "tokenizer.json: pre_tokenizer is not the Qwen3.5 or Qwen2 split pattern (this tokenizer implements those two)", NULL, NULL);
        return -1;
    }
    jval *dec = json_get(root, "decoder");
    if (dec && dec->t == J_OBJ) {
        jval *dt = json_get(dec, "type");
        if (!is_str(dt) || strcmp(dt->str, "ByteLevel") != 0) {
            seterr(err, errlen, "tokenizer.json: decoder type %s is not supported (only ByteLevel)", is_str(dt) ? dt->str : "missing", NULL);
            return -1;
        }
    }
    return 0;
}

static int jbool(jval *v, int dflt)
{
    return v && v->t == J_BOOL ? v->boolean : dflt;
}

/* The Gemma 4 schema: normalizer Replace " " -> U+2581, pre_tokenizer Split on the literal
 * " " (MergedWithPrevious), decoder Sequence[Replace U+2581 -> " ", ByteFallback, Fuse].
 * Anything else changes the ids and is refused. */
static int validate_gemma(jval *root, jval *model, char *err, size_t errlen)
{
    jval *nz = json_get(root, "normalizer");
    jval *pat = json_get(nz, "pattern");
    jval *ps = pat ? json_get(pat, "String") : NULL;
    jval *content = json_get(nz, "content");
    if (!is_str(ps) || strcmp(ps->str, " ") || !is_str(content) || strcmp(content->str, "\xE2\x96\x81")) {
        seterr(err, errlen, "tokenizer.json: normalizer Replace must map \" \" to U+2581 (the only Replace this tokenizer implements)", NULL, NULL);
        return -1;
    }
    jval *pt = json_get(root, "pre_tokenizer");
    jval *ptt = pt && pt->t == J_OBJ ? json_get(pt, "type") : NULL;
    jval *ptp = pt && pt->t == J_OBJ ? json_get(pt, "pattern") : NULL;
    jval *pts = ptp ? json_get(ptp, "String") : NULL;
    jval *beh = pt && pt->t == J_OBJ ? json_get(pt, "behavior") : NULL;
    if (!is_str(ptt) || strcmp(ptt->str, "Split") || !is_str(pts) || strcmp(pts->str, " ") ||
        !is_str(beh) || strcmp(beh->str, "MergedWithPrevious") || jbool(json_get(pt, "invert"), 0)) {
        seterr(err, errlen, "tokenizer.json: pre_tokenizer is not Split on \" \" (MergedWithPrevious), the Gemma pre-tokenizer", NULL, NULL);
        return -1;
    }
    jval *dec = json_get(root, "decoder");
    jval *dt = dec && dec->t == J_OBJ ? json_get(dec, "type") : NULL;
    jval *dl = dt ? json_get(dec, "decoders") : NULL;
    static const char *const want[3] = { "Replace", "ByteFallback", "Fuse" };
    int ok = is_str(dt) && !strcmp(dt->str, "Sequence") && dl && dl->t == J_ARR && dl->len == 3;
    for (int i = 0; ok && i < 3; i++) {
        jval *ty = json_get(dl->kids[i], "type");
        ok = is_str(ty) && !strcmp(ty->str, want[i]);
    }
    if (ok) {
        jval *dp = json_get(json_get(dl->kids[0], "pattern"), "String");
        jval *dc = json_get(dl->kids[0], "content");
        ok = is_str(dp) && !strcmp(dp->str, "\xE2\x96\x81") && is_str(dc) && !strcmp(dc->str, " ");
    }
    if (!ok) {
        seterr(err, errlen, "tokenizer.json: decoder is not Sequence[Replace U+2581 -> \" \", ByteFallback, Fuse]", NULL, NULL);
        return -1;
    }
    if (jbool(json_get(model, "ignore_merges"), 0)) {
        seterr(err, errlen, "tokenizer.json: model.ignore_merges true is not supported", NULL, NULL);
        return -1;
    }
    static const char *const unsupported[] = { "continuing_subword_prefix", "end_of_word_suffix", "dropout" };
    for (int i = 0; i < 3; i++) {
        jval *v = json_get(model, unsupported[i]);
        if (v && v->t != J_NULL && !(v->t == J_STR && v->str && !v->str[0])) {
            seterr(err, errlen, "tokenizer.json: model.%s is not supported", unsupported[i], NULL);
            return -1;
        }
    }
    jval *unk = json_get(model, "unk_token");
    if (unk && unk->t != J_NULL && !is_str(unk)) {
        seterr(err, errlen, "tokenizer.json: model.unk_token must be a string or null", NULL, NULL);
        return -1;
    }
    return 0;
}

static uint64_t pair_key(int l, int r) { return ((uint64_t)(unsigned)l << 32) | (uint64_t)(unsigned)r | (1ULL << 63); }

static size_t pair_slot(uint64_t key, size_t cap)
{
    uint64_t h = key * 0x9E3779B97F4A7C15ULL;
    return (size_t)(h >> 20) & (cap - 1);
}

/* Build the id-keyed merge table; every merge must name vocab entries and its concatenation
 * must be one too (Hugging Face refuses such a file; so do we). */
static int build_gemma(LmTok *t, char *err, size_t errlen)
{
    jval *model = json_get(t->t.json_root_, "model");
    jval *merges = json_get(model, "merges");
    t->pair_cap = 1;
    while (t->pair_cap < (size_t)merges->len * 2) t->pair_cap <<= 1;
    t->pair = (PairEnt *)calloc(t->pair_cap, sizeof(PairEnt));
    if (!t->pair) { seterr(err, errlen, "out of memory", NULL, NULL); return -1; }
    char *cat = NULL;
    size_t catcap = 0;
    for (int i = 0; i < merges->len; i++) {
        jval *m = merges->kids[i];
        const char *l, *r;
        size_t ll, rl;
        if (m->t == J_STR) {
            const char *sp = strchr(m->str, ' ');
            l = m->str; ll = (size_t)(sp - m->str); r = sp + 1; rl = strlen(r);
        } else {
            l = m->kids[0]->str; ll = strlen(l); r = m->kids[1]->str; rl = strlen(r);
        }
        if (ll + rl > catcap) {
            free(cat);
            catcap = (ll + rl) * 2 + 64;
            cat = (char *)malloc(catcap);
            if (!cat) { seterr(err, errlen, "out of memory", NULL, NULL); return -1; }
        }
        memcpy(cat, l, ll);
        memcpy(cat + ll, r, rl);
        int a = hm_get(&t->t.vocab, l, (int)ll), b = hm_get(&t->t.vocab, r, (int)rl);
        int c = hm_get(&t->t.vocab, cat, (int)(ll + rl));
        if (a < 0 || b < 0 || c < 0) {
            char num[32];
            snprintf(num, sizeof num, "%d", i);
            seterr(err, errlen, "tokenizer.json: merge %s names a token (or a result) missing from model.vocab", num, NULL);
            free(cat);
            return -1;
        }
        uint64_t key = pair_key(a, b);
        size_t s = pair_slot(key, t->pair_cap);
        while (t->pair[s].key && t->pair[s].key != key) s = (s + 1) & (t->pair_cap - 1);
        t->pair[s].key = key; /* a duplicated pair keeps its LAST rank, like HF's map */
        t->pair[s].rank = i;
        t->pair[s].id = c;
    }
    free(cat);
    jval *unk = json_get(model, "unk_token");
    t->byte_fb = jbool(json_get(model, "byte_fallback"), 0);
    t->fuse_unk = jbool(json_get(model, "fuse_unk"), 0);
    t->unk_id = -1;
    if (is_str(unk)) {
        t->unk_id = hm_get(&t->t.vocab, unk->str, (int)strlen(unk->str));
        if (t->unk_id < 0) { seterr(err, errlen, "tokenizer.json: unk_token %s is not in model.vocab", unk->str, NULL); return -1; }
    }
    for (int b = 0; b < 256; b++) {
        char name[8];
        snprintf(name, sizeof name, "<0x%02X>", b);
        t->byte_id[b] = hm_get(&t->t.vocab, name, 6);
    }
    for (int i = 0; i < t->t.nsp; i++)
        if (t->t.sp[i].len > 0) t->sp_first[(unsigned char)t->t.sp[i].str[0]] = 1;
    return 0;
}

LmTok *lm_tok_open(const char *path, char *err, size_t errlen)
{
    if (err && errlen) err[0] = 0;
    char *buf = read_file(path, err, errlen);
    if (!buf) return NULL;
    jval *root = json_parse(buf, NULL);
    free(buf);
    if (!root) {
        seterr(err, errlen, "tokenizer file %s is not valid JSON (malformed or truncated)", path, NULL);
        return NULL;
    }
    int nfc = 0, family = FAM_QWEN;
    int bad = validate(root, &nfc, &family, err, errlen);
    json_free_tree(root);
    if (bad) return NULL;

    LmTok *t = (LmTok *)calloc(1, sizeof *t);
    if (!t) { seterr(err, errlen, "out of memory", NULL, NULL); return NULL; }
    /* validate() covered every exit() path in tok_load(); the file is parsed a second time
     * there, and that tree stays alive for the table pointers until tok_free(). */
    tok_load(&t->t, path);
    t->nfc = nfc;
    t->family = family;
    if (family == FAM_GEMMA) {
        if (build_gemma(t, err, errlen)) { lm_tok_close(t); return NULL; }
        return t;
    }
    if (!t->t.qwen35 && !t->t.qwen2) { /* unreachable after validate(); defends the tok.h contract */
        seterr(err, errlen, "tokenizer.json: Qwen pre-tokenizer not recognised", NULL, NULL);
        tok_free(&t->t);
        free(t);
        return NULL;
    }
    return t;
}

/* ---- the tokenizer of a GGUF file --------------------------------------------------------
 * GGUF stores the BPE tables (tokenizer.ggml.tokens in the byte-level spelling, .merges as
 * "a b" strings, .token_type: 1 normal, 3 control, 4 user defined, 5 unused) but not the
 * pre-tokenizer regex, only its name (tokenizer.ggml.pre). The tables are written out as the
 * tokenizer.json document tok.h already loads, with the Qwen2 split pattern and an NFC
 * normalizer (the Hugging Face tokenizer of every Qwen2 model normalises to NFC). Control
 * tokens become special added tokens, user-defined ones non-special added tokens; unused
 * placeholders are dropped. */
typedef struct { char *p; size_t n, cap; int bad; } Sb;

static void sb_put(Sb *b, const char *s, size_t n)
{
    if (b->bad) return;
    if (b->n + n + 1 > b->cap) {
        size_t nc = b->cap ? b->cap * 2 : 1 << 16;
        while (nc < b->n + n + 1) nc *= 2;
        char *np = (char *)realloc(b->p, nc);
        if (!np) { b->bad = 1; return; }
        b->p = np;
        b->cap = nc;
    }
    memcpy(b->p + b->n, s, n);
    b->n += n;
    b->p[b->n] = 0;
}
static void sb_lit(Sb *b, const char *s) { sb_put(b, s, strlen(s)); }

static void sb_json(Sb *b, const char *s, size_t n)
{
    sb_put(b, "\"", 1);
    size_t from = 0;
    for (size_t i = 0; i < n; i++) {
        const unsigned char c = (unsigned char)s[i];
        if (c != '"' && c != '\\' && c >= 0x20) continue;
        sb_put(b, s + from, i - from);
        from = i + 1;
        char e[8];
        if (c == '"') sb_lit(b, "\\\"");
        else if (c == '\\') sb_lit(b, "\\\\");
        else { snprintf(e, sizeof e, "\\u%04x", c); sb_lit(b, e); }
    }
    sb_put(b, s + from, n - from);
    sb_put(b, "\"", 1);
}

#define QWEN2_SPLIT "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\\\r\\\\n\\\\p{L}\\\\p{N}]?\\\\p{L}+|\\\\p{N}| ?[^\\\\s\\\\p{L}\\\\p{N}]+[\\\\r\\\\n]*|\\\\s*[\\\\r\\\\n]+|\\\\s+(?!\\\\S)|\\\\s+"

LmTok *lm_tok_open_gguf(const LmGguf *g, char *err, size_t errlen)
{
    if (err && errlen) err[0] = 0;
    LmGgufStr model = { NULL, 0 }, pre = { NULL, 0 };
    if (lm_gguf_get_str(g, "tokenizer.ggml.model", &model) != 0 || model.n != 4 || memcmp(model.s, "gpt2", 4) != 0) {
        seterr(err, errlen, "%s: tokenizer.ggml.model is not \"gpt2\" (byte-level BPE); only that is supported", g->path, NULL);
        return NULL;
    }
    if (lm_gguf_get_str(g, "tokenizer.ggml.pre", &pre) != 0 || pre.n != 5 || memcmp(pre.s, "qwen2", 5) != 0) {
        seterr(err, errlen, "%s: tokenizer.ggml.pre is not \"qwen2\"; only the Qwen2 split pattern is implemented", g->path, NULL);
        return NULL;
    }
    int nt = 0, nm = 0, ny = 0;
    LmGgufStr *toks = lm_gguf_str_array(g, "tokenizer.ggml.tokens", &nt, err, errlen);
    if (!toks) return NULL;
    LmGgufStr *merges = lm_gguf_str_array(g, "tokenizer.ggml.merges", &nm, err, errlen);
    int64_t *types = merges ? lm_gguf_int_array(g, "tokenizer.ggml.token_type", &ny, err, errlen) : NULL;
    if (!merges || !types) { free(toks); free(merges); free(types); return NULL; }
    LmTok *res = NULL;
    Sb b = { NULL, 0, 0, 0 };
    if (ny != nt || nt < 1) {
        seterr(err, errlen, "%s: tokenizer.ggml.token_type does not have one entry per token", g->path, NULL);
        goto done;
    }
    sb_lit(&b, "{\"normalizer\":{\"type\":\"NFC\"},\"pre_tokenizer\":{\"type\":\"Sequence\",\"pretokenizers\":["
               "{\"type\":\"Split\",\"pattern\":{\"Regex\":\"" QWEN2_SPLIT "\"},\"behavior\":\"Isolated\",\"invert\":false},"
               "{\"type\":\"ByteLevel\",\"add_prefix_space\":false}]},\"decoder\":{\"type\":\"ByteLevel\"},"
               "\"model\":{\"type\":\"BPE\",\"ignore_merges\":false,\"vocab\":{");
    int first = 1;
    for (int i = 0; i < nt; i++) {
        if (types[i] != 1) continue;
        if (!first) sb_lit(&b, ",");
        first = 0;
        sb_json(&b, toks[i].s, toks[i].n);
        char idb[24];
        snprintf(idb, sizeof idb, ":%d", i);
        sb_lit(&b, idb);
    }
    sb_lit(&b, "},\"merges\":[");
    for (int i = 0; i < nm; i++) {
        if (i) sb_lit(&b, ",");
        sb_json(&b, merges[i].s, merges[i].n);
    }
    sb_lit(&b, "]},\"added_tokens\":[");
    first = 1;
    for (int i = 0; i < nt; i++) {
        if (types[i] != 3 && types[i] != 4) continue;
        if (!first) sb_lit(&b, ",");
        first = 0;
        char idb[48];
        snprintf(idb, sizeof idb, "{\"id\":%d,\"content\":", i);
        sb_lit(&b, idb);
        sb_json(&b, toks[i].s, toks[i].n);
        sb_lit(&b, types[i] == 3 ? ",\"special\":true}" : ",\"special\":false}");
    }
    sb_lit(&b, "]}");
    if (b.bad) { seterr(err, errlen, "out of memory", NULL, NULL); goto done; }

    jval *root = json_parse(b.p, NULL);
    if (!root) { seterr(err, errlen, "%s: the tokenizer tables do not form a valid document", g->path, NULL); goto done; }
    int nfc = 0, family = FAM_QWEN;
    const int bad = validate(root, &nfc, &family, err, errlen);
    json_free_tree(root);
    if (bad) goto done;
    res = (LmTok *)calloc(1, sizeof *res);
    if (!res) { seterr(err, errlen, "out of memory", NULL, NULL); goto done; }
    tok_load_text(&res->t, b.p);   /* takes b.p */
    b.p = NULL;
    res->nfc = nfc;
    res->family = family;
    if (!res->t.qwen2) { seterr(err, errlen, "internal: the Qwen2 pre-tokenizer was not recognised", NULL, NULL); lm_tok_close(res); res = NULL; }
done:
    free(b.p);
    free(toks); free(merges); free(types);
    return res;
}

LmTok *lm_tok_open_model(const char *model, char *err, size_t errlen)
{
    char gp[4096];
    if (lm_gguf_locate(model, gp, sizeof gp)) {
        LmGguf g;
        if (lm_gguf_open(&g, gp, err, errlen) != 0) return NULL;
        LmTok *t = lm_tok_open_gguf(&g, err, errlen);
        lm_gguf_close(&g);
        return t;
    }
    char path[4096];
    snprintf(path, sizeof path, "%s/tokenizer.json", model);
    return lm_tok_open(path, err, errlen);
}

void lm_tok_close(LmTok *t)
{
    if (!t) return;
    free(t->pair);
    tok_free(&t->t);
    free(t);
}

int lm_tok_n_ids(const LmTok *t) { return t->t.n_ids; }

/* ---- Gemma: BPE with byte fallback over one word ---------------------------------------
 * The normalizer turns every " " into U+2581 and the Split pre-tokenizer never fires after
 * that, so a text segment (the text between added tokens) is ONE word. Symbols are token
 * ids; merges come from the id-pair table. A doubly linked list plus a min-heap of
 * (rank, position) makes a word of n symbols cost O(n log n): the pair with the lowest
 * rank is merged first, the leftmost on ties, which is HF's order. Heap entries go stale
 * when a neighbour merges; they are re-validated on pop instead of being deleted. */

typedef struct { int rank, i, j, li, rj; } HeapEnt;

typedef struct {
    int *ids, *prev, *next;
    HeapEnt *heap;
    int hn;
    size_t cap; /* symbols */
} Work;

static int work_reserve(Work *w, size_t n)
{
    if (n <= w->cap) return 0;
    free(w->ids); free(w->prev); free(w->next); free(w->heap);
    w->ids = (int *)malloc(n * sizeof(int));
    w->prev = (int *)malloc(n * sizeof(int));
    w->next = (int *)malloc(n * sizeof(int));
    w->heap = (HeapEnt *)malloc((3 * n + 4) * sizeof(HeapEnt));
    w->cap = (w->ids && w->prev && w->next && w->heap) ? n : 0;
    return w->cap ? 0 : -1;
}

static int heap_less(const HeapEnt *a, const HeapEnt *b)
{
    return a->rank != b->rank ? a->rank < b->rank : a->i < b->i;
}

static void heap_push(Work *w, HeapEnt e)
{
    int k = w->hn++;
    while (k > 0) {
        int up = (k - 1) / 2;
        if (!heap_less(&e, &w->heap[up])) break;
        w->heap[k] = w->heap[up];
        k = up;
    }
    w->heap[k] = e;
}

static HeapEnt heap_pop(Work *w)
{
    HeapEnt top = w->heap[0], e = w->heap[--w->hn];
    int k = 0;
    for (;;) {
        int c = 2 * k + 1;
        if (c >= w->hn) break;
        if (c + 1 < w->hn && heap_less(&w->heap[c + 1], &w->heap[c])) c++;
        if (!heap_less(&w->heap[c], &e)) break;
        w->heap[k] = w->heap[c];
        k = c;
    }
    if (w->hn > 0) w->heap[k] = e;
    return top;
}

static const PairEnt *pair_find(const LmTok *t, int l, int r)
{
    uint64_t key = pair_key(l, r);
    size_t s = pair_slot(key, t->pair_cap);
    while (t->pair[s].key) {
        if (t->pair[s].key == key) return &t->pair[s];
        s = (s + 1) & (t->pair_cap - 1);
    }
    return NULL;
}

static void queue_pair(const LmTok *t, Work *w, int i)
{
    int j = w->next[i];
    if (j < 0) return;
    const PairEnt *pe = pair_find(t, w->ids[i], w->ids[j]);
    if (pe) {
        HeapEnt e = { pe->rank, i, j, w->ids[i], w->ids[j] };
        heap_push(w, e);
    }
}

/* Length of the strictly valid UTF-8 sequence at p (<= avail), 0 if p starts an invalid one. */
static int u8_strict_len(const unsigned char *p, size_t avail)
{
    unsigned char c = p[0];
    int need;
    if (c < 0x80) return 1;
    if (c >= 0xC2 && c <= 0xDF) need = 2;
    else if (c >= 0xE0 && c <= 0xEF) need = 3;
    else if (c >= 0xF0 && c <= 0xF4) need = 4;
    else return 0;
    if ((size_t)need > avail) return 0;
    for (int k = 1; k < need; k++) {
        unsigned lo = 0x80, hi = 0xBF;
        if (k == 1) {
            if (c == 0xE0) lo = 0xA0;
            else if (c == 0xED) hi = 0x9F;
            else if (c == 0xF0) lo = 0x90;
            else if (c == 0xF4) hi = 0x8F;
        }
        if (p[k] < lo || p[k] > hi) return 0;
    }
    return need;
}

static const unsigned char SPACE_U2581[3] = { 0xE2, 0x96, 0x81 };

/* Encode one word (raw text bytes p[0..len)) and append its ids to out[*no]. */
static int gemma_word(const LmTok *t, Work *w, const unsigned char *p, size_t len, int *out, int *no)
{
    if (!len) return 0;
    if (work_reserve(w, len)) return -1;
    int n = 0, unk_run = 0;
    for (size_t i = 0; i < len;) {
        const unsigned char *c;
        int cl, adv;
        if (p[i] == ' ') { c = SPACE_U2581; cl = 3; adv = 1; }
        else {
            adv = u8_strict_len(p + i, len - i);
            if (!adv) adv = 1; /* invalid byte: its own "character" (HF never sees one) */
            c = p + i; cl = adv;
        }
        i += (size_t)adv;
        int id = hm_get((hmap *)&t->t.vocab, (const char *)c, cl);
        if (id >= 0) { w->ids[n++] = id; unk_run = 0; continue; }
        if (t->byte_fb) {
            int ok = 1;
            for (int k = 0; k < cl; k++) if (t->byte_id[c[k]] < 0) ok = 0;
            if (ok) {
                for (int k = 0; k < cl; k++) w->ids[n++] = t->byte_id[c[k]];
                unk_run = 0;
                continue;
            }
        }
        if (t->unk_id >= 0) {
            if (!(unk_run && t->fuse_unk)) w->ids[n++] = t->unk_id;
            unk_run = 1;
        }
    }
    for (int i = 0; i < n; i++) { w->prev[i] = i - 1; w->next[i] = i + 1 < n ? i + 1 : -1; }
    w->hn = 0;
    for (int i = 0; i + 1 < n; i++) queue_pair(t, w, i);
    while (w->hn) {
        HeapEnt e = heap_pop(w);
        int i = e.i, j = e.j;
        /* stale: i or j merged away (their ids changed or they left the list) */
        if (w->next[i] != j || w->ids[i] != e.li || w->ids[j] != e.rj || w->ids[j] < 0) continue;
        const PairEnt *pe = pair_find(t, e.li, e.rj);
        if (!pe || pe->rank != e.rank) continue;
        w->ids[i] = pe->id;
        w->ids[j] = -1; /* dead */
        w->next[i] = w->next[j];
        if (w->next[j] >= 0) w->prev[w->next[j]] = i;
        if (w->prev[i] >= 0) queue_pair(t, w, w->prev[i]);
        queue_pair(t, w, i);
    }
    for (int i = 0; i >= 0 && i < n; i = w->next[i]) out[(*no)++] = w->ids[i];
    return 0;
}

/* Encode with added tokens parsed (special != 0: HF's behaviour) or as plain text. The
 * added tokens are matched on the raw text, leftmost then longest. */
static int *gemma_encode(const LmTok *t, const char *text, size_t len, int special, int *n)
{
    if (len > (size_t)INT_MAX / 8) return NULL;
    int *out = (int *)malloc((len + 1) * sizeof(int));
    if (!out) return NULL;
    Work w;
    memset(&w, 0, sizeof w);
    const unsigned char *p = (const unsigned char *)text;
    int no = 0;
    size_t seg = 0;
    int bad = 0;
    for (size_t i = 0; i < len && special;) {
        int hit = -1;
        size_t hl = 0;
        if (t->sp_first[p[i]]) {
            for (int k = 0; k < t->t.nsp; k++) { /* sp is sorted longest first */
                size_t sl = (size_t)t->t.sp[k].len;
                if (sl && i + sl <= len && !memcmp(p + i, t->t.sp[k].str, sl)) { hit = t->t.sp[k].id; hl = sl; break; }
            }
        }
        if (hit < 0) { i++; continue; }
        if (gemma_word(t, &w, p + seg, i - seg, out, &no)) { bad = 1; break; }
        out[no++] = hit;
        i += hl;
        seg = i;
    }
    if (!bad && gemma_word(t, &w, p + seg, len - seg, out, &no)) bad = 1;
    free(w.ids); free(w.prev); free(w.next); free(w.heap);
    if (bad) { free(out); return NULL; }
    *n = no;
    return out;
}

int *lm_tok_encode_special(const LmTok *t, const char *text, size_t len, int *n)
{
    *n = 0;
    if (t->family == FAM_GEMMA) return gemma_encode(t, text, len, 1, n);
    if (len > (size_t)INT_MAX / 8) return NULL;
    char *norm = NULL;
    if (t->nfc && len) {
        size_t nl = 0;
        norm = lm_nfc(text, len, &nl);
        if (!norm) return NULL;
        text = norm;
        len = nl;
    }
    int *out = (int *)malloc(((size_t)len + 1) * sizeof(int));
    if (out && len) *n = tok_encode_mode((Tok *)&t->t, text, (int)len, out, (int)len + 1, 1);
    free(norm);
    return out;
}

int *lm_tok_encode(const LmTok *t, const char *text, size_t len, int *n)
{
    *n = 0;
    if (t->family == FAM_GEMMA) return gemma_encode(t, text, len, 0, n);
    if (len > (size_t)INT_MAX / 8) return NULL;
    char *norm = NULL;
    if (t->nfc && len) {
        size_t nl = 0;
        norm = lm_nfc(text, len, &nl);
        if (!norm) return NULL;
        text = norm;
        len = nl;
    }
    /* A token covers at least one byte, so len + 1 slots can never be truncated by
     * tok.h's silent max. */
    int *out = (int *)malloc(((size_t)len + 1) * sizeof(int));
    if (out && len) *n = tok_encode_mode((Tok *)&t->t, text, (int)len, out, (int)len + 1, 0);
    free(norm);
    return out;
}

int lm_tok_special_id(const LmTok *t, const char *content)
{
    return tok_id_of((Tok *)&t->t, content);
}

int lm_tok_require(const LmTok *t, const char *content, int *id, char *err, size_t errlen)
{
    int v = lm_tok_special_id(t, content);
    if (v < 0) {
        seterr(err, errlen, "required special token %s is missing from tokenizer.json", content, NULL);
        return -1;
    }
    *id = v;
    return 0;
}

int lm_tok_is_special(const LmTok *t, int id)
{
    return id >= 0 && id < t->t.n_ids && t->t.id_special[id];
}

int lm_tok_vocab_id(const LmTok *t, const char *bytelevel)
{
    return hm_get((hmap *)&t->t.vocab, bytelevel, (int)strlen(bytelevel));
}

/* "<0xXX>" -> the byte, or -1 (exactly the shape HF's ByteFallback decoder accepts). */
static int byte_token(const char *s)
{
    if (strlen(s) != 6 || memcmp(s, "<0x", 3) || s[5] != '>') return -1;
    int v = 0;
    for (int k = 3; k < 5; k++) {
        int c = (unsigned char)s[k], d;
        if (c >= '0' && c <= '9') d = c - '0';
        else if (c >= 'A' && c <= 'F') d = c - 'A' + 10;
        else if (c >= 'a' && c <= 'f') d = c - 'a' + 10;
        else return -1;
        v = v * 16 + d;
    }
    return v;
}

/* Gemma piece bytes: a byte token is its one byte, anything else its string with each
 * U+2581 turned into a space. Never longer than strlen(s). */
static int gemma_piece(const char *s, char *out)
{
    int b = byte_token(s);
    if (b >= 0) { out[0] = (char)b; return 1; }
    int o = 0;
    for (const char *q = s; *q;) {
        if ((unsigned char)q[0] == SPACE_U2581[0] && (unsigned char)q[1] == SPACE_U2581[1] &&
            (unsigned char)q[2] == SPACE_U2581[2]) { out[o++] = ' '; q += 3; } /* && stops at the NUL */
        else out[o++] = *q++;
    }
    return o;
}

int lm_tok_piece(const LmTok *t, int id, char *out, size_t cap)
{
    if (id < 0 || id >= t->t.n_ids) return -1;
    const char *s = t->t.id2str[id];
    if (!s) return 0;
    size_t bound = strlen(s); /* decoded length never exceeds the byte-level string */
    if (bound > cap) return -1;
    if (t->family == FAM_GEMMA) return gemma_piece(s, out);
    return tok_decode((Tok *)&t->t, &id, 1, out, (int)cap);
}

/* Gemma decode, HF semantics: a maximal run of byte tokens is decoded as UTF-8 and, if the
 * run is not valid, EVERY byte of it becomes U+FFFD. */
static char *gemma_decode(const LmTok *t, const int *ids, int n, size_t *len)
{
    size_t cap = 1;
    for (int i = 0; i < n; i++)
        if (ids[i] >= 0 && ids[i] < t->t.n_ids && t->t.id2str[ids[i]]) cap += strlen(t->t.id2str[ids[i]]);
    char *out = (char *)malloc(cap + 1);
    if (!out) return NULL;
    size_t o = 0;
    for (int i = 0; i < n;) {
        const char *s = (ids[i] >= 0 && ids[i] < t->t.n_ids) ? t->t.id2str[ids[i]] : NULL;
        if (!s) { i++; continue; }
        if (byte_token(s) < 0) { o += (size_t)gemma_piece(s, out + o); i++; continue; }
        int j = i;
        size_t run = 0, o0 = o;
        for (; j < n; j++) {
            const char *sj = (ids[j] >= 0 && ids[j] < t->t.n_ids) ? t->t.id2str[ids[j]] : NULL;
            if (!sj) continue; /* an id with no token is skipped, it does not end the run */
            int b = byte_token(sj);
            if (b < 0) break;
            out[o0 + run++] = (char)b;
        }
        int valid = 1;
        for (size_t k = 0; k < run && valid;) {
            int l = u8_strict_len((const unsigned char *)out + o0 + k, run - k);
            if (!l) valid = 0; else k += (size_t)l;
        }
        o = o0 + run;
        if (!valid) { /* one U+FFFD per byte; fits because each byte token spells 6 chars */
            for (size_t k = 0; k < run; k++) memcpy(out + o0 + 3 * k, "\xEF\xBF\xBD", 3);
            o = o0 + run * 3;
        }
        i = j;
    }
    out[o] = 0;
    if (len) *len = o;
    return out;
}

char *lm_tok_decode_alloc(const LmTok *t, const int *ids, int n, size_t *len)
{
    if (t->family == FAM_GEMMA) return gemma_decode(t, ids, n, len);
    size_t cap = 1;
    for (int i = 0; i < n; i++) {
        int id = ids[i];
        if (id >= 0 && id < t->t.n_ids && t->t.id2str[id]) cap += strlen(t->t.id2str[id]);
    }
    char *out = (char *)malloc(cap + 1);
    if (!out) return NULL;
    int o = tok_decode((Tok *)&t->t, ids, n, out, (int)cap);
    out[o] = 0;
    if (len) *len = (size_t)o;
    return out;
}

/* ---- streaming UTF-8 ---------------------------------------------------------- */

static int u8_seq_len(unsigned char b)
{
    if (b >= 0xC2 && b <= 0xDF) return 2;
    if (b >= 0xE0 && b <= 0xEF) return 3;
    if (b >= 0xF0 && b <= 0xF4) return 4;
    return 0;
}

/* Is b a valid next byte of the held sequence (second byte has lead-specific bounds that
 * exclude overlong forms, surrogates and code points above U+10FFFF). */
static int u8_cont_ok(const LmUtf8 *u, unsigned char b)
{
    unsigned lo = 0x80, hi = 0xBF;
    if (u->pend_len == 1) {
        switch (u->pend[0]) {
        case 0xE0: lo = 0xA0; break;
        case 0xED: hi = 0x9F; break;
        case 0xF0: lo = 0x90; break;
        case 0xF4: hi = 0x8F; break;
        default: break;
        }
    }
    return b >= lo && b <= hi;
}

void lm_utf8_init(LmUtf8 *u) { memset(u, 0, sizeof *u); }

static size_t put_fffd(char *out)
{
    out[0] = (char)0xEF; out[1] = (char)0xBF; out[2] = (char)0xBD;
    return 3;
}

size_t lm_utf8_push(LmUtf8 *u, const char *in, size_t n, char *out)
{
    size_t o = 0;
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)in[i];
        for (;;) {
            if (u->pend_len == 0) {
                if (c < 0x80) out[o++] = (char)c;
                else if ((u->need = u8_seq_len(c)) != 0) { u->pend[0] = c; u->pend_len = 1; }
                else o += put_fffd(out + o);
                break;
            }
            if (u8_cont_ok(u, c)) {
                u->pend[u->pend_len++] = c;
                if (u->pend_len == u->need) {
                    memcpy(out + o, u->pend, (size_t)u->pend_len);
                    o += (size_t)u->pend_len;
                    u->pend_len = 0;
                }
                break;
            }
            o += put_fffd(out + o); /* held sequence was cut short: reprocess c fresh */
            u->pend_len = 0;
        }
    }
    return o;
}

size_t lm_utf8_flush(LmUtf8 *u, char *out)
{
    if (u->pend_len == 0) return 0;
    u->pend_len = 0;
    return put_fffd(out);
}
