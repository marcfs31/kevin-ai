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

struct LmTok {
    Tok t;
    int nfc; /* normalizer is NFC */
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
static int validate(jval *root, int *nfc, char *err, size_t errlen)
{
    if (!root || root->t != J_OBJ) { seterr(err, errlen, "tokenizer.json: top level is not a JSON object", NULL, NULL); return -1; }
    if (check_normalizer(json_get(root, "normalizer"), nfc, err, errlen)) return -1;
    jval *model = json_get(root, "model");
    if (!model || model->t != J_OBJ) { seterr(err, errlen, "tokenizer.json: missing \"model\" object", NULL, NULL); return -1; }
    jval *type = json_get(model, "type");
    if (!is_str(type) || strcmp(type->str, "BPE") != 0) {
        seterr(err, errlen, "tokenizer.json: model.type is %s, only BPE is supported", is_str(type) ? type->str : "missing", NULL);
        return -1;
    }
    jval *bf = json_get(model, "byte_fallback");
    if (bf && bf->t == J_BOOL && bf->boolean) {
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
    int qwen = 0;
    jval *pt = json_get(root, "pre_tokenizer");
    jval *ps = pt ? json_get(pt, "pretokenizers") : NULL;
    if (ps && ps->t == J_ARR) {
        for (int i = 0; i < ps->len; i++) {
            jval *pat = json_get(ps->kids[i], "pattern");
            jval *rx = pat ? json_get(pat, "Regex") : NULL;
            if (is_str(rx) && strstr(rx->str, "[\\p{L}\\p{M}]+") && !strstr(rx->str, "\\p{Lu}")) qwen = 1;
        }
    }
    if (!qwen) {
        seterr(err, errlen, "tokenizer.json: pre_tokenizer is not the Qwen3.5 split pattern (this tokenizer only implements that one)", NULL, NULL);
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
    int nfc = 0;
    int bad = validate(root, &nfc, err, errlen);
    json_free_tree(root);
    if (bad) return NULL;

    LmTok *t = (LmTok *)calloc(1, sizeof *t);
    if (!t) { seterr(err, errlen, "out of memory", NULL, NULL); return NULL; }
    /* validate() covered every exit() path in tok_load(); the file is parsed a second time
     * there, and that tree stays alive for the table pointers until tok_free(). */
    tok_load(&t->t, path);
    t->nfc = nfc;
    if (!t->t.qwen35) { /* unreachable after validate(); defends the tok.h contract */
        seterr(err, errlen, "tokenizer.json: Qwen3.5 pre-tokenizer not recognised", NULL, NULL);
        tok_free(&t->t);
        free(t);
        return NULL;
    }
    return t;
}

void lm_tok_close(LmTok *t)
{
    if (!t) return;
    tok_free(&t->t);
    free(t);
}

int lm_tok_n_ids(const LmTok *t) { return t->t.n_ids; }

int *lm_tok_encode(const LmTok *t, const char *text, size_t len, int *n)
{
    *n = 0;
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

int lm_tok_piece(const LmTok *t, int id, char *out, size_t cap)
{
    if (id < 0 || id >= t->t.n_ids) return -1;
    const char *s = t->t.id2str[id];
    if (!s) return 0;
    size_t bound = strlen(s); /* decoded length never exceeds the byte-level string */
    if (bound > cap) return -1;
    return tok_decode((Tok *)&t->t, &id, 1, out, (int)cap);
}

char *lm_tok_decode_alloc(const LmTok *t, const int *ids, int n, size_t *len)
{
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
