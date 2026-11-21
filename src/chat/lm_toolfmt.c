/* lm_toolfmt.c -- see lm_toolfmt.h. Mirrors the tool parts of the models' own
 * chat_template.jinja files: Qwen3.5 (`tool | tojson`, <function=...> calls) and Gemma 4
 * (format_function_declaration, format_parameters, format_argument). Original to this
 * repository. */
#include "chat/lm_toolfmt.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- string builder ----------------------------------------------------------------- */

void lm_sb_add(LmSb *b, const char *s, size_t n)
{
    if (b->oom || !n) return;
    if (b->n + n + 1 > b->cap) {
        size_t cap = b->cap ? b->cap : 256;
        while (cap < b->n + n + 1) cap *= 2;
        char *p = (char *)realloc(b->s, cap);
        if (!p) { b->oom = 1; return; }
        b->s = p;
        b->cap = cap;
    }
    memcpy(b->s + b->n, s, n);
    b->n += n;
    b->s[b->n] = 0;
}

void lm_sb_str(LmSb *b, const char *s) { lm_sb_add(b, s, strlen(s)); }
void lm_sb_free(LmSb *b) { free(b->s); memset(b, 0, sizeof *b); }
static void sb_ch(LmSb *b, char c) { lm_sb_add(b, &c, 1); }

/* user text for the Gemma builder: the sentinel byte is dropped so it cannot forge <|"|> */
void lm_sb_user(LmSb *b, const char *s)
{
    size_t run = 0, i = 0;
    for (; s[i]; i++) {
        if (s[i] != LM_QS) continue;
        lm_sb_add(b, s + run, i - run);
        run = i + 1;
    }
    lm_sb_add(b, s + run, i - run);
}

/* ---- JSON out ----------------------------------------------------------------------- */

void lm_json_string(LmSb *b, const char *s, size_t n)
{
    sb_ch(b, '"');
    size_t run = 0;
    for (size_t i = 0; i < n; i++) {
        unsigned char ch = (unsigned char)s[i];
        if (ch >= 0x20 && ch != '"' && ch != '\\') continue;
        lm_sb_add(b, s + run, i - run);
        run = i + 1;
        char t[8];
        if (ch == '"') lm_sb_str(b, "\\\"");
        else if (ch == '\\') lm_sb_str(b, "\\\\");
        else if (ch == '\n') lm_sb_str(b, "\\n");
        else if (ch == '\r') lm_sb_str(b, "\\r");
        else if (ch == '\t') lm_sb_str(b, "\\t");
        else if (ch == '\b') lm_sb_str(b, "\\b");
        else if (ch == '\f') lm_sb_str(b, "\\f");
        else if (ch) { snprintf(t, sizeof t, "\\u%04x", ch); lm_sb_str(b, t); }
    }
    lm_sb_add(b, s + run, n - run);
    sb_ch(b, '"');
}

static void num_text(LmSb *b, double x)
{
    char t[40];
    if (x == floor(x) && fabs(x) < 1e15) {
        snprintf(t, sizeof t, "%.0f", x == 0 ? 0.0 : x);
    } else {
        for (int prec = 15; prec <= 17; prec++) {
            snprintf(t, sizeof t, "%.*g", prec, x);
            if (strtod(t, NULL) == x) break;
        }
    }
    lm_sb_str(b, t);
}

void lm_json_dump(LmSb *b, const jval *v, int py)
{
    if (!v) { lm_sb_str(b, "null"); return; }
    switch (v->t) {
    case J_NULL: lm_sb_str(b, "null"); break;
    case J_BOOL: lm_sb_str(b, v->boolean ? "true" : "false"); break;
    case J_NUM: num_text(b, v->num); break;
    case J_STR: lm_json_string(b, v->str, strlen(v->str)); break;
    case J_ARR:
        sb_ch(b, '[');
        for (int i = 0; i < v->len; i++) {
            if (i) lm_sb_str(b, py ? ", " : ",");
            lm_json_dump(b, v->kids[i], py);
        }
        sb_ch(b, ']');
        break;
    case J_OBJ:
        sb_ch(b, '{');
        for (int i = 0; i < v->len; i++) {
            if (i) lm_sb_str(b, py ? ", " : ",");
            lm_json_string(b, v->keys[i], strlen(v->keys[i]));
            lm_sb_str(b, py ? ": " : ":");
            lm_json_dump(b, v->kids[i], py);
        }
        sb_ch(b, '}');
        break;
    }
}

/* ---- small jval helpers -------------------------------------------------------------- */

static const jval *get(const jval *o, const char *key)
{
    if (!o || o->t != J_OBJ) return NULL;
    for (int i = 0; i < o->len; i++) if (!strcmp(o->keys[i], key)) return o->kids[i];
    return NULL;
}

static int truthy(const jval *v)
{
    if (!v) return 0;
    switch (v->t) {
    case J_NULL: return 0;
    case J_BOOL: return v->boolean;
    case J_NUM: return v->num != 0;
    case J_STR: return v->str[0] != 0;
    default: return v->len > 0;
    }
}

static int is_map(const jval *v) { return v && v->t == J_OBJ; }

/* v's "type" as upper-case text, "" if it is not a string */
static const char *type_up(const jval *v, char *buf, size_t n)
{
    const jval *t = get(v, "type");
    size_t i = 0;
    if (t && t->t == J_STR)
        for (; t->str[i] && i + 1 < n; i++) buf[i] = (char)((t->str[i] >= 'a' && t->str[i] <= 'z') ? t->str[i] - 32 : t->str[i]);
    buf[i] = 0;
    return buf;
}

static int lower(int c) { return c >= 'A' && c <= 'Z' ? c + 32 : c; }

/* Indices of an object's entries in jinja's dictsort order (keys compared case-insensitively,
 * ties in input order). The caller frees. */
static int *sorted(const jval *o)
{
    int *ord = (int *)malloc((size_t)(o->len ? o->len : 1) * sizeof *ord);
    if (!ord) return NULL;
    for (int i = 0; i < o->len; i++) {
        int j = i;
        for (; j > 0; j--) {
            const char *a = o->keys[ord[j - 1]], *b = o->keys[i];
            int c = 0;
            for (; *a && *b && lower((unsigned char)*a) == lower((unsigned char)*b); a++, b++) {}
            c = lower((unsigned char)*a) - lower((unsigned char)*b);
            if (c <= 0) break;
            ord[j] = ord[j - 1];
        }
        ord[j] = i;
    }
    return ord;
}

/* ---- Qwen ---------------------------------------------------------------------------- */

void lm_qwen_arg_text(LmSb *b, const jval *v)
{
    switch (v->t) {
    case J_STR: lm_sb_str(b, v->str); break;
    case J_BOOL: lm_sb_str(b, v->boolean ? "True" : "False"); break;
    case J_NULL: lm_sb_str(b, "None"); break;
    case J_NUM: num_text(b, v->num); break;
    default: lm_json_dump(b, v, 1); break;
    }
}

/* ---- Gemma: declarations and arguments ------------------------------------------------ */

#define QS_STR "\x01"

void lm_gemma_argument(LmSb *b, const jval *v, int escape_keys)
{
    switch (v->t) {
    case J_STR: lm_sb_str(b, QS_STR); lm_sb_user(b, v->str); lm_sb_str(b, QS_STR); break;
    case J_BOOL: lm_sb_str(b, v->boolean ? "true" : "false"); break;
    case J_NULL: lm_sb_str(b, "None"); break;
    case J_NUM: num_text(b, v->num); break;
    case J_OBJ: {
        sb_ch(b, '{');
        int *ord = sorted(v);
        if (!ord) { b->oom = 1; return; }
        for (int k = 0; k < v->len; k++) {
            if (k) sb_ch(b, ',');
            if (escape_keys) { lm_sb_str(b, QS_STR); lm_sb_user(b, v->keys[ord[k]]); lm_sb_str(b, QS_STR); }
            else lm_sb_user(b, v->keys[ord[k]]);
            sb_ch(b, ':');
            lm_gemma_argument(b, v->kids[ord[k]], escape_keys);
        }
        free(ord);
        sb_ch(b, '}');
        break;
    }
    case J_ARR:
        sb_ch(b, '[');
        for (int i = 0; i < v->len; i++) {
            if (i) sb_ch(b, ',');
            lm_gemma_argument(b, v->kids[i], escape_keys);
        }
        sb_ch(b, ']');
        break;
    }
}

static void quoted(LmSb *b, const char *s) { lm_sb_str(b, QS_STR); lm_sb_user(b, s); lm_sb_str(b, QS_STR); }

static void required_list(LmSb *b, const jval *req)
{
    lm_sb_str(b, "required:[");
    for (int i = 0; i < req->len; i++) {
        if (i) sb_ch(b, ',');
        if (req->kids[i]->t == J_STR) quoted(b, req->kids[i]->str);
    }
    sb_ch(b, ']');
}

static void params(LmSb *b, const jval *props, int depth);

/* the "items:{...}" body of an array parameter */
static void items_body(LmSb *b, const jval *items, int depth)
{
    int *ord = sorted(items);
    if (!ord) { b->oom = 1; return; }
    int first = 0;
    for (int k = 0; k < items->len; k++) {
        const char *key = items->keys[ord[k]];
        const jval *iv = items->kids[ord[k]];
        if (iv->t == J_NULL) continue;
        if (first) sb_ch(b, ',');
        first = 1;
        if (!strcmp(key, "properties")) {
            lm_sb_str(b, "properties:{");
            if (is_map(iv)) params(b, iv, depth + 1);
            sb_ch(b, '}');
        } else if (!strcmp(key, "required")) {
            if (iv->t == J_ARR) required_list(b, iv);
            else lm_sb_str(b, "required:[]");
        } else if (!strcmp(key, "type")) {
            lm_sb_str(b, "type:");
            if (iv->t == J_STR) {
                char up[64];
                jval tmp = { J_STR, 0, 0, NULL, NULL, NULL, 0 };
                size_t i = 0;
                for (; iv->str[i] && i + 1 < sizeof up; i++) up[i] = (char)((iv->str[i] >= 'a' && iv->str[i] <= 'z') ? iv->str[i] - 32 : iv->str[i]);
                up[i] = 0;
                tmp.str = up;
                lm_gemma_argument(b, &tmp, 1);
            } else {
                lm_gemma_argument(b, iv, 1);
            }
        } else {
            lm_sb_user(b, key);
            sb_ch(b, ':');
            lm_gemma_argument(b, iv, 1);
        }
    }
    free(ord);
}

/* format_parameters of the jinja: the properties of an object schema, one "name:{...}" each */
static void params(LmSb *b, const jval *props, int depth)
{
    static const char *const std[] = { "description", "type", "properties", "required", "nullable" };
    if (!is_map(props) || depth > 16) return;
    int *ord = sorted(props);
    if (!ord) { b->oom = 1; return; }
    int found = 0;
    for (int k = 0; k < props->len; k++) {
        const char *key = props->keys[ord[k]];
        const jval *value = props->kids[ord[k]];
        int skip = 0;
        for (size_t s = 0; s < sizeof std / sizeof *std; s++) if (!strcmp(key, std[s])) skip = 1;
        if (skip) continue;
        if (found) sb_ch(b, ',');
        found = 1;
        lm_sb_user(b, key);
        lm_sb_str(b, ":{");
        int add = 0;
        char up[64];
        const jval *desc = get(value, "description");
        if (truthy(desc) && desc->t == J_STR) {
            lm_sb_str(b, "description:");
            quoted(b, desc->str);
            add = 1;
        }
        type_up(value, up, sizeof up);
        if (!strcmp(up, "STRING")) {
            const jval *en = get(value, "enum");
            if (truthy(en)) {
                if (add) sb_ch(b, ','); else add = 1;
                lm_sb_str(b, "enum:");
                lm_gemma_argument(b, en, 1);
            }
        } else if (!strcmp(up, "ARRAY")) {
            const jval *items = get(value, "items");
            if (is_map(items) && items->len > 0) {
                if (add) sb_ch(b, ','); else add = 1;
                lm_sb_str(b, "items:{");
                items_body(b, items, depth);
                sb_ch(b, '}');
            }
        }
        if (truthy(get(value, "nullable"))) {
            if (add) sb_ch(b, ','); else add = 1;
            lm_sb_str(b, "nullable:true");
        }
        if (!strcmp(up, "OBJECT")) {
            const jval *pp = get(value, "properties");
            if (is_map(pp)) {
                if (add) sb_ch(b, ','); else add = 1;
                lm_sb_str(b, "properties:{");
                params(b, pp, depth + 1);
                sb_ch(b, '}');
            } else if (is_map(value)) {
                if (add) sb_ch(b, ','); else add = 1;
                lm_sb_str(b, "properties:{");
                params(b, value, depth + 1);
                sb_ch(b, '}');
            }
            const jval *req = get(value, "required");
            if (truthy(req) && req->t == J_ARR) {
                if (add) sb_ch(b, ','); else add = 1;
                required_list(b, req);
            }
        }
        if (add) sb_ch(b, ',');
        lm_sb_str(b, "type:");
        quoted(b, up);
        sb_ch(b, '}');
    }
    free(ord);
}

void lm_gemma_declaration(LmSb *b, const jval *tool)
{
    const jval *fn = get(tool, "function");
    const jval *name = get(fn, "name"), *desc = get(fn, "description");
    lm_sb_str(b, "declaration:");
    if (name && name->t == J_STR) lm_sb_user(b, name->str);
    lm_sb_str(b, "{description:");
    quoted(b, desc && desc->t == J_STR ? desc->str : "");
    const jval *pr = get(fn, "parameters");
    if (truthy(pr)) {
        lm_sb_str(b, ",parameters:{");
        const jval *props = get(pr, "properties"), *req = get(pr, "required");
        if (truthy(props)) {
            lm_sb_str(b, "properties:{");
            params(b, props, 0);
            lm_sb_str(b, "},");
        }
        if (truthy(req) && req->t == J_ARR) {
            required_list(b, req);
            sb_ch(b, ',');
        }
        if (truthy(get(pr, "type"))) {
            char up[64];
            lm_sb_str(b, "type:");
            quoted(b, type_up(pr, up, sizeof up));
            sb_ch(b, '}');
        }
    }
    sb_ch(b, '}');
}

/* ---- parsing a generated Qwen call -------------------------------------------------------
 *   \n<function=NAME>\n<parameter=KEY>\nVALUE\n</parameter>\n...</function>\n */

static int is_ws(char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; }

static void fail(char *err, size_t n, const char *msg)
{
    if (err && n) snprintf(err, n, "%s", msg);
}

/* the declared type of parameter key of function name, "" when unknown */
static const char *param_type(const jval *tools, const char *name, const char *key)
{
    if (!tools || tools->t != J_ARR) return "";
    for (int i = 0; i < tools->len; i++) {
        const jval *fn = get(tools->kids[i], "function"), *nm = get(fn, "name");
        if (!nm || nm->t != J_STR || strcmp(nm->str, name)) continue;
        const jval *p = get(get(get(fn, "parameters"), "properties"), key);
        const jval *t = get(p, "type");
        if (t && t->t == J_STR) return t->str;
        if (t && t->t == J_ARR)
            for (int k = 0; k < t->len; k++)
                if (t->kids[k]->t == J_STR && strcmp(t->kids[k]->str, "null")) return t->kids[k]->str;
        return "";
    }
    return "";
}

int lm_qwen_parse_call(const char *raw, const jval *tools, LmToolCall *out, char *err, size_t errlen)
{
    const char *p = raw;
    while (is_ws(*p)) p++;
    if (strncmp(p, "<function=", 10)) { fail(err, errlen, "a tool call must start with <function=NAME>"); return -1; }
    p += 10;
    const char *gt = strchr(p, '>');
    if (!gt || gt == p) { fail(err, errlen, "the tool call has no function name"); return -1; }
    char *name = (char *)malloc((size_t)(gt - p) + 1);
    if (!name) { fail(err, errlen, "out of memory"); return -1; }
    memcpy(name, p, (size_t)(gt - p));
    name[gt - p] = 0;
    p = gt + 1;
    LmSb a = { 0 };
    sb_ch(&a, '{');
    int nargs = 0, closed = 0;
    for (;;) {
        while (is_ws(*p)) p++;
        if (!strncmp(p, "</function>", 11)) { p += 11; closed = 1; break; }
        if (strncmp(p, "<parameter=", 11)) break;
        p += 11;
        const char *kend = strchr(p, '>');
        if (!kend || kend == p) break;
        char *key = (char *)malloc((size_t)(kend - p) + 1);
        if (!key) { a.oom = 1; break; }
        memcpy(key, p, (size_t)(kend - p));
        key[kend - p] = 0;
        const char *v = kend + 1, *ve = strstr(v, "</parameter>");
        if (!ve) { free(key); break; }
        if (*v == '\n') v++;
        size_t vn = (size_t)(ve - v);
        if (vn && v[vn - 1] == '\n') vn--;
        if (nargs++) sb_ch(&a, ',');
        lm_json_string(&a, key, strlen(key));
        sb_ch(&a, ':');
        const char *ty = param_type(tools, name, key);
        char *val = (char *)malloc(vn + 1);
        if (!val) { a.oom = 1; free(key); break; }
        memcpy(val, v, vn);
        val[vn] = 0;
        int as_string = !strcmp(ty, "string") || !strcmp(ty, "") ;
        jval *parsed = as_string ? NULL : json_parse(val, NULL);
        if (parsed) { lm_json_dump(&a, parsed, 0); json_free_tree(parsed); }
        else lm_json_string(&a, val, vn);
        free(val);
        free(key);
        p = ve + 12;
    }
    sb_ch(&a, '}');
    while (closed && is_ws(*p)) p++;
    if (a.oom) { free(name); lm_sb_free(&a); fail(err, errlen, "out of memory"); return -1; }
    if (!closed || *p) {
        free(name);
        lm_sb_free(&a);
        fail(err, errlen, "the tool call is not <function=NAME> with <parameter=KEY> blocks closed by </function>");
        return -1;
    }
    out->name = name;
    out->arguments = a.s;
    return 0;
}

/* ---- parsing a generated Gemma call ------------------------------------------------------
 *   call:NAME{key:<|"|>text<|"|>,key:123,key:{k:v},key:[v,v]}   (<|"|> as LM_QS) */

typedef struct { const char *p; LmSb *out; int err; int depth; } GP;

static void gws(GP *g) { while (is_ws(*g->p)) g->p++; }
static void gvalue(GP *g);

static int json_number(const char *s, size_t n)
{
    if (!n || n > 40) return 0;
    char t[48];
    memcpy(t, s, n);
    t[n] = 0;
    size_t i = 0;
    if (t[i] == '-') i++;
    if (!(t[i] >= '0' && t[i] <= '9')) return 0;
    if (t[i] == '0' && t[i + 1] >= '0' && t[i + 1] <= '9') return 0;
    for (; i < n; i++) if (!((t[i] >= '0' && t[i] <= '9') || t[i] == '.' || t[i] == 'e' || t[i] == 'E' || t[i] == '+' || t[i] == '-')) return 0;
    char *end;
    strtod(t, &end);
    return *end == 0 && t[n - 1] != '.';
}

static void gkey(GP *g)
{
    gws(g);
    if (*g->p == LM_QS) {
        const char *e = strchr(g->p + 1, LM_QS);
        if (!e) { g->err = 1; return; }
        lm_json_string(g->out, g->p + 1, (size_t)(e - g->p - 1));
        g->p = e + 1;
    } else {
        const char *e = strchr(g->p, ':');
        if (!e || e == g->p) { g->err = 1; return; }
        const char *k = g->p, *ke = e;
        while (ke > k && is_ws(ke[-1])) ke--;
        lm_json_string(g->out, k, (size_t)(ke - k));
        g->p = e;
    }
    gws(g);
    if (*g->p != ':') { g->err = 1; return; }
    g->p++;
}

static void gobject(GP *g)
{
    /* g->p is after '{' */
    sb_ch(g->out, '{');
    int n = 0;
    for (;;) {
        gws(g);
        if (*g->p == '}') { g->p++; break; }
        if (n++) {
            if (*g->p != ',') { g->err = 1; return; }
            g->p++;
            sb_ch(g->out, ',');
            gws(g);
            if (*g->p == '}') { g->err = 1; return; } /* a trailing comma */
        }
        gkey(g);
        if (g->err) return;
        sb_ch(g->out, ':');
        gvalue(g);
        if (g->err) return;
    }
    sb_ch(g->out, '}');
}

static void gvalue(GP *g)
{
    if (++g->depth > 32) { g->err = 1; return; }
    gws(g);
    if (*g->p == LM_QS) {
        const char *e = strchr(g->p + 1, LM_QS);
        if (!e) { g->err = 1; g->depth--; return; }
        lm_json_string(g->out, g->p + 1, (size_t)(e - g->p - 1));
        g->p = e + 1;
    } else if (*g->p == '{') {
        g->p++;
        gobject(g);
    } else if (*g->p == '[') {
        g->p++;
        sb_ch(g->out, '[');
        int n = 0;
        for (;;) {
            gws(g);
            if (*g->p == ']') { g->p++; break; }
            if (n++) {
                if (*g->p != ',') { g->err = 1; break; }
                g->p++;
                sb_ch(g->out, ',');
            }
            gvalue(g);
            if (g->err) break;
        }
        sb_ch(g->out, ']');
    } else {
        const char *s = g->p, *e = s;
        while (*e && *e != ',' && *e != '}' && *e != ']') e++;
        const char *te = e;
        while (te > s && is_ws(te[-1])) te--;
        size_t n = (size_t)(te - s);
        if (!n) g->err = 1;
        else if ((n == 4 && !memcmp(s, "true", 4)) || (n == 5 && !memcmp(s, "false", 5)) || (n == 4 && !memcmp(s, "null", 4)))
            lm_sb_add(g->out, s, n);
        else if (n == 4 && !memcmp(s, "None", 4)) lm_sb_str(g->out, "null");
        else if (json_number(s, n)) lm_sb_add(g->out, s, n);
        else lm_json_string(g->out, s, n);
        g->p = e;
    }
    g->depth--;
}

int lm_gemma_parse_call(const char *raw, LmToolCall *out, char *err, size_t errlen)
{
    const char *p = raw;
    while (is_ws(*p)) p++;
    if (strncmp(p, "call:", 5)) { fail(err, errlen, "a tool call must start with call:NAME{...}"); return -1; }
    p += 5;
    const char *br = strchr(p, '{');
    if (!br || br == p) { fail(err, errlen, "the tool call has no function name or no argument braces"); return -1; }
    const char *ne = br;
    while (ne > p && is_ws(ne[-1])) ne--;
    char *name = (char *)malloc((size_t)(ne - p) + 1);
    if (!name) { fail(err, errlen, "out of memory"); return -1; }
    memcpy(name, p, (size_t)(ne - p));
    name[ne - p] = 0;
    LmSb a = { 0 };
    GP g = { br + 1, &a, 0, 0 };
    gobject(&g);
    if (!g.err) { gws(&g); if (*g.p) g.err = 1; }
    if (a.oom || g.err) {
        free(name);
        lm_sb_free(&a);
        fail(err, errlen, a.oom ? "out of memory" : "the tool call arguments are not call:NAME{key:value,...}");
        return -1;
    }
    out->name = name;
    out->arguments = a.s ? a.s : strdup("{}");
    return 0;
}
