/* lm_template.c -- Qwen3.5 chat format. Original to this repository.
 *
 * Normative source: Qwen3.5-4B's chat_template.jinja, text-only path (no tools, no
 * images). Everything below mirrors that file; the comments name the jinja it mirrors.
 *
 * The jinja produces one string that the Hugging Face tokenizer then splits on added
 * tokens. Here the structure is built directly: the template's own markers are emitted
 * as special ids looked up by content, and every piece of message text goes through
 * lm_tok_encode, which never yields a special id. So a user who types "<|im_start|>"
 * gets those characters as ordinary tokens, where Hugging Face would turn the literal into
 * the control token. Text between two markers is encoded as one string, exactly the
 * segments Hugging Face would produce for a well-formed conversation. */
#include "chat/lm_template.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- id vector ---------------------------------------------------------------- */

int lm_ids_append(LmIds *v, const int *ids, int n)
{
    if (n <= 0) return 0;
    if (v->n + n > v->cap) {
        int cap = v->cap ? v->cap : 64;
        while (cap < v->n + n) cap *= 2;
        int *p = (int *)realloc(v->ids, (size_t)cap * sizeof *p);
        if (!p) return -1;
        v->ids = p;
        v->cap = cap;
    }
    memcpy(v->ids + v->n, ids, (size_t)n * sizeof *ids);
    v->n += n;
    return 0;
}

int lm_ids_push(LmIds *v, int id) { return lm_ids_append(v, &id, 1); }

void lm_ids_free(LmIds *v)
{
    free(v->ids);
    v->ids = NULL;
    v->n = v->cap = 0;
}

/* ---- string helpers (Python semantics) ------------------------------------------ */

typedef struct { const char *p; size_t n; } Sl;

static Sl sl_of(const char *s) { Sl r = { s ? s : "", s ? strlen(s) : 0 }; return r; }

/* Length of the whitespace character ending at p[n] going backwards, or starting at p[0]
 * going forwards: Python's str.isspace() set (ASCII, U+0085, U+00A0, U+1680, U+2000-200A,
 * U+2028/2029, U+202F, U+205F, U+3000). Returns the byte length, 0 if not whitespace. */
static int ws_fwd(const char *p, size_t n)
{
    unsigned char c = (unsigned char)p[0];
    if (c == ' ' || (c >= 9 && c <= 13) || (c >= 0x1C && c <= 0x1F)) return 1;
    if (c == 0xC2 && n >= 2 && ((unsigned char)p[1] == 0x85 || (unsigned char)p[1] == 0xA0)) return 2;
    if (n >= 3) {
        unsigned char d = (unsigned char)p[1], e = (unsigned char)p[2];
        if (c == 0xE1 && d == 0x9A && e == 0x80) return 3;                     /* U+1680 */
        if (c == 0xE2 && d == 0x80 && ((e >= 0x80 && e <= 0x8A) || e == 0xA8 || e == 0xA9 || e == 0xAF)) return 3;
        if (c == 0xE2 && d == 0x81 && e == 0x9F) return 3;                     /* U+205F */
        if (c == 0xE3 && d == 0x80 && e == 0x80) return 3;                     /* U+3000 */
    }
    return 0;
}

static int ws_back(const char *p, size_t n)
{
    for (int k = 1; k <= 3 && (size_t)k <= n; k++) {
        if (ws_fwd(p + n - (size_t)k, (size_t)k) == k) return k;
    }
    return 0;
}

static Sl sl_trim(Sl s)
{
    for (;;) { int k = s.n ? ws_fwd(s.p, s.n) : 0; if (!k) break; s.p += k; s.n -= (size_t)k; }
    for (;;) { int k = s.n ? ws_back(s.p, s.n) : 0; if (!k) break; s.n -= (size_t)k; }
    return s;
}

static Sl sl_lstrip_nl(Sl s) { while (s.n && s.p[0] == '\n') { s.p++; s.n--; } return s; }
static Sl sl_rstrip_nl(Sl s) { while (s.n && s.p[s.n - 1] == '\n') s.n--; return s; }

static const char *sl_find(Sl s, const char *needle)
{
    size_t m = strlen(needle);
    if (m > s.n) return NULL;
    for (size_t i = 0; i + m <= s.n; i++) if (!memcmp(s.p + i, needle, m)) return s.p + i;
    return NULL;
}

static const char *sl_rfind(Sl s, const char *needle)
{
    size_t m = strlen(needle);
    if (m > s.n) return NULL;
    for (size_t i = s.n - m + 1; i-- > 0;) if (!memcmp(s.p + i, needle, m)) return s.p + i;
    return NULL;
}

/* ---- the implementation object -------------------------------------------------- */

#define MAX_STOP 8

typedef struct {
    const LmTok *tok;
    int im_start, im_end, think, think_end, tool_call, tool_call_end;
    int stop[MAX_STOP];
    int n_stop;
} Qwen;

/* ---- builder: text segments between special ids --------------------------------- */

typedef struct {
    const Qwen *q;
    LmIds *out;
    char *txt;
    size_t n, cap;
    int oom;
} Bld;

static void b_text_n(Bld *b, const char *s, size_t n)
{
    if (b->oom || !n) return;
    if (b->n + n + 1 > b->cap) {
        size_t cap = b->cap ? b->cap : 256;
        while (cap < b->n + n + 1) cap *= 2;
        char *p = (char *)realloc(b->txt, cap);
        if (!p) { b->oom = 1; return; }
        b->txt = p;
        b->cap = cap;
    }
    memcpy(b->txt + b->n, s, n);
    b->n += n;
}

static void b_text(Bld *b, const char *s) { b_text_n(b, s, strlen(s)); }
static void b_sl(Bld *b, Sl s) { b_text_n(b, s.p, s.n); }

static void b_flush(Bld *b)
{
    if (b->oom || !b->n) return;
    int n = 0;
    int *ids = lm_tok_encode(b->q->tok, b->txt, b->n, &n);
    if (!ids || lm_ids_append(b->out, ids, n)) b->oom = 1;
    free(ids);
    b->n = 0;
}

static void b_special(Bld *b, int id)
{
    b_flush(b);
    if (!b->oom && lm_ids_push(b->out, id)) b->oom = 1;
}

/* '<|im_start|>assistant\n' + ('<think>\n' | '<think>\n\n</think>\n\n') -- the generation
 * prompt, and, for verbatim history, the opening of an assistant turn. */
static void b_assistant_open(Bld *b, int thinking)
{
    b_special(b, b->q->im_start);
    b_text(b, "assistant\n");
    b_special(b, b->q->think);
    if (thinking) {
        b_text(b, "\n");
    } else {
        b_text(b, "\n\n");
        b_special(b, b->q->think_end);
        b_text(b, "\n\n");
    }
}

static int q_render(const LmTemplate *t, const LmMsg *msgs, int n, const LmRenderOpts *opts,
                    LmIds *out, char *err, size_t errlen)
{
    const Qwen *q = (const Qwen *)t->impl;
#define FAIL(m) do { if (err && errlen) snprintf(err, errlen, "%s", m); return -1; } while (0)
    out->n = 0;
    if (n <= 0) FAIL("No messages provided.");

    /* last_query_index: the last user message that is not a tool response. Assistant
     * turns after it keep their reasoning in the prompt; earlier ones lose it. */
    int last_query = -1;
    for (int i = n - 1; i >= 0 && last_query < 0; i--) {
        if (msgs[i].role != LM_ROLE_USER) continue;
        Sl c = sl_trim(sl_of(msgs[i].content));
        static const char open[] = "<tool_response>", close[] = "</tool_response>";
        int is_resp = c.n >= sizeof open - 1 && !memcmp(c.p, open, sizeof open - 1) &&
                      c.n >= sizeof close - 1 && !memcmp(c.p + c.n - (sizeof close - 1), close, sizeof close - 1);
        if (!is_resp) last_query = i;
    }
    if (last_query < 0) FAIL("No user query found in messages.");
    for (int i = 1; i < n; i++)
        if (msgs[i].role == LM_ROLE_SYSTEM) FAIL("System message must be at the beginning.");

    Bld b = { q, out, NULL, 0, 0, 0 };
    for (int i = 0; i < n; i++) {
        const LmMsg *m = &msgs[i];
        Sl content = sl_trim(sl_of(m->content));
        if (m->role == LM_ROLE_SYSTEM) {
            b_special(&b, q->im_start);
            b_text(&b, "system\n");
            b_sl(&b, content);
            b_special(&b, q->im_end);
            b_text(&b, "\n");
        } else if (m->role == LM_ROLE_USER) {
            b_special(&b, q->im_start);
            b_text(&b, "user\n");
            b_sl(&b, content);
            b_special(&b, q->im_end);
            b_text(&b, "\n");
        } else {
            if (opts->verbatim_history && m->gen_ids) {
                b_assistant_open(&b, m->gen_thinking);
                b_flush(&b);
                if (!b.oom && lm_ids_append(out, m->gen_ids, m->n_gen_ids)) b.oom = 1;
            } else {
                Sl reasoning = { "", 0 };
                if (m->reasoning) {
                    reasoning = sl_of(m->reasoning);
                } else if (sl_find(content, "</think>")) {
                    /* reasoning = content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n')
                     * content   = content.split('</think>')[-1].lstrip('\n') */
                    const char *first = sl_find(content, "</think>");
                    Sl head = { content.p, (size_t)(first - content.p) };
                    head = sl_rstrip_nl(head);
                    const char *open = sl_rfind(head, "<think>");
                    if (open) { head.n -= (size_t)(open + 7 - head.p); head.p = open + 7; }
                    reasoning = sl_lstrip_nl(head);
                    const char *last = sl_rfind(content, "</think>");
                    content.n -= (size_t)(last + 8 - content.p);
                    content.p = last + 8;
                    content = sl_lstrip_nl(content);
                }
                reasoning = sl_trim(reasoning);
                if (i > last_query) {
                    b_special(&b, q->im_start);
                    b_text(&b, "assistant\n");
                    b_special(&b, q->think);
                    b_text(&b, "\n");
                    b_sl(&b, reasoning);
                    b_text(&b, "\n");
                    b_special(&b, q->think_end);
                    b_text(&b, "\n\n");
                    b_sl(&b, content);
                } else {
                    b_special(&b, q->im_start);
                    b_text(&b, "assistant\n");
                    b_sl(&b, content);
                }
            }
            b_special(&b, q->im_end);
            b_text(&b, "\n");
        }
    }
    b_assistant_open(&b, opts->enable_thinking);
    b_flush(&b);
    free(b.txt);
    if (b.oom) FAIL("out of memory");
    return 0;
#undef FAIL
}

/* ---- streaming parser ------------------------------------------------------------ */

#define HELD_MAX 4096

typedef struct {
    int started;
    char *held;
    size_t nheld, cap;
} Strm;

enum { ST_THINK, ST_ANSWER, ST_TOOL };

typedef struct {
    LmParser base;
    const Qwen *q;
    int state;
    LmUtf8 u;
    Strm reason, answer;
} QParser;

static int is_ws_byte(char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f'; }

static void strm_reset(Strm *s) { s->started = 0; s->nheld = 0; }

/* Forward bytes, withholding leading whitespace (dropped) and trailing whitespace (held
 * until something follows it; dropped if nothing does). */
static void strm_push(Strm *s, LmOutKind kind, const char *in, size_t n, LmSink sink, void *user)
{
    char *out = (char *)malloc(s->nheld + n + 1);
    if (!out) return;
    size_t o = 0;
    for (size_t i = 0; i < n; i++) {
        char c = in[i];
        if (is_ws_byte(c)) {
            if (!s->started) continue;
            if (s->nheld == HELD_MAX) { memcpy(out + o, s->held, s->nheld); o += s->nheld; s->nheld = 0; }
            if (s->nheld == s->cap) {
                size_t cap = s->cap ? s->cap * 2 : 64;
                char *p = (char *)realloc(s->held, cap);
                if (!p) continue;
                s->held = p;
                s->cap = cap;
            }
            s->held[s->nheld++] = c;
        } else {
            s->started = 1;
            if (s->nheld) { memcpy(out + o, s->held, s->nheld); o += s->nheld; s->nheld = 0; }
            out[o++] = c;
        }
    }
    if (o && sink) sink(user, kind, out, o);
    free(out);
}

static void route(QParser *p, const char *bytes, size_t n, LmSink sink, void *user)
{
    if (!n) return;
    if (p->state == ST_THINK) strm_push(&p->reason, LM_OUT_REASONING, bytes, n, sink, user);
    else if (p->state == ST_ANSWER) strm_push(&p->answer, LM_OUT_ANSWER, bytes, n, sink, user);
    /* ST_TOOL: the body of an unsupported tool call is not forwarded */
}

static void flush_utf8(QParser *p, LmSink sink, void *user)
{
    char tmp[3];
    size_t k = lm_utf8_flush(&p->u, tmp);
    route(p, tmp, k, sink, user);
}

static void enter(QParser *p, int state)
{
    p->state = state;
    p->base.thinking_open = state == ST_THINK;
    if (state == ST_THINK) strm_reset(&p->reason);
    if (state == ST_ANSWER) strm_reset(&p->answer);
}

static int qp_feed(LmParser *base, int id, LmSink sink, void *user)
{
    QParser *p = (QParser *)base;
    const Qwen *q = p->q;
    for (int i = 0; i < q->n_stop; i++) {
        if (q->stop[i] == id) { flush_utf8(p, sink, user); return LM_PARSE_STOP; }
    }
    if (id == q->think_end) {
        flush_utf8(p, sink, user);
        if (p->state == ST_THINK) enter(p, ST_ANSWER);
        return LM_PARSE_CONT;
    }
    if (id == q->think) {
        flush_utf8(p, sink, user);
        if (p->state != ST_TOOL) enter(p, ST_THINK);
        return LM_PARSE_CONT;
    }
    if (id == q->tool_call) {
        flush_utf8(p, sink, user);
        if (p->state != ST_TOOL) {
            enter(p, ST_TOOL);
            p->base.tool_calls++;
            static const char note[] = "tool calls are not supported";
            if (sink) sink(user, LM_OUT_TOOLCALL, note, sizeof note - 1);
        }
        return LM_PARSE_CONT;
    }
    if (id == q->tool_call_end) {
        flush_utf8(p, sink, user);
        return p->state == ST_TOOL ? LM_PARSE_STOP_AFTER : LM_PARSE_CONT;
    }
    if (lm_tok_is_special(q->tok, id)) return LM_PARSE_CONT; /* control token: no text */
    char piece[LM_TOK_PIECE_MAX];
    int n = lm_tok_piece(q->tok, id, piece, sizeof piece);
    if (n <= 0) return LM_PARSE_CONT;
    char *text = (char *)malloc(LM_UTF8_OUT_MAX(n));
    if (!text) return LM_PARSE_CONT;
    size_t k = lm_utf8_push(&p->u, piece, (size_t)n, text);
    route(p, text, k, sink, user);
    free(text);
    return LM_PARSE_CONT;
}

static void qp_finish(LmParser *base, LmSink sink, void *user)
{
    flush_utf8((QParser *)base, sink, user);
}

static void qp_destroy(LmParser *base)
{
    QParser *p = (QParser *)base;
    if (!p) return;
    free(p->reason.held);
    free(p->answer.held);
    free(p);
}

static LmParser *q_parser_new(const LmTemplate *t, int thinking)
{
    QParser *p = (QParser *)calloc(1, sizeof *p);
    if (!p) return NULL;
    p->q = (const Qwen *)t->impl;
    p->base.feed = qp_feed;
    p->base.finish = qp_finish;
    p->base.destroy = qp_destroy;
    lm_utf8_init(&p->u);
    enter(p, thinking ? ST_THINK : ST_ANSWER);
    return &p->base;
}

static int q_is_stop(const LmTemplate *t, int id)
{
    const Qwen *q = (const Qwen *)t->impl;
    for (int i = 0; i < q->n_stop; i++) if (q->stop[i] == id) return 1;
    return 0;
}

static void q_destroy(LmTemplate *t)
{
    free(t->impl);
}

LmTemplate *lm_template_qwen35(const LmTok *tok, const int *eos_ids, int n_eos, char *err, size_t errlen)
{
    Qwen *q = (Qwen *)calloc(1, sizeof *q);
    LmTemplate *t = (LmTemplate *)calloc(1, sizeof *t);
    if (!q || !t) {
        free(q); free(t);
        if (err && errlen) snprintf(err, errlen, "out of memory");
        return NULL;
    }
    q->tok = tok;
    if (lm_tok_require(tok, "<|im_start|>", &q->im_start, err, errlen) ||
        lm_tok_require(tok, "<|im_end|>", &q->im_end, err, errlen) ||
        lm_tok_require(tok, "<think>", &q->think, err, errlen) ||
        lm_tok_require(tok, "</think>", &q->think_end, err, errlen) ||
        lm_tok_require(tok, "<tool_call>", &q->tool_call, err, errlen) ||
        lm_tok_require(tok, "</tool_call>", &q->tool_call_end, err, errlen)) {
        free(q); free(t);
        return NULL;
    }
    q->stop[q->n_stop++] = q->im_end;
    for (int i = 0; i < n_eos; i++) {
        int dup = 0;
        for (int j = 0; j < q->n_stop; j++) if (q->stop[j] == eos_ids[i]) dup = 1;
        if (!dup && eos_ids[i] >= 0 && q->n_stop < MAX_STOP) q->stop[q->n_stop++] = eos_ids[i];
    }
    t->name = "qwen3.5";
    t->render = q_render;
    t->parser_new = q_parser_new;
    t->is_stop = q_is_stop;
    t->destroy = q_destroy;
    t->impl = q;
    return t;
}

void lm_template_free(LmTemplate *t)
{
    if (!t) return;
    if (t->destroy) t->destroy(t);
    free(t);
}
