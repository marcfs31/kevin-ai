/* lm_template.c -- Qwen3.5 and Gemma 4 chat formats. Original to this repository.
 *
 * Normative sources: Qwen3.5-4B's chat_template.jinja and Gemma 4's chat_template.jinja,
 * text-only path (no tools, no images). Everything below mirrors those files; the comments
 * name the jinja they mirror.
 *
 * The jinja produces one string that the Hugging Face tokenizer then splits on added
 * tokens. Here the structure is built directly: the template's own markers are emitted
 * as special ids looked up by content, and every piece of message text goes through
 * lm_tok_encode, which never yields a special id. So a user who types "<|im_start|>"
 * gets those characters as ordinary tokens, where Hugging Face would turn the literal into
 * the control token. Text between two markers is encoded as one string, exactly the
 * segments Hugging Face would produce for a well-formed conversation. */
#include "chat/lm_template.h"

#include "chat/lm_toolfmt.h"

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
    int im_start, im_end, think, think_end, tool_call, tool_call_end, tool_resp, tool_resp_end;
    int vis_start, vis_pad, vis_end; /* -1 = vision unset */
    int stop[MAX_STOP];
    int n_stop;
} Qwen;

static int span_push(const LmRenderOpts *opts, int id_start, int n_ids, int img_index)
{
    if (!opts || !opts->spans || !opts->n_spans) return 0;
    const int n = *opts->n_spans;
    LmImgSpan *p = (LmImgSpan *)realloc(*opts->spans, (size_t)(n + 1) * sizeof *p);
    if (!p) return -1;
    p[n].id_start = id_start;
    p[n].n_ids = n_ids;
    p[n].img_index = img_index;
    *opts->spans = p;
    *opts->n_spans = n + 1;
    return 0;
}

/* ---- builder: text segments between special ids --------------------------------- */

typedef struct {
    const LmTok *tok;
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
static void b_special(Bld *b, int id);
static void b_sl(Bld *b, Sl s) { b_text_n(b, s.p, s.n); }

/* Fixed template text that names the model's own markers: each literal <tool_call> and
 * </tool_call> becomes its id, as the Hugging Face tokenizer does with the jinja's output. */
static void b_marked_calls(Bld *b, int open_id, int close_id, const char *s)
{
    for (;;) {
        const char *o = strstr(s, "<tool_call>"), *c = strstr(s, "</tool_call>");
        const char *at = !o ? c : !c ? o : (o < c ? o : c);
        if (!at) break;
        b_text_n(b, s, (size_t)(at - s));
        if (at == o) { b_special(b, open_id); s = at + 11; }
        else { b_special(b, close_id); s = at + 12; }
    }
    b_text(b, s);
}

static void b_flush(Bld *b)
{
    if (b->oom || !b->n) return;
    int n = 0;
    int *ids = lm_tok_encode(b->tok, b->txt, b->n, &n);
    if (!ids || lm_ids_append(b->out, ids, n)) b->oom = 1;
    free(ids);
    b->n = 0;
}

static void b_special(Bld *b, int id)
{
    b_flush(b);
    if (!b->oom && lm_ids_push(b->out, id)) b->oom = 1;
}

/* Emit user content: plain text (trimmed) or multimodal parts. *img_i is the running
 * image index across the conversation. Returns 0, or -1 with a message. */
static int emit_user_content(Bld *b, const LmMsg *m, int vis_start, int vis_pad, int vis_end,
                             const LmRenderOpts *opts, int *img_i, char *err, size_t errlen)
{
    if (!m->parts || m->n_parts <= 0) {
        b_sl(b, sl_trim(sl_of(m->content)));
        return 0;
    }
    for (int i = 0; i < m->n_parts; i++) {
        const LmMsgPart *p = &m->parts[i];
        if (p->kind == LM_PART_TEXT) {
            if (p->text) b_text(b, p->text);
            continue;
        }
        if (p->kind != LM_PART_IMAGE) {
            if (err && errlen) snprintf(err, errlen, "unknown content part kind");
            return -1;
        }
        if (vis_start < 0 || vis_pad < 0 || vis_end < 0) {
            if (err && errlen) snprintf(err, errlen, "vision tokens are not configured");
            return -1;
        }
        if (p->n_rows < 1) {
            if (err && errlen) snprintf(err, errlen, "image part has no rows");
            return -1;
        }
        b_flush(b);
        const int id_start = b->out->n;
        if (lm_ids_push(b->out, vis_start)) { b->oom = 1; return -1; }
        for (int k = 0; k < p->n_rows; k++) {
            if (lm_ids_push(b->out, vis_pad)) { b->oom = 1; return -1; }
        }
        if (lm_ids_push(b->out, vis_end)) { b->oom = 1; return -1; }
        if (span_push(opts, id_start, p->n_rows + 2, *img_i)) {
            if (err && errlen) snprintf(err, errlen, "out of memory");
            return -1;
        }
        (*img_i)++;
    }
    return 0;
}

/* '<|im_start|>assistant\n' + ('<think>\n' | '<think>\n\n</think>\n\n') -- the generation
 * prompt, and, for verbatim history, the opening of an assistant turn. */
static void b_assistant_open(Bld *b, const Qwen *q, int thinking)
{
    b_special(b, q->im_start);
    b_text(b, "assistant\n");
    b_special(b, q->think);
    if (thinking) {
        b_text(b, "\n");
    } else {
        b_text(b, "\n\n");
        b_special(b, q->think_end);
        b_text(b, "\n\n");
    }
}

static int q_render(const LmTemplate *t, const LmMsg *msgs, int n, const LmRenderOpts *opts,
                    LmIds *out, char *err, size_t errlen)
{
    const Qwen *q = (const Qwen *)t->impl;
#define FAIL(m) do { if (err && errlen) snprintf(err, errlen, "%s", m); return -1; } while (0)
    out->n = 0;
    if (opts && opts->n_spans) *opts->n_spans = 0;
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
    for (int i = 0; i < n; i++) {
        if (msgs[i].role == LM_ROLE_SYSTEM && msgs[i].parts && msgs[i].n_parts > 0) {
            for (int k = 0; k < msgs[i].n_parts; k++)
                if (msgs[i].parts[k].kind == LM_PART_IMAGE)
                    FAIL("System messages may not contain images.");
        }
    }

    Bld b = { q->tok, out, NULL, 0, 0, 0 };
    int img_i = 0;
    const int has_tools = opts && opts->tools && opts->tools->t == J_ARR && opts->tools->len > 0;
    if (has_tools) {
        b_special(&b, q->im_start);
        b_text(&b, "system\n# Tools\n\nYou have access to the following functions:\n\n<tools>");
        for (int k = 0; k < opts->tools->len; k++) {
            LmSb d = { 0 };
            lm_json_dump(&d, opts->tools->kids[k], 1);
            b_text(&b, "\n");
            if (d.oom) b.oom = 1; else b_text_n(&b, d.s, d.n);
            lm_sb_free(&d);
        }
        b_text(&b, "\n</tools>");
        b_marked_calls(&b, q->tool_call, q->tool_call_end,
                       "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
                       "<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n"
                       "</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\n"
                       "that can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\n"
                       "Reminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> "
                       "block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be "
                       "specified\n- You may provide optional reasoning for your function call in natural language BEFORE "
                       "the function call, but NOT after\n- If there is no function call available, answer the question "
                       "like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>");
        if (msgs[0].role == LM_ROLE_SYSTEM) {
            Sl sc = sl_trim(sl_of(msgs[0].content));
            if (sc.n) { b_text(&b, "\n\n"); b_sl(&b, sc); }
        }
        b_special(&b, q->im_end);
        b_text(&b, "\n");
    }
    for (int i = 0; i < n; i++) {
        const LmMsg *m = &msgs[i];
        Sl content = sl_trim(sl_of(m->content));
        if (m->role == LM_ROLE_SYSTEM) {
            if (has_tools) continue;
            b_special(&b, q->im_start);
            b_text(&b, "system\n");
            b_sl(&b, content);
            b_special(&b, q->im_end);
            b_text(&b, "\n");
        } else if (m->role == LM_ROLE_USER) {
            b_special(&b, q->im_start);
            b_text(&b, "user\n");
            if (emit_user_content(&b, m, q->vis_start, q->vis_pad, q->vis_end, opts, &img_i, err, errlen)) {
                free(b.txt);
                return -1;
            }
            b_special(&b, q->im_end);
            b_text(&b, "\n");
        } else if (m->role == LM_ROLE_ASSISTANT) {
            if (opts->verbatim_history && m->gen_ids) {
                b_assistant_open(&b, q, m->gen_thinking);
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
                for (int k = 0; k < m->n_calls; k++) {
                    jval *args = json_parse(m->calls[k].arguments ? m->calls[k].arguments : "{}", NULL);
                    if (!args || args->t != J_OBJ) {
                        json_free_tree(args);
                        free(b.txt);
                        FAIL("a tool call's arguments must be a JSON object");
                    }
                    if (k == 0) { if (content.n) b_text(&b, "\n\n"); }
                    else b_text(&b, "\n");
                    b_special(&b, q->tool_call);
                    b_text(&b, "\n<function=");
                    b_text(&b, m->calls[k].name);
                    b_text(&b, ">\n");
                    for (int a = 0; a < args->len; a++) {
                        LmSb v = { 0 };
                        lm_qwen_arg_text(&v, args->kids[a]);
                        b_text(&b, "<parameter=");
                        b_text(&b, args->keys[a]);
                        b_text(&b, ">\n");
                        if (v.oom) b.oom = 1; else b_text_n(&b, v.s, v.n);
                        lm_sb_free(&v);
                        b_text(&b, "\n</parameter>\n");
                    }
                    b_text(&b, "</function>\n");
                    b_special(&b, q->tool_call_end);
                    json_free_tree(args);
                }
            }
            b_special(&b, q->im_end);
            b_text(&b, "\n");
        } else {
            /* a tool message: consecutive ones share one user turn of <tool_response> blocks */
            if (i == 0 || msgs[i - 1].role != LM_ROLE_TOOL) {
                b_special(&b, q->im_start);
                b_text(&b, "user");
            }
            b_text(&b, "\n");
            b_special(&b, q->tool_resp);
            b_text(&b, "\n");
            b_sl(&b, content);
            b_text(&b, "\n");
            b_special(&b, q->tool_resp_end);
            if (i == n - 1 || msgs[i + 1].role != LM_ROLE_TOOL) {
                b_special(&b, q->im_end);
                b_text(&b, "\n");
            }
        }
    }
    b_assistant_open(&b, q, opts->enable_thinking);
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

/* What the parser needs to know about a family's markers. */
typedef struct {
    const LmTok *tok;
    int think, think_end, tool_call, tool_call_end;
    int stop[MAX_STOP];
    int n_stop;
    /* Gemma: the text after the think-open id up to the first newline names the channel
     * ("thought") and is not reasoning. */
    int channel_header;
    int quote; /* Gemma <|"|>: kept as LM_QS inside a collected tool call; -1 = none */
} PFmt;

typedef struct {
    LmParser base;
    PFmt f;
    int hdr_pending; /* still inside the channel name */
    int state;
    LmUtf8 u;
    Strm reason, answer;
    char *tb;        /* the tool call being collected (LmParser.collect_calls) */
    size_t tn, tcap;
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

static void tb_add(QParser *p, const char *s, size_t n)
{
    if (p->tn + n + 1 > p->tcap) {
        size_t cap = p->tcap ? p->tcap : 256;
        while (cap < p->tn + n + 1) cap *= 2;
        char *q = (char *)realloc(p->tb, cap);
        if (!q) return;
        p->tb = q;
        p->tcap = cap;
    }
    memcpy(p->tb + p->tn, s, n);
    p->tn += n;
    p->tb[p->tn] = 0;
}

/* hand the collected call body to the sink */
static void tb_emit(QParser *p, LmSink sink, void *user)
{
    if (p->tn && sink) sink(user, LM_OUT_TOOLCALL, p->tb, p->tn);
    p->tn = 0;
}

static void route(QParser *p, const char *bytes, size_t n, LmSink sink, void *user)
{
    if (p->hdr_pending) {
        const char *nl = (const char *)memchr(bytes, '\n', n);
        if (!nl) return;
        p->hdr_pending = 0;
        n -= (size_t)(nl + 1 - bytes);
        bytes = nl + 1;
    }
    if (!n) return;
    if (p->state == ST_THINK) strm_push(&p->reason, LM_OUT_REASONING, bytes, n, sink, user);
    else if (p->state == ST_ANSWER) strm_push(&p->answer, LM_OUT_ANSWER, bytes, n, sink, user);
    else if (p->state == ST_TOOL && p->base.collect_calls) tb_add(p, bytes, n);
    /* ST_TOOL otherwise: the body of an unsupported tool call is not forwarded */
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
    p->hdr_pending = state == ST_THINK && p->f.channel_header;
    if (state == ST_THINK) strm_reset(&p->reason);
    if (state == ST_ANSWER) strm_reset(&p->answer);
}

static int qp_feed(LmParser *base, int id, LmSink sink, void *user)
{
    QParser *p = (QParser *)base;
    const PFmt *q = &p->f;
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
            p->tn = 0;
            p->base.tool_calls++;
            static const char note[] = "tool calls are not supported";
            if (sink && !p->base.collect_calls) sink(user, LM_OUT_TOOLCALL, note, sizeof note - 1);
        }
        return LM_PARSE_CONT;
    }
    if (id == q->tool_call_end) {
        flush_utf8(p, sink, user);
        if (p->state != ST_TOOL) return LM_PARSE_CONT;
        if (!p->base.collect_calls) return LM_PARSE_STOP_AFTER;
        tb_emit(p, sink, user);
        enter(p, ST_ANSWER);
        return LM_PARSE_CONT;
    }
    if (q->quote >= 0 && id == q->quote && p->state == ST_TOOL && p->base.collect_calls) {
        flush_utf8(p, sink, user);
        const char qs = LM_QS;
        tb_add(p, &qs, 1);
        return LM_PARSE_CONT;
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
    QParser *p = (QParser *)base;
    flush_utf8(p, sink, user);
    if (p->state == ST_TOOL && p->base.collect_calls) tb_emit(p, sink, user); /* a call cut short */
}

static void qp_destroy(LmParser *base)
{
    QParser *p = (QParser *)base;
    if (!p) return;
    free(p->reason.held);
    free(p->answer.held);
    free(p->tb);
    free(p);
}

static LmParser *parser_new(const PFmt *f, int start_in_think)
{
    QParser *p = (QParser *)calloc(1, sizeof *p);
    if (!p) return NULL;
    p->f = *f;
    p->base.feed = qp_feed;
    p->base.finish = qp_finish;
    p->base.destroy = qp_destroy;
    lm_utf8_init(&p->u);
    enter(p, start_in_think ? ST_THINK : ST_ANSWER);
    return &p->base;
}

static LmParser *q_parser_new(const LmTemplate *t, int thinking)
{
    const Qwen *q = (const Qwen *)t->impl;
    PFmt f = { q->tok, q->think, q->think_end, q->tool_call, q->tool_call_end, { 0 }, q->n_stop, 0, -1 };
    memcpy(f.stop, q->stop, sizeof f.stop);
    return parser_new(&f, thinking); /* the Qwen generation prompt itself opens the block */
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

static int q_parse_call(const LmTemplate *t, const char *raw, const jval *tools, LmToolCall *out,
                        char *err, size_t errlen)
{
    (void)t;
    return lm_qwen_parse_call(raw, tools, out, err, errlen);
}

void lm_toolcall_free(LmToolCall *c)
{
    free((void *)c->name);
    free((void *)c->arguments);
    c->name = c->arguments = NULL;
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
    q->vis_start = q->vis_pad = q->vis_end = -1;
    if (lm_tok_require(tok, "<|im_start|>", &q->im_start, err, errlen) ||
        lm_tok_require(tok, "<|im_end|>", &q->im_end, err, errlen) ||
        lm_tok_require(tok, "<think>", &q->think, err, errlen) ||
        lm_tok_require(tok, "</think>", &q->think_end, err, errlen) ||
        lm_tok_require(tok, "<tool_call>", &q->tool_call, err, errlen) ||
        lm_tok_require(tok, "</tool_call>", &q->tool_call_end, err, errlen) ||
        lm_tok_require(tok, "<tool_response>", &q->tool_resp, err, errlen) ||
        lm_tok_require(tok, "</tool_response>", &q->tool_resp_end, err, errlen)) {
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
    t->parse_call = q_parse_call;
    t->destroy = q_destroy;
    t->impl = q;
    return t;
}

/* ---- Gemma 4 --------------------------------------------------------------------
 * Mirrors chat_template.jinja of google/gemma-4-E4B-it, text only:
 *   <bos>
 *   [<|turn>system\n [<|think|>\n] trim(system) <turn|>\n]      when thinking or a system message
 *   (<|turn>user|model\n trim(content) <turn|>\n)*
 *   <|turn>model\n                                              the generation prompt
 * Thinking is a flag of the system turn, not of the generation prompt. A model turn's
 * content passes through strip_thinking: every <|channel>...<channel|> block is dropped. */

typedef struct {
    const LmTok *tok;
    int bos, turn, turn_end, think, channel, channel_end, tool_call, tool_call_end;
    int tool_open, tool_close, tool_resp, tool_resp_end, quote;
    int vis_start, vis_pad, vis_end; /* -1 = vision unset */
    int stop[MAX_STOP];
    int n_stop;
} Gemma;

/* strip_thinking(text) of the jinja: split on '<channel|>'; in each part keep what precedes
 * the first '<|channel>'; join; trim. Returns a malloc'd NUL-terminated string. */
static char *strip_thinking(Sl s, size_t *outn)
{
    char *res = (char *)malloc(s.n + 1);
    if (!res) return NULL;
    size_t o = 0;
    Sl rest = s;
    for (;;) {
        const char *end = sl_find(rest, "<channel|>");
        Sl part = { rest.p, end ? (size_t)(end - rest.p) : rest.n };
        const char *open = sl_find(part, "<|channel>");
        size_t keep = open ? (size_t)(open - part.p) : part.n;
        memcpy(res + o, part.p, keep);
        o += keep;
        if (!end) break;
        rest.n -= (size_t)(end + 10 - rest.p);
        rest.p = end + 10;
    }
    Sl t = sl_trim((Sl){ res, o });
    memmove(res, t.p, t.n);
    res[t.n] = 0;
    *outn = t.n;
    return res;
}

/* text whose LM_QS bytes stand for the <|"|> control token */
static void b_qs(Bld *b, int quote, const char *s, size_t n)
{
    size_t run = 0;
    for (size_t i = 0; i < n; i++) {
        if (s[i] != LM_QS) continue;
        b_text_n(b, s + run, i - run);
        b_special(b, quote);
        run = i + 1;
    }
    b_text_n(b, s + run, n - run);
}

/* the <|tool_response> ... <tool_response|> block of one tool message (format_tool_response_block) */
static void g_tool_response(Bld *b, const Gemma *g, const LmMsg *tm)
{
    LmSb d = { 0 };
    jval v = { J_STR, 0, 0, (char *)(tm->content ? tm->content : ""), NULL, NULL, 0 };
    lm_sb_str(&d, "response:");
    lm_sb_user(&d, tm->tool_name && tm->tool_name[0] ? tm->tool_name : "unknown");
    lm_sb_str(&d, "{value:");
    lm_gemma_argument(&d, &v, 0);
    lm_sb_str(&d, "}");
    b_special(b, g->tool_resp);
    if (d.oom) b->oom = 1; else b_qs(b, g->quote, d.s, d.n);
    lm_sb_free(&d);
    b_special(b, g->tool_resp_end);
}

/* <|tool_call>call:NAME{key:value,...}<tool_call|>, keys sorted */
static int g_tool_call(Bld *b, const Gemma *g, const LmToolCall *c, char *err, size_t errlen)
{
    jval *args = json_parse(c->arguments ? c->arguments : "{}", NULL);
    if (!args || args->t != J_OBJ) {
        json_free_tree(args);
        if (err && errlen) snprintf(err, errlen, "a tool call's arguments must be a JSON object");
        return -1;
    }
    LmSb d = { 0 };
    lm_sb_str(&d, "call:");
    lm_sb_user(&d, c->name);
    lm_gemma_argument(&d, args, 0); /* writes {k:v,...} with its own braces */
    json_free_tree(args);
    b_special(b, g->tool_call);
    if (d.oom) b->oom = 1; else b_qs(b, g->quote, d.s, d.n);
    lm_sb_free(&d);
    b_special(b, g->tool_call_end);
    return 0;
}

static int g_render(const LmTemplate *t, const LmMsg *msgs, int n, const LmRenderOpts *opts,
                    LmIds *out, char *err, size_t errlen)
{
    const Gemma *g = (const Gemma *)t->impl;
#define FAIL(m) do { if (err && errlen) snprintf(err, errlen, "%s", m); return -1; } while (0)
    out->n = 0;
    if (opts && opts->n_spans) *opts->n_spans = 0;
    if (n <= 0) FAIL("No messages provided.");
    int last_user = -1;
    for (int i = 0; i < n; i++) if (msgs[i].role == LM_ROLE_USER) last_user = i;
    if (last_user < 0) FAIL("No user query found in messages.");
    for (int i = 1; i < n; i++) {
        if (msgs[i].role == LM_ROLE_SYSTEM) FAIL("System message must be at the beginning.");
        /* the jinja merges consecutive assistant messages into one model turn (tool flows) */
        if (msgs[i].role == LM_ROLE_ASSISTANT && msgs[i - 1].role == LM_ROLE_ASSISTANT)
            FAIL("Consecutive assistant messages are not supported.");
    }
    for (int i = 0; i < n; i++) {
        if (msgs[i].role == LM_ROLE_SYSTEM && msgs[i].parts && msgs[i].n_parts > 0) {
            for (int k = 0; k < msgs[i].n_parts; k++)
                if (msgs[i].parts[k].kind == LM_PART_IMAGE)
                    FAIL("System messages may not contain images.");
        }
    }

    Bld b = { g->tok, out, NULL, 0, 0, 0 };
    int first_sys = msgs[0].role == LM_ROLE_SYSTEM;
    int img_i = 0;
    const int has_tools = opts && opts->tools && opts->tools->t == J_ARR && opts->tools->len > 0;
    b_special(&b, g->bos);
    if (opts->enable_thinking || has_tools || first_sys) {
        b_special(&b, g->turn);
        b_text(&b, "system\n");
        if (opts->enable_thinking) { b_special(&b, g->think); b_text(&b, "\n"); }
        if (first_sys) b_sl(&b, sl_trim(sl_of(msgs[0].content)));
        if (has_tools) {
            for (int k = 0; k < opts->tools->len; k++) {
                LmSb d = { 0 };
                lm_gemma_declaration(&d, opts->tools->kids[k]);
                b_special(&b, g->tool_open);
                if (d.oom) b.oom = 1; else b_qs(&b, g->quote, d.s, d.n);
                lm_sb_free(&d);
                b_special(&b, g->tool_close);
            }
        }
        b_special(&b, g->turn_end);
        b_text(&b, "\n");
    }
    int open_call = 0;  /* the last model turn ended in a tool call: the answer continues it */
    int prev_role = -1; /* role of the previous message that is not a tool result */
    for (int i = first_sys ? 1 : 0; i < n; i++) {
        const LmMsg *m = &msgs[i];
        if (m->role == LM_ROLE_TOOL) continue; /* rendered with the assistant message that made the call */
        const int model = m->role == LM_ROLE_ASSISTANT;
        const int has_calls = model && m->n_calls > 0;
        if (!(model && prev_role == LM_ROLE_ASSISTANT)) {
            b_special(&b, g->turn);
            b_text(&b, model ? "model\n" : "user\n");
        }
        open_call = 0;
        if (!model) {
            if (emit_user_content(&b, m, g->vis_start, g->vis_pad, g->vis_end, opts, &img_i, err, errlen)) {
                free(b.txt);
                return -1;
            }
            b_special(&b, g->turn_end);
            b_text(&b, "\n");
        } else {
            const int verbatim = opts->verbatim_history && m->gen_ids;
            if (verbatim) {
                b_flush(&b);
                if (!b.oom && lm_ids_append(out, m->gen_ids, m->n_gen_ids)) b.oom = 1;
            } else {
                /* reasoning is rendered only for a model turn after the last user message */
                if (m->reasoning && m->reasoning[0] && i > last_user) {
                    b_special(&b, g->channel);
                    b_text(&b, "thought\n");
                    b_text(&b, m->reasoning);
                    b_text(&b, "\n");
                    b_special(&b, g->channel_end);
                }
                for (int k = 0; k < m->n_calls && !b.oom; k++) {
                    if (g_tool_call(&b, g, &m->calls[k], err, errlen)) { free(b.txt); return -1; }
                }
            }
            int answered = 0;
            if (has_calls) {
                for (int k = i + 1; k < n && msgs[k].role == LM_ROLE_TOOL; k++) {
                    g_tool_response(&b, g, &msgs[k]);
                    answered = 1;
                }
            }
            int has_content = 0;
            if (!verbatim) {
                size_t cn = 0;
                char *c = strip_thinking(sl_of(m->content), &cn);
                if (!c) b.oom = 1;
                else b_text_n(&b, c, cn);
                free(c);
                has_content = m->content && m->content[0];
            }
            /* the jinja: a call with no result yet ends in the <|tool_response> the model writes
             * next; a call with results and no text after it leaves the turn open */
            if (has_calls && !answered) {
                b_special(&b, g->tool_resp);
                open_call = 1;
            } else if (answered && !has_content) {
                open_call = 1;
            } else {
                b_special(&b, g->turn_end);
                b_text(&b, "\n");
            }
        }
        prev_role = m->role;
    }
    if (!open_call) {
        b_special(&b, g->turn);
        b_text(&b, "model\n");
    }
    b_flush(&b);
    free(b.txt);
    if (b.oom) FAIL("out of memory");
    return 0;
#undef FAIL
}

/* The generation prompt never opens a channel, so output starts as the answer even with
 * thinking on; the model opens "<|channel>thought\n" itself when it thinks. */
static LmParser *g_parser_new(const LmTemplate *t, int thinking)
{
    (void)thinking;
    const Gemma *g = (const Gemma *)t->impl;
    PFmt f = { g->tok, g->channel, g->channel_end, g->tool_call, g->tool_call_end, { 0 }, g->n_stop, 1, g->quote };
    memcpy(f.stop, g->stop, sizeof f.stop);
    return parser_new(&f, 0);
}

static int g_is_stop(const LmTemplate *t, int id)
{
    const Gemma *g = (const Gemma *)t->impl;
    for (int i = 0; i < g->n_stop; i++) if (g->stop[i] == id) return 1;
    return 0;
}

static void g_destroy(LmTemplate *t) { free(t->impl); }

static int g_parse_call(const LmTemplate *t, const char *raw, const jval *tools, LmToolCall *out,
                        char *err, size_t errlen)
{
    (void)t;
    (void)tools;
    return lm_gemma_parse_call(raw, out, err, errlen);
}

LmTemplate *lm_template_gemma4(const LmTok *tok, const int *eos_ids, int n_eos, char *err, size_t errlen)
{
    Gemma *g = (Gemma *)calloc(1, sizeof *g);
    LmTemplate *t = (LmTemplate *)calloc(1, sizeof *t);
    if (!g || !t) {
        free(g); free(t);
        if (err && errlen) snprintf(err, errlen, "out of memory");
        return NULL;
    }
    g->tok = tok;
    g->vis_start = g->vis_pad = g->vis_end = -1;
    if (lm_tok_require(tok, "<bos>", &g->bos, err, errlen) ||
        lm_tok_require(tok, "<|turn>", &g->turn, err, errlen) ||
        lm_tok_require(tok, "<turn|>", &g->turn_end, err, errlen) ||
        lm_tok_require(tok, "<|think|>", &g->think, err, errlen) ||
        lm_tok_require(tok, "<|channel>", &g->channel, err, errlen) ||
        lm_tok_require(tok, "<channel|>", &g->channel_end, err, errlen) ||
        lm_tok_require(tok, "<|tool_call>", &g->tool_call, err, errlen) ||
        lm_tok_require(tok, "<tool_call|>", &g->tool_call_end, err, errlen) ||
        lm_tok_require(tok, "<|tool>", &g->tool_open, err, errlen) ||
        lm_tok_require(tok, "<tool|>", &g->tool_close, err, errlen) ||
        lm_tok_require(tok, "<|tool_response>", &g->tool_resp, err, errlen) ||
        lm_tok_require(tok, "<tool_response|>", &g->tool_resp_end, err, errlen) ||
        lm_tok_require(tok, "<|\"|>", &g->quote, err, errlen)) {
        free(g); free(t);
        return NULL;
    }
    g->stop[g->n_stop++] = g->turn_end;
    g->stop[g->n_stop++] = g->tool_resp; /* what the model writes after a tool call: the result is the caller's */
    for (int i = 0; i < n_eos; i++) {
        int dup = 0;
        for (int j = 0; j < g->n_stop; j++) if (g->stop[j] == eos_ids[i]) dup = 1;
        if (!dup && eos_ids[i] >= 0 && g->n_stop < MAX_STOP) g->stop[g->n_stop++] = eos_ids[i];
    }
    t->name = "gemma4";
    t->render = g_render;
    t->parser_new = g_parser_new;
    t->is_stop = g_is_stop;
    t->parse_call = g_parse_call;
    t->destroy = g_destroy;
    t->impl = g;
    return t;
}

void lm_template_free(LmTemplate *t)
{
    if (!t) return;
    if (t->destroy) t->destroy(t);
    free(t);
}

void lm_template_set_vision(LmTemplate *t, int start, int pad, int end)
{
    if (!t || !t->impl) return;
    if (t->name && !strcmp(t->name, "qwen3.5")) {
        Qwen *q = (Qwen *)t->impl;
        q->vis_start = start;
        q->vis_pad = pad;
        q->vis_end = end;
        return;
    }
    if (t->name && !strcmp(t->name, "gemma4")) {
        Gemma *g = (Gemma *)t->impl;
        g->vis_start = start;
        g->vis_pad = pad;
        g->vis_end = end;
    }
}
