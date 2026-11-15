/* lm_chat.c -- see lm_chat.h. Original to this repository; the shape of the loop follows
 * kimi-k3-in-c src/chat/k3_chat.c (Apache-2.0) only in spirit, nothing is copied. */
#include "chat/lm_chat.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "lm/lm.h"

typedef struct {
    LmRole role;
    char *content;
    char *reasoning; /* assistant only */
    int *gen_ids;    /* assistant only; never NULL for an assistant turn made by this chat */
    int n_gen;
    int gen_thinking;
} Hist;

typedef struct { char *s; size_t n, cap; } Buf;

struct LmChat {
    LmChatConfig cfg;
    char *system;
    LmSampler smp;
    Hist *h;
    int nh, caph;
    LmIds fed;       /* ids the backend holds, in order */
    int dirty;       /* backend state unknown (error mid-feed): next turn must reset */
    int reprefills;
    float *logits;
};

static void set_err(char *err, size_t n, const char *fmt, ...) __attribute__((format(printf, 3, 4)));
static void set_err(char *err, size_t n, const char *fmt, ...)
{
    if (!err || !n) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(err, n, fmt, ap);
    va_end(ap);
}

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static char *dup_str(const char *s) { size_t n = strlen(s ? s : ""); char *p = (char *)malloc(n + 1); if (p) memcpy(p, s ? s : "", n + 1); return p; }

static void hist_free_one(Hist *h) { free(h->content); free(h->reasoning); free(h->gen_ids); memset(h, 0, sizeof *h); }

static void hist_clear(LmChat *c)
{
    for (int i = 0; i < c->nh; i++) hist_free_one(&c->h[i]);
    c->nh = 0;
}

static int hist_push(LmChat *c, const Hist *h)
{
    if (c->nh == c->caph) {
        int cap = c->caph ? c->caph * 2 : 8;
        Hist *p = (Hist *)realloc(c->h, (size_t)cap * sizeof *p);
        if (!p) return -1;
        c->h = p;
        c->caph = cap;
    }
    c->h[c->nh++] = *h;
    return 0;
}

LmChat *lm_chat_new(const LmChatConfig *cfg, char *err, size_t errlen)
{
    if (!cfg || !cfg->tok || !cfg->tpl || !cfg->backend.feed || !cfg->backend.reset ||
        !cfg->backend.pos || cfg->backend.n_vocab <= 0 || cfg->backend.n_ctx <= 0) {
        set_err(err, errlen, "lm_chat_new: incomplete configuration or backend");
        return NULL;
    }
    LmChat *c = (LmChat *)calloc(1, sizeof *c);
    if (!c) { set_err(err, errlen, "out of memory"); return NULL; }
    c->cfg = *cfg;
    c->system = cfg->system ? dup_str(cfg->system) : NULL;
    c->logits = (float *)malloc((size_t)cfg->backend.n_vocab * sizeof *c->logits);
    if ((cfg->system && !c->system) || !c->logits) {
        set_err(err, errlen, "out of memory");
        free(c->system); free(c->logits); free(c);
        return NULL;
    }
    c->cfg.system = c->system;
    lm_sampler_init(&c->smp, cfg->temperature, cfg->top_p, cfg->top_k, cfg->seed);
    return c;
}

void lm_chat_free(LmChat *c)
{
    if (!c) return;
    hist_clear(c);
    free(c->h);
    lm_ids_free(&c->fed);
    lm_sampler_free(&c->smp);
    free(c->system);
    free(c->logits);
    free(c);
}

void lm_chat_reset(LmChat *c)
{
    hist_clear(c);
    c->fed.n = 0;
    c->dirty = 0;
    c->cfg.backend.reset(c->cfg.backend.ctx);
}

int lm_chat_reprefills(const LmChat *c) { return c->reprefills; }
int lm_chat_fed(const LmChat *c) { return c->fed.n; }

int lm_chat_sync(LmChat *c, const LmMsg *msgs, int n, char *err, size_t errlen)
{
    const int first = n > 0 && msgs[0].role == LM_ROLE_SYSTEM;
    for (int i = 0; i < n; i++) {
        if (i >= first && msgs[i].role == LM_ROLE_SYSTEM) { set_err(err, errlen, "a system message must come first"); return -1; }
        if (!msgs[i].content) { set_err(err, errlen, "message %d has no text", i); return -1; }
    }
    const int nh = n - first;
    /* how many leading messages match the history (those entries are kept as they are) */
    int keep = 0;
    while (keep < nh && keep < c->nh && c->h[keep].role == msgs[first + keep].role &&
           !strcmp(c->h[keep].content, msgs[first + keep].content))
        keep++;
    Hist *h = (Hist *)calloc((size_t)(nh > 0 ? nh : 1), sizeof *h);
    char *sys = first ? dup_str(msgs[0].content) : NULL;
    int oom = !h || (first && !sys);
    for (int i = keep; i < nh && !oom; i++) {
        h[i].role = msgs[first + i].role;
        if (!(h[i].content = dup_str(msgs[first + i].content))) oom = 1;
    }
    if (oom) {
        if (h) for (int i = keep; i < nh; i++) free(h[i].content);
        free(h);
        free(sys);
        set_err(err, errlen, "out of memory");
        return -1;
    }
    for (int i = 0; i < keep; i++) h[i] = c->h[i];
    for (int i = keep; i < c->nh; i++) hist_free_one(&c->h[i]);
    free(c->h);
    c->h = h;
    c->nh = nh;
    c->caph = nh > 0 ? nh : 1;
    free(c->system);
    c->system = sys;
    c->cfg.system = sys;
    return 0;
}

void lm_chat_set_sampling(LmChat *c, double temperature, double top_p, int top_k, uint64_t seed)
{
    lm_sampler_free(&c->smp);
    lm_sampler_init(&c->smp, temperature, top_p, top_k, seed);
    c->cfg.temperature = temperature;
    c->cfg.top_p = top_p;
    c->cfg.top_k = top_k;
    c->cfg.seed = seed;
}

/* ---- collecting sink --------------------------------------------------------------- */

static int buf_add(Buf *b, const char *s, size_t n)
{
    if (b->n + n + 1 > b->cap) {
        size_t cap = b->cap ? b->cap : 256;
        while (cap < b->n + n + 1) cap *= 2;
        char *p = (char *)realloc(b->s, cap);
        if (!p) return -1;
        b->s = p;
        b->cap = cap;
    }
    memcpy(b->s + b->n, s, n);
    b->n += n;
    b->s[b->n] = 0;
    return 0;
}

typedef struct {
    LmSink sink;
    void *user;
    Buf reasoning, answer;
    int oom;
} Collect;

static void collect(void *u, LmOutKind kind, const char *s, size_t n)
{
    Collect *c = (Collect *)u;
    if (kind == LM_OUT_REASONING) { if (buf_add(&c->reasoning, s, n)) c->oom = 1; }
    else if (kind == LM_OUT_ANSWER) { if (buf_add(&c->answer, s, n)) c->oom = 1; }
    if (c->sink) c->sink(c->user, kind, s, n);
}

/* ---- the turn ---------------------------------------------------------------------- */

static void pop_user(LmChat *c)
{
    if (c->nh > 0) hist_free_one(&c->h[--c->nh]);
}

int lm_chat_turn(LmChat *c, const char *user_text, const LmTurnOpts *opts, LmTurn *res,
                 char *err, size_t errlen)
{
    LmTurn local;
    if (!res) res = &local;
    memset(res, 0, sizeof *res);
    static const LmTurnOpts none;
    if (!opts) opts = &none;
    const LmBackend *be = &c->cfg.backend;

    Hist u = { LM_ROLE_USER, dup_str(user_text), NULL, NULL, 0, 0 };
    if (!u.content || hist_push(c, &u)) { free(u.content); set_err(err, errlen, "out of memory"); return -1; }

    /* Render the whole transcript. */
    LmMsg *msgs = (LmMsg *)calloc((size_t)c->nh + 1, sizeof *msgs);
    if (!msgs) { pop_user(c); set_err(err, errlen, "out of memory"); return -1; }
    int nm = 0;
    if (c->system) { msgs[nm].role = LM_ROLE_SYSTEM; msgs[nm].content = c->system; nm++; }
    for (int i = 0; i < c->nh; i++) {
        msgs[nm].role = c->h[i].role;
        msgs[nm].content = c->h[i].content;
        msgs[nm].reasoning = NULL; /* the template re-derives it; verbatim turns use gen_ids */
        msgs[nm].gen_ids = c->h[i].gen_ids;
        msgs[nm].n_gen_ids = c->h[i].n_gen;
        msgs[nm].gen_thinking = c->h[i].gen_thinking;
        nm++;
    }
    LmRenderOpts ro = { c->cfg.thinking, !c->cfg.strict_history };
    LmIds prompt = { 0 };
    int rc = c->cfg.tpl->render(c->cfg.tpl, msgs, nm, &ro, &prompt, err, errlen);
    free(msgs);
    if (rc) { lm_ids_free(&prompt); pop_user(c); return -1; }
    res->n_prompt = prompt.n;

    /* Is what the backend holds an exact prefix of the prompt, with at least one new id
     * left to feed (the last prompt id must be fed with logits)? */
    int reuse = !c->dirty && c->fed.n < prompt.n && be->pos(be->ctx) == c->fed.n &&
                (c->fed.n == 0 || !memcmp(c->fed.ids, prompt.ids, (size_t)c->fed.n * sizeof(int)));
    int start = reuse ? c->fed.n : 0;
    if (prompt.n > be->n_ctx) {
        set_err(err, errlen, "the prompt is %d tokens and does not fit the %d-token context", prompt.n, be->n_ctx);
        res->stop = LM_STOP_CTX;
        lm_ids_free(&prompt);
        pop_user(c);
        return LM_ERR_CTX;
    }
    if (!reuse) {
        if (c->fed.n > 0 || c->dirty) { res->reprefill = 1; c->reprefills++; }
        if (c->fed.n > 0 || c->dirty || be->pos(be->ctx) != 0) be->reset(be->ctx);
        c->fed.n = 0;
        c->dirty = 0;
    }

    /* Prefill: ids without logits, the last one with. A backend with prefill takes chunks, polling
     * cancellation between them; one without is fed token by token. */
    double t0 = now_s();
    const int chunk = be->prefill ? LM_CHAT_PREFILL_CHUNK : 1;
    for (int i = start; i < prompt.n; i += chunk) {
        if (opts->cancelled && opts->cancelled(opts->cancel_user)) {
            res->stop = LM_STOP_CANCEL;
            res->prefill_s = now_s() - t0;
            lm_ids_free(&prompt);
            pop_user(c); /* what was fed stays a valid prefix if the same message is sent again */
            return LM_OK;
        }
        const int n = prompt.n - i < chunk ? prompt.n - i : chunk;
        const int last = i + n == prompt.n;
        int frc = be->prefill ? be->prefill(be->ctx, prompt.ids + i, n, last ? c->logits : NULL)
                              : be->feed(be->ctx, prompt.ids[i], last ? c->logits : NULL);
        if (frc != LM_OK) {
            if (frc != LM_ERR_CTX) c->dirty = 1;
            else res->stop = LM_STOP_CTX;
            set_err(err, errlen, "backend feed failed with code %d while prefilling position %d of %d", frc, i, prompt.n);
            lm_ids_free(&prompt);
            pop_user(c);
            return frc;
        }
        for (int k = 0; k < n; k++) {
            if (lm_ids_push(&c->fed, prompt.ids[i + k])) {
                c->dirty = 1; /* the backend advanced but we lost count */
                lm_ids_free(&prompt);
                pop_user(c);
                set_err(err, errlen, "out of memory");
                return -1;
            }
        }
        res->n_prefill += n;
    }
    res->prefill_s = now_s() - t0;
    lm_ids_free(&prompt);

    /* Generate. */
    LmParser *parser = c->cfg.tpl->parser_new(c->cfg.tpl, c->cfg.thinking);
    Collect col = { opts->sink, opts->user, { NULL, 0, 0 }, { NULL, 0, 0 }, 0 };
    LmIds gen = { 0 };
    int result = LM_OK, internal = 0;
    if (!parser) { pop_user(c); set_err(err, errlen, "out of memory"); return -1; }
    t0 = now_s();
    for (;;) {
        if (opts->cancelled && opts->cancelled(opts->cancel_user)) { res->stop = LM_STOP_CANCEL; break; }
        int id;
        if (lm_sampler_next(&c->smp, c->logits, be->n_vocab, &id)) {
            set_err(err, errlen, "sampling failed: non-finite logits or invalid sampler settings");
            internal = 1;
            break;
        }
        int pr = parser->feed(parser, id, collect, &col);
        if (pr == LM_PARSE_STOP) { res->stop = LM_STOP_EOS; break; }
        if (lm_ids_push(&gen, id)) { set_err(err, errlen, "out of memory"); internal = 1; break; }
        if (pr == LM_PARSE_STOP_AFTER) { res->stop = LM_STOP_TOOLCALL; break; }
        if (opts->max_tokens > 0 && gen.n >= opts->max_tokens) { res->stop = LM_STOP_MAX; break; }
        int frc = be->feed(be->ctx, id, c->logits);
        if (frc == LM_ERR_CTX) { res->stop = LM_STOP_CTX; result = LM_ERR_CTX; set_err(err, errlen, "context window full (%d tokens)", be->n_ctx); break; }
        if (frc != LM_OK) {
            c->dirty = 1;
            result = frc;
            set_err(err, errlen, "backend feed failed with code %d while decoding", frc);
            break;
        }
        if (lm_ids_push(&c->fed, id)) { c->dirty = 1; set_err(err, errlen, "out of memory"); internal = 1; break; }
    }
    parser->finish(parser, collect, &col);
    res->decode_s = now_s() - t0;
    res->n_gen = gen.n;
    res->tool_calls = parser->tool_calls;
    res->thinking_open = parser->thinking_open;
    parser->destroy(parser);

    if (internal || col.oom || (result != LM_OK && result != LM_ERR_CTX)) {
        /* nothing from this turn is kept; if the backend advanced, c->fed says how far */
        lm_ids_free(&gen);
        free(col.reasoning.s);
        free(col.answer.s);
        pop_user(c);
        if (!internal && col.oom) set_err(err, errlen, "out of memory");
        return internal || col.oom ? -1 : result;
    }

    Hist a = { LM_ROLE_ASSISTANT, NULL, NULL, NULL, gen.n, c->cfg.thinking };
    a.content = col.answer.s ? col.answer.s : dup_str("");
    a.reasoning = col.reasoning.s ? col.reasoning.s : dup_str("");
    a.gen_ids = gen.ids ? gen.ids : (int *)malloc(sizeof(int)); /* non-NULL marks a verbatim turn */
    if (!a.content || !a.reasoning || !a.gen_ids || hist_push(c, &a)) {
        free(a.content); free(a.reasoning); free(a.gen_ids);
        pop_user(c);
        set_err(err, errlen, "out of memory");
        return -1;
    }
    res->answer = c->h[c->nh - 1].content;
    res->reasoning = c->h[c->nh - 1].reasoning;
    return result;
}
