/* test_chat.c -- the generation loop (lm_chat) against a scripted fake backend, with the
 * fixture tokenizer and the Qwen template. Run from the repo root. */
#include "lm_test.h"
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "chat/lm_chat.h"
#include "lm/lm.h"

#define FIXTURE "tests/fixtures/tok/qwen-tokenizer.json"

static int g_fail = 0, g_checks = 0;
#undef CHECK /* lm_test.h defines its own; this file counts with g_checks/g_fail */
#define CHECK(c) do { g_checks++; if (!(c)) { g_fail++; fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #c); } } while (0)
#undef CHECK_MSG
#define CHECK_MSG(c, ...) do { g_checks++; if (!(c)) { g_fail++; fprintf(stderr, "%s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); } } while (0)

static LmTok *g_tok;
static LmTemplate *g_tpl;

static void bl(const char *s, size_t n, char *out)
{
    static int map[256], init;
    if (!init) {
        int direct[256] = { 0 }, k = 0;
        for (int b = 33; b <= 126; b++) direct[b] = 1;
        for (int b = 161; b <= 172; b++) direct[b] = 1;
        for (int b = 174; b <= 255; b++) direct[b] = 1;
        for (int b = 0; b < 256; b++) map[b] = direct[b] ? b : 256 + k++;
        init = 1;
    }
    size_t o = 0;
    for (size_t i = 0; i < n; i++) {
        int cp = map[(unsigned char)s[i]];
        if (cp < 0x80) out[o++] = (char)cp;
        else { out[o++] = (char)(0xC0 | (cp >> 6)); out[o++] = (char)(0x80 | (cp & 0x3F)); }
    }
    out[o] = 0;
}

static int pid(const char *p)
{
    if (p[0] == '<' && lm_tok_special_id(g_tok, p) >= 0) return lm_tok_special_id(g_tok, p);
    char buf[256];
    bl(p, strlen(p), buf);
    return lm_tok_vocab_id(g_tok, buf);
}

/* ---- the fake engine ------------------------------------------------------------------ */

enum { MAXLOG = 4096 };
typedef struct {
    int n_vocab, n_ctx, pos;
    int log[MAXLOG], null_logits[MAXLOG], nlog;   /* every successful feed, in order */
    int resets;
    int script[256], ns, cursor;                  /* argmax of the k-th logits row handed out */
    int feeds, fail_at;                           /* fail_at: the feed() call (0-based) returning LM_ERR_IO */
    int calls;
    int pf_calls, pf_max;                         /* prefill backend: calls and the largest chunk handed to it */
    int st[MAXLOG];                               /* the "state": the id at every position held */
    int saves, restores, drops, live, maxlive;    /* prefix-cache hooks: calls, snapshots alive (most at once) */
    int save_fail, restore_fail;                  /* make save return NULL / restore fail */
    int saved_pos;                                /* position of the last snapshot taken */
} Fake;

static int fk_feed(void *c, int token, float *logits)
{
    Fake *f = (Fake *)c;
    int call = f->calls++;
    if (f->fail_at >= 0 && call == f->fail_at) return LM_ERR_IO;
    if (f->pos >= f->n_ctx) return LM_ERR_CTX;
    f->st[f->pos] = token;
    f->log[f->nlog] = token;
    f->null_logits[f->nlog++] = logits == NULL;
    f->pos++;
    if (logits) {
        for (int i = 0; i < f->n_vocab; i++) logits[i] = -5.0f;
        int next = f->script[f->cursor < f->ns ? f->cursor : f->ns - 1];
        f->cursor++;
        logits[next] = 5.0f;
    }
    return LM_OK;
}
/* a prefill backend is n feeds, logits after the last id only; it records how it was called */
static int fk_prefill(void *c, const int *ids, int n, float *logits)
{
    Fake *f = (Fake *)c;
    f->pf_calls++;
    if (n > f->pf_max) f->pf_max = n;
    for (int i = 0; i < n; i++) {
        const int rc = fk_feed(c, ids[i], i == n - 1 ? logits : NULL);
        if (rc != LM_OK) return rc;
    }
    return LM_OK;
}
static int g_use_prefill; /* mk() gives the backend the prefill call when set */
typedef struct { int pos; int st[MAXLOG]; } FkSnap;
static void *fk_save(void *c)
{
    Fake *f = (Fake *)c;
    f->saves++;
    if (f->save_fail) return NULL;
    FkSnap *s = (FkSnap *)malloc(sizeof *s);
    if (!s) return NULL;
    s->pos = f->saved_pos = f->pos;
    memcpy(s->st, f->st, sizeof s->st);
    if (++f->live > f->maxlive) f->maxlive = f->live;
    return s;
}
static int fk_restore(void *c, void *snap)
{
    Fake *f = (Fake *)c;
    f->restores++;
    if (f->restore_fail) return LM_ERR_IO;
    const FkSnap *s = (const FkSnap *)snap;
    f->pos = s->pos;
    memcpy(f->st, s->st, sizeof f->st);
    return LM_OK;
}
static void fk_drop(void *c, void *snap) { ((Fake *)c)->drops++; ((Fake *)c)->live--; free(snap); }
static size_t fk_bytes(void *c, void *snap) { (void)c; return 1000 + (size_t)((const FkSnap *)snap)->pos; }
static int g_use_cache; /* mk() gives the backend the prefix-cache hooks when set */
static size_t g_cache_bytes; /* and this budget */
static void fk_reset(void *c) { Fake *f = (Fake *)c; f->pos = 0; f->resets++; f->nlog = 0; }
static int fk_pos(void *c) { return ((Fake *)c)->pos; }

static void fake_init(Fake *f, int n_ctx)
{
    memset(f, 0, sizeof *f);
    f->n_vocab = lm_tok_n_ids(g_tok);
    f->n_ctx = n_ctx;
    f->fail_at = -1;
}

static void script(Fake *f, ...)
{
    va_list ap;
    va_start(ap, f);
    f->ns = 0;
    f->cursor = 0;
    for (const char *p; (p = va_arg(ap, const char *)) != NULL;) f->script[f->ns++] = pid(p);
    va_end(ap);
}

static LmChat *mk(Fake *f, int thinking, int strict, const char *system)
{
    LmChatConfig cc;
    memset(&cc, 0, sizeof cc);
    cc.tok = g_tok;
    cc.tpl = g_tpl;
    cc.backend = (LmBackend){ f, f->n_vocab, f->n_ctx, fk_feed, fk_reset, fk_pos, g_use_prefill ? fk_prefill : NULL,
                              g_use_cache ? fk_save : NULL, g_use_cache ? fk_restore : NULL, g_use_cache ? fk_drop : NULL,
                              g_use_cache ? fk_bytes : NULL };
    cc.cache_bytes = g_cache_bytes;
    cc.temperature = 0;
    cc.top_p = 1;
    cc.seed = 1;
    cc.thinking = thinking;
    cc.strict_history = strict;
    cc.system = system;
    char err[200];
    LmChat *c = lm_chat_new(&cc, err, sizeof err);
    if (!c) fprintf(stderr, "lm_chat_new: %s\n", err);
    return c;
}

static char g_ans[512], g_rea[512], g_tool[128];
static void sink(void *u, LmOutKind k, const char *s, size_t n)
{
    (void)u;
    char *d = k == LM_OUT_ANSWER ? g_ans : k == LM_OUT_REASONING ? g_rea : g_tool;
    strncat(d, s, n);
}
static void clear_out(void) { g_ans[0] = g_rea[0] = g_tool[0] = 0; }

static int turn(LmChat *c, const char *text, int max_tokens, LmTurn *t, int (*cancel)(void *), void *cu)
{
    LmTurnOpts o = { max_tokens, sink, NULL, cancel, cu, NULL };
    char err[256] = "";
    clear_out();
    int rc = lm_chat_turn(c, text, &o, t, err, sizeof err);
    if (rc != LM_OK && rc != LM_ERR_CTX && rc != LM_ERR_IO) fprintf(stderr, "  turn rc=%d err=%s\n", rc, err);
    return rc;
}

static int render_n(LmChat *c, int dummy) { (void)c; (void)dummy; return 0; }

/* ---- tests --------------------------------------------------------------------------- */

static void test_prefix_and_tail(void)
{
    Fake f;
    fake_init(&f, 4096);
    LmChat *c = mk(&f, 0, 0, NULL);
    LmTurn t;
    script(&f, "Hello", "<|im_end|>", NULL);
    CHECK(turn(c, "Hi", 0, &t, NULL, NULL) == LM_OK);
    CHECK(t.stop == LM_STOP_EOS && t.n_gen == 1 && !strcmp(g_ans, "Hello") && !strcmp(t.answer, "Hello"));
    CHECK(t.reprefill == 0 && f.resets == 0);

    /* turn 1 fed exactly the prompt, then the generated token; the stop id was not fed */
    int p1[] = { pid("<|im_start|>"), pid("user"), pid("\n"), pid("Hi"), pid("<|im_end|>"), pid("\n"),
                 pid("<|im_start|>"), pid("assistant"), pid("\n"), pid("<think>"), pid("\n\n"), pid("</think>"), pid("\n\n") };
    int n1 = (int)(sizeof p1 / sizeof *p1);
    CHECK(t.n_prompt == n1 && t.n_prefill == n1 && f.nlog == n1 + 1);
    CHECK(!memcmp(f.log, p1, sizeof p1) && f.log[n1] == pid("Hello"));
    /* prefill ids carry no logits, only the last prompt id and the generated id do */
    int only_last = 1;
    for (int i = 0; i < n1 - 1; i++) if (!f.null_logits[i]) only_last = 0;
    CHECK(only_last && !f.null_logits[n1 - 1] && !f.null_logits[n1]);
    CHECK(lm_chat_fed(c) == n1 + 1 && f.pos == n1 + 1);

    /* turn 2: the fed ids are a prefix of the new prompt, so only the tail is fed */
    f.nlog = 0;
    script(&f, "Yes", "<|im_end|>", NULL);
    CHECK(turn(c, "Hi", 0, &t, NULL, NULL) == LM_OK);
    int tail[] = { pid("<|im_end|>"), pid("\n"), pid("<|im_start|>"), pid("user"), pid("\n"), pid("Hi"), pid("<|im_end|>"), pid("\n"),
                   pid("<|im_start|>"), pid("assistant"), pid("\n"), pid("<think>"), pid("\n\n"), pid("</think>"), pid("\n\n") };
    int nt = (int)(sizeof tail / sizeof *tail);
    CHECK(t.reprefill == 0 && f.resets == 0 && t.n_prefill == nt && t.n_prompt == n1 + 1 + nt);
    CHECK(f.nlog == nt + 1 && !memcmp(f.log, tail, sizeof tail));
    CHECK(!strcmp(g_ans, "Yes") && lm_chat_reprefills(c) == 0);
    lm_chat_free(c);
    (void)render_n;
}

static void test_mismatch_resets(void)
{
    Fake f;
    fake_init(&f, 4096);
    LmChat *c = mk(&f, 0, 1 /* strict: the model's own history rendering */, NULL);
    LmTurn t;
    script(&f, "Hello", "<|im_end|>", NULL);
    CHECK(turn(c, "Hi", 0, &t, NULL, NULL) == LM_OK && t.reprefill == 0);
    int fed1 = f.pos;
    script(&f, "Yes", "<|im_end|>", NULL);
    CHECK(turn(c, "Hi", 0, &t, NULL, NULL) == LM_OK);
    /* strict history drops the empty think block of the earlier turn: not a prefix */
    CHECK(t.reprefill == 1 && f.resets == 1 && t.n_prefill == t.n_prompt && f.pos == t.n_prompt + 1);
    CHECK(t.n_prompt > fed1 && lm_chat_reprefills(c) == 1);
    int first = f.log[0] == pid("<|im_start|>");
    CHECK(first && f.nlog == t.n_prompt + 1);

    /* an explicit reset starts a new conversation and resets the backend */
    lm_chat_reset(c);
    CHECK(f.resets == 2 && f.pos == 0 && lm_chat_fed(c) == 0);
    script(&f, "Hello", "<|im_end|>", NULL);
    CHECK(turn(c, "Hi", 0, &t, NULL, NULL) == LM_OK && t.reprefill == 0 && t.n_prefill == t.n_prompt);
    lm_chat_free(c);
}

static void three_turns(int thinking, int strict, int *reprefills, int *fed_total)
{
    Fake f;
    fake_init(&f, 4096);
    LmChat *c = mk(&f, thinking, strict, "OK");
    LmTurn t;
    const char *users[3] = { "Hi", "How are you?", "Thanks" };
    *fed_total = 0;
    for (int i = 0; i < 3; i++) {
        if (thinking) script(&f, "Think", "\n", "</think>", "\n\n", "Hello", "<|im_end|>", NULL);
        else script(&f, "Hello", "<|im_end|>", NULL);
        int before = f.nlog;
        (void)before;
        CHECK(turn(c, users[i], 0, &t, NULL, NULL) == LM_OK && t.stop == LM_STOP_EOS);
        CHECK(!strcmp(g_ans, "Hello") && (thinking ? !strcmp(g_rea, "Think") : g_rea[0] == 0));
        *fed_total += t.n_prefill + t.n_gen - (t.stop == LM_STOP_EOS ? 0 : 0);
    }
    *reprefills = lm_chat_reprefills(c);
    lm_chat_free(c);
}

static void test_reprefill_report(void)
{
    int rp, fed;
    for (int thinking = 0; thinking <= 1; thinking++) {
        int rp_v, fed_v, rp_s, fed_s;
        three_turns(thinking, 0, &rp_v, &fed_v);
        three_turns(thinking, 1, &rp_s, &fed_s);
        printf("3-turn chat, thinking %s: re-prefills verbatim history = %d (%d ids fed in total), strict model template = %d (%d ids fed in total)\n",
               thinking ? "on " : "off", rp_v, fed_v, rp_s, fed_s);
        CHECK(rp_v == 0);
        CHECK(rp_s == 2);
        CHECK(fed_s > fed_v);
    }
    (void)rp; (void)fed;
}

static void test_ctx_full(void)
{
    Fake f;
    fake_init(&f, 4096);
    LmChat *probe = mk(&f, 0, 0, NULL);
    LmTurn t;
    script(&f, "Hello", "<|im_end|>", NULL);
    turn(probe, "Hi", 0, &t, NULL, NULL);
    int np = t.n_prompt;
    lm_chat_free(probe);

    /* the window fills while generating: reported as LM_ERR_CTX, partial answer kept */
    fake_init(&f, np + 1);
    LmChat *c = mk(&f, 0, 0, NULL);
    script(&f, "Hello", "Hello", "Hello", "<|im_end|>", NULL);
    int rc = turn(c, "Hi", 0, &t, NULL, NULL);
    CHECK(rc == LM_ERR_CTX && t.stop == LM_STOP_CTX);
    CHECK(t.n_gen == 2 && !strcmp(g_ans, "HelloHello") && f.pos == np + 1);
    CHECK(t.answer && !strcmp(t.answer, "HelloHello"));
    lm_chat_free(c);

    /* the prompt itself does not fit: nothing is fed, the history is untouched */
    fake_init(&f, 5);
    c = mk(&f, 0, 0, NULL);
    script(&f, "Hello", "<|im_end|>", NULL);
    rc = turn(c, "Hi", 0, &t, NULL, NULL);
    CHECK(rc == LM_ERR_CTX && t.stop == LM_STOP_CTX && f.nlog == 0 && f.resets == 0 && lm_chat_fed(c) == 0);
    lm_chat_free(c);
}

static void test_backend_error(void)
{
    Fake f;
    fake_init(&f, 4096);
    LmChat *c = mk(&f, 0, 0, NULL);
    LmTurn t;
    script(&f, "Hello", "<|im_end|>", NULL);
    f.fail_at = 3; /* inside the prefill */
    CHECK(turn(c, "Hi", 0, &t, NULL, NULL) == LM_ERR_IO);
    /* the state is invalid: the next turn resets and feeds everything, and works */
    f.fail_at = -1;
    script(&f, "Hello", "<|im_end|>", NULL);
    CHECK(turn(c, "Hi", 0, &t, NULL, NULL) == LM_OK);
    CHECK(f.resets == 1 && t.reprefill == 1 && t.n_prefill == t.n_prompt && !strcmp(g_ans, "Hello"));
    /* the failed message left no trace: the transcript has exactly one user turn */
    CHECK(t.n_prompt == 13);
    /* an error while decoding propagates too */
    f.fail_at = f.calls + 3 + 0;
    script(&f, "Hello", "Hello", "Hello", "<|im_end|>", NULL);
    int rc = turn(c, "Hi", 0, &t, NULL, NULL);
    CHECK(rc != LM_OK && rc != LM_ERR_CTX);
    lm_chat_free(c);
}

/* A prompt long enough for several prefill chunks: "Hello" 40 times. */
static const char *LONG_MSG =
    "Hello Hello Hello Hello Hello Hello Hello Hello Hello Hello Hello Hello Hello Hello Hello Hello "
    "Hello Hello Hello Hello Hello Hello Hello Hello Hello Hello Hello Hello Hello Hello Hello Hello "
    "Hello Hello Hello Hello Hello Hello Hello Hello";

typedef struct { int after, seen; } Cancel;
static int cancel_after(void *u) { Cancel *c = (Cancel *)u; return ++c->seen > c->after; }
static int cancel_after_x(void *u) { return cancel_after(u); }

static void test_cancel_and_max(void)
{
    Fake f;
    fake_init(&f, 4096);
    LmChat *c = mk(&f, 0, 0, NULL);
    LmTurn t;
    script(&f, "Hello", "Hello", "Hello", "Hello", "Hello", "<|im_end|>", NULL);
    Cancel cn = { 15, 0 }; /* 13 polls during prefill, then two tokens */
    CHECK(turn(c, "Hi", 0, &t, cancel_after, &cn) == LM_OK);
    CHECK(t.stop == LM_STOP_CANCEL && t.n_gen == 2 && !strcmp(g_ans, "HelloHello"));
    CHECK(f.pos == lm_chat_fed(c));

    /* the next turn extends what was fed (cancelled output is in the history) */
    f.nlog = 0;
    script(&f, "Yes", "<|im_end|>", NULL);
    CHECK(turn(c, "Hi", 0, &t, NULL, NULL) == LM_OK && t.reprefill == 0 && f.log[0] == pid("<|im_end|>"));
    lm_chat_free(c);

    /* max_tokens: the last sampled token is not fed, and is the first of the next tail */
    fake_init(&f, 4096);
    c = mk(&f, 0, 0, NULL);
    script(&f, "Hello", "Hello", "Hello", "Hello", "Hello", "<|im_end|>", NULL);
    CHECK(turn(c, "Hi", 3, &t, NULL, NULL) == LM_OK);
    CHECK(t.stop == LM_STOP_MAX && t.n_gen == 3 && f.nlog == t.n_prompt + 2);
    f.nlog = 0;
    script(&f, "Yes", "<|im_end|>", NULL);
    CHECK(turn(c, "Hi", 0, &t, NULL, NULL) == LM_OK && t.reprefill == 0 && f.log[0] == pid("Hello") && f.log[1] == pid("<|im_end|>"));
    lm_chat_free(c);
}

static int cancel_after_x(void *u);

static void test_prefill_backend(void)
{
    /* the same turn with and without a prefill backend: identical ids fed, identical logits flags,
     * identical text; the prefill calls are chunks of at most LM_CHAT_PREFILL_CHUNK */
    Fake a, b;
    LmTurn ta, tb;
    char ans_a[512];
    g_use_prefill = 0;
    fake_init(&a, 4096);
    LmChat *ca = mk(&a, 0, 0, NULL);
    script(&a, "Hello", "<|im_end|>", NULL);
    CHECK(turn(ca, LONG_MSG, 0, &ta, NULL, NULL) == LM_OK);
    snprintf(ans_a, sizeof ans_a, "%s", g_ans);
    g_use_prefill = 1;
    fake_init(&b, 4096);
    LmChat *cb = mk(&b, 0, 0, NULL);
    script(&b, "Hello", "<|im_end|>", NULL);
    CHECK(turn(cb, LONG_MSG, 0, &tb, NULL, NULL) == LM_OK);
    CHECK(ta.n_prompt > 3 * LM_CHAT_PREFILL_CHUNK);                       /* the test needs several chunks */
    CHECK(ta.n_prompt == tb.n_prompt && ta.n_prefill == tb.n_prefill && ta.n_gen == tb.n_gen);
    CHECK(a.nlog == b.nlog && !memcmp(a.log, b.log, (size_t)a.nlog * sizeof(int)));
    CHECK(!memcmp(a.null_logits, b.null_logits, (size_t)a.nlog * sizeof(int)));
    CHECK(!strcmp(ans_a, g_ans));
    CHECK(a.pf_calls == 0);
    CHECK(b.pf_calls == (tb.n_prompt + LM_CHAT_PREFILL_CHUNK - 1) / LM_CHAT_PREFILL_CHUNK && b.pf_max == LM_CHAT_PREFILL_CHUNK);

    /* the second turn's tail is one short prefill call */
    b.nlog = 0;
    const int calls_before = b.pf_calls;
    script(&b, "Yes", "<|im_end|>", NULL);
    CHECK(turn(cb, "Hi", 0, &tb, NULL, NULL) == LM_OK && tb.reprefill == 0);
    CHECK(b.pf_calls == calls_before + (tb.n_prefill + LM_CHAT_PREFILL_CHUNK - 1) / LM_CHAT_PREFILL_CHUNK);
    lm_chat_free(ca);
    lm_chat_free(cb);

    /* cancellation is polled between chunks: after the first chunk, exactly one chunk is fed,
     * and sending the same message again continues from there without a reset */
    fake_init(&b, 4096);
    cb = mk(&b, 0, 0, NULL);
    script(&b, "Hello", "<|im_end|>", NULL);
    Cancel cn = { 1, 0 };
    CHECK(turn(cb, LONG_MSG, 0, &tb, cancel_after_x, &cn) == LM_OK);
    CHECK(tb.stop == LM_STOP_CANCEL && tb.n_prefill == LM_CHAT_PREFILL_CHUNK && b.pf_calls == 1);
    CHECK(b.pos == LM_CHAT_PREFILL_CHUNK && lm_chat_fed(cb) == LM_CHAT_PREFILL_CHUNK);
    script(&b, "Hello", "<|im_end|>", NULL);
    CHECK(turn(cb, LONG_MSG, 0, &tb, NULL, NULL) == LM_OK);
    CHECK(tb.reprefill == 0 && b.resets == 0 && tb.n_prefill == tb.n_prompt - LM_CHAT_PREFILL_CHUNK);
    lm_chat_free(cb);

    /* NEGATIVE: a failure inside a chunk leaves the state unknown: the next turn resets and re-feeds
     * everything, like a failed single feed */
    fake_init(&b, 4096);
    cb = mk(&b, 0, 0, NULL);
    script(&b, "Hello", "<|im_end|>", NULL);
    b.fail_at = 20; /* inside the second chunk */
    CHECK(turn(cb, LONG_MSG, 0, &tb, NULL, NULL) == LM_ERR_IO);
    b.fail_at = -1;
    script(&b, "Hello", "<|im_end|>", NULL);
    CHECK(turn(cb, LONG_MSG, 0, &tb, NULL, NULL) == LM_OK);
    CHECK(b.resets == 1 && tb.reprefill == 1 && tb.n_prefill == tb.n_prompt);
    lm_chat_free(cb);
    g_use_prefill = 0;
}

static void test_thinking_and_tools(void)
{
    Fake f;
    fake_init(&f, 4096);
    LmChat *c = mk(&f, 1, 0, NULL);
    LmTurn t;
    /* NEGATIVE: cut off inside the think block */
    script(&f, "Think", "Think", "Think", "<|im_end|>", NULL);
    CHECK(turn(c, "Hi", 2, &t, NULL, NULL) == LM_OK);
    CHECK(t.stop == LM_STOP_MAX && t.thinking_open && !strcmp(g_rea, "ThinkThink") && g_ans[0] == 0);
    /* and the conversation carries on without a re-feed */
    f.nlog = 0;
    script(&f, "Think", "\n", "</think>", "\n\n", "Yes", "<|im_end|>", NULL);
    CHECK(turn(c, "Hi", 0, &t, NULL, NULL) == LM_OK && t.reprefill == 0 && !t.thinking_open && !strcmp(g_ans, "Yes"));
    lm_chat_free(c);

    /* a tool-call attempt is reported, not dropped, and generation stops after it */
    fake_init(&f, 4096);
    c = mk(&f, 0, 0, NULL);
    script(&f, "Sure", "<tool_call>", "Hello", "</tool_call>", "Hello", "<|im_end|>", NULL);
    CHECK(turn(c, "Hi", 0, &t, NULL, NULL) == LM_OK);
    CHECK(t.stop == LM_STOP_TOOLCALL && t.tool_calls == 1 && !strcmp(g_tool, "tool calls are not supported"));
    CHECK(!strcmp(g_ans, "Sure") && t.n_gen == 4);
    f.nlog = 0;
    script(&f, "Yes", "<|im_end|>", NULL);
    CHECK(turn(c, "Hi", 0, &t, NULL, NULL) == LM_OK && t.reprefill == 0);
    lm_chat_free(c);
}

static void test_same_seed_same_text(void)
{
    /* a stochastic sampler through the loop is reproducible */
    char outs[2][512];
    for (int r = 0; r < 2; r++) {
        Fake f;
        fake_init(&f, 4096);
        f.n_vocab = lm_tok_n_ids(g_tok);
        LmChatConfig cc;
        memset(&cc, 0, sizeof cc);
        cc.tok = g_tok; cc.tpl = g_tpl;
        cc.backend = (LmBackend){ &f, f.n_vocab, f.n_ctx, fk_feed, fk_reset, fk_pos, NULL, NULL, NULL, NULL, NULL };
        cc.temperature = 1.0; cc.top_p = 0.9; cc.top_k = 5; cc.seed = 42;
        char err[100];
        LmChat *c = lm_chat_new(&cc, err, sizeof err);
        script(&f, "Hello", "Hello", "Hello", "<|im_end|>", NULL);
        LmTurn t;
        turn(c, "Hi", 3, &t, NULL, NULL);
        strcpy(outs[r], g_ans);
        lm_chat_free(c);
    }
    CHECK(!strcmp(outs[0], outs[1]));
}

/* lm_chat_sync: a client that sends the whole conversation back (lm-serve) keeps the fed state
 * when its history matches what this chat generated, and re-feeds when it does not. */
static void test_sync(void)
{
    Fake f;
    fake_init(&f, 4096);
    LmChat *c = mk(&f, 0, 0, NULL);
    LmTurn t;
    char err[256] = "";
    script(&f, "Hello", "<|im_end|>", NULL);
    CHECK(turn(c, "Hi", 0, &t, NULL, NULL) == LM_OK && !strcmp(t.answer, "Hello"));

    LmMsg same[] = { { LM_ROLE_USER, "Hi", NULL, NULL, 0, 0 }, { LM_ROLE_ASSISTANT, "Hello", NULL, NULL, 0, 0 } };
    CHECK(lm_chat_sync(c, same, 2, err, sizeof err) == 0);
    script(&f, "Yes", "<|im_end|>", NULL);
    CHECK(turn(c, "Again", 0, &t, NULL, NULL) == LM_OK);
    CHECK(t.reprefill == 0 && f.resets == 0 && t.n_prefill < t.n_prompt && !strcmp(t.answer, "Yes"));

    /* NEGATIVE: the client edited the earlier answer, so the state is not a prefix any more */
    LmMsg edited[] = { { LM_ROLE_USER, "Hi", NULL, NULL, 0, 0 }, { LM_ROLE_ASSISTANT, "Hello!", NULL, NULL, 0, 0 } };
    CHECK(lm_chat_sync(c, edited, 2, err, sizeof err) == 0);
    script(&f, "Yes", "<|im_end|>", NULL);
    CHECK(turn(c, "Again", 0, &t, NULL, NULL) == LM_OK);
    CHECK(t.reprefill == 1 && f.resets == 1 && t.n_prefill == t.n_prompt);

    /* the conversation just generated (edited history + "Again" + "Yes") is matched again */
    LmMsg longer[] = { { LM_ROLE_USER, "Hi", NULL, NULL, 0, 0 }, { LM_ROLE_ASSISTANT, "Hello!", NULL, NULL, 0, 0 },
                       { LM_ROLE_USER, "Again", NULL, NULL, 0, 0 }, { LM_ROLE_ASSISTANT, "Yes", NULL, NULL, 0, 0 } };
    CHECK(lm_chat_sync(c, longer, 4, err, sizeof err) == 0);
    script(&f, "Hello", "<|im_end|>", NULL);
    CHECK(turn(c, "Hi", 0, &t, NULL, NULL) == LM_OK && t.reprefill == 0 && f.resets == 1);

    /* a system message replaces the chat's and opens the prompt; a new system means a re-feed */
    LmMsg sys[] = { { LM_ROLE_SYSTEM, "Be brief", NULL, NULL, 0, 0 } };
    CHECK(lm_chat_sync(c, sys, 1, err, sizeof err) == 0);
    f.nlog = 0;
    script(&f, "Hello", "<|im_end|>", NULL);
    CHECK(turn(c, "Hi", 0, &t, NULL, NULL) == LM_OK && t.reprefill == 1);
    CHECK(f.log[0] == pid("<|im_start|>") && f.log[1] == pid("system"));

    /* an empty list is a new conversation without a system message */
    CHECK(lm_chat_sync(c, NULL, 0, err, sizeof err) == 0);
    f.nlog = 0;
    script(&f, "Hello", "<|im_end|>", NULL);
    CHECK(turn(c, "Hi", 0, &t, NULL, NULL) == LM_OK && t.reprefill == 1 && f.log[1] == pid("user"));

    /* NEGATIVE: a system message that is not first, or a message without text, is refused and
     * leaves the conversation as it was (the next turn still extends the state) */
    LmMsg late[] = { { LM_ROLE_USER, "Hi", NULL, NULL, 0, 0 }, { LM_ROLE_SYSTEM, "x", NULL, NULL, 0, 0 } };
    err[0] = 0;
    CHECK(lm_chat_sync(c, late, 2, err, sizeof err) == -1 && strstr(err, "system"));
    LmMsg empty[] = { { LM_ROLE_USER, NULL, NULL, NULL, 0, 0 } };
    CHECK(lm_chat_sync(c, empty, 1, err, sizeof err) == -1);
    script(&f, "Yes", "<|im_end|>", NULL);
    CHECK(turn(c, "Again", 0, &t, NULL, NULL) == LM_OK && t.reprefill == 0);
    lm_chat_free(c);
}

/* lm_chat_set_sampling replaces the settings for the next turn: invalid ones make it fail,
 * valid ones work again. */
static void test_set_sampling(void)
{
    Fake f;
    fake_init(&f, 4096);
    LmChat *c = mk(&f, 0, 0, NULL);
    LmTurn t;
    lm_chat_set_sampling(c, 0.7, 0.0 /* invalid top_p */, 0, 7);
    script(&f, "Hello", "<|im_end|>", NULL);
    CHECK(turn(c, "Hi", 0, &t, NULL, NULL) == -1);
    lm_chat_set_sampling(c, 0.0, 1.0, 0, 7);
    script(&f, "Hello", "<|im_end|>", NULL);
    CHECK(turn(c, "Hi", 0, &t, NULL, NULL) == LM_OK && !strcmp(t.answer, "Hello"));
    lm_chat_free(c);
}

/* a NUL byte from the model (the <0x00> byte token) does not cut the kept answer */
static void test_nul_byte(void)
{
    Fake f;
    fake_init(&f, 4096);
    LmChat *c = mk(&f, 0, 0, NULL);
    LmTurn t;
    char nul[8];
    bl("\0", 1, nul);
    const int nul_id = lm_tok_vocab_id(g_tok, nul);
    CHECK(nul_id >= 0);
    script(&f, "A", "A", "B", "<|im_end|>", NULL);
    f.script[1] = nul_id;
    CHECK(turn(c, "Hi", 0, &t, NULL, NULL) == LM_OK && t.n_gen == 3);
    CHECK(!strcmp(t.answer, "AB"));
    lm_chat_free(c);
}

/* ---- prefix cache ------------------------------------------------------------------------ */

/* The state a chat without any cache reaches for one turn: the oracle a cached turn must equal. */
static void oracle(Fake *o, const char *system, const char *user)
{
    const int keep = g_use_cache;
    g_use_cache = 0;
    fake_init(o, 4096);
    LmChat *c = mk(o, 0, 0, system);
    LmTurn t;
    script(o, "Hello", "<|im_end|>", NULL);
    CHECK(turn(c, user, 0, &t, NULL, NULL) == LM_OK);
    lm_chat_free(c);
    g_use_cache = keep;
}

static int same_state(const Fake *a, const Fake *b) { return a->pos == b->pos && !memcmp(a->st, b->st, (size_t)a->pos * sizeof(int)); }

/* where the cache snapshots: the common prefix of two conversations with this system prompt
 * and different first messages, i.e. the system block and the user header, spelled out */
static int shared_prefix(const char *system)
{
    Fake a, b;
    oracle(&a, system, "Hi");
    oracle(&b, system, "Yo");
    int n = 0;
    while (n < a.pos && n < b.pos && a.st[n] == b.st[n]) n++;
    /* <|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n */
    CHECK(n > 6 && a.st[0] == pid("<|im_start|>") && a.st[n - 3] == pid("<|im_start|>") && a.st[n - 2] == pid("user"));
    return n;
}

static void test_prefix_cache(void)
{
    const char *sys = "You are terse. Answer in one short line.";
    const int L = shared_prefix(sys);
    CHECK(L >= LM_CHAT_CACHE_MIN); /* the system block and the next turn's opening */
    for (int pf = 0; pf < 2; pf++) {
        g_use_prefill = pf;
        g_use_cache = 1;
        Fake f, o;
        fake_init(&f, 4096);
        LmChat *c = mk(&f, 0, 0, sys);
        LmTurn t;
        /* conversation A: nothing to compare with yet, nothing cached */
        script(&f, "Hello", "<|im_end|>", NULL);
        CHECK(turn(c, "Hi", 0, &t, NULL, NULL) == LM_OK);
        CHECK(f.saves == 0 && lm_chat_cached(c) == 0 && t.n_cached == 0);

        /* /reset, conversation B: it shares L ids with A, so the prefill snapshots there */
        lm_chat_reset(c);
        script(&f, "Hello", "<|im_end|>", NULL);
        CHECK(turn(c, "Yo", 0, &t, NULL, NULL) == LM_OK);
        CHECK_MSG(f.saves == 1 && lm_chat_cached(c) == L && f.saved_pos == L,
                  "pf=%d: saves %d cached %d saved at %d, want %d", pf, f.saves, lm_chat_cached(c), f.saved_pos, L);
        CHECK_MSG(t.n_cached == 0 && t.n_prefill == t.n_prompt, "pf=%d: cached %d prefill %d prompt %d", pf, t.n_cached, t.n_prefill, t.n_prompt);
        oracle(&o, sys, "Yo");
        CHECK_MSG(same_state(&f, &o), "pf=%d: taking a snapshot changed what was fed", pf);

        /* /reset, conversation C: restored from the cache, only the rest fed, same state */
        lm_chat_reset(c);
        script(&f, "Hello", "<|im_end|>", NULL);
        f.nlog = 0;
        CHECK(turn(c, "Hey", 0, &t, NULL, NULL) == LM_OK);
        CHECK_MSG(t.n_cached == L && t.n_prefill == t.n_prompt - L && f.restores == 1,
                  "pf=%d: cached %d prefill %d prompt %d", pf, t.n_cached, t.n_prefill, t.n_prompt);
        CHECK(f.saves == 1 && f.nlog == t.n_prompt - L + 1);
        oracle(&o, sys, "Hey");
        CHECK_MSG(same_state(&f, &o), "pf=%d: the restored conversation differs from a fresh one", pf);
        CHECK(!strcmp(g_ans, "Hello"));

        /* the next turn of C extends the state: no cache involved */
        script(&f, "Hello", "<|im_end|>", NULL);
        CHECK(turn(c, "More", 0, &t, NULL, NULL) == LM_OK && t.n_cached == 0 && t.reprefill == 0 && f.restores == 1);
        lm_chat_free(c);
        CHECK(f.live == 0); /* the chat dropped its snapshot */
    }
    g_use_prefill = 0;
    g_use_cache = 0;
}

static void test_prefix_cache_sync(void)
{
    /* lm-serve: no reset, each request replaces the conversation */
    g_use_cache = 1;
    Fake f, o;
    fake_init(&f, 4096);
    LmChat *c = mk(&f, 0, 0, NULL);
    char err[200];
    LmTurn t;
    LmMsg a[] = { { LM_ROLE_SYSTEM, "You are terse. Answer in one short line.", NULL, NULL, 0, 0 }, { LM_ROLE_USER, "Hi", NULL, NULL, 0, 0 } };
    CHECK(lm_chat_sync(c, a, 1, err, sizeof err) == 0);
    script(&f, "Hello", "<|im_end|>", NULL);
    CHECK(turn(c, "Hi", 0, &t, NULL, NULL) == LM_OK && f.saves == 0);
    LmMsg b[] = { { LM_ROLE_SYSTEM, "You are terse. Answer in one short line.", NULL, NULL, 0, 0 } };
    CHECK(lm_chat_sync(c, b, 1, err, sizeof err) == 0);
    script(&f, "Hello", "<|im_end|>", NULL);
    CHECK(turn(c, "Yo", 0, &t, NULL, NULL) == LM_OK);
    const int L = lm_chat_cached(c);
    CHECK(t.reprefill == 1 && f.saves == 1 && L == shared_prefix("You are terse. Answer in one short line."));
    CHECK(lm_chat_sync(c, b, 1, err, sizeof err) == 0);
    const int reprefills = lm_chat_reprefills(c);
    script(&f, "Hello", "<|im_end|>", NULL);
    CHECK(turn(c, "Hey", 0, &t, NULL, NULL) == LM_OK && t.n_cached == L);
    CHECK(t.reprefill == 0 && lm_chat_reprefills(c) == reprefills); /* restored, not re-fed from scratch */
    oracle(&o, "You are terse. Answer in one short line.", "Hey");
    CHECK(same_state(&f, &o));

    /* NEGATIVE: an edited answer shares the whole conversation with what was fed, but the
     * snapshot still stops at the system message (its size is bounded by it) */
    LmMsg e[] = { { LM_ROLE_SYSTEM, "You are terse. Answer in one short line.", NULL, NULL, 0, 0 },
                  { LM_ROLE_USER, "Hey", NULL, NULL, 0, 0 }, { LM_ROLE_ASSISTANT, "Edited", NULL, NULL, 0, 0 } };
    CHECK(lm_chat_sync(c, e, 3, err, sizeof err) == 0);
    script(&f, "Hello", "<|im_end|>", NULL);
    const int saves0 = f.saves;
    CHECK(turn(c, "More", 0, &t, NULL, NULL) == LM_OK && t.n_cached == L && f.saves == saves0 && lm_chat_cached(c) == L);

    /* NEGATIVE: another system prompt is not served from the cache; the state is still right */
    LmMsg d[] = { { LM_ROLE_SYSTEM, "You are verbose. Answer in many long lines.", NULL, NULL, 0, 0 } };
    CHECK(lm_chat_sync(c, d, 1, err, sizeof err) == 0);
    script(&f, "Hello", "<|im_end|>", NULL);
    const int restores = f.restores, saves = f.saves;
    CHECK(turn(c, "Hey", 0, &t, NULL, NULL) == LM_OK && t.n_cached == 0 && f.restores == restores);
    oracle(&o, "You are verbose. Answer in many long lines.", "Hey");
    CHECK(same_state(&f, &o));
    /* NEGATIVE: the few ids the two share (the system header) did not evict the cache, which
     * still serves the first system prompt */
    CHECK(f.saves == saves && lm_chat_cached(c) == L);
    CHECK(lm_chat_sync(c, b, 1, err, sizeof err) == 0);
    script(&f, "Hello", "<|im_end|>", NULL);
    CHECK(turn(c, "Hey", 0, &t, NULL, NULL) == LM_OK && t.n_cached == L);

    /* two conversations in a row with the other system prompt replace the cache, and the old
     * snapshot is freed before the new one is taken: never two alive at once */
    CHECK(lm_chat_sync(c, d, 1, err, sizeof err) == 0);
    script(&f, "Hello", "<|im_end|>", NULL);
    CHECK(turn(c, "Hey", 0, &t, NULL, NULL) == LM_OK && t.n_cached == 0);
    CHECK(lm_chat_sync(c, d, 1, err, sizeof err) == 0);
    script(&f, "Hello", "<|im_end|>", NULL);
    CHECK(turn(c, "Yo", 0, &t, NULL, NULL) == LM_OK);
    CHECK(f.saves == saves + 1 && lm_chat_cached(c) == shared_prefix("You are verbose. Answer in many long lines."));
    CHECK(f.maxlive == 1);
    lm_chat_free(c);
    CHECK(f.live == 0);
    g_use_cache = 0;
}

/* lm-serve style: each request is one conversation with its own system prompt */
static int conv(LmChat *c, Fake *f, const char *system, const char *user, LmTurn *t)
{
    char err[200];
    LmMsg m[] = { { LM_ROLE_SYSTEM, system, NULL, NULL, 0, 0 } };
    if (lm_chat_sync(c, m, 1, err, sizeof err) != 0) return -1;
    script(f, "Hello", "<|im_end|>", NULL);
    return turn(c, user, 0, t, NULL, NULL);
}

static void test_prefix_cache_slots(void)
{
    const char *A = "You are terse. Answer in one short line.", *B = "You are verbose. Answer in many long lines.",
               *C = "You are a pirate. Answer like one, briefly.";
    const int LA = shared_prefix(A), LB = shared_prefix(B), LC = shared_prefix(C);
    g_use_cache = 1;
    g_cache_bytes = 2 * 1000 + LA + LB + 50;   /* room for two snapshots (fk_bytes: 1000 + ids), not three */
    Fake f, o;
    fake_init(&f, 4096);
    LmChat *c = mk(&f, 0, 0, NULL);
    LmTurn t;
    /* A is cached after its second conversation, as with one slot */
    CHECK(conv(c, &f, A, "Hi", &t) == LM_OK && conv(c, &f, A, "Yo", &t) == LM_OK && lm_chat_cache_slots(c) == 1);
    /* B gets its own slot and A's stays */
    CHECK(conv(c, &f, B, "Hi", &t) == LM_OK && conv(c, &f, B, "Yo", &t) == LM_OK);
    CHECK(lm_chat_cache_slots(c) == 2 && lm_chat_cached(c) == LB && f.maxlive == 2);
    CHECK(lm_chat_cache_bytes(c) == 2000 + (size_t)LA + LB);
    /* alternating clients restore each time: no saves, the state as if fed */
    const int saves = f.saves;
    for (int k = 0; k < 2; k++) {
        CHECK(conv(c, &f, A, "Hey", &t) == LM_OK && t.n_cached == LA && lm_chat_cached(c) == LA);
        oracle(&o, A, "Hey");
        CHECK(same_state(&f, &o));
        CHECK(conv(c, &f, B, "Hey", &t) == LM_OK && t.n_cached == LB && lm_chat_cached(c) == LB);
        oracle(&o, B, "Hey");
        CHECK(same_state(&f, &o));
    }
    CHECK(f.saves == saves);
    /* a third system prompt evicts the least recently used one (A), never past the budget */
    CHECK(conv(c, &f, C, "Hi", &t) == LM_OK && conv(c, &f, C, "Yo", &t) == LM_OK);
    CHECK(lm_chat_cache_slots(c) == 2 && lm_chat_cached(c) == LC && lm_chat_cache_bytes(c) <= g_cache_bytes);
    CHECK(f.live == 2);
    CHECK(conv(c, &f, B, "Hey", &t) == LM_OK && t.n_cached == LB);
    CHECK(conv(c, &f, A, "Hey", &t) == LM_OK && t.n_cached == 0);   /* NEGATIVE: evicted */
    oracle(&o, A, "Hey");
    CHECK(same_state(&f, &o));
    /* a slot that fails to restore is dropped, the turn fed in full */
    f.restore_fail = 1;
    const int slots = lm_chat_cache_slots(c);
    CHECK(conv(c, &f, B, "Hey", &t) == LM_OK && t.n_cached == 0 && lm_chat_cache_slots(c) == slots - 1);
    oracle(&o, B, "Hey");
    CHECK(same_state(&f, &o));
    lm_chat_free(c);
    CHECK(f.live == 0);
    g_use_cache = 0;
    g_cache_bytes = 0;
}

static void test_prefix_cache_failures(void)
{
    g_use_cache = 1;
    Fake f, o;
    LmTurn t;
    /* save fails: no cache, the turns are unaffected */
    fake_init(&f, 4096);
    f.save_fail = 1;
    LmChat *c = mk(&f, 0, 0, "You are terse. Answer in one short line.");
    script(&f, "Hello", "<|im_end|>", NULL);
    CHECK(turn(c, "Hi", 0, &t, NULL, NULL) == LM_OK);
    lm_chat_reset(c);
    script(&f, "Hello", "<|im_end|>", NULL);
    CHECK(turn(c, "Yo", 0, &t, NULL, NULL) == LM_OK && f.saves == 1 && lm_chat_cached(c) == 0);
    oracle(&o, "You are terse. Answer in one short line.", "Yo");
    CHECK(same_state(&f, &o));
    lm_chat_free(c);

    /* restore fails: the backend is reset and everything fed */
    fake_init(&f, 4096);
    c = mk(&f, 0, 0, "You are terse. Answer in one short line.");
    script(&f, "Hello", "<|im_end|>", NULL);
    CHECK(turn(c, "Hi", 0, &t, NULL, NULL) == LM_OK);
    lm_chat_reset(c);
    script(&f, "Hello", "<|im_end|>", NULL);
    CHECK(turn(c, "Yo", 0, &t, NULL, NULL) == LM_OK && lm_chat_cached(c) > 0);
    lm_chat_reset(c);
    f.restore_fail = 1;
    const int resets = f.resets;
    script(&f, "Hello", "<|im_end|>", NULL);
    CHECK(turn(c, "Hey", 0, &t, NULL, NULL) == LM_OK && f.restores == 1 && t.n_cached == 0);
    CHECK(t.n_prefill == t.n_prompt && f.resets == resets + 1);
    oracle(&o, "You are terse. Answer in one short line.", "Hey");
    CHECK(same_state(&f, &o));
    lm_chat_free(c);
    CHECK(f.live == 0);

    /* NEGATIVE: without a system message nothing is cached, however much two conversations share */
    fake_init(&f, 4096);
    c = mk(&f, 0, 0, NULL);
    script(&f, "Hello", "<|im_end|>", NULL);
    CHECK(turn(c, "The same long opening line for both of these conversations, then Hi", 0, &t, NULL, NULL) == LM_OK);
    lm_chat_reset(c);
    script(&f, "Hello", "<|im_end|>", NULL);
    CHECK(turn(c, "The same long opening line for both of these conversations, then Yo", 0, &t, NULL, NULL) == LM_OK);
    CHECK(f.saves == 0 && lm_chat_cached(c) == 0);
    lm_chat_free(c);

    /* the hooks come all four or none */
    LmChatConfig cc;
    memset(&cc, 0, sizeof cc);
    cc.tok = g_tok; cc.tpl = g_tpl;
    fake_init(&f, 4096);
    cc.backend = (LmBackend){ &f, f.n_vocab, f.n_ctx, fk_feed, fk_reset, fk_pos, NULL, fk_save, fk_restore, NULL, fk_bytes };
    char err[100];
    CHECK(lm_chat_new(&cc, err, sizeof err) == NULL);
    cc.backend = (LmBackend){ &f, f.n_vocab, f.n_ctx, fk_feed, fk_reset, fk_pos, NULL, fk_save, fk_restore, fk_drop, NULL };
    CHECK(lm_chat_new(&cc, err, sizeof err) == NULL);
    g_use_cache = 0;
}

/* Purge drops the transcript, the system prompt and every prefix snapshot. */
static void test_purge(void)
{
    g_use_cache = 1;
    Fake f;
    LmTurn t;
    fake_init(&f, 4096);
    LmChat *c = mk(&f, 0, 0, "You are terse. Answer in one short line.");
    script(&f, "Hello", "<|im_end|>", NULL);
    CHECK(turn(c, "Hi", 0, &t, NULL, NULL) == LM_OK);
    lm_chat_reset(c);
    script(&f, "Hello", "<|im_end|>", NULL);
    CHECK(turn(c, "Yo", 0, &t, NULL, NULL) == LM_OK && lm_chat_cached(c) > 0);
    lm_chat_purge(c);
    CHECK(lm_chat_cached(c) == 0 && lm_chat_cache_slots(c) == 0 && lm_chat_fed(c) == 0);
    CHECK(lm_chat_set_system(c, "You are terse. Answer in one short line.") == 0);
    script(&f, "Hello", "<|im_end|>", NULL);
    CHECK(turn(c, "Hi", 0, &t, NULL, NULL) == LM_OK && t.n_cached == 0);
    lm_chat_free(c);
    CHECK(f.live == 0);
    g_use_cache = 0;
}

int main(void)
{
    char err[256];
    g_tok = lm_tok_open(FIXTURE, err, sizeof err);
    if (!g_tok) { fprintf(stderr, "cannot open fixture: %s\n", err); return 1; }
    int eos[1] = { lm_tok_special_id(g_tok, "<|endoftext|>") };
    g_tpl = lm_template_qwen35(g_tok, eos, 1, err, sizeof err);
    if (!g_tpl) { fprintf(stderr, "cannot build template: %s\n", err); return 1; }

    test_prefix_and_tail();
    test_mismatch_resets();
    test_reprefill_report();
    test_ctx_full();
    test_backend_error();
    test_cancel_and_max();
    test_prefill_backend();
    test_thinking_and_tools();
    test_same_seed_same_text();
    test_sync();
    test_set_sampling();
    test_nul_byte();
    test_prefix_cache();
    test_prefix_cache_sync();
    test_prefix_cache_failures();
    test_prefix_cache_slots();
    test_purge();

    lm_template_free(g_tpl);
    lm_tok_close(g_tok);
    printf("test_chat: %d checks, %d failed\n", g_checks, g_fail);
    return g_fail ? 1 : 0;
}
