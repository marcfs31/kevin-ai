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
} Fake;

static int fk_feed(void *c, int token, float *logits)
{
    Fake *f = (Fake *)c;
    int call = f->calls++;
    if (f->fail_at >= 0 && call == f->fail_at) return LM_ERR_IO;
    if (f->pos >= f->n_ctx) return LM_ERR_CTX;
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
    cc.backend = (LmBackend){ f, f->n_vocab, f->n_ctx, fk_feed, fk_reset, fk_pos };
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
    LmTurnOpts o = { max_tokens, sink, NULL, cancel, cu };
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

typedef struct { int after, seen; } Cancel;
static int cancel_after(void *u) { Cancel *c = (Cancel *)u; return ++c->seen > c->after; }

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
        cc.backend = (LmBackend){ &f, f.n_vocab, f.n_ctx, fk_feed, fk_reset, fk_pos };
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
    test_thinking_and_tools();
    test_same_seed_same_text();

    lm_template_free(g_tpl);
    lm_tok_close(g_tok);
    printf("test_chat: %d checks, %d failed\n", g_checks, g_fail);
    return g_fail ? 1 : 0;
}
