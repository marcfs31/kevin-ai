/* test_chat_gemma.c -- the Gemma 4 chat format: template rendering, the streaming parser,
 * the model-directory profile (family, stop ids, sampling defaults) and the generation loop
 * with verbatim history, all against the synthetic fixture tests/fixtures/tok/
 * gemma-tokenizer.json. Run from the repo root.
 *
 * Expected prompts are STRINGS written by hand from chat_template.jinja (the model's own
 * template). Each is turned into ids by lm_tok_encode_special, which splits on added tokens
 * exactly as the Hugging Face tokenizer does on the rendered jinja output (and is pinned to
 * the Python reference by test_tok_gemma), so a pass means "the template builds the ids HF
 * would build from the jinja's string". */
#include "lm_test.h"
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "chat/lm_chat.h"
#include "chat/lm_profile.h"
#include "chat/lm_template.h"
#include "lm/lm.h"

#define FIX "tests/fixtures/tok/gemma-tokenizer.json"

static LmTok *g_tok;
static LmTemplate *g_tpl;

static int sid(const char *s) { return lm_tok_special_id(g_tok, s); }

static int *enc_special(const char *text, int *n) { return lm_tok_encode_special(g_tok, text, strlen(text), n); }
static int *enc_plain(const char *text, int *n) { return lm_tok_encode(g_tok, text, strlen(text), n); }

static int ids_eq(const LmIds *got, const int *want, int nw)
{
    return got->n == nw && (nw == 0 || !memcmp(got->ids, want, (size_t)nw * sizeof(int)));
}

static void dump(const char *what, const int *ids, int n)
{
    fprintf(stderr, "  %s:", what);
    for (int i = 0; i < n; i++) fprintf(stderr, " %d", ids[i]);
    fprintf(stderr, "\n");
}

/* Render msgs and compare with the encoding of the jinja's output string. */
static int renders(const LmMsg *msgs, int n, int thinking, int verbatim, const char *jinja_out)
{
    LmRenderOpts ro = { thinking, verbatim };
    LmIds got = { 0 };
    char err[200] = "";
    int rc = g_tpl->render(g_tpl, msgs, n, &ro, &got, err, sizeof err);
    int nw;
    int *want = enc_special(jinja_out, &nw);
    int ok = rc == 0 && want && ids_eq(&got, want, nw);
    if (!ok) {
        fprintf(stderr, "  render mismatch for %s (rc=%d %s)\n", jinja_out, rc, err);
        dump("got ", got.ids, got.n);
        dump("want", want, nw);
    }
    free(want);
    lm_ids_free(&got);
    return ok;
}

static LmMsg U(const char *c) { LmMsg m; memset(&m, 0, sizeof m); m.role = LM_ROLE_USER; m.content = c; return m; }
static LmMsg S(const char *c) { LmMsg m; memset(&m, 0, sizeof m); m.role = LM_ROLE_SYSTEM; m.content = c; return m; }
static LmMsg A(const char *c) { LmMsg m; memset(&m, 0, sizeof m); m.role = LM_ROLE_ASSISTANT; m.content = c; return m; }

static void test_render(void)
{
    /* no system message, thinking off: only <bos> and the turns */
    LmMsg m1[] = { U("Hello") };
    CHECK(renders(m1, 1, 0, 0, "<bos><|turn>user\nHello<turn|>\n<|turn>model\n"));

    /* system message, trimmed (Python str.strip, incl. NBSP and U+3000), user trimmed */
    LmMsg m2[] = { S("  Be brief.\n\xC2\xA0"), U("\n hi there \xE3\x80\x80") };
    CHECK(renders(m2, 2, 0, 0, "<bos><|turn>system\nBe brief.<turn|>\n<|turn>user\nhi there<turn|>\n<|turn>model\n"));

    /* thinking on, no system message: the system turn exists and holds only <|think|>\n */
    CHECK(renders(m1, 1, 1, 0, "<bos><|turn>system\n<|think|>\n<turn|>\n<|turn>user\nHello<turn|>\n<|turn>model\n"));

    /* thinking on with a system message: <|think|>\n comes first in the same turn */
    LmMsg m3[] = { S("Be brief."), U("Hello") };
    CHECK(renders(m3, 2, 1, 0, "<bos><|turn>system\n<|think|>\nBe brief.<turn|>\n<|turn>user\nHello<turn|>\n<|turn>model\n"));

    /* multi-turn: the model turn is re-rendered as its answer, thought blocks stripped,
     * then trimmed; the generation prompt follows the last user turn */
    LmMsg m4[] = { U("Hi"), A("Hello! <|channel>thought\nx<channel|>there "), U("Bye") };
    CHECK(renders(m4, 3, 0, 0,
                  "<bos><|turn>user\nHi<turn|>\n<|turn>model\nHello! there<turn|>\n<|turn>user\nBye<turn|>\n<|turn>model\n"));

    /* strip_thinking: two blocks, and an unterminated one (its tail survives only up to the opener) */
    LmMsg m5[] = { U("q"), A("<|channel>a<channel|>X<|channel>b<channel|>Y"), U("r") };
    CHECK(renders(m5, 3, 0, 0, "<bos><|turn>user\nq<turn|>\n<|turn>model\nXY<turn|>\n<|turn>user\nr<turn|>\n<|turn>model\n"));
    LmMsg m6[] = { U("q"), A("keep<|channel>unterminated"), U("r") };
    CHECK(renders(m6, 3, 0, 0, "<bos><|turn>user\nq<turn|>\n<|turn>model\nkeep<turn|>\n<|turn>user\nr<turn|>\n<|turn>model\n"));

    /* a reasoning field is rendered only for a model turn after the last user message */
    LmMsg m7[] = { U("q"), A("answer") };
    m7[1].reasoning = "R";
    CHECK(renders(m7, 2, 0, 0, "<bos><|turn>user\nq<turn|>\n<|turn>model\n<|channel>thought\nR\n<channel|>answer<turn|>\n<|turn>model\n"));
    LmMsg m8[] = { U("q"), A("answer"), U("r") };
    m8[1].reasoning = "R";
    CHECK(renders(m8, 3, 0, 0, "<bos><|turn>user\nq<turn|>\n<|turn>model\nanswer<turn|>\n<|turn>user\nr<turn|>\n<|turn>model\n"));

    /* empty content still opens and closes its turn */
    LmMsg m9[] = { U("") };
    CHECK(renders(m9, 1, 0, 0, "<bos><|turn>user\n<turn|>\n<|turn>model\n"));

    /* message text is ordinary text: a spelled "<turn|>" or "<bos>" is never a control id.
     * HF would split on them; this renderer must not, so the expected ids are built by hand:
     * the segment between two structural ids is encoded as one string. */
    {
        LmMsg m[] = { U("<turn|><bos>x") };
        LmRenderOpts ro = { 0, 0 };
        LmIds got = { 0 };
        char err[100];
        CHECK(g_tpl->render(g_tpl, m, 1, &ro, &got, err, sizeof err) == 0);
        LmIds want = { 0 };
        int n;
        lm_ids_push(&want, sid("<bos>"));
        lm_ids_push(&want, sid("<|turn>"));
        int *u = enc_plain("user\n<turn|><bos>x", &n);
        lm_ids_append(&want, u, n);
        free(u);
        lm_ids_push(&want, sid("<turn|>"));
        u = enc_plain("\n", &n);
        lm_ids_append(&want, u, n);
        free(u);
        lm_ids_push(&want, sid("<|turn>"));
        u = enc_plain("model\n", &n);
        lm_ids_append(&want, u, n);
        free(u);
        CHECK(ids_eq(&got, want.ids, want.n));
        int control = 0;
        for (int i = 0; i < got.n; i++) if (got.ids[i] == sid("<turn|>")) control++;
        CHECK(control == 1); /* only the template's own */
        lm_ids_free(&got);
        lm_ids_free(&want);
    }

    /* errors */
    LmRenderOpts ro = { 0, 0 };
    LmIds out = { 0 };
    char err[200];
    CHECK(g_tpl->render(g_tpl, m1, 0, &ro, &out, err, sizeof err) == -1 && strstr(err, "No messages"));
    LmMsg only_sys[] = { S("x") };
    CHECK(g_tpl->render(g_tpl, only_sys, 1, &ro, &out, err, sizeof err) == -1 && strstr(err, "No user query"));
    LmMsg sys_late[] = { U("a"), S("x") };
    CHECK(g_tpl->render(g_tpl, sys_late, 2, &ro, &out, err, sizeof err) == -1 && strstr(err, "System message must be at the beginning"));
    LmMsg two_a[] = { U("a"), A("b"), A("c") };
    CHECK(g_tpl->render(g_tpl, two_a, 3, &ro, &out, err, sizeof err) == -1 && strstr(err, "Consecutive assistant"));
    lm_ids_free(&out);
}

static void test_verbatim_render(void)
{
    /* an assistant message that carries its generated ids is rendered from them */
    int gen[] = { sid("<|channel>"), 777, 778, sid("<channel|>"), 779 };
    LmMsg m[] = { U("Hi"), A("ignored"), U("Bye") };
    m[1].gen_ids = gen;
    m[1].n_gen_ids = 5;
    LmRenderOpts ro = { 0, 1 };
    LmIds got = { 0 };
    char err[100];
    CHECK(g_tpl->render(g_tpl, m, 3, &ro, &got, err, sizeof err) == 0);
    LmIds want = { 0 };
    int n;
    int *a = enc_special("<bos><|turn>user\nHi<turn|>\n<|turn>model\n", &n);
    lm_ids_append(&want, a, n);
    free(a);
    lm_ids_append(&want, gen, 5);
    a = enc_special("<turn|>\n<|turn>user\nBye<turn|>\n<|turn>model\n", &n);
    lm_ids_append(&want, a, n);
    free(a);
    CHECK(ids_eq(&got, want.ids, want.n));
    lm_ids_free(&got);
    /* the same conversation with verbatim off drops the thought block and uses the text */
    ro.verbatim_history = 0;
    CHECK(renders(m, 3, 0, 0, "<bos><|turn>user\nHi<turn|>\n<|turn>model\nignored<turn|>\n<|turn>user\nBye<turn|>\n<|turn>model\n"));
    lm_ids_free(&want);
}

/* ---- the parser --------------------------------------------------------------------- */

typedef struct { char rea[256], ans[256], tool[128]; } Got;
static void gsink(void *u, LmOutKind k, const char *s, size_t n)
{
    Got *g = (Got *)u;
    strncat(k == LM_OUT_REASONING ? g->rea : k == LM_OUT_ANSWER ? g->ans : g->tool, s, n);
}

/* Feed ids until a stop; returns the parse code of the last feed. */
static int feed_all(LmParser *p, const int *ids, int n, Got *g)
{
    int rc = LM_PARSE_CONT;
    for (int i = 0; i < n && rc == LM_PARSE_CONT; i++) rc = p->feed(p, ids[i], gsink, g);
    p->finish(p, gsink, g);
    return rc;
}

static int parse_seq(int thinking, Got *g, int *thinking_open, int *tool_calls, int nparts, ...)
{
    LmParser *p = g_tpl->parser_new(g_tpl, thinking);
    LmIds all = { 0 };
    va_list ap;
    va_start(ap, nparts);
    for (int i = 0; i < nparts; i++) {
        const char *part = va_arg(ap, const char *);
        if (part[0] == '<' && sid(part) >= 0) lm_ids_push(&all, sid(part));
        else { int n; int *x = enc_plain(part, &n); lm_ids_append(&all, x, n); free(x); }
    }
    va_end(ap);
    memset(g, 0, sizeof *g);
    int rc = feed_all(p, all.ids, all.n, g);
    if (thinking_open) *thinking_open = p->thinking_open;
    if (tool_calls) *tool_calls = p->tool_calls;
    p->destroy(p);
    lm_ids_free(&all);
    return rc;
}

static void test_parser(void)
{
    Got g;
    int open, tools;
    /* a plain answer ends at <turn|>, which is not part of the output */
    CHECK(parse_seq(0, &g, &open, &tools, 3, "  Hello there", "<turn|>", "ignored") == LM_PARSE_STOP);
    CHECK(!strcmp(g.ans, "Hello there") && !g.rea[0] && !open && tools == 0);

    /* a thought channel: the channel name "thought\n" is not reasoning; reasoning and answer
     * are trimmed; thinking_open reflects an unclosed block */
    CHECK(parse_seq(1, &g, &open, &tools, 5, "<|channel>", "thought\nI think so. ", "<channel|>", "Yes.", "<turn|>") == LM_PARSE_STOP);
    CHECK(!strcmp(g.rea, "I think so.") && !strcmp(g.ans, "Yes.") && !open);
    CHECK(parse_seq(1, &g, &open, &tools, 2, "<|channel>", "thought\nstill going") == LM_PARSE_CONT);
    CHECK(!strcmp(g.rea, "still going") && !g.ans[0] && open);
    /* with thinking off the output still starts as the answer (the prompt opens no block) */
    CHECK(parse_seq(0, &g, &open, &tools, 4, "<|channel>", "thought\n", "<channel|>", "Hi") == LM_PARSE_CONT);
    CHECK(!g.rea[0] && !strcmp(g.ans, "Hi"));
    /* the same with thinking on: the parser does not assume a block */
    CHECK(parse_seq(1, &g, &open, &tools, 1, "Hi") == LM_PARSE_CONT);
    CHECK(!strcmp(g.ans, "Hi") && !g.rea[0]);

    /* every configured stop id stops: <eos>, <turn|>, <|tool_response> */
    int stops[] = { 1, 106, 50 };
    for (int i = 0; i < 3; i++) {
        LmParser *p = g_tpl->parser_new(g_tpl, 0);
        CHECK(p->feed(p, stops[i], gsink, &g) == LM_PARSE_STOP);
        p->destroy(p);
        CHECK(g_tpl->is_stop(g_tpl, stops[i]));
    }
    CHECK(!g_tpl->is_stop(g_tpl, 107) && !g_tpl->is_stop(g_tpl, 0) && !g_tpl->is_stop(g_tpl, sid("<bos>")));

    /* a tool call is reported as unsupported and ends the turn after <tool_call|> */
    CHECK(parse_seq(0, &g, &open, &tools, 4, "<|tool_call>", "call:f{x:1}", "<tool_call|>", "more") == LM_PARSE_STOP_AFTER);
    CHECK(tools == 1 && strstr(g.tool, "not supported") && !g.ans[0]);
}

/* ---- the profile: family, stop ids, sampling defaults --------------------------------- */

static void write_cfg(const char *dir, const char *leaf, const char *text)
{
    char p[1024];
    lt_path(p, dir, leaf);
    lt_write_text(p, text);
}

static void test_profile(void)
{
    char dir[512], err[300];
    lt_mkdtemp(dir);

    /* Gemma: stop ids and sampling from generation_config.json (the release's content) */
    write_cfg(dir, "config.json", "{\"model_type\":\"gemma4\",\"eos_token_id\":[1,106],\"text_config\":{\"model_type\":\"gemma4_text\"}}");
    write_cfg(dir, "generation_config.json",
              "{\"bos_token_id\":2,\"do_sample\":true,\"eos_token_id\":[1,106,50],\"pad_token_id\":0,"
              "\"temperature\":1.0,\"top_k\":64,\"top_p\":0.95}");
    LmProfile p;
    CHECK(lm_profile_read(dir, &p, err, sizeof err) == 0);
    CHECK(!strcmp(p.model_type, "gemma4") && p.n_eos == 3 && p.eos[0] == 1 && p.eos[1] == 106 && p.eos[2] == 50);
    CHECK(p.has_temperature && p.temperature == 1.0 && p.has_top_k && p.top_k == 64 && p.has_top_p && p.top_p == 0.95);
    LmTemplate *t = lm_profile_template(&p, g_tok, err, sizeof err);
    CHECK(t && !strcmp(t->name, "gemma4"));
    if (t) {
        CHECK(t->is_stop(t, 1) && t->is_stop(t, 106) && t->is_stop(t, 50) && !t->is_stop(t, 2));
        lm_template_free(t);
    }

    /* no generation_config.json: stop ids from config.json, no sampling defaults */
    char path[1024];
    lt_path(path, dir, "generation_config.json");
    CHECK(unlink(path) == 0);
    CHECK(lm_profile_read(dir, &p, err, sizeof err) == 0);
    CHECK(p.n_eos == 2 && p.eos[0] == 1 && p.eos[1] == 106 && !p.has_temperature && !p.has_top_k && !p.has_top_p);

    /* a partial generation_config.json gives only what it has */
    write_cfg(dir, "generation_config.json", "{\"top_k\":40}");
    CHECK(lm_profile_read(dir, &p, err, sizeof err) == 0);
    CHECK(p.has_top_k && p.top_k == 40 && !p.has_temperature && !p.has_top_p && p.n_eos == 2);

    /* invalid values and malformed files are errors, not defaults */
    write_cfg(dir, "generation_config.json", "{\"top_p\":2}");
    CHECK_ERR(lm_profile_read(dir, &p, err, sizeof err), err, "top_p");
    write_cfg(dir, "generation_config.json", "{\"temperature\":-1}");
    CHECK_ERR(lm_profile_read(dir, &p, err, sizeof err), err, "temperature");
    write_cfg(dir, "generation_config.json", "{\"top_k\":1.5}");
    CHECK_ERR(lm_profile_read(dir, &p, err, sizeof err), err, "top_k");
    write_cfg(dir, "generation_config.json", "{\"top_k\":");
    CHECK_ERR(lm_profile_read(dir, &p, err, sizeof err), err, "generation_config.json");

    /* Qwen: generation_config.json is not read, so phase-1 behaviour is unchanged */
    write_cfg(dir, "config.json", "{\"model_type\":\"qwen3_5\",\"text_config\":{\"eos_token_id\":248044}}");
    write_cfg(dir, "generation_config.json", "{\"temperature\":0.1,\"top_k\":3}");
    CHECK(lm_profile_read(dir, &p, err, sizeof err) == 0);
    CHECK(!strcmp(p.model_type, "qwen3_5") && p.n_eos == 1 && p.eos[0] == 248044 && !p.has_temperature && !p.has_top_k);
    char qerr[200];
    LmTok *qtok = lm_tok_open("tests/fixtures/tok/qwen-tokenizer.json", qerr, sizeof qerr);
    CHECK(qtok != NULL);
    if (qtok) {
        t = lm_profile_template(&p, qtok, err, sizeof err);
        CHECK(t && !strcmp(t->name, "qwen3.5"));
        lm_template_free(t);
        /* a Qwen tokenizer cannot serve a Gemma template */
        LmProfile gp;
        memset(&gp, 0, sizeof gp);
        strcpy(gp.model_type, "gemma4");
        CHECK(lm_profile_template(&gp, qtok, err, sizeof err) == NULL && strstr(err, "required special token"));
        lm_tok_close(qtok);
    }

    /* an unknown model_type is an error naming it; so is a config without one */
    write_cfg(dir, "config.json", "{\"model_type\":\"llama\"}");
    CHECK(lm_profile_read(dir, &p, err, sizeof err) == 0);
    CHECK(lm_profile_template(&p, g_tok, err, sizeof err) == NULL && strstr(err, "\"llama\"") && strstr(err, "no chat template"));
    write_cfg(dir, "config.json", "{\"hidden_size\":1}");
    CHECK_ERR(lm_profile_read(dir, &p, err, sizeof err), err, "no model_type");
    write_cfg(dir, "config.json", "[1]");
    CHECK_ERR(lm_profile_read(dir, &p, err, sizeof err), err, "config.json");
    lt_path(path, dir, "config.json");
    CHECK(unlink(path) == 0);
    CHECK_ERR(lm_profile_read(dir, &p, err, sizeof err), err, "cannot open");

    lt_rmtree(dir);
}

/* ---- the loop: verbatim history extends what was fed ----------------------------------- */

enum { MAXLOG = 8192 };
typedef struct {
    int n_vocab, n_ctx, pos;
    int log[MAXLOG], nlog, resets;
    int script[256], ns, cursor;
} Fake;

static int fk_feed(void *c, int token, float *logits)
{
    Fake *f = (Fake *)c;
    if (f->pos >= f->n_ctx) return LM_ERR_CTX;
    f->log[f->nlog++] = token;
    f->pos++;
    if (logits) {
        for (int i = 0; i < f->n_vocab; i++) logits[i] = -5.0f;
        logits[f->script[f->cursor < f->ns ? f->cursor : f->ns - 1]] = 5.0f;
        f->cursor++;
    }
    return LM_OK;
}
static void fk_reset(void *c) { Fake *f = (Fake *)c; f->pos = 0; f->resets++; f->nlog = 0; }
static int fk_pos(void *c) { return ((Fake *)c)->pos; }

static LmChat *mk_chat(Fake *f, int strict)
{
    LmChatConfig cc;
    memset(&cc, 0, sizeof cc);
    cc.tok = g_tok;
    cc.tpl = g_tpl;
    cc.backend = (LmBackend){ f, f->n_vocab, f->n_ctx, fk_feed, fk_reset, fk_pos, NULL, NULL, NULL, NULL, NULL };
    cc.temperature = 0;
    cc.top_p = 1;
    cc.seed = 1;
    cc.strict_history = strict;
    char err[200];
    LmChat *c = lm_chat_new(&cc, err, sizeof err);
    if (!c) fprintf(stderr, "lm_chat_new: %s\n", err);
    return c;
}

/* script: <|channel> thought\nhmm <channel|> Hello there <turn|> */
static void load_script(Fake *f, int with_thought)
{
    LmIds s = { 0 };
    int n;
    if (with_thought) {
        lm_ids_push(&s, sid("<|channel>"));
        int *x = enc_plain("thought\nhmm", &n);
        lm_ids_append(&s, x, n);
        free(x);
        lm_ids_push(&s, sid("<channel|>"));
    }
    int *x = enc_plain("Hello there", &n);
    lm_ids_append(&s, x, n);
    free(x);
    lm_ids_push(&s, sid("<turn|>"));
    memcpy(f->script, s.ids, (size_t)s.n * sizeof(int));
    f->ns = s.n;
    f->cursor = 0;
    lm_ids_free(&s);
}

static int run_turn(LmChat *c, const char *text, LmTurn *t)
{
    LmTurnOpts o = { 100, NULL, NULL, NULL, NULL, NULL };
    char err[200] = "";
    int rc = lm_chat_turn(c, text, &o, t, err, sizeof err);
    if (rc != LM_OK) fprintf(stderr, "  turn rc=%d err=%s\n", rc, err);
    return rc;
}

static void test_verbatim_prefix(void)
{
    static Fake f;
    memset(&f, 0, sizeof f);
    f.n_vocab = lm_tok_n_ids(g_tok);
    f.n_ctx = MAXLOG;
    LmChat *c = mk_chat(&f, 0);
    CHECK(c != NULL);
    if (!c) return;

    /* turn 1: the model thinks, answers, and ends its turn with <turn|> */
    load_script(&f, 1);
    LmTurn t1, t2, t3;
    CHECK(run_turn(c, "hi", &t1) == LM_OK && t1.stop == LM_STOP_EOS);
    CHECK(t1.answer && !strcmp(t1.answer, "Hello there") && t1.reasoning && !strcmp(t1.reasoning, "hmm"));
    int fed1 = lm_chat_fed(c);
    CHECK(fed1 == f.nlog && fed1 == t1.n_prompt + t1.n_gen); /* the stop id is not fed */
    int *snap = (int *)malloc((size_t)fed1 * sizeof(int));
    memcpy(snap, f.log, (size_t)fed1 * sizeof(int));
    CHECK(f.log[t1.n_prompt - 1] != sid("<turn|>") && f.log[fed1 - 1] != sid("<turn|>"));

    /* turn 2: the new prompt EXTENDS what was fed. The thought block the model generated is
     * still in it (verbatim), then <turn|> and "\n" close the model turn, then the user turn */
    load_script(&f, 1);
    CHECK(run_turn(c, "bye", &t2) == LM_OK);
    CHECK(lm_chat_reprefills(c) == 0 && !t2.reprefill && f.resets == 0);
    CHECK(t2.n_prefill == t2.n_prompt - fed1);                       /* only the tail was fed */
    CHECK(f.nlog >= fed1 && !memcmp(f.log, snap, (size_t)fed1 * sizeof(int)));
    int nn;
    int *tail = enc_special("<turn|>\n<|turn>user\nbye<turn|>\n<|turn>model\n", &nn);
    CHECK(tail && nn >= 4 && !memcmp(f.log + fed1, tail, (size_t)nn * sizeof(int)));
    CHECK(f.log[fed1] == sid("<turn|>"));
    free(tail);

    /* turn 3 the same */
    int fed2 = lm_chat_fed(c);
    load_script(&f, 0);
    CHECK(run_turn(c, "again", &t3) == LM_OK);
    CHECK(lm_chat_reprefills(c) == 0 && t3.n_prefill == t3.n_prompt - fed2);
    lm_chat_free(c);
    free(snap);

    /* strict history renders the model's own template (thought block dropped): the prompt is
     * no longer an extension of what was fed, so the second turn re-feeds from scratch */
    memset(&f, 0, sizeof f);
    f.n_vocab = lm_tok_n_ids(g_tok);
    f.n_ctx = MAXLOG;
    c = mk_chat(&f, 1);
    load_script(&f, 1);
    CHECK(c && run_turn(c, "hi", &t1) == LM_OK);
    load_script(&f, 1);
    CHECK(run_turn(c, "bye", &t2) == LM_OK);
    CHECK(lm_chat_reprefills(c) == 1 && t2.reprefill && t2.n_prefill == t2.n_prompt);
    lm_chat_free(c);
}

int main(void)
{
    char err[256];
    g_tok = lm_tok_open(FIX, err, sizeof err);
    if (!g_tok) { fprintf(stderr, "cannot open fixture: %s\n", err); return 1; }
    int eos[] = { 1, 106, 50 };
    g_tpl = lm_template_gemma4(g_tok, eos, 3, err, sizeof err);
    if (!g_tpl) { fprintf(stderr, "cannot build template: %s\n", err); return 1; }
    test_render();
    test_verbatim_render();
    test_parser();
    test_profile();
    test_verbatim_prefix();
    lm_template_free(g_tpl);
    lm_tok_close(g_tok);
    return lt_finish("test_chat_gemma");
}
