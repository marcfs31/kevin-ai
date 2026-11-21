/* test_tools.c -- tool calling in both chat families, against the fixture tokenizers:
 * the JSON the templates print, the prompt rendered for tools / calls / results (expected
 * strings written by hand from each model's chat_template.jinja and encoded by
 * lm_tok_encode_special, as test_template.c does), the parsers that turn a generated call
 * back into a name and arguments, and the generation loop with a scripted fake backend:
 * a call, a call answered by a tool message (no re-feed), several calls, a malformed call
 * (kept as text) and a reply with no call at all. Run from the repo root. */
#include "lm_test.h"

#include "chat/lm_chat.h"
#include "chat/lm_template.h"
#include "chat/lm_toolfmt.h"
#include "lm/lm.h"

typedef struct { LmTok *tok; LmTemplate *tpl; } Fam;
static Fam g_qwen, g_gemma;

static const char *TOOLS_JSON =
    "[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"description\":\"Get the weather\","
    "\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\",\"description\":\"City name\"},"
    "\"days\":{\"type\":\"integer\"}},\"required\":[\"city\"]}}}]";

static int streq(const char *a, const char *b) { return a && b && !strcmp(a, b); }

/* ---- JSON text ------------------------------------------------------------------------ */

static void test_json(void)
{
    jval *v = json_parse("{\"a\":[1,2.5,true,null],\"b\":{\"c\":\"x\\ny\"}}", NULL);
    LmSb b = { 0 };
    lm_json_dump(&b, v, 1);
    CHECK(streq(b.s, "{\"a\": [1, 2.5, true, null], \"b\": {\"c\": \"x\\ny\"}}"));
    lm_sb_free(&b);
    lm_json_dump(&b, v, 0);
    CHECK(streq(b.s, "{\"a\":[1,2.5,true,null],\"b\":{\"c\":\"x\\ny\"}}"));
    lm_sb_free(&b);
    json_free_tree(v);

    /* Qwen's str() of a value: Python spellings for scalars, JSON for containers */
    jval *s = json_parse("[\"t\",true,false,null,3,[1,\"a\"]]", NULL);
    const char *want[] = { "t", "True", "False", "None", "3", "[1, \"a\"]" };
    for (int i = 0; i < s->len; i++) {
        lm_qwen_arg_text(&b, s->kids[i]);
        CHECK_MSG(streq(b.s, want[i]), "arg %d: %s", i, b.s);
        lm_sb_free(&b);
    }
    json_free_tree(s);
}

/* ---- parsing a generated call ---------------------------------------------------------- */

static void test_parse(void)
{
    char err[200];
    LmToolCall c;
    jval *tools = json_parse(TOOLS_JSON, NULL);

    /* Qwen: parameters typed by the declaration; a string stays a string, an integer is one */
    CHECK(lm_qwen_parse_call("\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n"
                             "<parameter=days>\n3\n</parameter>\n</function>\n", tools, &c, err, sizeof err) == 0);
    CHECK(streq(c.name, "get_weather") && streq(c.arguments, "{\"city\":\"Paris\",\"days\":3}"));
    lm_toolcall_free(&c);
    /* a string parameter that looks like a number is not turned into one; a multi-line value keeps its lines */
    CHECK(lm_qwen_parse_call("<function=get_weather>\n<parameter=city>\n123\nsecond line\n</parameter>\n</function>",
                             tools, &c, err, sizeof err) == 0);
    CHECK(streq(c.arguments, "{\"city\":\"123\\nsecond line\"}"));
    lm_toolcall_free(&c);
    /* no arguments */
    CHECK(lm_qwen_parse_call("<function=ping>\n</function>", NULL, &c, err, sizeof err) == 0 && streq(c.arguments, "{}"));
    lm_toolcall_free(&c);
    /* NEGATIVE: malformed calls are refused with a reason, never half parsed */
    CHECK(lm_qwen_parse_call("just text", tools, &c, err, sizeof err) == -1 && strstr(err, "<function="));
    CHECK(lm_qwen_parse_call("<function=f>\n<parameter=a>\nx\n", tools, &c, err, sizeof err) == -1);
    CHECK(lm_qwen_parse_call("<function=f>\n<parameter=a>\nx\n</parameter>\n", tools, &c, err, sizeof err) == -1);
    CHECK(lm_qwen_parse_call("<function=>\n</function>", tools, &c, err, sizeof err) == -1);
    CHECK(lm_qwen_parse_call("<function=f>\n</function> trailing", tools, &c, err, sizeof err) == -1);

    /* Gemma: call:NAME{key:value,...}, <|"|> (here \x01) around strings, nested values */
    CHECK(lm_gemma_parse_call("call:get_weather{city:\x01Paris\x01,days:3,opts:{a:true,b:[1,\x01x\x01],c:null}}",
                              &c, err, sizeof err) == 0);
    CHECK(streq(c.name, "get_weather") &&
          streq(c.arguments, "{\"city\":\"Paris\",\"days\":3,\"opts\":{\"a\":true,\"b\":[1,\"x\"],\"c\":null}}"));
    lm_toolcall_free(&c);
    CHECK(lm_gemma_parse_call("call:ping{}", &c, err, sizeof err) == 0 && streq(c.arguments, "{}"));
    lm_toolcall_free(&c);
    /* a string may hold the structural characters */
    CHECK(lm_gemma_parse_call("call:f{q:\x01" "a,b}{c\x01}", &c, err, sizeof err) == 0 && streq(c.arguments, "{\"q\":\"a,b}{c\"}"));
    lm_toolcall_free(&c);
    /* NEGATIVE */
    CHECK(lm_gemma_parse_call("get_weather{}", &c, err, sizeof err) == -1);
    CHECK(lm_gemma_parse_call("call:f{city:\x01Paris}", &c, err, sizeof err) == -1);
    CHECK(lm_gemma_parse_call("call:f{a:1,}", &c, err, sizeof err) == -1);
    CHECK(lm_gemma_parse_call("call:f{a:1} junk", &c, err, sizeof err) == -1);
    CHECK(lm_gemma_parse_call("call:{a:1}", &c, err, sizeof err) == -1);
    json_free_tree(tools);
}

/* ---- rendering ------------------------------------------------------------------------- */

static int *enc_special(const Fam *f, const char *text, int *n) { return lm_tok_encode_special(f->tok, text, strlen(text), n); }

static int renders(const Fam *f, const LmMsg *msgs, int n, const jval *tools, int thinking, int verbatim, const char *jinja_out)
{
    LmRenderOpts ro;
    memset(&ro, 0, sizeof ro);
    ro.enable_thinking = thinking;
    ro.verbatim_history = verbatim;
    ro.tools = tools;
    LmIds got = { 0 };
    char err[200] = "";
    int rc = f->tpl->render(f->tpl, msgs, n, &ro, &got, err, sizeof err);
    int nw;
    int *want = enc_special(f, jinja_out, &nw);
    int ok = rc == 0 && want && got.n == nw && !memcmp(got.ids, want, (size_t)nw * sizeof(int));
    if (!ok) fprintf(stderr, "  render mismatch (rc=%d %s) for %s\n", rc, err, jinja_out);
    free(want);
    lm_ids_free(&got);
    return ok;
}

#define QWEN_TOOLS_HEAD \
    "<|im_start|>system\n# Tools\n\nYou have access to the following functions:\n\n<tools>\n" \
    "{\"type\": \"function\", \"function\": {\"name\": \"get_weather\", \"description\": \"Get the weather\", " \
    "\"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\", \"description\": " \
    "\"City name\"}, \"days\": {\"type\": \"integer\"}}, \"required\": [\"city\"]}}}\n</tools>" \
    "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n" \
    "<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n" \
    "<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n" \
    "</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified " \
    "format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n" \
    "- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural " \
    "language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question " \
    "like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>"

static void test_render_qwen(void)
{
    jval *tools = json_parse(TOOLS_JSON, NULL);
    LmMsg u = { LM_ROLE_USER, "Weather in Paris?" };
    LmMsg sys = { LM_ROLE_SYSTEM, "Be brief." };
    LmToolCall call = { "get_weather", "{\"city\":\"Paris\",\"days\":2}" };
    LmMsg a = { LM_ROLE_ASSISTANT, "" };
    a.calls = &call;
    a.n_calls = 1;
    LmMsg tr = { LM_ROLE_TOOL, "18C sunny" };
    tr.tool_name = "get_weather";

    /* tools alone, then with a system message, which joins the tools block */
    CHECK(renders(&g_qwen, &u, 1, tools, 0, 0, QWEN_TOOLS_HEAD "<|im_end|>\n<|im_start|>user\nWeather in Paris?<|im_end|>\n"
                                                "<|im_start|>assistant\n<think>\n\n</think>\n\n"));
    LmMsg su[] = { sys, u };
    CHECK(renders(&g_qwen, su, 2, tools, 1, 0, QWEN_TOOLS_HEAD "\n\nBe brief.<|im_end|>\n<|im_start|>user\nWeather in Paris?<|im_end|>\n"
                                               "<|im_start|>assistant\n<think>\n"));
    /* no tools: the same conversation renders without the block (NEGATIVE: tools never leak) */
    CHECK(renders(&g_qwen, &u, 1, NULL, 0, 0, "<|im_start|>user\nWeather in Paris?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"));

    /* a call and its result, from the text of the history (the model template's own form) */
    LmMsg conv[] = { u, a, tr };
    CHECK(renders(&g_qwen, conv, 3, tools, 0, 0,
                  QWEN_TOOLS_HEAD "<|im_end|>\n<|im_start|>user\nWeather in Paris?<|im_end|>\n"
                  "<|im_start|>assistant\n<think>\n\n</think>\n\n<tool_call>\n<function=get_weather>\n"
                  "<parameter=city>\nParis\n</parameter>\n<parameter=days>\n2\n</parameter>\n</function>\n</tool_call><|im_end|>\n"
                  "<|im_start|>user\n<tool_response>\n18C sunny\n</tool_response><|im_end|>\n"
                  "<|im_start|>assistant\n<think>\n\n</think>\n\n"));
    /* text before the call, two calls, two results sharing one user turn */
    LmToolCall two[] = { { "get_weather", "{\"city\":\"Paris\"}" }, { "get_weather", "{\"city\":\"Rome\"}" } };
    LmMsg a2 = { LM_ROLE_ASSISTANT, "Checking." };
    a2.calls = two;
    a2.n_calls = 2;
    LmMsg tr2 = { LM_ROLE_TOOL, "21C" };
    tr2.tool_name = "get_weather";
    LmMsg conv2[] = { u, a2, tr, tr2 };
    CHECK(renders(&g_qwen, conv2, 4, tools, 0, 0,
                  QWEN_TOOLS_HEAD "<|im_end|>\n<|im_start|>user\nWeather in Paris?<|im_end|>\n"
                  "<|im_start|>assistant\n<think>\n\n</think>\n\nChecking.\n\n<tool_call>\n<function=get_weather>\n"
                  "<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=get_weather>\n"
                  "<parameter=city>\nRome\n</parameter>\n</function>\n</tool_call><|im_end|>\n"
                  "<|im_start|>user\n<tool_response>\n18C sunny\n</tool_response>\n<tool_response>\n21C\n</tool_response><|im_end|>\n"
                  "<|im_start|>assistant\n<think>\n\n</think>\n\n"));
    /* NEGATIVE: arguments that are not a JSON object are refused, not rendered */
    LmToolCall bad = { "f", "[1]" };
    LmMsg ab = { LM_ROLE_ASSISTANT, "" };
    ab.calls = &bad;
    ab.n_calls = 1;
    LmMsg badconv[] = { u, ab, tr };
    LmRenderOpts ro;
    memset(&ro, 0, sizeof ro);
    LmIds out = { 0 };
    char err[200] = "";
    CHECK(g_qwen.tpl->render(g_qwen.tpl, badconv, 3, &ro, &out, err, sizeof err) == -1 && strstr(err, "JSON object"));
    lm_ids_free(&out);
    json_free_tree(tools);
}

static int count_id(const LmIds *v, int id)
{
    int n = 0;
    for (int i = 0; i < v->n; i++) n += v->ids[i] == id;
    return n;
}

static void test_render_gemma(void)
{
    jval *tools = json_parse(TOOLS_JSON, NULL);
    LmMsg u = { LM_ROLE_USER, "Weather in Paris?" };
    LmToolCall call = { "get_weather", "{\"days\":2,\"city\":\"Paris\"}" };
    LmMsg a = { LM_ROLE_ASSISTANT, "" };
    a.calls = &call;
    a.n_calls = 1;
    LmMsg tr = { LM_ROLE_TOOL, "18C sunny" };
    tr.tool_name = "get_weather";
    const char *decl =
        "<|tool>declaration:get_weather{description:<|\"|>Get the weather<|\"|>,parameters:{properties:{"
        "city:{description:<|\"|>City name<|\"|>,type:<|\"|>STRING<|\"|>},days:{type:<|\"|>INTEGER<|\"|>}},"
        "required:[<|\"|>city<|\"|>],type:<|\"|>OBJECT<|\"|>}}<tool|>";
    char buf[2048];

    snprintf(buf, sizeof buf, "<bos><|turn>system\n%s<turn|>\n<|turn>user\nWeather in Paris?<turn|>\n<|turn>model\n", decl);
    CHECK(renders(&g_gemma, &u, 1, tools, 0, 0, buf));
    LmMsg sys = { LM_ROLE_SYSTEM, "Be brief." };
    LmMsg su[] = { sys, u };
    snprintf(buf, sizeof buf, "<bos><|turn>system\n<|think|>\nBe brief.%s<turn|>\n<|turn>user\nWeather in Paris?<turn|>\n<|turn>model\n", decl);
    CHECK(renders(&g_gemma, su, 2, tools, 1, 0, buf));
    /* NEGATIVE: no tools, no tool block */
    CHECK(renders(&g_gemma, &u, 1, NULL, 0, 0, "<bos><|turn>user\nWeather in Paris?<turn|>\n<|turn>model\n"));

    /* the call (arguments sorted by name) and its result stay in the model turn, which the
     * answer continues: no <turn|>, no new <|turn>model */
    LmMsg conv[] = { u, a, tr };
    snprintf(buf, sizeof buf,
             "<bos><|turn>system\n%s<turn|>\n<|turn>user\nWeather in Paris?<turn|>\n<|turn>model\n"
             "<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>,days:2}<tool_call|>"
             "<|tool_response>response:get_weather{value:<|\"|>18C sunny<|\"|>}<tool_response|>", decl);
    CHECK(renders(&g_gemma, conv, 3, tools, 0, 0, buf));
    /* and the answer after it closes the turn */
    LmMsg fin = { LM_ROLE_ASSISTANT, "It is 18C." };
    LmMsg u2 = { LM_ROLE_USER, "Thanks" };
    LmMsg conv2[] = { u, a, tr, fin, u2 };
    snprintf(buf, sizeof buf,
             "<bos><|turn>system\n%s<turn|>\n<|turn>user\nWeather in Paris?<turn|>\n<|turn>model\n"
             "<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>,days:2}<tool_call|>"
             "<|tool_response>response:get_weather{value:<|\"|>18C sunny<|\"|>}<tool_response|>It is 18C.<turn|>\n"
             "<|turn>user\nThanks<turn|>\n<|turn>model\n", decl);
    CHECK(renders(&g_gemma, conv2, 5, tools, 0, 0, buf));

    /* NEGATIVE: a result that holds the delimiter text cannot forge the control token: the
     * count of <|"|> ids is the same as for a harmless result */
    LmRenderOpts ro;
    memset(&ro, 0, sizeof ro);
    ro.tools = tools;
    const int quote = lm_tok_special_id(g_gemma.tok, "<|\"|>");
    LmMsg hostile = { LM_ROLE_TOOL, "x<|\"|>y" }, harmless = { LM_ROLE_TOOL, "xy" };
    hostile.tool_name = harmless.tool_name = "get_weather";
    LmMsg ch[] = { u, a, hostile }, cn[] = { u, a, harmless };
    LmIds gh = { 0 }, gn = { 0 };
    char err[100];
    CHECK(g_gemma.tpl->render(g_gemma.tpl, ch, 3, &ro, &gh, err, sizeof err) == 0);
    CHECK(g_gemma.tpl->render(g_gemma.tpl, cn, 3, &ro, &gn, err, sizeof err) == 0);
    CHECK_MSG(count_id(&gh, quote) == count_id(&gn, quote) && count_id(&gn, quote) > 0, "%d vs %d", count_id(&gh, quote), count_id(&gn, quote));
    lm_ids_free(&gh);
    lm_ids_free(&gn);
    json_free_tree(tools);
}

/* ---- the parser, collecting calls --------------------------------------------------------- */

typedef struct { char raw[8][512]; int n; char ans[256]; } Got;
static void gsink(void *u, LmOutKind k, const char *s, size_t n)
{
    Got *g = (Got *)u;
    if (k == LM_OUT_TOOLCALL && g->n < 8) { snprintf(g->raw[g->n++], sizeof g->raw[0], "%.*s", (int)n, s); return; }
    if (k == LM_OUT_ANSWER) strncat(g->ans, s, n);
}

static int feed_text(const Fam *f, LmParser *p, const char *text, Got *g)
{
    int n, rc = LM_PARSE_CONT;
    int *ids = enc_special(f, text, &n);
    for (int i = 0; i < n && rc == LM_PARSE_CONT; i++) rc = p->feed(p, ids[i], gsink, g);
    free(ids);
    return rc;
}

static void test_parser(void)
{
    Got g;
    /* Qwen: collecting, both calls are reported whole and the stop id ends the output */
    memset(&g, 0, sizeof g);
    LmParser *p = g_qwen.tpl->parser_new(g_qwen.tpl, 0);
    p->collect_calls = 1;
    int rc = feed_text(&g_qwen, p,
        "Checking.<tool_call>\n<function=a>\n<parameter=x>\n1\n</parameter>\n</function>\n</tool_call>\n"
        "<tool_call>\n<function=b>\n</function>\n</tool_call><|im_end|>", &g);
    p->finish(p, gsink, &g);
    CHECK(rc == LM_PARSE_STOP && p->tool_calls == 2 && g.n == 2 && !strcmp(g.ans, "Checking."));
    CHECK(strstr(g.raw[0], "<function=a>") && strstr(g.raw[1], "<function=b>"));
    p->destroy(p);
    /* not collecting (the default): the first call ends the output, as before */
    memset(&g, 0, sizeof g);
    p = g_qwen.tpl->parser_new(g_qwen.tpl, 0);
    rc = feed_text(&g_qwen, p, "<tool_call>\n<function=a>\n</function>\n</tool_call>more", &g);
    CHECK(rc == LM_PARSE_STOP_AFTER && g.n == 1 && !strcmp(g.raw[0], "tool calls are not supported"));
    p->destroy(p);
    /* a call cut off by the end of the stream is still handed over, so it can be shown as text */
    memset(&g, 0, sizeof g);
    p = g_qwen.tpl->parser_new(g_qwen.tpl, 0);
    p->collect_calls = 1;
    feed_text(&g_qwen, p, "<tool_call>\n<function=a>\n<parameter=x>\n1", &g);
    p->finish(p, gsink, &g);
    CHECK(g.n == 1 && strstr(g.raw[0], "<parameter=x>"));
    p->destroy(p);
    /* NEGATIVE: no call, nothing reported */
    memset(&g, 0, sizeof g);
    p = g_qwen.tpl->parser_new(g_qwen.tpl, 0);
    p->collect_calls = 1;
    feed_text(&g_qwen, p, "Just an answer.<|im_end|>", &g);
    CHECK(g.n == 0 && p->tool_calls == 0 && !strcmp(g.ans, "Just an answer."));
    p->destroy(p);

    /* Gemma: <|"|> comes through as \x01, and <|tool_response> stops the output */
    memset(&g, 0, sizeof g);
    p = g_gemma.tpl->parser_new(g_gemma.tpl, 0);
    p->collect_calls = 1;
    rc = feed_text(&g_gemma, p, "<|tool_call>call:f{q:<|\"|>hi<|\"|>}<tool_call|><|tool_call>call:g{}<tool_call|><|tool_response>", &g);
    p->finish(p, gsink, &g);
    CHECK(rc == LM_PARSE_STOP && g.n == 2 && !strcmp(g.raw[0], "call:f{q:\x01hi\x01}") && !strcmp(g.raw[1], "call:g{}"));
    p->destroy(p);
    memset(&g, 0, sizeof g);
    p = g_gemma.tpl->parser_new(g_gemma.tpl, 0);
    p->collect_calls = 1;
    feed_text(&g_gemma, p, "Hello<turn|>", &g);
    CHECK(g.n == 0 && !strcmp(g.ans, "Hello"));
    p->destroy(p);
}

/* ---- the generation loop ---------------------------------------------------------------- */

enum { MAXLOG = 4096 };
typedef struct {
    int n_vocab, n_ctx, pos, resets;
    int script[512], ns, cursor;
} Fake;

static int fk_feed(void *c, int token, float *logits)
{
    Fake *f = (Fake *)c;
    (void)token;
    if (f->pos >= f->n_ctx) return LM_ERR_CTX;
    f->pos++;
    if (logits) {
        for (int i = 0; i < f->n_vocab; i++) logits[i] = -5.0f;
        logits[f->script[f->cursor < f->ns ? f->cursor : f->ns - 1]] = 5.0f;
        f->cursor++;
    }
    return LM_OK;
}
static void fk_reset(void *c) { Fake *f = (Fake *)c; f->pos = 0; f->resets++; }
static int fk_pos(void *c) { return ((Fake *)c)->pos; }

static void script_text(const Fam *fam, Fake *f, const char *text)
{
    int n;
    int *ids = enc_special(fam, text, &n);
    memcpy(f->script, ids, (size_t)n * sizeof(int));
    f->ns = n;
    f->cursor = 0;
    free(ids);
}

static LmChat *mk_chat(const Fam *fam, Fake *f)
{
    LmChatConfig cc;
    memset(&cc, 0, sizeof cc);
    cc.tok = fam->tok;
    cc.tpl = fam->tpl;
    cc.backend = (LmBackend){ f, f->n_vocab, f->n_ctx, fk_feed, fk_reset, fk_pos, NULL, NULL, NULL, NULL, NULL, NULL };
    cc.temperature = 0;
    cc.top_p = 1;
    cc.seed = 1;
    char err[200];
    LmChat *c = lm_chat_new(&cc, err, sizeof err);
    if (!c) fprintf(stderr, "lm_chat_new: %s\n", err);
    return c;
}

typedef struct { int n; char name[4][32], args[4][128]; } Seen;
static void on_call(void *u, int i, const LmToolCall *c)
{
    Seen *s = (Seen *)u;
    if (i < 4) { snprintf(s->name[i], sizeof s->name[i], "%s", c->name); snprintf(s->args[i], sizeof s->args[i], "%s", c->arguments); }
    s->n++;
}
static char g_ans[256];
static void asink(void *u, LmOutKind k, const char *s, size_t n)
{
    (void)u;
    if (k == LM_OUT_ANSWER) strncat(g_ans, s, n);
}

static int run(LmChat *c, const char *user, Seen *seen, LmTurn *t, char *err)
{
    LmTurnOpts o = { 200, asink, seen, NULL, NULL, on_call };
    memset(seen, 0, sizeof *seen);
    g_ans[0] = 0;
    return user ? lm_chat_turn(c, user, &o, t, err, 200) : lm_chat_continue(c, &o, t, err, 200);
}

static void chat_flow(const Fam *fam, const char *call_text, const char *two_calls_text, const char *tail,
                      const char *want_args, const char *answer_text)
{
    jval *tools = json_parse(TOOLS_JSON, NULL);
    static Fake f;
    memset(&f, 0, sizeof f);
    f.n_vocab = lm_tok_n_ids(fam->tok);
    f.n_ctx = MAXLOG;
    LmChat *c = mk_chat(fam, &f);
    CHECK(c != NULL);
    if (!c) { json_free_tree(tools); return; }
    LmTurn t;
    Seen seen;
    char err[200] = "";
    char text[1024];

    /* 1. the model calls a tool: one call, reported as it finishes, nothing but the call as text */
    lm_chat_set_tools(c, tools);
    snprintf(text, sizeof text, "%s%s", call_text, tail);
    script_text(fam, &f, text);
    CHECK(run(c, "Weather in Paris?", &seen, &t, err) == LM_OK);
    CHECK_MSG(t.stop == LM_STOP_EOS && t.n_calls == 1 && t.tool_calls == 1, "stop %d calls %d", t.stop, t.n_calls);
    CHECK(seen.n == 1 && streq(seen.name[0], "get_weather") && streq(seen.args[0], want_args));
    CHECK(t.n_calls == 1 && streq(t.calls[0].name, "get_weather") && streq(t.calls[0].arguments, want_args));
    CHECK(streq(t.answer, "") && !g_ans[0]);

    /* 2. the client sends the call back with the result: the state is extended, not re-fed */
    LmToolCall call = { "get_weather", want_args };
    LmMsg msgs[3] = { { LM_ROLE_USER, "Weather in Paris?" }, { LM_ROLE_ASSISTANT, "" }, { LM_ROLE_TOOL, "18C sunny" } };
    msgs[1].calls = &call;
    msgs[1].n_calls = 1;
    msgs[2].tool_name = "get_weather";
    CHECK(lm_chat_sync(c, msgs, 3, err, sizeof err) == 0);
    script_text(fam, &f, answer_text);
    CHECK(run(c, NULL, &seen, &t, err) == LM_OK);
    CHECK_MSG(t.stop == LM_STOP_EOS && t.n_calls == 0 && t.reprefill == 0 && streq(t.answer, "It is 18C."),
              "stop %d calls %d reprefill %d answer '%s'", t.stop, t.n_calls, t.reprefill, t.answer ? t.answer : "");
    CHECK(streq(g_ans, "It is 18C.") && t.n_prefill < t.n_prompt);
    /* the answer is in the history: the next user turn extends it too */
    LmMsg more[5] = { msgs[0], msgs[1], msgs[2], { LM_ROLE_ASSISTANT, "It is 18C." }, { LM_ROLE_USER, "Thanks" } };
    more[1].calls = &call;
    more[1].n_calls = 1;
    more[2].tool_name = "get_weather";
    CHECK(lm_chat_sync(c, more, 4, err, sizeof err) == 0);
    script_text(fam, &f, answer_text);
    CHECK(run(c, "Thanks", &seen, &t, err) == LM_OK && t.reprefill == 0 && f.resets == 0);

    /* NEGATIVE: a call that differs from what the chat made is a different conversation (re-fed) */
    LmToolCall other = { "get_weather", "{\"city\":\"Rome\"}" };
    LmMsg edited[3] = { msgs[0], msgs[1], msgs[2] };
    edited[1].calls = &other;
    CHECK(lm_chat_sync(c, edited, 3, err, sizeof err) == 0);
    script_text(fam, &f, answer_text);
    CHECK(run(c, NULL, &seen, &t, err) == LM_OK && t.reprefill == 1);

    /* 3. several calls in one reply */
    lm_chat_purge(c);
    snprintf(text, sizeof text, "%s%s", two_calls_text, tail);
    script_text(fam, &f, text);
    CHECK(run(c, "Weather?", &seen, &t, err) == LM_OK && t.n_calls == 2 && seen.n == 2 && streq(seen.name[1], "ping"));

    /* 4. NEGATIVE: the model does not call anything: a plain answer, finish by stop */
    lm_chat_purge(c);
    snprintf(text, sizeof text, "Hello there.%s", fam == &g_qwen ? "<|im_end|>" : "<turn|>");
    script_text(fam, &f, text);
    CHECK(run(c, "Hi", &seen, &t, err) == LM_OK);
    CHECK(t.stop == LM_STOP_EOS && t.n_calls == 0 && t.tool_calls == 0 && seen.n == 0 && streq(t.answer, "Hello there."));

    /* 5. NEGATIVE: a malformed call is not dropped: its text becomes part of the answer */
    lm_chat_purge(c);
    snprintf(text, sizeof text, "%s", fam == &g_qwen ? "<tool_call>\ngarbage here\n</tool_call><|im_end|>" : "<|tool_call>garbage here<tool_call|><|tool_response>");
    script_text(fam, &f, text);
    CHECK(run(c, "Hi", &seen, &t, err) == LM_OK);
    CHECK_MSG(t.n_calls == 0 && t.tool_calls == 1 && seen.n == 0 && t.answer && strstr(t.answer, "garbage here"),
              "calls %d attempted %d answer '%s'", t.n_calls, t.tool_calls, t.answer ? t.answer : "");

    /* 6. without tools the old behaviour holds: the first call ends the turn, none is parsed */
    lm_chat_purge(c);
    lm_chat_set_tools(c, NULL);
    snprintf(text, sizeof text, "%s%s", call_text, tail);
    script_text(fam, &f, text);
    CHECK(run(c, "Hi", &seen, &t, err) == LM_OK && t.n_calls == 0 && t.tool_calls == 1 && t.stop == LM_STOP_TOOLCALL && seen.n == 0);

    /* continue needs something to answer */
    lm_chat_purge(c);
    CHECK(run(c, NULL, &seen, &t, err) == -1 && strstr(err, "must end with"));
    lm_chat_free(c);
    json_free_tree(tools);
}

static void test_chat(void)
{
    chat_flow(&g_qwen,
              "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>",
              "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>\n"
              "<tool_call>\n<function=ping>\n</function>\n</tool_call>",
              "<|im_end|>", "{\"city\":\"Paris\"}", "It is 18C.<|im_end|>");
    chat_flow(&g_gemma,
              "<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}<tool_call|>",
              "<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}<tool_call|><|tool_call>call:ping{}<tool_call|>",
              "<|tool_response>", "{\"city\":\"Paris\"}", "It is 18C.<turn|>");
}

int main(void)
{
    char err[256];
    g_qwen.tok = lm_tok_open("tests/fixtures/tok/qwen-tokenizer.json", err, sizeof err);
    g_gemma.tok = lm_tok_open("tests/fixtures/tok/gemma-tokenizer.json", err, sizeof err);
    if (!g_qwen.tok || !g_gemma.tok) { fprintf(stderr, "cannot open the fixture tokenizers: %s\n", err); return 1; }
    int qeos[] = { lm_tok_special_id(g_qwen.tok, "<|endoftext|>") };
    int geos[] = { 1, 106, 50 };
    g_qwen.tpl = lm_template_qwen35(g_qwen.tok, qeos, 1, err, sizeof err);
    g_gemma.tpl = lm_template_gemma4(g_gemma.tok, geos, 3, err, sizeof err);
    if (!g_qwen.tpl || !g_gemma.tpl) { fprintf(stderr, "cannot build the templates: %s\n", err); return 1; }
    test_json();
    test_parse();
    test_render_qwen();
    test_render_gemma();
    test_parser();
    test_chat();
    lm_template_free(g_qwen.tpl);
    lm_template_free(g_gemma.tpl);
    lm_tok_close(g_qwen.tok);
    lm_tok_close(g_gemma.tok);
    return lt_finish("test_tools");
}
