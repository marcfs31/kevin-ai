/* lm-chat -- chat with a model from the terminal.
 *
 *   lm-chat --model DIR [--weights FILE] [--prompt TEXT] [--system TEXT]
 *           [--think | --no-think] [--show-thinking] [--temperature T] [--top-p P]
 *           [--top-k K] [--seed N] [--max-tokens N] [--ctx N] [--threads N]
 *           [--strict-template] [--stats]
 *
 * DIR holds config.json and tokenizer.json (Hugging Face layout). --weights names a q8
 * file from lm-quant; without it the bf16 safetensors in DIR are streamed (exact, slow).
 * With --prompt the single answer is streamed to stdout and the program exits; without it
 * lines are read from stdin, one user turn per line: "/reset" starts over, "/exit" (or
 * end of input) quits. stdout carries only the answer; notices, reasoning (with
 * --show-thinking) and statistics go to stderr. Ctrl-C ends the answer being generated
 * and keeps the process (at the prompt it quits).
 *
 * Thinking is off by default: a think block costs hundreds of tokens at a few tokens/s.
 * Sampling defaults when a flag is absent: temperature 0.7, top-p 0.8 without thinking;
 * 0.6, 0.95 with; top-k 20. Original to this repository. */
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "chat/lm_chat.h"
#include "chat/lm_template.h"
#include "lm/lm.h"
#include "tokenizer/lm_tok.h"

/* json.h is a header-only library of file-scope functions; the suppression is scoped to
 * this include so first-party dead code still warns. */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#include "json.h"
#pragma GCC diagnostic pop

static volatile sig_atomic_t g_cancel = 0;
static volatile sig_atomic_t g_in_turn = 0;

static void on_sigint(int sig)
{
    (void)sig;
    if (g_in_turn) { g_cancel = 1; return; }
    static const char nl[] = "\n";
    (void)!write(2, nl, sizeof nl - 1);
    _exit(130);
}

static int cancelled(void *user) { (void)user; return g_cancel != 0; }

/* ---- LmModel as an LmBackend (lm.h only) ----------------------------------------- */

static int be_feed(void *ctx, int token, float *logits) { return lm_feed((LmModel *)ctx, token, logits); }
static void be_reset(void *ctx) { lm_reset((LmModel *)ctx); }
static int be_pos(void *ctx) { return lm_pos((const LmModel *)ctx); }

/* ---- config.json: model_type and eos ids ------------------------------------------ */

typedef struct { char model_type[64]; int eos[8]; int n_eos; } ModelInfo;

static int read_all(const char *path, char **out, char *err, size_t errlen)
{
    FILE *f = fopen(path, "rb");
    if (!f) { snprintf(err, errlen, "cannot open %s: %s", path, strerror(errno)); return -1; }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *b = n >= 0 && n < (1L << 26) ? (char *)malloc((size_t)n + 1) : NULL;
    if (!b || fread(b, 1, (size_t)n, f) != (size_t)n) {
        snprintf(err, errlen, "cannot read %s", path);
        free(b); fclose(f);
        return -1;
    }
    fclose(f);
    b[n] = 0;
    *out = b;
    return 0;
}

static void add_eos(ModelInfo *mi, const jval *v)
{
    if (v && v->t == J_NUM && v->num >= 0 && mi->n_eos < 8) mi->eos[mi->n_eos++] = (int)v->num;
}

static int read_model_info(const char *dir, ModelInfo *mi, char *err, size_t errlen)
{
    char path[4096], *buf = NULL;
    snprintf(path, sizeof path, "%s/config.json", dir);
    if (read_all(path, &buf, err, errlen)) return -1;
    jval *root = json_parse(buf, NULL);
    free(buf);
    if (!root || root->t != J_OBJ) { snprintf(err, errlen, "%s is not valid JSON", path); json_free_tree(root); return -1; }
    memset(mi, 0, sizeof *mi);
    jval *mt = json_get(root, "model_type");
    if (!mt || mt->t != J_STR || !mt->str) { snprintf(err, errlen, "%s has no model_type", path); json_free_tree(root); return -1; }
    snprintf(mi->model_type, sizeof mi->model_type, "%s", mt->str);
    jval *tc = json_get(root, "text_config");
    jval *e = json_get(tc && tc->t == J_OBJ ? tc : root, "eos_token_id");
    if (!e) e = json_get(root, "eos_token_id");
    if (e && e->t == J_ARR) for (int i = 0; i < e->len; i++) add_eos(mi, e->kids[i]);
    else add_eos(mi, e);
    json_free_tree(root);
    return 0;
}

/* The one place that maps a model family to its chat format. */
static LmTemplate *make_template(const ModelInfo *mi, const LmTok *tok, char *err, size_t errlen)
{
    if (!strcmp(mi->model_type, "qwen3_5")) return lm_template_qwen35(tok, mi->eos, mi->n_eos, err, errlen);
    snprintf(err, errlen, "no chat template for model_type \"%s\" (phase 1 supports qwen3_5)", mi->model_type);
    return NULL;
}

/* ---- output ----------------------------------------------------------------------- */

typedef struct { int show_thinking; int in_reasoning; int wrote_answer; int color; } Out;

static void sink(void *user, LmOutKind kind, const char *s, size_t n)
{
    Out *o = (Out *)user;
    if (kind == LM_OUT_REASONING) {
        if (!o->show_thinking) return;
        if (!o->in_reasoning) { fputs(o->color ? "\033[2m[thinking]\n" : "[thinking]\n", stderr); o->in_reasoning = 1; }
        fwrite(s, 1, n, stderr);
    } else if (kind == LM_OUT_ANSWER) {
        if (o->in_reasoning) { fputs(o->color ? "\n[/thinking]\033[0m\n" : "\n[/thinking]\n", stderr); o->in_reasoning = 0; }
        fwrite(s, 1, n, stdout);
        fflush(stdout);
        o->wrote_answer = 1;
    } else {
        if (o->in_reasoning) { fputs(o->color ? "\n\033[0m" : "\n", stderr); o->in_reasoning = 0; }
        fprintf(stderr, "\n[%.*s]\n", (int)n, s);
    }
}

static void usage(FILE *f)
{
    fputs("usage: lm-chat --model DIR [--weights FILE] [--prompt TEXT] [--system TEXT]\n"
          "       [--think | --no-think] [--show-thinking] [--temperature T] [--top-p P]\n"
          "       [--top-k K] [--seed N] [--max-tokens N] [--ctx N] [--threads N]\n"
          "       [--strict-template] [--stats]\n", f);
}

static int parse_double(const char *s, double *out)
{
    char *end;
    errno = 0;
    double v = strtod(s, &end);
    if (end == s || *end || errno || !isfinite(v)) return -1;
    *out = v;
    return 0;
}

static int parse_long(const char *s, long lo, long hi, long *out)
{
    char *end;
    errno = 0;
    long v = strtol(s, &end, 10);
    if (end == s || *end || errno || v < lo || v > hi) return -1;
    *out = v;
    return 0;
}

typedef struct { LmChat *chat; Out out; int stats; int max_tokens; } Session;

/* One user turn. Returns 0 to go on, 1 on a fatal backend error. */
static int run_turn(Session *s, const char *text)
{
    char err[512] = "";
    LmTurn t;
    LmTurnOpts to = { s->max_tokens, sink, &s->out, cancelled, NULL };
    s->out.in_reasoning = 0;
    s->out.wrote_answer = 0;
    g_cancel = 0;
    g_in_turn = 1;
    int rc = lm_chat_turn(s->chat, text, &to, &t, err, sizeof err);
    g_in_turn = 0;
    if (s->out.in_reasoning) fputs(s->out.color ? "\n[/thinking]\033[0m\n" : "\n[/thinking]\n", stderr);
    if (s->out.wrote_answer) fputc('\n', stdout);
    fflush(stdout);
    if (rc == LM_ERR_CTX) {
        fprintf(stderr, "[context full: %s%s]\n", err[0] ? err : "the window is exhausted",
                " -- /reset to start over, or raise --ctx");
    } else if (rc != LM_OK) {
        fprintf(stderr, "lm-chat: %s\n", err[0] ? err : "generation failed");
        return 1;
    } else if (t.stop == LM_STOP_MAX) {
        fprintf(stderr, "[stopped: --max-tokens reached%s]\n", t.thinking_open ? " while still thinking; no answer yet" : "");
    } else if (t.stop == LM_STOP_CANCEL) {
        fputs("[cancelled]\n", stderr);
    }
    if (s->stats)
        fprintf(stderr, "[prompt %d tokens, fed %d%s: %.2f tok/s | generated %d tokens: %.2f tok/s]\n",
                t.n_prompt, t.n_prefill, t.reprefill ? " (re-fed from scratch)" : "",
                t.prefill_s > 0 ? t.n_prefill / t.prefill_s : 0.0, t.n_gen, t.decode_s > 0 ? t.n_gen / t.decode_s : 0.0);
    return 0;
}

static const char *need_value(int argc, char **argv, int *i)
{
    if (*i + 1 >= argc) {
        fprintf(stderr, "lm-chat: %s needs a value\n", argv[*i]);
        usage(stderr);
        exit(2);
    }
    return argv[++*i];
}

int main(int argc, char **argv)
{
    const char *model = NULL, *weights = NULL, *prompt = NULL, *system = NULL;
    int think = 0, show_thinking = 0, stats = 0, strict = 0;
    double temperature = -1, top_p = -1;
    long top_k = -1, seed = 1, max_tokens = 512, ctx = 0, threads = 0;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
#define VAL need_value(argc, argv, &i)
#define BAD do { fprintf(stderr, "lm-chat: invalid value for %s\n", a); return 2; } while (0)
        if (!strcmp(a, "--model")) model = VAL;
        else if (!strcmp(a, "--weights")) weights = VAL;
        else if (!strcmp(a, "--prompt")) prompt = VAL;
        else if (!strcmp(a, "--system")) system = VAL;
        else if (!strcmp(a, "--think")) think = 1;
        else if (!strcmp(a, "--no-think")) think = 0;
        else if (!strcmp(a, "--show-thinking")) show_thinking = 1;
        else if (!strcmp(a, "--strict-template")) strict = 1;
        else if (!strcmp(a, "--stats")) stats = 1;
        else if (!strcmp(a, "--temperature")) { if (parse_double(VAL, &temperature) || temperature < 0) BAD; }
        else if (!strcmp(a, "--top-p")) { if (parse_double(VAL, &top_p) || top_p <= 0 || top_p > 1) BAD; }
        else if (!strcmp(a, "--top-k")) { if (parse_long(VAL, 0, 1 << 20, &top_k)) BAD; }
        else if (!strcmp(a, "--seed")) { if (parse_long(VAL, 0, LONG_MAX, &seed)) BAD; }
        else if (!strcmp(a, "--max-tokens")) { if (parse_long(VAL, 1, 1 << 24, &max_tokens)) BAD; }
        else if (!strcmp(a, "--ctx")) { if (parse_long(VAL, 1, 1 << 24, &ctx)) BAD; }
        else if (!strcmp(a, "--threads")) { if (parse_long(VAL, 1, 4096, &threads)) BAD; }
        else if (!strcmp(a, "--help") || !strcmp(a, "-h")) { usage(stdout); return 0; }
        else { fprintf(stderr, "lm-chat: unknown option %s\n", a); usage(stderr); return 2; }
    }
    if (!model) { fputs("lm-chat: --model DIR is required\n", stderr); usage(stderr); return 2; }
    if (temperature < 0) temperature = think ? 0.6 : 0.7;
    if (top_p < 0) top_p = think ? 0.95 : 0.8;
    if (top_k < 0) top_k = 20;

    char err[1024] = "", path[4096];
    ModelInfo mi;
    if (read_model_info(model, &mi, err, sizeof err)) { fprintf(stderr, "lm-chat: %s\n", err); return 1; }
    snprintf(path, sizeof path, "%s/tokenizer.json", model);
    LmTok *tok = lm_tok_open(path, err, sizeof err);
    if (!tok) { fprintf(stderr, "lm-chat: %s\n", err); return 1; }
    LmTemplate *tpl = make_template(&mi, tok, err, sizeof err);
    if (!tpl) { fprintf(stderr, "lm-chat: %s\n", err); lm_tok_close(tok); return 1; }

    LmOpts lo = { weights, (int)ctx, (int)threads };
    LmModel *m = lm_open(model, &lo, err, sizeof err);
    if (!m) { fprintf(stderr, "lm-chat: %s\n", err); lm_template_free(tpl); lm_tok_close(tok); return 1; }

    LmChatConfig cc;
    memset(&cc, 0, sizeof cc);
    cc.tok = tok;
    cc.tpl = tpl;
    cc.backend.ctx = m;
    cc.backend.n_vocab = lm_n_vocab(m);
    cc.backend.n_ctx = lm_n_ctx(m);
    cc.backend.feed = be_feed;
    cc.backend.reset = be_reset;
    cc.backend.pos = be_pos;
    cc.temperature = temperature;
    cc.top_p = top_p;
    cc.top_k = (int)top_k;
    cc.seed = (uint64_t)seed;
    cc.thinking = think;
    cc.strict_history = strict;
    cc.system = system;
    LmChat *chat = lm_chat_new(&cc, err, sizeof err);
    if (!chat) { fprintf(stderr, "lm-chat: %s\n", err); lm_close(m); lm_template_free(tpl); lm_tok_close(tok); return 1; }

    signal(SIGINT, on_sigint);
    Session s = { chat, { show_thinking, 0, 0, isatty(2) }, stats, (int)max_tokens };
    int status = 0;
    if (prompt) {
        status = run_turn(&s, prompt);
    } else {
        int tty = isatty(0);
        char *line = NULL;
        size_t cap = 0;
        for (;;) {
            if (tty) { fputs("> ", stderr); fflush(stderr); }
            ssize_t n = getline(&line, &cap, stdin);
            if (n < 0) break;
            while (n > 0 && (line[n - 1] == '\n' || line[n - 1] == '\r')) line[--n] = 0;
            if (!strcmp(line, "/exit") || !strcmp(line, "/quit")) break;
            if (!strcmp(line, "/reset")) { lm_chat_reset(chat); fputs("[conversation reset]\n", stderr); continue; }
            if (!n) continue;
            if (run_turn(&s, line)) { status = 1; break; }
        }
        free(line);
    }
    lm_chat_free(chat);
    lm_close(m);
    lm_template_free(tpl);
    lm_tok_close(tok);
    return status;
}
