/* lm-serve -- an OpenAI-compatible HTTP endpoint for one model, one request at a time.
 *
 *   lm-serve --model DIR [--weights FILE] [--host ADDR] [--port N] [--name ID]
 *            [--think | --no-think] [--max-tokens N] [--seed N] [--ctx N] [--threads N]
 *            [--timeout S] [--fast]
 *
 *   GET  /health               {"status":"ok"} once the model is loaded (no key needed)
 *   GET  /v1/models            the one model, as id --name (default: DIR's last component)
 *   POST /v1/chat/completions  messages, stream, max_tokens / max_completion_tokens,
 *                              temperature, top_p, top_k, seed, stream_options.include_usage
 *
 * Text chat only: tools, stop strings, n > 1, logprobs, penalties and JSON modes are refused
 * with 400 rather than ignored. The answer is "content"; a think block, when --think is on,
 * comes back as "reasoning_content". Without max_tokens a reply runs until the model stops
 * or the context fills (--max-tokens N sets a server-wide default instead). A reply is
 * stopped as soon as its client goes away, so an abandoned request does not hold the server.
 * A reply that could not be finished (shutdown, a tool call) is an error, never a "stop".
 *
 * The engine cannot rewind, so the state is reused when the request's history begins with
 * the conversation it last served: a client that sends back the previous answer unchanged
 * only pays for the new message (lm_chat_sync). Anything else re-feeds the whole prompt.
 *
 * Listens on 127.0.0.1 unless --host says otherwise; --port 0 picks a free port. The bound
 * address is printed on stderr. If LM_SERVE_API_KEY is set (in the environment, so it never
 * shows up in ps), every /v1 path requires "Authorization: Bearer <key>", checked before
 * the body is read. A request has --timeout seconds (default 30) to arrive in full, so a slow
 * client cannot hold the server. Requests are logged as counts and timings only, never their
 * text. SIGINT/SIGTERM stop the reply being generated and exit. Original to this repository. */
#include <arpa/inet.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <netdb.h>
#include <netinet/in.h>
#include <poll.h>
#include <signal.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#include "chat/lm_chat.h"
#include "chat/lm_profile.h"
#include "chat/lm_template.h"
#include "json.h"
#include "lm/lm.h"
#include "tokenizer/lm_tok.h"

#ifndef MSG_NOSIGNAL
#define MSG_NOSIGNAL 0 /* macOS: SIGPIPE is ignored process-wide instead */
#endif

enum { MAX_HEAD = 16 * 1024, MAX_BODY = 4 * 1024 * 1024 };

static volatile sig_atomic_t g_stop = 0;
static void on_signal(int sig) { (void)sig; g_stop = 1; }

static double mono_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* ---- growable text ---------------------------------------------------------------- */

typedef struct { char *s; size_t n, cap; int oom; } Buf;

static void buf_add(Buf *b, const char *s, size_t n)
{
    if (b->oom) return;
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
static void buf_str(Buf *b, const char *s) { buf_add(b, s, strlen(s)); }
static void buf_fmt(Buf *b, const char *fmt, ...) __attribute__((format(printf, 2, 3)));
static void buf_fmt(Buf *b, const char *fmt, ...)
{
    char tmp[512];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(tmp, sizeof tmp, fmt, ap);
    va_end(ap);
    if (n < 0 || (size_t)n >= sizeof tmp) { b->oom = 1; return; }
    buf_add(b, tmp, (size_t)n);
}
/* a JSON string literal, quotes included; the text is UTF-8 and passes through. NUL bytes are
 * dropped: \u0000 would make the client's next request (with this text in it) unparseable. */
static void buf_json(Buf *b, const char *s, size_t n)
{
    buf_add(b, "\"", 1);
    size_t run = 0;
    for (size_t i = 0; i < n; i++) {
        unsigned char ch = (unsigned char)s[i];
        if (ch >= 0x20 && ch != '"' && ch != '\\') continue;
        buf_add(b, s + run, i - run);
        run = i + 1;
        if (ch == '"') buf_add(b, "\\\"", 2);
        else if (ch == '\\') buf_add(b, "\\\\", 2);
        else if (ch == '\n') buf_add(b, "\\n", 2);
        else if (ch == '\r') buf_add(b, "\\r", 2);
        else if (ch == '\t') buf_add(b, "\\t", 2);
        else if (ch) buf_fmt(b, "\\u%04x", ch);
    }
    buf_add(b, s + run, n - run);
    buf_add(b, "\"", 1);
}
static void buf_jstr(Buf *b, const char *s) { buf_json(b, s, strlen(s)); }

/* ---- the connection --------------------------------------------------------------- */

typedef struct {
    int fd;
    int broken;          /* a write failed or the peer closed: the client is gone */
    double deadline;     /* the whole request must have arrived by then */
    char method[16], path[256];
    char *auth;          /* the Authorization header value, or NULL */
    long content_length;
    int expect_continue;
    size_t head_n, have; /* head bytes in raw; bytes received after the head */
    char *body;
    size_t body_n;
    char *raw;           /* the bytes received with the head */
} Conn;

static int send_all(Conn *c, const char *s, size_t n)
{
    while (n > 0 && !c->broken) {
        ssize_t w = send(c->fd, s, n, MSG_NOSIGNAL);
        if (w < 0 && errno == EINTR) continue;
        if (w <= 0) { c->broken = 1; break; }
        s += w;
        n -= (size_t)w;
    }
    return c->broken ? -1 : 0;
}

/* 1 when the socket has data (or EOF) before the request deadline, 0 when time ran out or the
 * server is stopping */
static int wait_readable(const Conn *c)
{
    for (;;) {
        if (g_stop) return 0;
        const double left = c->deadline - mono_s();
        if (left <= 0) return 0;
        struct pollfd p = { c->fd, POLLIN, 0 };
        int r = poll(&p, 1, (int)(left * 1000) + 1);
        if (r > 0) return 1;
        if (r == 0) return 0;
        if (errno != EINTR) return 0;
    }
}

/* The client closed its side (a request that is complete never sends more, so readable
 * means EOF or a reset). Polled between tokens, so an abandoned reply stops early. */
static int client_gone(Conn *c)
{
    struct pollfd p = { c->fd, POLLIN, 0 };
    if (poll(&p, 1, 0) <= 0) return 0;
    if (p.revents & (POLLHUP | POLLERR)) return 1;
    char b;
    ssize_t r = recv(c->fd, &b, 1, MSG_PEEK | MSG_DONTWAIT);
    return r == 0 || (r < 0 && errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR);
}

static const char *reason(int status)
{
    switch (status) {
    case 200: return "OK";
    case 400: return "Bad Request";
    case 401: return "Unauthorized";
    case 404: return "Not Found";
    case 405: return "Method Not Allowed";
    case 408: return "Request Timeout";
    case 411: return "Length Required";
    case 413: return "Content Too Large";
    case 431: return "Request Header Fields Too Large";
    case 500: return "Internal Server Error";
    case 503: return "Service Unavailable";
    default:  return "Error";
    }
}

static void respond(Conn *c, int status, const char *extra_headers, const char *body, size_t n)
{
    char head[512];
    int h = snprintf(head, sizeof head,
                     "HTTP/1.1 %d %s\r\nContent-Type: application/json\r\nContent-Length: %zu\r\n"
                     "Cache-Control: no-store\r\nConnection: close\r\n%s\r\n",
                     status, reason(status), n, extra_headers ? extra_headers : "");
    if (send_all(c, head, (size_t)h) == 0) send_all(c, body, n);
}

static void error_json(Buf *b, const char *type, const char *code, const char *msg)
{
    buf_str(b, "{\"error\":{\"message\":");
    buf_jstr(b, msg);
    buf_str(b, ",\"type\":");
    buf_jstr(b, type);
    buf_str(b, ",\"code\":");
    if (code) buf_jstr(b, code); else buf_str(b, "null");
    buf_str(b, "}}");
}

static void respond_error(Conn *c, int status, const char *code, const char *fmt, ...) __attribute__((format(printf, 4, 5)));
static void respond_error(Conn *c, int status, const char *code, const char *fmt, ...)
{
    char msg[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(msg, sizeof msg, fmt, ap);
    va_end(ap);
    Buf b = { 0 };
    error_json(&b, status == 401 ? "authentication_error" : status >= 500 ? "server_error" : "invalid_request_error", code, msg);
    if (b.oom) { respond(c, 500, NULL, "{}", 2); free(b.s); return; }
    respond(c, status, status == 401 ? "WWW-Authenticate: Bearer\r\n" : NULL, b.s, b.n);
    free(b.s);
}

/* Walk the header lines of a NUL-terminated head. Returns 0, or 400 for a line that is not
 * "name: value" with a name free of whitespace, or for a repeated Content-Length,
 * Transfer-Encoding or Authorization (ambiguous framing or credentials are refused, never
 * resolved by picking one). Copies the values of those three into the out pointers. */
static int scan_headers(const char *head, char **cl, char **te, char **au, int *expect_continue)
{
    static const char *const names[3] = { "Content-Length", "Transfer-Encoding", "Authorization" };
    char **outs[3] = { cl, te, au };
    const char *line = strstr(head, "\r\n");
    while (line && strncmp(line, "\r\n\r\n", 4)) {
        line += 2;
        const char *end = strstr(line, "\r\n");
        if (!end) break;
        const char *colon = memchr(line, ':', (size_t)(end - line));
        if (!colon || colon == line) return 400;
        for (const char *p = line; p < colon; p++) if (*p == ' ' || *p == '\t') return 400;
        const size_t len = (size_t)(colon - line);
        const char *v = colon + 1, *e = end;
        while (*v == ' ' || *v == '\t') v++;
        while (e > v && (e[-1] == ' ' || e[-1] == '\t')) e--;
        for (int k = 0; k < 3; k++) {
            if (len != strlen(names[k]) || strncasecmp(line, names[k], len)) continue;
            if (*outs[k]) return 400;
            if (!(*outs[k] = (char *)malloc((size_t)(e - v) + 1))) return 500;
            memcpy(*outs[k], v, (size_t)(e - v));
            (*outs[k])[e - v] = 0;
        }
        if (len == 6 && !strncasecmp(line, "Expect", 6) && e - v == 12 && !strncasecmp(v, "100-continue", 12))
            *expect_continue = 1;
        line = end;
    }
    return 0;
}

/* Read the request head. Returns 0, an HTTP status to answer with, or -1 to close without
 * answering (the client went away, or sent nothing in time). */
static int read_head(Conn *c)
{
    size_t n = 0;
    c->raw = (char *)malloc(MAX_HEAD + 1);
    if (!c->raw) return 500;
    char *eoh = NULL;
    while (!eoh) {
        if (n == MAX_HEAD) return 431;
        if (!wait_readable(c)) return n ? 408 : -1;
        ssize_t r = recv(c->fd, c->raw + n, MAX_HEAD - n, 0);
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) return -1;
        n += (size_t)r;
        c->raw[n] = 0;
        eoh = strstr(c->raw, "\r\n\r\n");
    }
    c->head_n = (size_t)(eoh - c->raw) + 4;
    c->have = n - c->head_n;
    if (sscanf(c->raw, "%15s %255s HTTP/1.%*c", c->method, c->path) != 2) return 400;
    char *q = strchr(c->path, '?');
    if (q) *q = 0;

    const char saved = c->raw[c->head_n];
    c->raw[c->head_n] = 0; /* header lookups stay inside the head */
    char *cl = NULL, *te = NULL;
    int bad = scan_headers(c->raw, &cl, &te, &c->auth, &c->expect_continue);
    c->raw[c->head_n] = saved;
    if (!bad && te) bad = 411;
    if (!bad && cl) {
        char *end;
        errno = 0;
        c->content_length = strtol(cl, &end, 10);
        if (end == cl || *end || errno || c->content_length < 0) bad = 400;
        else if (c->content_length > MAX_BODY) bad = 413;
    }
    free(cl);
    free(te);
    return bad;
}

/* Read the body announced by Content-Length, within the same deadline. Same returns. */
static int read_body(Conn *c)
{
    const size_t len = (size_t)c->content_length;
    c->body = (char *)malloc(len + 1);
    if (!c->body) return 500;
    size_t have = c->have < len ? c->have : len;
    memcpy(c->body, c->raw + c->head_n, have);
    if (have < len && c->expect_continue && send_all(c, "HTTP/1.1 100 Continue\r\n\r\n", 25)) return -1;
    while (have < len) {
        if (!wait_readable(c)) return 408;
        ssize_t r = recv(c->fd, c->body + have, len - have, 0);
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) return -1;
        have += (size_t)r;
    }
    c->body[len] = 0;
    c->body_n = len;
    return 0;
}

/* After the response: close our side, then read and discard what the client still sends
 * (an unread body would turn close() into a reset that can swallow the response), for at
 * most a second. */
static void finish_conn(Conn *c)
{
    shutdown(c->fd, SHUT_WR);
    c->deadline = mono_s() + 1.0;
    char scratch[4096];
    while (wait_readable(c) && recv(c->fd, scratch, sizeof scratch, 0) > 0) {}
}

/* ---- the server --------------------------------------------------------------------- */

typedef struct {
    LmChat *chat;
    const char *name;
    const char *api_key;     /* NULL = no key */
    int max_tokens;          /* server default, 0 = until stop or context */
    double temperature, top_p;
    int top_k;
    uint64_t seed;
    unsigned long requests;
} Server;

/* compare without an early exit, so the time taken does not reveal how much of a key matched */
static int key_ok(const char *got, const char *want)
{
    size_t gn = strlen(got), wn = strlen(want);
    unsigned diff = (unsigned)(gn != wn);
    for (size_t i = 0; i < wn; i++) diff |= (unsigned char)want[i] ^ (unsigned char)(i < gn ? got[i] : 0);
    return diff == 0;
}

static int authorised(const Server *s, const Conn *c)
{
    if (!s->api_key) return 1;
    if (!c->auth || strncasecmp(c->auth, "Bearer ", 7)) return 0;
    return key_ok(c->auth + 7, s->api_key);
}

/* A streamed reply: headers go out with the first text, so an error found before any output
 * can still be a plain HTTP error. */
typedef struct {
    Conn *conn;
    const Server *srv;
    int stream, started;
    char id[64];
    long created;
} Reply;

static void chunk_head(Buf *b, const Reply *r)
{
    buf_str(b, "data: {\"id\":");
    buf_jstr(b, r->id);
    buf_fmt(b, ",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":", r->created);
    buf_jstr(b, r->srv->name);
}

static void sse_send(Reply *r, Buf *b)
{
    buf_str(b, "\n\n");
    if (b->oom) r->conn->broken = 1;
    else send_all(r->conn, b->s, b->n);
    free(b->s);
}

static void sse_start(Reply *r)
{
    if (r->started) return;
    r->started = 1;
    static const char head[] = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-store\r\n"
                               "Connection: close\r\n\r\n";
    if (send_all(r->conn, head, sizeof head - 1)) return;
    Buf b = { 0 };
    chunk_head(&b, r);
    buf_str(&b, ",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"},\"finish_reason\":null}]}");
    sse_send(r, &b);
}

static void sink(void *user, LmOutKind kind, const char *s, size_t n)
{
    Reply *r = (Reply *)user;
    if (!r->stream || kind == LM_OUT_TOOLCALL || r->conn->broken) return;
    sse_start(r);
    Buf b = { 0 };
    chunk_head(&b, r);
    buf_str(&b, kind == LM_OUT_REASONING ? ",\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":"
                                         : ",\"choices\":[{\"index\":0,\"delta\":{\"content\":");
    buf_json(&b, s, n);
    buf_str(&b, "},\"finish_reason\":null}]}");
    sse_send(r, &b);
}

static Reply *g_reply; /* the reply in progress, for cancellation */
static int cancelled(void *user)
{
    (void)user;
    if (g_stop) return 1;
    if (!g_reply) return 0;
    if (!g_reply->conn->broken && client_gone(g_reply->conn)) g_reply->conn->broken = 1;
    return g_reply->conn->broken;
}

/* message text: a string, an array of {"type":"text","text":...} parts joined, or null (an
 * assistant turn with no text). NULL with a message in err for anything else. */
static char *msg_text(jval *content, int i, char *err, size_t errlen)
{
    Buf b = { 0 };
    buf_str(&b, "");
    if (!content || content->t == J_NULL) {
    } else if (content->t == J_STR) {
        buf_str(&b, content->str);
    } else if (content->t == J_ARR) {
        for (int k = 0; k < content->len; k++) {
            jval *p = content->kids[k], *ty = json_get(p, "type"), *tx = json_get(p, "text");
            if (!ty || ty->t != J_STR || strcmp(ty->str, "text") || !tx || tx->t != J_STR) {
                snprintf(err, errlen, "messages[%d].content: only text parts are supported", i);
                free(b.s);
                return NULL;
            }
            buf_str(&b, tx->str);
        }
    } else {
        snprintf(err, errlen, "messages[%d].content must be a string or an array of text parts", i);
        free(b.s);
        return NULL;
    }
    if (b.oom) { snprintf(err, errlen, "out of memory"); free(b.s); return NULL; }
    return b.s;
}

static int is_int(const jval *v, double lo, double hi)
{
    return v && v->t == J_NUM && v->num == floor(v->num) && v->num >= lo && v->num <= hi;
}

/* a parameter this server cannot honour must be absent or neutral: refuse, never ignore */
static const char *unsupported(jval *req)
{
    static const char *const keys[] = { "tools", "functions", "tool_choice", "function_call", "logit_bias", "stop" };
    for (size_t i = 0; i < sizeof keys / sizeof *keys; i++) {
        jval *v = json_get(req, keys[i]);
        if (!v || v->t == J_NULL || ((v->t == J_ARR || v->t == J_OBJ) && v->len == 0)) continue;
        if (!strcmp(keys[i], "tool_choice") && v->t == J_STR && !strcmp(v->str, "none")) continue;
        return keys[i];
    }
    jval *v;
    if ((v = json_get(req, "n")) && v->t != J_NULL && !(v->t == J_NUM && v->num == 1)) return "n";
    if ((v = json_get(req, "logprobs")) && v->t == J_BOOL && v->boolean) return "logprobs";
    if ((v = json_get(req, "frequency_penalty")) && v->t == J_NUM && v->num != 0) return "frequency_penalty";
    if ((v = json_get(req, "presence_penalty")) && v->t == J_NUM && v->num != 0) return "presence_penalty";
    if ((v = json_get(req, "response_format")) && v->t == J_OBJ) {
        jval *t = json_get(v, "type");
        if (!t || t->t != J_STR || strcmp(t->str, "text")) return "response_format";
    }
    return NULL;
}

typedef struct { int status, n_prompt, n_cached, n_prefill, n_gen, reprefill; double prefill_s, decode_s; } Log;

static void chat_completions(Server *s, Conn *c, Log *log)
{
    char err[512] = "";
    jval *req = json_parse(c->body, NULL);
    LmMsg *msgs = NULL;
    char **texts = NULL;
    int nm = 0;
    jval *v;
    if (!req || req->t != J_OBJ) { respond_error(c, log->status = 400, NULL, "the body is not a JSON object"); goto out; }
    const char *bad = unsupported(req);
    if (bad) { respond_error(c, log->status = 400, "unsupported_parameter", "%s is not supported by this server (text chat only)", bad); goto out; }

    jval *jm = json_get(req, "messages");
    if (!jm || jm->t != J_ARR || jm->len == 0) { respond_error(c, log->status = 400, NULL, "messages must be a non-empty array"); goto out; }
    nm = jm->len;
    msgs = (LmMsg *)calloc((size_t)nm, sizeof *msgs);
    texts = (char **)calloc((size_t)nm, sizeof *texts);
    if (!msgs || !texts) { respond_error(c, log->status = 500, NULL, "out of memory"); goto out; }
    for (int i = 0; i < nm; i++) {
        jval *m = jm->kids[i], *role = json_get(m, "role");
        if (!role || role->t != J_STR) { respond_error(c, log->status = 400, NULL, "messages[%d] needs a role", i); goto out; }
        if (!strcmp(role->str, "system") || !strcmp(role->str, "developer")) msgs[i].role = LM_ROLE_SYSTEM;
        else if (!strcmp(role->str, "user")) msgs[i].role = LM_ROLE_USER;
        else if (!strcmp(role->str, "assistant")) msgs[i].role = LM_ROLE_ASSISTANT;
        else { respond_error(c, log->status = 400, NULL, "messages[%d]: role \"%.64s\" is not supported (text chat only)", i, role->str); goto out; }
        if (!(texts[i] = msg_text(json_get(m, "content"), i, err, sizeof err))) { respond_error(c, log->status = 400, NULL, "%s", err); goto out; }
        msgs[i].content = texts[i];
    }
    if (msgs[nm - 1].role != LM_ROLE_USER) { respond_error(c, log->status = 400, NULL, "the last message must be from the user"); goto out; }

    int max_tokens = s->max_tokens;
    if (!(v = json_get(req, "max_completion_tokens")) || v->t == J_NULL) v = json_get(req, "max_tokens");
    if (v && v->t != J_NULL) {
        if (!is_int(v, 1, 1 << 24)) { respond_error(c, log->status = 400, NULL, "max_tokens must be a positive integer"); goto out; }
        max_tokens = (int)v->num;
    }
    double temperature = s->temperature, top_p = s->top_p;
    int top_k = s->top_k;
    uint64_t seed = s->seed + s->requests; /* a different draw per request unless the client fixes it */
    if ((v = json_get(req, "temperature")) && v->t != J_NULL) {
        if (v->t != J_NUM || !(v->num >= 0 && v->num <= 100)) { respond_error(c, log->status = 400, NULL, "temperature must be a number >= 0"); goto out; }
        temperature = v->num;
    }
    if ((v = json_get(req, "top_p")) && v->t != J_NULL) {
        if (v->t != J_NUM || !(v->num > 0 && v->num <= 1)) { respond_error(c, log->status = 400, NULL, "top_p must be in (0, 1]"); goto out; }
        top_p = v->num;
    }
    if ((v = json_get(req, "top_k")) && v->t != J_NULL) {
        if (!is_int(v, 0, 1 << 20)) { respond_error(c, log->status = 400, NULL, "top_k must be an integer >= 0"); goto out; }
        top_k = (int)v->num;
    }
    if ((v = json_get(req, "seed")) && v->t != J_NULL) {
        if (!is_int(v, 0, 9007199254740991.0)) { respond_error(c, log->status = 400, NULL, "seed must be an integer >= 0"); goto out; }
        seed = (uint64_t)v->num;
    }
    int stream = 0, usage_in_stream = 0;
    if ((v = json_get(req, "stream")) && v->t != J_NULL) {
        if (v->t != J_BOOL) { respond_error(c, log->status = 400, NULL, "stream must be true or false"); goto out; }
        stream = v->boolean;
    }
    if ((v = json_get(req, "stream_options")) && v->t == J_OBJ && (v = json_get(v, "include_usage")) && v->t == J_BOOL)
        usage_in_stream = v->boolean;

    if (lm_chat_sync(s->chat, msgs, nm - 1, err, sizeof err)) { respond_error(c, log->status = 400, NULL, "%s", err); goto out; }
    lm_chat_set_sampling(s->chat, temperature, top_p, top_k, seed);

    Reply r;
    memset(&r, 0, sizeof r);
    r.conn = c;
    r.srv = s;
    r.stream = stream;
    r.created = (long)time(NULL);
    snprintf(r.id, sizeof r.id, "chatcmpl-%ld-%lu", r.created, s->requests);
    LmTurnOpts to = { max_tokens, sink, &r, cancelled, NULL };
    LmTurn t;
    g_reply = &r;
    int rc = lm_chat_turn(s->chat, msgs[nm - 1].content, &to, &t, err, sizeof err);
    g_reply = NULL;
    log->n_prompt = t.n_prompt;
    log->n_prefill = t.n_prefill;
    log->n_cached = t.n_cached;
    log->n_gen = t.n_gen;
    log->reprefill = t.reprefill;
    log->prefill_s = t.prefill_s;
    log->decode_s = t.decode_s;

    /* A reply that could not be finished is an error, never a "stop" that looks complete. An
     * LM_ERR_CTX after the prompt went in is a reply cut by the window: finish_reason "length". */
    const char *fail = NULL, *code = NULL;
    int fail_status = 500;
    if (rc == LM_OK && t.stop == LM_STOP_CANCEL) {
        if (c->broken) { log->status = 499; goto out; } /* nobody to answer */
        fail = "the server is shutting down; the reply was not finished";
        fail_status = 503;
    } else if (rc == LM_OK && t.stop == LM_STOP_TOOLCALL) {
        fail = "the model attempted a tool call, which this server does not support";
    } else if (rc == LM_ERR_CTX && t.n_prefill == 0) {
        fail = err;
        fail_status = 400;
        code = "context_length_exceeded";
    } else if (rc != LM_OK && rc != LM_ERR_CTX) {
        fail = err[0] ? err : "generation failed";
    }
    if (fail && !r.started) { respond_error(c, log->status = fail_status, code, "%s", fail); goto out; }
    log->status = fail ? fail_status : 200;
    if (stream) {
        sse_start(&r);
        Buf b = { 0 };
        if (fail) {
            buf_str(&b, "data: ");
            error_json(&b, "server_error", code, fail);
        } else {
            chunk_head(&b, &r);
            buf_fmt(&b, ",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"%s\"}]}",
                    t.stop == LM_STOP_EOS ? "stop" : "length");
        }
        sse_send(&r, &b);
        if (usage_in_stream && !fail) {
            Buf u = { 0 };
            chunk_head(&u, &r);
            buf_fmt(&u, ",\"choices\":[],\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,\"total_tokens\":%d}}",
                    t.n_prompt, t.n_gen, t.n_prompt + t.n_gen);
            sse_send(&r, &u);
        }
        send_all(c, "data: [DONE]\n\n", 14);
    } else {
        Buf b = { 0 };
        buf_str(&b, "{\"id\":");
        buf_jstr(&b, r.id);
        buf_fmt(&b, ",\"object\":\"chat.completion\",\"created\":%ld,\"model\":", r.created);
        buf_jstr(&b, s->name);
        buf_str(&b, ",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":");
        buf_jstr(&b, t.answer ? t.answer : "");
        if (t.reasoning && t.reasoning[0]) { buf_str(&b, ",\"reasoning_content\":"); buf_jstr(&b, t.reasoning); }
        buf_fmt(&b, "},\"finish_reason\":\"%s\"}],\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,\"total_tokens\":%d}}",
                t.stop == LM_STOP_EOS ? "stop" : "length", t.n_prompt, t.n_gen, t.n_prompt + t.n_gen);
        if (b.oom) respond_error(c, log->status = 500, NULL, "out of memory");
        else respond(c, 200, NULL, b.s, b.n);
        free(b.s);
    }
out:
    if (texts) for (int i = 0; i < nm; i++) free(texts[i]);
    free(texts);
    free(msgs);
    json_free_tree(req);
}

static void respond_name(Conn *c, Log *log, const char *pre, const char *name, const char *post)
{
    Buf b = { 0 };
    buf_str(&b, pre);
    buf_jstr(&b, name);
    buf_str(&b, post);
    if (b.oom) respond_error(c, log->status = 500, NULL, "out of memory");
    else respond(c, log->status = 200, NULL, b.s, b.n);
    free(b.s);
}

static void handle(Server *s, Conn *c, int timeout_s)
{
    const double t0 = mono_s();
    c->deadline = t0 + timeout_s;
    Log log;
    memset(&log, 0, sizeof log);
    const int rh = read_head(c);
    if (rh < 0) return;
    if (rh > 0) {
        respond_error(c, log.status = rh, NULL, "%s", reason(rh));
    } else if (!strcmp(c->path, "/health")) {
        if (strcmp(c->method, "GET")) respond_error(c, log.status = 405, NULL, "use GET");
        else respond_name(c, &log, "{\"status\":\"ok\",\"model\":", s->name, "}");
    } else if (strncmp(c->path, "/v1/", 4)) {
        respond_error(c, log.status = 404, NULL, "no such endpoint: %s", c->path);
    } else if (!authorised(s, c)) { /* before the body: no key, no 4 MiB read */
        respond_error(c, log.status = 401, "invalid_api_key", "a valid \"Authorization: Bearer <key>\" header is required");
    } else if (!strcmp(c->path, "/v1/models")) {
        if (strcmp(c->method, "GET")) respond_error(c, log.status = 405, NULL, "use GET");
        else respond_name(c, &log, "{\"object\":\"list\",\"data\":[{\"id\":", s->name,
                          ",\"object\":\"model\",\"created\":0,\"owned_by\":\"local\"}]}");
    } else if (!strcmp(c->path, "/v1/chat/completions")) {
        if (strcmp(c->method, "POST")) respond_error(c, log.status = 405, NULL, "use POST");
        else {
            const int rb = read_body(c);
            if (rb > 0) respond_error(c, log.status = rb, NULL, "%s", reason(rb));
            else if (rb == 0) { s->requests++; chat_completions(s, c, &log); }
        }
    } else {
        respond_error(c, log.status = 404, NULL, "no such endpoint: %s", c->path);
    }
    /* the path is the client's: no control bytes reach the log */
    char path[sizeof c->path];
    size_t i = 0;
    for (; c->path[i]; i++) path[i] = (unsigned char)c->path[i] < 0x20 || c->path[i] == 0x7f ? '?' : c->path[i];
    path[i] = 0;
    const double dt = mono_s() - t0;
    if (log.n_prompt)
        fprintf(stderr, "lm-serve: %s %s %d%s | prompt %d, %d from cache, fed %d%s at %.2f tok/s | generated %d at %.2f tok/s | %.1f s\n",
                c->method, path, log.status, c->broken ? " (client gone)" : "", log.n_prompt, log.n_cached, log.n_prefill,
                log.reprefill && !log.n_cached ? " (re-fed)" : "", log.prefill_s > 0 ? log.n_prefill / log.prefill_s : 0.0,
                log.n_gen, log.decode_s > 0 ? log.n_gen / log.decode_s : 0.0, dt);
    else
        fprintf(stderr, "lm-serve: %s %s %d | %.3f s\n", c->method, path, log.status, dt);
}

/* ---- setup ------------------------------------------------------------------------- */

static int be_feed(void *ctx, int token, float *logits) { return lm_feed((LmModel *)ctx, token, logits); }
static int be_prefill(void *ctx, const int *ids, int n, float *logits) { return lm_prefill((LmModel *)ctx, ids, n, logits); }
static void be_reset(void *ctx) { lm_reset((LmModel *)ctx); }
static int be_pos(void *ctx) { return lm_pos((const LmModel *)ctx); }
static void *be_save(void *ctx) { return lm_snapshot((const LmModel *)ctx); }
static int be_restore(void *ctx, void *snap) { return lm_restore((LmModel *)ctx, (const LmSnapshot *)snap); }
static void be_drop(void *ctx, void *snap) { (void)ctx; lm_snapshot_free((LmSnapshot *)snap); }

/* Listen on host:port. *loopback says whether only this machine can connect. */
static int listen_on(const char *host, const char *port, char *bound, size_t boundlen, int *loopback)
{
    struct addrinfo hints, *ai = NULL;
    memset(&hints, 0, sizeof hints);
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE;
    int rc = getaddrinfo(host, port, &hints, &ai);
    if (rc) { fprintf(stderr, "lm-serve: cannot resolve %s: %s\n", host, gai_strerror(rc)); return -1; }
    int fd = -1;
    for (struct addrinfo *p = ai; p && fd < 0; p = p->ai_next) {
        fd = socket(p->ai_family, p->ai_socktype, p->ai_protocol);
        if (fd < 0) continue;
        int one = 1;
        setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
        if (bind(fd, p->ai_addr, p->ai_addrlen) || listen(fd, 16)) {
            fprintf(stderr, "lm-serve: cannot listen on %s port %s: %s\n", host, port, strerror(errno));
            close(fd);
            fd = -1;
        }
    }
    freeaddrinfo(ai);
    if (fd < 0) return -1;
    struct sockaddr_storage ss;
    socklen_t sl = sizeof ss;
    char ip[INET6_ADDRSTRLEN] = "?";
    *loopback = 0;
    if (!getsockname(fd, (struct sockaddr *)&ss, &sl)) {
        if (ss.ss_family == AF_INET6) {
            struct sockaddr_in6 *a = (struct sockaddr_in6 *)&ss;
            inet_ntop(AF_INET6, &a->sin6_addr, ip, sizeof ip);
            snprintf(bound, boundlen, "http://[%s]:%d", ip, ntohs(a->sin6_port));
            *loopback = IN6_IS_ADDR_LOOPBACK(&a->sin6_addr);
        } else {
            struct sockaddr_in *a = (struct sockaddr_in *)&ss;
            inet_ntop(AF_INET, &a->sin_addr, ip, sizeof ip);
            snprintf(bound, boundlen, "http://%s:%d", ip, ntohs(a->sin_port));
            *loopback = (ntohl(a->sin_addr.s_addr) >> 24) == 127;
        }
    }
    return fd;
}

static void usage(FILE *f)
{
    fputs("usage: lm-serve --model DIR [--weights FILE] [--host ADDR] [--port N] [--name ID]\n"
          "       [--think | --no-think] [--max-tokens N] [--seed N] [--ctx N] [--threads N]\n"
          "       [--timeout S] [--fast]\n"
          "       (set LM_SERVE_API_KEY to require a bearer key on /v1 paths)\n", f);
}

static const char *need_value(int argc, char **argv, int *i)
{
    if (*i + 1 >= argc) { fprintf(stderr, "lm-serve: %s needs a value\n", argv[*i]); usage(stderr); exit(2); }
    return argv[++*i];
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

int main(int argc, char **argv)
{
    const char *model = NULL, *weights = NULL, *host = "127.0.0.1", *port = "8080", *name = NULL;
    int think = 0, fast = 0;
    long max_tokens = 0, seed = 1, ctx = 0, threads = 0, port_n, timeout_s = 30;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
#define VAL need_value(argc, argv, &i)
#define BAD do { fprintf(stderr, "lm-serve: invalid value for %s\n", a); return 2; } while (0)
        if (!strcmp(a, "--model")) model = VAL;
        else if (!strcmp(a, "--weights")) weights = VAL;
        else if (!strcmp(a, "--host")) host = VAL;
        else if (!strcmp(a, "--port")) { port = VAL; if (parse_long(port, 0, 65535, &port_n)) BAD; }
        else if (!strcmp(a, "--name")) name = VAL;
        else if (!strcmp(a, "--think")) think = 1;
        else if (!strcmp(a, "--no-think")) think = 0;
        else if (!strcmp(a, "--fast")) fast = 1;
        else if (!strcmp(a, "--max-tokens")) { if (parse_long(VAL, 0, 1 << 24, &max_tokens)) BAD; }
        else if (!strcmp(a, "--seed")) { if (parse_long(VAL, 0, LONG_MAX, &seed)) BAD; }
        else if (!strcmp(a, "--ctx")) { if (parse_long(VAL, 1, 1 << 24, &ctx)) BAD; }
        else if (!strcmp(a, "--threads")) { if (parse_long(VAL, 1, 4096, &threads)) BAD; }
        else if (!strcmp(a, "--timeout")) { if (parse_long(VAL, 1, 3600, &timeout_s)) BAD; }
        else if (!strcmp(a, "--help") || !strcmp(a, "-h")) { usage(stdout); return 0; }
        else { fprintf(stderr, "lm-serve: unknown option %s\n", a); usage(stderr); return 2; }
    }
    if (!model) { fputs("lm-serve: --model DIR is required\n", stderr); usage(stderr); return 2; }
    char nb[256];
    if (!name) {
        size_t n = strlen(model);
        while (n > 1 && model[n - 1] == '/') n--;
        const char *b = model + n;
        while (b > model && b[-1] != '/') b--;
        snprintf(nb, sizeof nb, "%.*s", (int)(model + n - b), b);
        name = nb;
    }
    const char *key = getenv("LM_SERVE_API_KEY");
    if (key && !key[0]) key = NULL;

    char err[1024] = "", path[4096];
    LmProfile prof;
    if (lm_profile_read(model, &prof, err, sizeof err)) { fprintf(stderr, "lm-serve: %s\n", err); return 1; }
    snprintf(path, sizeof path, "%s/tokenizer.json", model);
    LmTok *tok = lm_tok_open(path, err, sizeof err);
    if (!tok) { fprintf(stderr, "lm-serve: %s\n", err); return 1; }
    LmTemplate *tpl = lm_profile_template(&prof, tok, err, sizeof err);
    if (!tpl) { fprintf(stderr, "lm-serve: %s\n", err); lm_tok_close(tok); return 1; }
    LmOpts lo = { weights, (int)ctx, (int)threads, fast };
    LmModel *m = lm_open(model, &lo, err, sizeof err);
    if (!m) { fprintf(stderr, "lm-serve: %s\n", err); lm_template_free(tpl); lm_tok_close(tok); return 1; }

    Server s;
    memset(&s, 0, sizeof s);
    s.name = name;
    s.api_key = key;
    s.max_tokens = (int)max_tokens;
    /* the model directory's sampling defaults, else lm-chat's */
    s.temperature = prof.has_temperature ? prof.temperature : think ? 0.6 : 0.7;
    s.top_p = prof.has_top_p ? prof.top_p : think ? 0.95 : 0.8;
    s.top_k = prof.has_top_k ? prof.top_k : 20;
    s.seed = (uint64_t)seed;

    LmChatConfig cc;
    memset(&cc, 0, sizeof cc);
    cc.tok = tok;
    cc.tpl = tpl;
    cc.backend = (LmBackend){ m, lm_n_vocab(m), lm_n_ctx(m), be_feed, be_reset, be_pos, be_prefill,
                              be_save, be_restore, be_drop };
    cc.temperature = s.temperature;
    cc.top_p = s.top_p;
    cc.top_k = s.top_k;
    cc.seed = s.seed;
    cc.thinking = think;
    int status = 1;
    s.chat = lm_chat_new(&cc, err, sizeof err);
    if (!s.chat) { fprintf(stderr, "lm-serve: %s\n", err); goto done; }

    char bound[128] = "?";
    int loopback = 0;
    const int lfd = listen_on(host, port, bound, sizeof bound, &loopback);
    if (lfd < 0) goto done;

    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_handler = on_signal;
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);
    signal(SIGPIPE, SIG_IGN);
    fprintf(stderr, "lm-serve: listening on %s (model %s, context %d%s)\n", bound, name, lm_n_ctx(m),
            key ? ", API key required" : "");
    if (!loopback && !key)
        fputs("lm-serve: warning: reachable from other machines and LM_SERVE_API_KEY is not set: "
              "anyone who can connect can use the model\n", stderr);

    while (!g_stop) {
        /* wait in short slices so a signal that lands just before accept() is seen */
        struct pollfd lp = { lfd, POLLIN, 0 };
        if (poll(&lp, 1, 500) <= 0) continue;
        int fd = accept(lfd, NULL, NULL);
        if (fd < 0) {
            if (errno == EINTR || errno == ECONNABORTED || errno == EAGAIN) continue;
            fprintf(stderr, "lm-serve: accept failed: %s\n", strerror(errno));
            break;
        }
        struct timeval tv = { timeout_s, 0 }; /* a client that stops reading cannot hold a write */
        setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof tv);
#ifdef SO_NOSIGPIPE
        int one = 1;
        setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof one);
#endif
        Conn c;
        memset(&c, 0, sizeof c);
        c.fd = fd;
        handle(&s, &c, (int)timeout_s);
        finish_conn(&c);
        free(c.raw);
        free(c.body);
        free(c.auth);
        close(fd);
    }
    close(lfd);
    status = g_stop ? 0 : 1;
    fputs("lm-serve: stopped\n", stderr);
done:
    lm_chat_free(s.chat);
    lm_close(m);
    lm_template_free(tpl);
    lm_tok_close(tok);
    return status;
}
