/* lm-serve -- an OpenAI-compatible HTTP endpoint for one model, one reply at a time.
 *
 *   lm-serve --model DIR [--weights FILE] [--host ADDR] [--port N] [--name ID]
 *            [--think | --no-think] [--max-tokens N] [--seed N] [--ctx N] [--threads N]
 *            [--timeout S] [--cache-mb N] [--fast]
 *
 *   --cache-mb N  memory for snapshots of system prompts (default 512): a request whose system
 *                 prompt was seen before restores it instead of feeding it again. 0 = one.
 *
 *   GET  /health               {"status":"ok"} once the model is loaded (no key needed)
 *   GET  /v1/models            the one model, as id --name (default: DIR's last component)
 *   POST /v1/chat/completions  messages, stream, max_tokens / max_completion_tokens,
 *                              temperature, top_p, top_k, seed, stream_options.include_usage,
 *                              tools, tool_choice ("auto" or "none")
 *
 * Text, image and tool chat: stop strings, n > 1, logprobs, penalties, JSON modes and the
 * legacy functions / function_call are refused with 400 rather than ignored.
 *
 * Tools (OpenAI shape). "tools" is rendered where the model's own chat template puts it
 * (Qwen3.5 and Gemma 4); a call the model writes comes back as message.tool_calls
 * (id, type "function", function.name, function.arguments as a JSON string) with
 * finish_reason "tool_calls", in a stream as one delta.tool_calls entry per call, whole,
 * when the call is complete. Send the results back as {"role":"tool","tool_call_id":...,
 * "content":...} messages after the assistant message that holds tool_calls. A tool
 * definition that is not well formed is 400 invalid_tools; a call the model wrote that does
 * not parse is returned as plain content, never dropped. tool_choice "none" keeps the tools
 * out of the prompt; "required" and a named function are refused. stream_options
 * .include_usage adds a last chunk with empty choices and usage (prompt_tokens,
 * completion_tokens, total_tokens, prompt_tokens_details.cached_tokens).
 * Message content may be a string or an array of
 * {"type":"text","text":...} and {"type":"image_url","image_url":{"url":"data:image/..."}}
 * parts (PNG/JPEG data URLs only; at most 8 images). The answer is "content"; a think block,
 * when --think is on, comes back as "reasoning_content". Without max_tokens a reply runs
 * until the model stops or the context fills (--max-tokens N sets a server-wide default
 * instead). A reply is stopped as soon as its client goes away, so an abandoned request
 * does not hold the server. A reply that could not be finished (shutdown, a tool call) is
 * an error, never a "stop".
 *
 * The engine cannot rewind, so the state is reused when the request's history begins with
 * the conversation it last served: a client that sends back the previous answer unchanged
 * only pays for the new message (lm_chat_sync). Anything else re-feeds the whole prompt.
 * Image SHAs are part of that match, so a different picture never reuses another.
 *
 * Listens on 127.0.0.1 unless --host says otherwise; --port 0 picks a free port. The bound
 * address is printed on stderr. If LM_SERVE_API_KEY is set (in the environment, so it never
 * shows up in ps), every /v1 path requires "Authorization: Bearer <key>", checked before
 * the body is read. A request has --timeout seconds (default 30) to arrive in full, so a slow
 * client cannot hold the server. Each connection is read on its own thread: /health,
 * /v1/models and refusals are answered there, even while a reply is being generated; chat
 * requests, once read in full, wait their turn for the model, which only the main thread
 * touches. The head of a request has at most ten seconds to arrive; the body shares the
 * --timeout deadline with it. SIGINT/SIGTERM stop the reply being generated and exit without
 * waiting out slow clients. Requests are logged as counts and timings only, never their
 * text. Original to this repository. */
#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <netdb.h>
#include <netinet/in.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <time.h>
#include <sys/stat.h>
#include <unistd.h>

#include "chat/lm_chat.h"
#include "chat/lm_profile.h"
#include "chat/lm_template.h"
#include "chat/lm_toolfmt.h"
#include "io/lm_image.h"
#include "io/lm_sha256.h"
#include "json.h"
#include "lm/lm.h"
#include "model/lm_vision.h"
#include "tokenizer/lm_tok.h"

#ifndef MSG_NOSIGNAL
#define MSG_NOSIGNAL 0 /* macOS: SIGPIPE is ignored process-wide instead */
#endif

enum { MAX_HEAD = 16 * 1024, MAX_BODY = 24 * 1024 * 1024, MAX_IMAGES = 8, MAX_TOOLS = 32, MAX_TOOL_CALLS = 32 };

/* read and written from several threads; lock-free on every platform this builds on */
static _Atomic int g_stop = 0;
/* a byte on the pipe wakes every poll() the moment a signal lands: the workers do not wait
 * out their request deadlines after SIGINT/SIGTERM */
static int g_stop_pipe[2] = { -1, -1 };
static long g_log_max = 1 << 20; /* stderr, when it is a file, is cut back to this */
static void on_signal(int sig)
{
    (void)sig;
    g_stop = 1;
    if (g_stop_pipe[1] >= 0 && write(g_stop_pipe[1], "x", 1) < 0) { /* full or closing: the flag is set anyway */ }
}

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
        struct pollfd p[2] = { { c->fd, POLLIN, 0 }, { g_stop_pipe[0], POLLIN, 0 } };
        int r = poll(p, 2, (int)(left * 1000) + 1);
        if (r > 0) return p[0].revents ? 1 : 0;
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
    LmModel *model;
    LmVision *vision;        /* NULL = vision unavailable */
    const char *name;
    const char *api_key;     /* NULL = no key */
    int max_tokens;          /* server default, 0 = until stop or context */
    int image_max_pixels;    /* default 1048576 */
    int n_ctx;
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

static void call_id(const Reply *r, int index, char *out, size_t n)
{
    snprintf(out, n, "call_%ld_%lu_%d", r->created, r->srv->requests, index);
}

/* A tool call the model finished writing. A stream carries it whole, as one delta. */
static void on_call(void *user, int index, const LmToolCall *call)
{
    Reply *r = (Reply *)user;
    if (!r->stream || r->conn->broken) return;
    sse_start(r);
    char id[64];
    call_id(r, index, id, sizeof id);
    Buf b = { 0 };
    chunk_head(&b, r);
    buf_fmt(&b, ",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":%d,\"id\":", index);
    buf_jstr(&b, id);
    buf_str(&b, ",\"type\":\"function\",\"function\":{\"name\":");
    buf_jstr(&b, call->name);
    buf_str(&b, ",\"arguments\":");
    buf_jstr(&b, call->arguments);
    buf_str(&b, "}}]},\"finish_reason\":null}]}");
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

typedef struct {
    char *fingerprint;   /* sync/reuse key (text + image SHAs) */
    LmMsgPart *parts;
    int n_parts;
    int n_images;
    char *err_code;      /* malloc'd OpenAI-style code, or NULL */
    LmToolCall *calls;   /* assistant: the tool_calls it carries, arguments canonical */
    char **call_ids;     /* their ids ("" when the client sent none) */
    int n_calls;
    char *tool_name;     /* tool: the function the result belongs to */
} ParsedMsg;

static void parsed_free(ParsedMsg *p)
{
    if (!p) return;
    if (p->parts) {
        for (int i = 0; i < p->n_parts; i++) {
            if (p->parts[i].kind == LM_PART_IMAGE) free((void *)p->parts[i].rows);
            else free((void *)p->parts[i].text);
        }
        free(p->parts);
    }
    for (int i = 0; i < p->n_calls; i++) { lm_toolcall_free(&p->calls[i]); free(p->call_ids[i]); }
    free(p->calls);
    free(p->call_ids);
    free(p->tool_name);
    free(p->fingerprint);
    free(p->err_code);
    memset(p, 0, sizeof *p);
}

static int encode_image(Server *s, const char *url, LmMsgPart *out, char *err, size_t errlen, char **code)
{
    memset(out, 0, sizeof *out);
    out->kind = LM_PART_IMAGE;
    if (!s->vision) {
        snprintf(err, errlen, "this model has no vision encoder");
        *code = strdup("image_unsupported");
        return -1;
    }
    if (!url || strncmp(url, "data:image/", 11)) {
        snprintf(err, errlen, "image_url must be a data:image/png|jpeg|jpg;base64,... URL");
        *code = strdup("image_invalid");
        return -1;
    }
    /* Hash the base64 payload bytes (decoded) for conversation reuse. */
    uint8_t *rgb = NULL;
    int w = 0, h = 0;
    if (lm_image_decode_data_url(url, &rgb, &w, &h, err, errlen)) {
        *code = strdup("image_invalid");
        return -1;
    }
    /* SHA of the resized? Prefer source compressed bytes: re-decode payload. */
    {
        const char *comma = strchr(url, ',');
        LmSha256 ctx;
        unsigned char dig[32];
        lm_sha256_init(&ctx);
        if (comma) lm_sha256_update(&ctx, comma + 1, strlen(comma + 1));
        lm_sha256_final(&ctx, dig);
        lm_sha256_hex(dig, out->sha_hex);
    }
    int ow = w, oh = h;
    const LmVisionInfo *vi = lm_vision_info(s->vision);
    if (vi->family == LM_VISION_QWEN) {
        /* factor 32 = patch 16 x merge 2; unit on LmVisionInfo is the patch size. */
        lm_image_smart_resize(h, w, 32, 65536, s->image_max_pixels, &oh, &ow);
    } else {
        const int max_patches = 2520; /* 280 soft tokens * 9 */
        if (lm_image_gemma_size(h, w, vi->unit, max_patches, &oh, &ow, err, errlen)) {
            free(rgb);
            *code = strdup("image_too_large");
            return -1;
        }
        if ((long long)oh * ow > s->image_max_pixels) {
            free(rgb);
            snprintf(err, errlen, "image exceeds --image-max-pixels (%d)", s->image_max_pixels);
            *code = strdup("image_too_large");
            return -1;
        }
    }
    uint8_t *resized = NULL;
    if (lm_image_resize(rgb, w, h, ow, oh, &resized, err, errlen)) {
        free(rgb);
        *code = strdup("image_invalid");
        return -1;
    }
    free(rgb);
    const int nt = lm_vision_tokens(s->vision, oh, ow);
    if (nt < 1) {
        free(resized);
        snprintf(err, errlen, "image size %dx%d is not valid for this vision encoder", ow, oh);
        *code = strdup("image_invalid");
        return -1;
    }
    float *rows = (float *)malloc((size_t)nt * (size_t)vi->out_dim * sizeof(float));
    if (!rows || lm_vision_encode(s->vision, resized, oh, ow, rows, err, errlen)) {
        free(resized); free(rows);
        if (!*err) snprintf(err, errlen, "vision encode failed");
        *code = strdup("image_invalid");
        return -1;
    }
    free(resized);
    out->rows = rows;
    out->n_rows = nt;
    out->width = vi->out_dim;
    out->grid_h = oh / vi->unit;
    out->grid_w = ow / vi->unit;
    return 0;
}

/* Parse message content into fingerprint + optional parts. On error sets *code. */
static int parse_msg_content(Server *s, jval *content, int msg_i, int allow_images,
                             ParsedMsg *out, char *err, size_t errlen)
{
    memset(out, 0, sizeof *out);
    Buf fp = { 0 };
    buf_str(&fp, "");
    if (!content || content->t == J_NULL) {
        out->fingerprint = fp.s ? fp.s : strdup("");
        return out->fingerprint ? 0 : -1;
    }
    if (content->t == J_STR) {
        buf_str(&fp, content->str);
        if (fp.oom) { free(fp.s); snprintf(err, errlen, "out of memory"); return -1; }
        out->fingerprint = fp.s;
        return 0;
    }
    if (content->t != J_ARR) {
        snprintf(err, errlen, "messages[%d].content must be a string or an array of parts", msg_i);
        free(fp.s);
        return -1;
    }
    LmMsgPart *parts = (LmMsgPart *)calloc((size_t)content->len, sizeof *parts);
    if (!parts) { free(fp.s); snprintf(err, errlen, "out of memory"); return -1; }
    int np = 0, ni = 0;
    for (int k = 0; k < content->len; k++) {
        jval *p = content->kids[k], *ty = json_get(p, "type");
        if (!ty || ty->t != J_STR) {
            snprintf(err, errlen, "messages[%d].content[%d]: missing type", msg_i, k);
            out->err_code = strdup("image_invalid");
            goto fail;
        }
        if (!strcmp(ty->str, "text")) {
            jval *tx = json_get(p, "text");
            if (!tx || tx->t != J_STR) {
                snprintf(err, errlen, "messages[%d].content[%d]: text part needs a string", msg_i, k);
                goto fail;
            }
            char *t = strdup(tx->str);
            if (!t) { snprintf(err, errlen, "out of memory"); goto fail; }
            parts[np].kind = LM_PART_TEXT;
            parts[np].text = t;
            buf_str(&fp, t);
            np++;
        } else if (!strcmp(ty->str, "image_url")) {
            if (!allow_images) {
                snprintf(err, errlen, "messages[%d]: images are not allowed in this role", msg_i);
                out->err_code = strdup("image_unsupported");
                goto fail;
            }
            if (!s->vision) {
                snprintf(err, errlen, "vision is unavailable for this model");
                out->err_code = strdup("image_unsupported");
                goto fail;
            }
            if (ni >= MAX_IMAGES) {
                snprintf(err, errlen, "at most %d images per request", MAX_IMAGES);
                out->err_code = strdup("too_many_images");
                goto fail;
            }
            jval *iu = json_get(p, "image_url"), *url = iu ? json_get(iu, "url") : NULL;
            if (!url || url->t != J_STR) {
                snprintf(err, errlen, "messages[%d].content[%d]: image_url needs a url string", msg_i, k);
                out->err_code = strdup("image_invalid");
                goto fail;
            }
            char *code = NULL;
            if (encode_image(s, url->str, &parts[np], err, errlen, &code)) {
                out->err_code = code;
                goto fail;
            }
            buf_fmt(&fp, "\n<#img:%s>", parts[np].sha_hex);
            np++;
            ni++;
        } else {
            snprintf(err, errlen, "messages[%d].content[%d]: unsupported part type \"%.32s\"", msg_i, k, ty->str);
            out->err_code = strdup("image_unsupported");
            goto fail;
        }
    }
    if (fp.oom) { snprintf(err, errlen, "out of memory"); goto fail; }
    out->fingerprint = fp.s;
    out->parts = parts;
    out->n_parts = np;
    out->n_images = ni;
    return 0;
fail:
    for (int i = 0; i < np; i++) {
        if (parts[i].kind == LM_PART_IMAGE) free((void *)parts[i].rows);
        else free((void *)parts[i].text);
    }
    free(parts);
    free(fp.s);
    return -1;
}

static int is_int(const jval *v, double lo, double hi)
{
    return v && v->t == J_NUM && v->num == floor(v->num) && v->num >= lo && v->num <= hi;
}

/* a parameter this server cannot honour must be absent or neutral: refuse, never ignore */
static const char *unsupported(jval *req)
{
    static const char *const keys[] = { "functions", "function_call", "logit_bias", "stop" };
    for (size_t i = 0; i < sizeof keys / sizeof *keys; i++) {
        jval *v = json_get(req, keys[i]);
        if (!v || v->t == J_NULL || ((v->t == J_ARR || v->t == J_OBJ) && v->len == 0)) continue;
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

/* a function name the two chat formats can carry: no markup or separators */
static int tool_name_ok(const char *n)
{
    size_t len = strlen(n);
    if (len == 0 || len > 64) return 0;
    for (; *n; n++) {
        const unsigned char c = (unsigned char)*n;
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' || c == '-' || c == '.'))
            return 0;
    }
    return 1;
}

/* the request's "tools": 0, or -1 with the reason in err (400 invalid_tools) */
static int check_tools(const jval *tools, char *err, size_t errlen)
{
    if (tools->t != J_ARR) { snprintf(err, errlen, "tools must be an array"); return -1; }
    if (tools->len > MAX_TOOLS) { snprintf(err, errlen, "at most %d tools per request", MAX_TOOLS); return -1; }
    for (int i = 0; i < tools->len; i++) {
        jval *t = tools->kids[i], *ty = json_get(t, "type"), *fn = json_get(t, "function");
        if (!t || t->t != J_OBJ || !ty || ty->t != J_STR || strcmp(ty->str, "function") || !fn || fn->t != J_OBJ) {
            snprintf(err, errlen, "tools[%d] must be {\"type\":\"function\",\"function\":{...}}", i);
            return -1;
        }
        jval *nm = json_get(fn, "name"), *ds = json_get(fn, "description"), *pr = json_get(fn, "parameters");
        if (!nm || nm->t != J_STR || !tool_name_ok(nm->str)) {
            snprintf(err, errlen, "tools[%d].function.name must be 1-64 characters of letters, digits, _ - .", i);
            return -1;
        }
        if (ds && ds->t != J_NULL && ds->t != J_STR) { snprintf(err, errlen, "tools[%d].function.description must be a string", i); return -1; }
        if (pr && pr->t != J_NULL) {
            jval *pt = json_get(pr, "type"), *pp = json_get(pr, "properties"), *rq = json_get(pr, "required");
            if (pr->t != J_OBJ || (pt && (pt->t != J_STR || strcmp(pt->str, "object"))) ||
                (pp && pp->t != J_OBJ) || (rq && rq->t != J_ARR)) {
                snprintf(err, errlen, "tools[%d].function.parameters must be a JSON schema object of type \"object\"", i);
                return -1;
            }
            for (int k = 0; rq && k < rq->len; k++)
                if (rq->kids[k]->t != J_STR) { snprintf(err, errlen, "tools[%d].function.parameters.required must list strings", i); return -1; }
            for (int k = 0; pp && k < pp->len; k++)
                if (pp->kids[k]->t != J_OBJ) { snprintf(err, errlen, "tools[%d].function.parameters.properties.%.64s must be an object", i, pp->keys[k]); return -1; }
        }
    }
    return 0;
}

/* An assistant message's "tool_calls" into p. 0, or -1 with the reason in err
 * (400 invalid_tool_call). Arguments are canonical compact JSON, so a client that
 * reformats them still matches the call the chat made. */
static int parse_tool_calls(const jval *tc, int msg_i, ParsedMsg *p, char *err, size_t errlen)
{
    if (!tc || tc->t == J_NULL) return 0;
    if (tc->t != J_ARR || tc->len > MAX_TOOL_CALLS) {
        snprintf(err, errlen, "messages[%d].tool_calls must be an array of at most %d calls", msg_i, MAX_TOOL_CALLS);
        return -1;
    }
    if (tc->len == 0) return 0;
    p->calls = (LmToolCall *)calloc((size_t)tc->len, sizeof *p->calls);
    p->call_ids = (char **)calloc((size_t)tc->len, sizeof *p->call_ids);
    if (!p->calls || !p->call_ids) { snprintf(err, errlen, "out of memory"); return -1; }
    for (int k = 0; k < tc->len; k++) {
        jval *c = tc->kids[k], *fn = json_get(c, "function"), *id = json_get(c, "id");
        jval *nm = fn ? json_get(fn, "name") : NULL, *ar = fn ? json_get(fn, "arguments") : NULL;
        if (!c || c->t != J_OBJ || !fn || fn->t != J_OBJ || !nm || nm->t != J_STR || !tool_name_ok(nm->str)) {
            snprintf(err, errlen, "messages[%d].tool_calls[%d] must be {\"id\":...,\"type\":\"function\",\"function\":{\"name\":...,\"arguments\":\"{...}\"}}", msg_i, k);
            return -1;
        }
        jval *args = NULL;
        if (!ar || ar->t == J_NULL) args = NULL;
        else if (ar->t == J_STR) {
            const char *t = ar->str;
            while (*t == ' ' || *t == '\t' || *t == '\n' || *t == '\r') t++;
            if (*t) {
                args = json_parse(ar->str, NULL);
                if (!args || args->t != J_OBJ) {
                    json_free_tree(args);
                    snprintf(err, errlen, "messages[%d].tool_calls[%d].function.arguments is not a JSON object", msg_i, k);
                    return -1;
                }
            }
        } else if (ar->t != J_OBJ) {
            snprintf(err, errlen, "messages[%d].tool_calls[%d].function.arguments must be a JSON object string", msg_i, k);
            return -1;
        }
        LmSb d = { 0 };
        if (args) lm_json_dump(&d, args, 0);
        else if (ar && ar->t == J_OBJ) lm_json_dump(&d, ar, 0);
        else lm_sb_str(&d, "{}");
        json_free_tree(args);
        p->calls[k].name = strdup(nm->str);
        p->calls[k].arguments = d.oom ? NULL : d.s;
        p->call_ids[k] = strdup(id && id->t == J_STR ? id->str : "");
        p->n_calls = k + 1;
        if (!p->calls[k].name || !p->calls[k].arguments || !p->call_ids[k]) { snprintf(err, errlen, "out of memory"); return -1; }
    }
    return 0;
}

typedef struct { int status, n_prompt, n_cached, n_rewound, n_prefill, n_gen, reprefill; double prefill_s, decode_s; } Log;

static void usage_json(Buf *b, const LmTurn *t)
{
    buf_fmt(b, "\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,\"total_tokens\":%d,"
               "\"prompt_tokens_details\":{\"cached_tokens\":%d}}",
            t->n_prompt, t->n_gen, t->n_prompt + t->n_gen, t->n_cached);
}

static void chat_completions(Server *s, Conn *c, Log *log)
{
    char err[512] = "";
    jval *req = json_parse(c->body, NULL);
    LmMsg *msgs = NULL;
    ParsedMsg *parsed = NULL;
    int nm = 0, total_images = 0;
    jval *v;
    if (!req || req->t != J_OBJ) { respond_error(c, log->status = 400, NULL, "the body is not a JSON object"); goto out; }
    const char *bad = unsupported(req);
    if (bad) { respond_error(c, log->status = 400, "unsupported_parameter", "%s is not supported by this server", bad); goto out; }

    jval *jm = json_get(req, "messages");
    if (!jm || jm->t != J_ARR || jm->len == 0) { respond_error(c, log->status = 400, NULL, "messages must be a non-empty array"); goto out; }
    nm = jm->len;
    msgs = (LmMsg *)calloc((size_t)nm, sizeof *msgs);
    parsed = (ParsedMsg *)calloc((size_t)nm, sizeof *parsed);
    if (!msgs || !parsed) { respond_error(c, log->status = 500, NULL, "out of memory"); goto out; }
    for (int i = 0; i < nm; i++) {
        jval *m = jm->kids[i], *role = json_get(m, "role");
        if (!role || role->t != J_STR) { respond_error(c, log->status = 400, NULL, "messages[%d] needs a role", i); goto out; }
        if (!strcmp(role->str, "system") || !strcmp(role->str, "developer")) msgs[i].role = LM_ROLE_SYSTEM;
        else if (!strcmp(role->str, "user")) msgs[i].role = LM_ROLE_USER;
        else if (!strcmp(role->str, "assistant")) msgs[i].role = LM_ROLE_ASSISTANT;
        else if (!strcmp(role->str, "tool")) msgs[i].role = LM_ROLE_TOOL;
        else { respond_error(c, log->status = 400, NULL, "messages[%d]: role \"%.64s\" is not supported", i, role->str); goto out; }
        const int allow_img = msgs[i].role == LM_ROLE_USER;
        if (parse_msg_content(s, json_get(m, "content"), i, allow_img, &parsed[i], err, sizeof err)) {
            respond_error(c, log->status = 400, parsed[i].err_code, "%s", err);
            goto out;
        }
        if (msgs[i].role == LM_ROLE_ASSISTANT && parse_tool_calls(json_get(m, "tool_calls"), i, &parsed[i], err, sizeof err)) {
            respond_error(c, log->status = 400, "invalid_tool_call", "%s", err);
            goto out;
        }
        if (msgs[i].role == LM_ROLE_TOOL) {
            /* a result belongs to a call the assistant message before it made; its name is what
             * Gemma's format carries, so the id must resolve */
            const ParsedMsg *pv = i > 0 ? &parsed[i - 1] : NULL;
            const int after_calls = i > 0 && ((msgs[i - 1].role == LM_ROLE_ASSISTANT && pv->n_calls > 0) || msgs[i - 1].role == LM_ROLE_TOOL);
            jval *tid = json_get(m, "tool_call_id"), *tn = json_get(m, "name");
            const char *name = NULL;
            for (int j = i - 1; j >= 0 && !name && tid && tid->t == J_STR; j--) {
                if (msgs[j].role == LM_ROLE_USER) break;
                for (int k = 0; k < parsed[j].n_calls; k++)
                    if (parsed[j].call_ids[k][0] && !strcmp(parsed[j].call_ids[k], tid->str)) name = parsed[j].calls[k].name;
            }
            if (!name && tn && tn->t == J_STR && tool_name_ok(tn->str)) name = tn->str;
            if (!after_calls || !name) {
                respond_error(c, log->status = 400, "invalid_tool_message",
                              !after_calls ? "messages[%d]: a tool message must follow the assistant message whose tool_calls it answers"
                                           : "messages[%d]: tool_call_id does not match a tool call of the assistant message before it",
                              i);
                goto out;
            }
            if (!(parsed[i].tool_name = strdup(name))) { respond_error(c, log->status = 500, NULL, "out of memory"); goto out; }
            msgs[i].tool_name = parsed[i].tool_name;
        }
        msgs[i].calls = parsed[i].calls;
        msgs[i].n_calls = parsed[i].n_calls;
        total_images += parsed[i].n_images;
        if (total_images > MAX_IMAGES) {
            respond_error(c, log->status = 400, "too_many_images", "at most %d images per request", MAX_IMAGES);
            goto out;
        }
        msgs[i].content = parsed[i].fingerprint;
        if (parsed[i].n_parts > 0) {
            msgs[i].parts = parsed[i].parts;
            msgs[i].n_parts = parsed[i].n_parts;
        }
    }
    if (msgs[nm - 1].role != LM_ROLE_USER && msgs[nm - 1].role != LM_ROLE_TOOL) {
        respond_error(c, log->status = 400, NULL, "the last message must be from the user or a tool");
        goto out;
    }
    const int ends_with_tool = msgs[nm - 1].role == LM_ROLE_TOOL;

    /* tools: validated whatever tool_choice says; "none" keeps them out of the prompt */
    const jval *tools = NULL;
    if ((v = json_get(req, "tools")) && v->t != J_NULL && !(v->t == J_ARR && v->len == 0)) {
        if (check_tools(v, err, sizeof err)) { respond_error(c, log->status = 400, "invalid_tools", "%s", err); goto out; }
        tools = v;
    }
    if ((v = json_get(req, "tool_choice")) && v->t != J_NULL) {
        if (v->t != J_STR || (strcmp(v->str, "auto") && strcmp(v->str, "none"))) {
            respond_error(c, log->status = 400, "unsupported_parameter", "tool_choice must be \"auto\" or \"none\" (\"required\" and a named function are not supported by this server)");
            goto out;
        }
        if (!strcmp(v->str, "none")) tools = NULL;
    }
    if (ends_with_tool && parsed[nm - 1].n_images) { respond_error(c, log->status = 400, NULL, "a tool message cannot hold images"); goto out; }

    int max_tokens = s->max_tokens;
    if (!(v = json_get(req, "max_completion_tokens")) || v->t == J_NULL) v = json_get(req, "max_tokens");
    if (v && v->t != J_NULL) {
        if (!is_int(v, 1, 1 << 24)) { respond_error(c, log->status = 400, NULL, "max_tokens must be a positive integer"); goto out; }
        max_tokens = (int)v->num;
    }
    double temperature = s->temperature, top_p = s->top_p;
    int top_k = s->top_k;
    uint64_t seed = s->seed + s->requests;
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

    if (lm_chat_sync(s->chat, msgs, ends_with_tool ? nm : nm - 1, err, sizeof err)) { respond_error(c, log->status = 400, NULL, "%s", err); goto out; }
    lm_chat_set_sampling(s->chat, temperature, top_p, top_k, seed);

    Reply r;
    memset(&r, 0, sizeof r);
    r.conn = c;
    r.srv = s;
    r.stream = stream;
    r.created = (long)time(NULL);
    snprintf(r.id, sizeof r.id, "chatcmpl-%ld-%lu", r.created, s->requests);
    LmTurnOpts to = { max_tokens, sink, &r, cancelled, NULL, on_call };
    LmTurn t;
    g_reply = &r;
    lm_chat_set_tools(s->chat, tools);
    int rc;
    if (ends_with_tool)
        rc = lm_chat_continue(s->chat, &to, &t, err, sizeof err);
    else if (parsed[nm - 1].n_parts > 0)
        rc = lm_chat_turn_parts(s->chat, parsed[nm - 1].parts, parsed[nm - 1].n_parts, &to, &t, err, sizeof err);
    else
        rc = lm_chat_turn(s->chat, msgs[nm - 1].content, &to, &t, err, sizeof err);
    lm_chat_set_tools(s->chat, NULL);
    g_reply = NULL;
    log->n_prompt = t.n_prompt;
    log->n_prefill = t.n_prefill;
    log->n_cached = t.n_cached;
    log->n_rewound = t.n_rewound;
    log->n_gen = t.n_gen;
    log->reprefill = t.reprefill;
    log->prefill_s = t.prefill_s;
    log->decode_s = t.decode_s;

    const char *fail = NULL, *code = NULL;
    int fail_status = 500;
    if (rc == LM_OK && t.stop == LM_STOP_CANCEL) {
        if (c->broken) { log->status = 499; goto out; }
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
    const char *finish = t.stop == LM_STOP_EOS ? (t.n_calls > 0 ? "tool_calls" : "stop") : "length";
    if (stream) {
        sse_start(&r);
        Buf b = { 0 };
        if (fail) {
            buf_str(&b, "data: ");
            error_json(&b, "server_error", code, fail);
        } else {
            chunk_head(&b, &r);
            buf_fmt(&b, ",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"%s\"}]}", finish);
        }
        sse_send(&r, &b);
        if (usage_in_stream && !fail) {
            Buf u = { 0 };
            chunk_head(&u, &r);
            buf_str(&u, ",\"choices\":[],");
            usage_json(&u, &t);
            buf_str(&u, "}");
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
        if (t.n_calls > 0 && !(t.answer && t.answer[0])) buf_str(&b, "null");
        else buf_jstr(&b, t.answer ? t.answer : "");
        if (t.reasoning && t.reasoning[0]) { buf_str(&b, ",\"reasoning_content\":"); buf_jstr(&b, t.reasoning); }
        if (t.n_calls > 0) {
            buf_str(&b, ",\"tool_calls\":[");
            for (int k = 0; k < t.n_calls; k++) {
                char id[64];
                call_id(&r, k, id, sizeof id);
                if (k) buf_str(&b, ",");
                buf_str(&b, "{\"id\":");
                buf_jstr(&b, id);
                buf_str(&b, ",\"type\":\"function\",\"function\":{\"name\":");
                buf_jstr(&b, t.calls[k].name);
                buf_str(&b, ",\"arguments\":");
                buf_jstr(&b, t.calls[k].arguments);
                buf_str(&b, "}}");
            }
            buf_str(&b, "]");
        }
        buf_fmt(&b, "},\"finish_reason\":\"%s\"}],", finish);
        usage_json(&b, &t);
        buf_str(&b, "}");
        if (b.oom) respond_error(c, log->status = 500, NULL, "out of memory");
        else respond(c, 200, NULL, b.s, b.n);
        free(b.s);
    }
out:
    if (parsed) for (int i = 0; i < nm; i++) parsed_free(&parsed[i]);
    free(parsed);
    free(msgs);
    json_free_tree(req);
}

/* One connection, from accept to close. A thread reads it and answers what needs no model;
 * the rest goes to the main thread through the queue. */
typedef struct Job {
    Conn c;
    Log log;
    double t0;
    struct Job *next;
} Job;

static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_cv = PTHREAD_COND_INITIALIZER; /* a job was queued or a connection ended */
static Job *g_head, *g_tail;                            /* read in full, waiting for the model */
static int g_conns;                                     /* connections open, queued ones included (shutdown waits for 0) */
static pthread_mutex_t g_log_mu = PTHREAD_MUTEX_INITIALIZER;

/* Read the request and answer it if it needs no model: 1 = the model thread must answer it,
 * 0 = answered, -1 = nothing to answer (the client left or sent nothing). */
static int answer_quick(Server *s, Job *j, int timeout_s)
{
    Conn *c = &j->c;
    Log *log = &j->log;
    j->t0 = mono_s();
    const double full = j->t0 + timeout_s;
    /* the head must arrive quickly: a client that trickles it ties up its thread for seconds,
     * not the whole request timeout, and the connection is fresh (no keep-alive), so a head
     * has no reason to be slow */
    const double head_deadline = timeout_s > 10 ? j->t0 + 10 : full;
    c->deadline = head_deadline;
    const int rh = read_head(c);
    c->deadline = full;
    if (rh < 0) return -1;
    if (rh > 0) {
        respond_error(c, log->status = rh, NULL, "%s", reason(rh));
    } else if (!strcmp(c->path, "/health")) {
        if (strcmp(c->method, "GET")) respond_error(c, log->status = 405, NULL, "use GET");
        else {
            Buf b = { 0 };
            buf_str(&b, "{\"status\":\"ok\",\"model\":");
            buf_jstr(&b, s->name);
            buf_fmt(&b, ",\"context_length\":%d,\"capabilities\":{\"tools\":{\"status\":\"ok\"},\"vision\":{\"status\":\"%s\",\"max_pixels\":%d",
                    s->n_ctx, s->vision ? "ok" : "unavailable", s->image_max_pixels);
            if (s->vision && lm_vision_info(s->vision)->family == LM_VISION_QWEN)
                buf_str(&b, ",\"pixels_per_token\":1024");
            else if (s->vision)
                buf_str(&b, ",\"tokens_per_image\":280");
            buf_str(&b, "}}}");
            if (b.oom) respond_error(c, log->status = 500, NULL, "out of memory");
            else respond(c, log->status = 200, NULL, b.s, b.n);
            free(b.s);
        }
    } else if (strncmp(c->path, "/v1/", 4)) {
        respond_error(c, log->status = 404, NULL, "no such endpoint: %s", c->path);
    } else if (!authorised(s, c)) { /* before the body: no key, no large body read */
        respond_error(c, log->status = 401, "invalid_api_key", "a valid \"Authorization: Bearer <key>\" header is required");
    } else if (!strcmp(c->path, "/v1/models")) {
        if (strcmp(c->method, "GET")) respond_error(c, log->status = 405, NULL, "use GET");
        else {
            Buf b = { 0 };
            buf_str(&b, "{\"object\":\"list\",\"data\":[{\"id\":");
            buf_jstr(&b, s->name);
            buf_fmt(&b, ",\"object\":\"model\",\"created\":0,\"owned_by\":\"local\",\"context_length\":%d,"
                        "\"capabilities\":{\"tools\":{\"status\":\"ok\"},\"vision\":{\"status\":\"%s\",\"max_pixels\":%d",
                    s->n_ctx, s->vision ? "ok" : "unavailable", s->image_max_pixels);
            if (s->vision && lm_vision_info(s->vision)->family == LM_VISION_QWEN)
                buf_str(&b, ",\"pixels_per_token\":1024");
            else if (s->vision)
                buf_str(&b, ",\"tokens_per_image\":280");
            buf_str(&b, "}}}]}");
            if (b.oom) respond_error(c, log->status = 500, NULL, "out of memory");
            else respond(c, log->status = 200, NULL, b.s, b.n);
            free(b.s);
        }
    } else if (!strcmp(c->path, "/v1/chat/reset")) {
        if (strcmp(c->method, "POST")) respond_error(c, log->status = 405, NULL, "use POST");
        else return 1;
    } else if (!strcmp(c->path, "/v1/chat/completions")) {
        if (strcmp(c->method, "POST")) respond_error(c, log->status = 405, NULL, "use POST");
        else {
            /* the whole body before the queue, so a slow client never holds the model */
            const int rb = read_body(c);
            if (rb > 0) respond_error(c, log->status = rb, NULL, "%s", reason(rb));
            else if (rb == 0) return 1;
        }
    } else {
        respond_error(c, log->status = 404, NULL, "no such endpoint: %s", c->path);
    }
    return 0;
}

/* on the main thread only: the model, the chat state and g_reply are its alone */
static void answer_model(Server *s, Job *j)
{
    Conn *c = &j->c;
    /* the client left while the request waited in the queue: nothing to answer, and the
     * prompt (and any image encoding) is not work the live requests should wait for */
    if (client_gone(c)) {
        c->broken = 1;
        j->log.status = 499;
        return;
    }
    if (!strcmp(c->path, "/v1/chat/reset")) {
        lm_chat_purge(s->chat);
        static const char body[] = "{\"status\":\"ok\"}";
        respond(c, j->log.status = 200, NULL, body, sizeof body - 1);
    } else {
        s->requests++;
        chat_completions(s, c, &j->log);
    }
}

static void log_request(const Job *j)
{
    const Conn *c = &j->c;
    const Log *log = &j->log;
    /* the path is the client's: no control bytes reach the log */
    char path[sizeof c->path];
    size_t i = 0;
    for (; c->path[i]; i++) path[i] = (unsigned char)c->path[i] < 0x20 || c->path[i] == 0x7f ? '?' : c->path[i];
    path[i] = 0;
    const double dt = mono_s() - j->t0;
    pthread_mutex_lock(&g_log_mu);
    /* A file log stays within g_log_max. The journal is not a regular file, so the unit
     * sends stderr to a file. Truncate, then write this request's line. */
    fflush(stderr);
    {
        const int fd = fileno(stderr);
        struct stat st;
        if (fd >= 0 && fstat(fd, &st) == 0 && S_ISREG(st.st_mode) && st.st_size > g_log_max && ftruncate(fd, 0) == 0)
            fseek(stderr, 0, SEEK_SET);
    }
    if (log->n_prompt)
        fprintf(stderr, "lm-serve: %s %s %d%s | prompt %d, %d from cache, %d kept by rewind, fed %d%s at %.2f tok/s | generated %d at %.2f tok/s | %.1f s\n",
                c->method, path, log->status, c->broken ? " (client gone)" : "", log->n_prompt, log->n_cached, log->n_rewound,
                log->n_prefill, log->reprefill ? " (re-fed)" : "", log->prefill_s > 0 ? log->n_prefill / log->prefill_s : 0.0,
                log->n_gen, log->decode_s > 0 ? log->n_gen / log->decode_s : 0.0, dt);
    else
        fprintf(stderr, "lm-serve: %s %s %d | %.3f s\n", c->method, path, log->status, dt);
    pthread_mutex_unlock(&g_log_mu);
}

/* close the connection and give its slot back; answered = it gets a log line */
static void job_done(Job *j, int answered)
{
    if (answered) log_request(j);
    finish_conn(&j->c);
    free(j->c.raw);
    free(j->c.body);
    free(j->c.auth);
    close(j->c.fd);
    free(j);
    pthread_mutex_lock(&g_mu);
    g_conns--;
    pthread_cond_broadcast(&g_cv);
    pthread_mutex_unlock(&g_mu);
}

typedef struct { Server *s; int timeout_s; Job *j; } ConnArg;

static void *conn_thread(void *arg)
{
    ConnArg a = *(ConnArg *)arg;
    free(arg);
    const int r = answer_quick(a.s, a.j, a.timeout_s);
    if (r != 1) { job_done(a.j, r == 0); return NULL; }
    pthread_mutex_lock(&g_mu);
    if (g_tail) g_tail->next = a.j;
    else g_head = a.j;
    g_tail = a.j;
    pthread_cond_broadcast(&g_cv);
    pthread_mutex_unlock(&g_mu);
    return NULL;
}

/* the next queued job, or NULL after up to half a second */
static Job *next_job(void)
{
    pthread_mutex_lock(&g_mu);
    if (!g_head) {
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        ts.tv_nsec += 500000000L;
        if (ts.tv_nsec >= 1000000000L) { ts.tv_sec++; ts.tv_nsec -= 1000000000L; }
        pthread_cond_timedwait(&g_cv, &g_mu, &ts);
    }
    Job *j = g_head;
    if (j) {
        g_head = j->next;
        if (!g_head) g_tail = NULL;
        j->next = NULL;
    }
    pthread_mutex_unlock(&g_mu);
    return j;
}

typedef struct { Server *s; int lfd, timeout_s, failed, spare; } Acceptor;

/* accepts while the server runs; at MAX_CONNS it waits, and new clients wait in the backlog */
static void *accept_thread(void *arg)
{
    Acceptor *ac = (Acceptor *)arg;
    while (!g_stop) {
        /* no cap on the count here: refusing to accept is what a stalled client wants —
         * it would keep /health out. The threads are cheap and each is bounded by its
         * deadline; the file-descriptor limit is the real bound, as before */
        /* wait in short slices so a signal that lands just before accept() is seen */
        struct pollfd lp = { ac->lfd, POLLIN, 0 };
        if (poll(&lp, 1, 500) <= 0) continue;
        int fd = accept(ac->lfd, NULL, NULL);
        if (fd < 0) {
            if (errno == EINTR || errno == ECONNABORTED || errno == EAGAIN) continue;
            /* out of descriptors (a peer holding connections open before they are read):
             * shed one connection and keep serving rather than stop. The spare descriptor
             * makes one accept() possible; the connection is closed at once */
            if ((errno == EMFILE || errno == ENFILE) && ac->spare >= 0) {
                close(ac->spare);
                int drop = accept(ac->lfd, NULL, NULL);
                if (drop >= 0) close(drop);
                ac->spare = open("/dev/null", O_RDONLY);
                continue;
            }
            fprintf(stderr, "lm-serve: accept failed: %s\n", strerror(errno));
            ac->failed = 1;
            g_stop = 1;
            break;
        }
        struct timeval tv = { ac->timeout_s, 0 }; /* a client that stops reading cannot hold a write */
        setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof tv);
#ifdef SO_NOSIGPIPE
        int one = 1;
        setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof one);
#endif
        Job *j = (Job *)calloc(1, sizeof *j);
        ConnArg *a = (ConnArg *)malloc(sizeof *a);
        pthread_t t;
        pthread_attr_t attr;
        int ok = j && a && pthread_attr_init(&attr) == 0;
        if (ok) {
            pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
            j->c.fd = fd;
            *a = (ConnArg){ ac->s, ac->timeout_s, j };
            pthread_mutex_lock(&g_mu);
            g_conns++;
            pthread_mutex_unlock(&g_mu);
            if (pthread_create(&t, &attr, conn_thread, a)) {
                pthread_mutex_lock(&g_mu);
                g_conns--;
                pthread_mutex_unlock(&g_mu);
                ok = 0;
            }
            pthread_attr_destroy(&attr);
        }
        if (!ok) {
            fputs("lm-serve: cannot start a connection thread\n", stderr);
            free(j);
            free(a);
            close(fd);
        }
    }
    return NULL;
}

/* ---- setup ------------------------------------------------------------------------- */

static int be_feed(void *ctx, int token, float *logits) { return lm_feed((LmModel *)ctx, token, logits); }
static int be_prefill(void *ctx, const int *ids, int n, float *logits) { return lm_prefill((LmModel *)ctx, ids, n, logits); }
static int be_prefill_image(void *ctx, const float *rows, int n, int width, int grid_h, int grid_w, float *logits)
{
    return lm_prefill_image((LmModel *)ctx, rows, n, width, grid_h, grid_w, logits, NULL);
}
static void be_reset(void *ctx) { lm_reset((LmModel *)ctx); }
static int be_pos(void *ctx) { return lm_pos((const LmModel *)ctx); }
static void *be_save(void *ctx) { return lm_snapshot((const LmModel *)ctx); }
static int be_restore(void *ctx, void *snap) { return lm_restore((LmModel *)ctx, (const LmSnapshot *)snap); }
static void be_drop(void *ctx, void *snap) { (void)ctx; lm_snapshot_free((LmSnapshot *)snap); }
static size_t be_snap_bytes(void *ctx, void *snap) { (void)ctx; return lm_snapshot_bytes((const LmSnapshot *)snap); }
static void *be_mark(void *ctx) { return lm_mark((const LmModel *)ctx); }

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
          "       [--timeout S] [--cache-mb N] [--image-max-pixels N] [--fast]\n"
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
    long max_tokens = 0, seed = 1, ctx = 0, threads = 0, port_n, timeout_s = 30, cache_mb = 512;
    long image_max_pixels = 1048576;
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
        else if (!strcmp(a, "--cache-mb")) { if (parse_long(VAL, 0, 1L << 20, &cache_mb)) BAD; }
        else if (!strcmp(a, "--image-max-pixels")) { if (parse_long(VAL, 1, 1L << 28, &image_max_pixels)) BAD; }
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
    const char *log_max = getenv("LM_SERVE_LOG_MAX");
    if (log_max && log_max[0]) {
        long v;
        if (parse_long(log_max, 1, 1L << 30, &v)) {
            fprintf(stderr, "lm-serve: invalid LM_SERVE_LOG_MAX\n");
            return 2;
        }
        g_log_max = v;
    }
    const char *key = getenv("LM_SERVE_API_KEY");
    if (key && !key[0]) key = NULL;

    char err[1024] = "";
    LmProfile prof;
    if (lm_profile_read(model, &prof, err, sizeof err)) { fprintf(stderr, "lm-serve: %s\n", err); return 1; }
    LmTok *tok = lm_tok_open_model(model, err, sizeof err);
    if (!tok) { fprintf(stderr, "lm-serve: %s\n", err); return 1; }
    LmTemplate *tpl = lm_profile_template(&prof, tok, err, sizeof err);
    if (!tpl) { fprintf(stderr, "lm-serve: %s\n", err); lm_tok_close(tok); return 1; }
    LmOpts lo = { weights, (int)ctx, (int)threads, fast, 0, 1 };
    LmModel *m = lm_open(model, &lo, err, sizeof err);
    if (!m) { fprintf(stderr, "lm-serve: %s\n", err); lm_template_free(tpl); lm_tok_close(tok); return 1; }

    Server s;
    memset(&s, 0, sizeof s);
    s.name = name;
    s.api_key = key;
    s.max_tokens = (int)max_tokens;
    s.image_max_pixels = (int)image_max_pixels;
    s.model = m;
    s.n_ctx = lm_n_ctx(m);
    /* the model directory's sampling defaults, else lm-chat's */
    s.temperature = prof.has_temperature ? prof.temperature : think ? 0.6 : 0.7;
    s.top_p = prof.has_top_p ? prof.top_p : think ? 0.95 : 0.8;
    s.top_k = prof.has_top_k ? prof.top_k : 20;
    s.seed = (uint64_t)seed;

    char verr[256] = "";
    s.vision = lm_vision_open(model, weights, verr, sizeof verr);
    if (s.vision) {
        const LmVisionInfo *vi = lm_vision_info(s.vision);
        lm_template_set_vision(tpl, vi->start_token, vi->image_token, vi->end_token);
    } else {
        fprintf(stderr, "lm-serve: vision unavailable (%s)\n", verr[0] ? verr : "no encoder");
    }

    LmChatConfig cc;
    memset(&cc, 0, sizeof cc);
    cc.tok = tok;
    cc.tpl = tpl;
    cc.backend = (LmBackend){ m, lm_n_vocab(m), lm_n_ctx(m), be_feed, be_reset, be_pos, be_prefill,
                              be_save, be_restore, be_drop, be_snap_bytes, be_prefill_image,
                              be_mark, be_restore };
    cc.cache_bytes = (size_t)cache_mb << 20;
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
    fprintf(stderr, "lm-serve: listening on %s (model %s, context %d, prefix cache %ld MiB%s)\n", bound, name,
            lm_n_ctx(m), cache_mb, key ? ", API key required" : "");
    if (!loopback && !key)
        fputs("lm-serve: warning: reachable from other machines and LM_SERVE_API_KEY is not set: "
              "anyone who can connect can use the model\n", stderr);

    if (pipe(g_stop_pipe)) {
        fputs("lm-serve: pipe failed\n", stderr);
        close(lfd);
        goto done;
    }
    /* the connection threads must not take SIGINT/SIGTERM: the main thread stops the reply */
    sigset_t block, old;
    sigemptyset(&block);
    sigaddset(&block, SIGINT);
    sigaddset(&block, SIGTERM);
    pthread_sigmask(SIG_BLOCK, &block, &old);
    Acceptor ac = { &s, lfd, (int)timeout_s, 0, open("/dev/null", O_RDONLY) };
    pthread_t acceptor;
    const int started = pthread_create(&acceptor, NULL, accept_thread, &ac) == 0;
    pthread_sigmask(SIG_SETMASK, &old, NULL);
    if (!started) fputs("lm-serve: cannot start the accept thread\n", stderr);
    while (started && !g_stop) {
        Job *j = next_job();
        if (!j) continue;
        answer_model(&s, j);
        job_done(j, 1);
    }
    if (started) pthread_join(acceptor, NULL);
    /* the connection threads see g_stop and end; queued requests are refused */
    pthread_mutex_lock(&g_mu);
    while (g_conns > 0) {
        Job *j = g_head;
        if (j) {
            g_head = j->next;
            if (!g_head) g_tail = NULL;
            pthread_mutex_unlock(&g_mu);
            respond_error(&j->c, j->log.status = 503, NULL, "the server is stopping");
            job_done(j, 1);
            pthread_mutex_lock(&g_mu);
            continue;
        }
        pthread_cond_wait(&g_cv, &g_mu);
    }
    pthread_mutex_unlock(&g_mu);
    close(lfd);
    status = started && !ac.failed ? 0 : 1;
    fputs("lm-serve: stopped\n", stderr);
done:
    lm_chat_free(s.chat);
    lm_vision_close(s.vision);
    lm_close(m);
    lm_template_free(tpl);
    lm_tok_close(tok);
    return status;
}
