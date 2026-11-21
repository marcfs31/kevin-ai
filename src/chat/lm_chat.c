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
    float *rows;
    int n_rows, width, grid_h, grid_w;
    char sha_hex[65];
} HistImg;

typedef struct {
    LmRole role;
    char *content;   /* fingerprint for sync (text + image SHAs) and plain text for assistants */
    char *reasoning; /* assistant only */
    int *gen_ids;    /* assistant only; never NULL for an assistant turn made by this chat */
    int n_gen;
    int gen_thinking;
    HistImg *imgs;   /* owned image blobs for user messages (re-prefill) */
    int nimgs;
    LmMsgPart *parts; /* owned part descriptors pointing at content slices / imgs */
    int n_parts;
    char *text_blob; /* owned concatenation of text parts (parts[].text point here) */
} Hist;

typedef struct { char *s; size_t n, cap; } Buf;

typedef struct { void *snap; LmIds ids; size_t bytes; } Slot;   /* a backend snapshot of ids */

struct LmChat {
    LmChatConfig cfg;
    char *system;
    LmSampler smp;
    Hist *h;
    int nh, caph;
    LmIds fed;       /* ids the backend holds, in order */
    char *fed_imgs;  /* '\n'-joined image SHAs in the fed prompt (empty if none) */
    int dirty;       /* backend state unknown (error mid-feed): next turn must reset */
    LmIds prev;      /* what was fed before the last lm_chat_reset */
    char *prev_imgs;
    Slot *slot;      /* the prefix cache, most recently used first */
    int nslot, capslot;
    size_t slot_bytes;   /* sum of slot[].bytes */
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

static void hist_free_imgs(Hist *h)
{
    if (h->imgs) {
        for (int i = 0; i < h->nimgs; i++) free(h->imgs[i].rows);
        free(h->imgs);
    }
    free(h->parts);
    free(h->text_blob);
    h->imgs = NULL;
    h->parts = NULL;
    h->text_blob = NULL;
    h->nimgs = h->n_parts = 0;
}

static void hist_free_one(Hist *h)
{
    free(h->content);
    free(h->reasoning);
    free(h->gen_ids);
    hist_free_imgs(h);
    memset(h, 0, sizeof *h);
}

/* Fingerprint for sync / reuse: text parts plus each image SHA. */
static char *parts_fingerprint(const LmMsgPart *parts, int n_parts)
{
    size_t cap = 64, n = 0;
    char *s = (char *)malloc(cap);
    if (!s) return NULL;
    s[0] = 0;
    for (int i = 0; i < n_parts; i++) {
        const LmMsgPart *p = &parts[i];
        const char *add = NULL;
        char buf[80];
        if (p->kind == LM_PART_TEXT) {
            add = p->text ? p->text : "";
        } else if (p->kind == LM_PART_IMAGE) {
            snprintf(buf, sizeof buf, "\n<#img:%s>", p->sha_hex);
            add = buf;
        } else {
            free(s);
            return NULL;
        }
        const size_t an = strlen(add);
        if (n + an + 1 > cap) {
            while (n + an + 1 > cap) cap *= 2;
            char *q = (char *)realloc(s, cap);
            if (!q) { free(s); return NULL; }
            s = q;
        }
        memcpy(s + n, add, an + 1);
        n += an;
    }
    return s;
}

/* Copy multimodal parts into hist (owned rows + part table). content must already be set. */
static int hist_own_parts(Hist *h, const LmMsgPart *parts, int n_parts)
{
    hist_free_imgs(h);
    if (!parts || n_parts <= 0) return 0;
    size_t text_n = 0;
    int nimgs = 0;
    for (int i = 0; i < n_parts; i++) {
        if (parts[i].kind == LM_PART_TEXT) text_n += parts[i].text ? strlen(parts[i].text) : 0;
        else if (parts[i].kind == LM_PART_IMAGE) nimgs++;
        else return -1;
    }
    char *blob = (char *)malloc(text_n + 1);
    LmMsgPart *op = (LmMsgPart *)calloc((size_t)n_parts, sizeof *op);
    HistImg *imgs = nimgs ? (HistImg *)calloc((size_t)nimgs, sizeof *imgs) : NULL;
    if (!blob || !op || (nimgs && !imgs)) { free(blob); free(op); free(imgs); return -1; }
    blob[0] = 0;
    size_t off = 0;
    int ii = 0;
    for (int i = 0; i < n_parts; i++) {
        op[i] = parts[i];
        if (parts[i].kind == LM_PART_TEXT) {
            const char *t = parts[i].text ? parts[i].text : "";
            const size_t tn = strlen(t);
            memcpy(blob + off, t, tn);
            op[i].text = blob + off;
            off += tn;
            blob[off] = 0;
        } else {
            const size_t nbytes = (size_t)parts[i].n_rows * (size_t)parts[i].width * sizeof(float);
            float *rows = (float *)malloc(nbytes ? nbytes : 1);
            if (!rows || (parts[i].n_rows > 0 && !parts[i].rows)) {
                for (int k = 0; k < ii; k++) free(imgs[k].rows);
                free(rows); free(blob); free(op); free(imgs);
                return -1;
            }
            if (nbytes) memcpy(rows, parts[i].rows, nbytes);
            imgs[ii].rows = rows;
            imgs[ii].n_rows = parts[i].n_rows;
            imgs[ii].width = parts[i].width;
            imgs[ii].grid_h = parts[i].grid_h;
            imgs[ii].grid_w = parts[i].grid_w;
            memcpy(imgs[ii].sha_hex, parts[i].sha_hex, 65);
            op[i].rows = rows;
            op[i].sha_hex[64] = 0;
            memcpy(op[i].sha_hex, parts[i].sha_hex, 65);
            ii++;
        }
    }
    h->text_blob = blob;
    h->parts = op;
    h->n_parts = n_parts;
    h->imgs = imgs;
    h->nimgs = nimgs;
    return 0;
}

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
        !cfg->backend.pos || cfg->backend.n_vocab <= 0 || cfg->backend.n_ctx <= 0 ||
        !cfg->backend.save != !cfg->backend.restore || !cfg->backend.save != !cfg->backend.drop ||
        !cfg->backend.save != !cfg->backend.snap_bytes) {
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

static void drop_slot(LmChat *c, int i)
{
    c->cfg.backend.drop(c->cfg.backend.ctx, c->slot[i].snap);
    lm_ids_free(&c->slot[i].ids);
    c->slot_bytes -= c->slot[i].bytes;
    memmove(c->slot + i, c->slot + i + 1, (size_t)(c->nslot - i - 1) * sizeof *c->slot);
    c->nslot--;
}

void lm_chat_free(LmChat *c)
{
    if (!c) return;
    hist_clear(c);
    free(c->h);
    lm_ids_free(&c->fed);
    lm_ids_free(&c->prev);
    free(c->fed_imgs);
    free(c->prev_imgs);
    while (c->nslot > 0) drop_slot(c, c->nslot - 1);
    free(c->slot);
    lm_sampler_free(&c->smp);
    free(c->system);
    free(c->logits);
    free(c);
}

void lm_chat_reset(LmChat *c)
{
    hist_clear(c);
    LmIds t = c->prev;
    c->prev = c->fed;
    c->fed = t;
    c->fed.n = 0;
    free(c->prev_imgs);
    c->prev_imgs = c->fed_imgs;
    c->fed_imgs = NULL;
    c->dirty = 0;
    c->cfg.backend.reset(c->cfg.backend.ctx);
}

void lm_chat_purge(LmChat *c)
{
    hist_clear(c);
    lm_ids_free(&c->fed);
    lm_ids_free(&c->prev);
    free(c->fed_imgs);
    free(c->prev_imgs);
    c->fed_imgs = c->prev_imgs = NULL;
    c->dirty = 0;
    while (c->nslot > 0) drop_slot(c, 0);
    free(c->system);
    c->system = NULL;
    c->cfg.system = NULL;
    c->cfg.backend.reset(c->cfg.backend.ctx);
}

int lm_chat_set_system(LmChat *c, const char *text)
{
    char *next = text ? dup_str(text) : NULL;
    if (text && !next) return -1;
    free(c->system);
    c->system = next;
    c->cfg.system = next;
    return 0;
}

int lm_chat_reprefills(const LmChat *c) { return c->reprefills; }
int lm_chat_fed(const LmChat *c) { return c->fed.n; }
int lm_chat_cached(const LmChat *c) { return c->nslot ? c->slot[0].ids.n : 0; }
int lm_chat_cache_slots(const LmChat *c) { return c->nslot; }
size_t lm_chat_cache_bytes(const LmChat *c) { return c->slot_bytes; }

int lm_chat_sync(LmChat *c, const LmMsg *msgs, int n, char *err, size_t errlen)
{
    const int first = n > 0 && msgs[0].role == LM_ROLE_SYSTEM;
    for (int i = 0; i < n; i++) {
        if (i >= first && msgs[i].role == LM_ROLE_SYSTEM) { set_err(err, errlen, "a system message must come first"); return -1; }
        if (!msgs[i].content) { set_err(err, errlen, "message %d has no text", i); return -1; }
        if (msgs[i].role == LM_ROLE_SYSTEM && msgs[i].parts) {
            for (int k = 0; k < msgs[i].n_parts; k++) {
                if (msgs[i].parts[k].kind == LM_PART_IMAGE) {
                    set_err(err, errlen, "system messages may not contain images");
                    return -1;
                }
            }
        }
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
        const LmMsg *m = &msgs[first + i];
        h[i].role = m->role;
        if (!(h[i].content = dup_str(m->content))) { oom = 1; break; }
        if (m->parts && m->n_parts > 0 && hist_own_parts(&h[i], m->parts, m->n_parts)) oom = 1;
    }
    if (oom) {
        if (h) for (int i = keep; i < nh; i++) hist_free_one(&h[i]);
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

/* NUL bytes (a model can emit the <0x00> byte token) are left out of the kept text: it is a C
 * string, so a NUL would silently cut it, and history matching compares it with strcmp. */
static void collect(void *u, LmOutKind kind, const char *s, size_t n)
{
    Collect *c = (Collect *)u;
    Buf *b = kind == LM_OUT_REASONING ? &c->reasoning : kind == LM_OUT_ANSWER ? &c->answer : NULL;
    for (size_t i = 0; b && i < n;) {
        size_t j = i;
        while (j < n && s[j]) j++;
        if (j > i && buf_add(b, s + i, j - i)) c->oom = 1;
        i = j + 1;
    }
    if (c->sink) c->sink(c->user, kind, s, n);
}

/* ---- the turn ---------------------------------------------------------------------- */

static void pop_user(LmChat *c)
{
    if (c->nh > 0) hist_free_one(&c->h[--c->nh]);
}

static int common_prefix(const LmIds *a, const LmIds *b)
{
    int n = 0;
    while (n < a->n && n < b->n && a->ids[n] == b->ids[n]) n++;
    return n;
}

/* Where the system message ends in a rendered prompt, the next turn's header included: where
 * two conversations of the system message and one different user message each part. 0 if the
 * template refuses them. */
static int system_end(const LmChat *c, const LmRenderOpts *ro)
{
    LmMsg a[2], b[2];
    memset(a, 0, sizeof a);
    memset(b, 0, sizeof b);
    a[0].role = LM_ROLE_SYSTEM; a[0].content = c->system;
    a[1].role = LM_ROLE_USER; a[1].content = "a";
    b[0].role = LM_ROLE_SYSTEM; b[0].content = c->system;
    b[1].role = LM_ROLE_USER; b[1].content = "b";
    /* Do not record image spans into the caller's list. */
    LmRenderOpts clean = *ro;
    clean.spans = NULL;
    clean.n_spans = NULL;
    LmIds x = { 0 }, y = { 0 };
    int n = 0;
    if (!c->cfg.tpl->render(c->cfg.tpl, a, 2, &clean, &x, NULL, 0) &&
        !c->cfg.tpl->render(c->cfg.tpl, b, 2, &clean, &y, NULL, 0))
        n = common_prefix(&x, &y);
    lm_ids_free(&x);
    lm_ids_free(&y);
    return n;
}

/* The backend is at position 0. Restore the longest slot that covers a prefix of the prompt
 * (leaving at least the last id to feed) and make it the most recently used; returns the ids
 * now held, 0 if none. A slot that fails to restore is dropped. */
static int restore_cache(LmChat *c, const LmIds *prompt)
{
    const LmBackend *be = &c->cfg.backend;
    int best = -1;
    for (int k = 0; k < c->nslot; k++) {
        const LmIds *ids = &c->slot[k].ids;
        if (ids->n < prompt->n && (best < 0 || ids->n > c->slot[best].ids.n) &&
            !memcmp(ids->ids, prompt->ids, (size_t)ids->n * sizeof(int)))
            best = k;
    }
    if (best < 0) return 0;
    const Slot s = c->slot[best];
    memmove(c->slot + 1, c->slot, (size_t)best * sizeof *c->slot);
    c->slot[0] = s;
    c->fed.n = 0;
    if (be->restore(be->ctx, s.snap) != LM_OK || lm_ids_append(&c->fed, s.ids.ids, s.ids.n)) {
        be->reset(be->ctx);
        c->fed.n = 0;
        drop_slot(c, 0);
        return 0;
    }
    return s.ids.n;
}

/* Add a snapshot of the state, which holds c->fed, as the most recently used slot, then drop
 * the least recently used ones while the slots exceed the budget (the new one always stays).
 * With no budget the old slot goes first: two snapshots alive at once could be what exhausts
 * memory. A failure adds nothing. */
static void save_cache(LmChat *c)
{
    const LmBackend *be = &c->cfg.backend;
    if (c->nslot == c->capslot) {
        const int nc = c->capslot ? 2 * c->capslot : 4;
        Slot *p = (Slot *)realloc(c->slot, (size_t)nc * sizeof *p);
        if (!p) return;
        c->slot = p;
        c->capslot = nc;
    }
    LmIds ids = { 0 };
    if (lm_ids_append(&ids, c->fed.ids, c->fed.n)) return;
    if (!c->cfg.cache_bytes) while (c->nslot > 0) drop_slot(c, c->nslot - 1);
    void *snap = be->save(be->ctx);
    if (!snap) { lm_ids_free(&ids); return; }
    memmove(c->slot + 1, c->slot, (size_t)c->nslot * sizeof *c->slot);
    c->slot[0] = (Slot){ snap, ids, be->snap_bytes(be->ctx, snap) };
    c->nslot++;
    c->slot_bytes += c->slot[0].bytes;
    while (c->nslot > 1 && c->slot_bytes > c->cfg.cache_bytes) drop_slot(c, c->nslot - 1);
}

/* Flat list of image blobs in conversation order (for prefill_image). */
typedef struct { const float *rows; int n_rows, width, grid_h, grid_w; char sha[65]; } TurnImg;

static char *imgs_fingerprint(const TurnImg *imgs, int n)
{
    size_t cap = (size_t)n * 66 + 1, off = 0;
    char *s = (char *)malloc(cap ? cap : 1);
    if (!s) return NULL;
    s[0] = 0;
    for (int i = 0; i < n; i++) {
        size_t L = strlen(imgs[i].sha);
        if (off + L + 2 > cap) { free(s); return NULL; }
        memcpy(s + off, imgs[i].sha, L);
        off += L;
        s[off++] = '\n';
        s[off] = 0;
    }
    return s;
}

static int collect_turn_imgs(const LmChat *c, TurnImg **out, int *nout)
{
    int n = 0;
    for (int i = 0; i < c->nh; i++) n += c->h[i].nimgs;
    TurnImg *imgs = n ? (TurnImg *)calloc((size_t)n, sizeof *imgs) : NULL;
    if (n && !imgs) return -1;
    int k = 0;
    for (int i = 0; i < c->nh; i++) {
        for (int j = 0; j < c->h[i].nimgs; j++) {
            const HistImg *hi = &c->h[i].imgs[j];
            imgs[k].rows = hi->rows;
            imgs[k].n_rows = hi->n_rows;
            imgs[k].width = hi->width;
            imgs[k].grid_h = hi->grid_h;
            imgs[k].grid_w = hi->grid_w;
            memcpy(imgs[k].sha, hi->sha_hex, 65);
            k++;
        }
    }
    *out = imgs;
    *nout = n;
    return 0;
}

static int turn_core(LmChat *c, const LmTurnOpts *opts, LmTurn *res, char *err, size_t errlen)
{
    const LmBackend *be = &c->cfg.backend;
    LmMsg *msgs = (LmMsg *)calloc((size_t)c->nh + 1, sizeof *msgs);
    if (!msgs) { pop_user(c); set_err(err, errlen, "out of memory"); return -1; }
    int nm = 0;
    if (c->system) { msgs[nm].role = LM_ROLE_SYSTEM; msgs[nm].content = c->system; nm++; }
    for (int i = 0; i < c->nh; i++) {
        msgs[nm].role = c->h[i].role;
        msgs[nm].content = c->h[i].content;
        msgs[nm].reasoning = NULL;
        msgs[nm].gen_ids = c->h[i].gen_ids;
        msgs[nm].n_gen_ids = c->h[i].n_gen;
        msgs[nm].gen_thinking = c->h[i].gen_thinking;
        msgs[nm].parts = c->h[i].parts;
        msgs[nm].n_parts = c->h[i].n_parts;
        nm++;
    }

    TurnImg *timgs = NULL;
    int ntimgs = 0;
    if (collect_turn_imgs(c, &timgs, &ntimgs)) {
        free(msgs); pop_user(c); set_err(err, errlen, "out of memory"); return -1;
    }
    char *img_fp = imgs_fingerprint(timgs, ntimgs);
    if (!img_fp) { free(timgs); free(msgs); pop_user(c); set_err(err, errlen, "out of memory"); return -1; }

    LmImgSpan *spans = NULL;
    int n_spans = 0;
    LmRenderOpts ro;
    memset(&ro, 0, sizeof ro);
    ro.enable_thinking = c->cfg.thinking;
    ro.verbatim_history = !c->cfg.strict_history;
    ro.spans = &spans;
    ro.n_spans = &n_spans;
    LmIds prompt = { 0 };
    int rc = c->cfg.tpl->render(c->cfg.tpl, msgs, nm, &ro, &prompt, err, errlen);
    free(msgs);
    if (rc) {
        free(spans); free(timgs); free(img_fp); lm_ids_free(&prompt); pop_user(c); return -1;
    }
    res->n_prompt = prompt.n;

    const int has_images = n_spans > 0;
    if (has_images && !be->prefill_image) {
        set_err(err, errlen, "this backend cannot prefill images");
        free(spans); free(timgs); free(img_fp); lm_ids_free(&prompt); pop_user(c); return -1;
    }
    /* Reuse only when ids and image SHAs both match (pad tokens alone are not enough). */
    const char *fed_fp = c->fed_imgs ? c->fed_imgs : "";
    int reuse = !c->dirty && c->fed.n < prompt.n && be->pos(be->ctx) == c->fed.n &&
                (c->fed.n == 0 || !memcmp(c->fed.ids, prompt.ids, (size_t)c->fed.n * sizeof(int))) &&
                !strcmp(fed_fp, img_fp);
    int start = reuse ? c->fed.n : 0;
    int save_at = 0;
    /* Prefix cache skips prompts that contain images (simpler; never restores wrong pixels). */
    if (be->save && c->system && !has_images && !(reuse && c->fed.n > 0)) {
        save_at = common_prefix(c->fed.n > 0 ? &c->fed : &c->prev, &prompt);
        int sys_end = system_end(c, &ro);
        if (save_at > sys_end) save_at = sys_end;
        if (save_at >= prompt.n) save_at = prompt.n - 1;
        if (save_at < LM_CHAT_CACHE_MIN) save_at = 0;
    }
    if (prompt.n > be->n_ctx) {
        set_err(err, errlen, "the prompt is %d tokens and does not fit the %d-token context", prompt.n, be->n_ctx);
        res->stop = LM_STOP_CTX;
        free(spans); free(timgs); free(img_fp); lm_ids_free(&prompt); pop_user(c);
        return LM_ERR_CTX;
    }
    if (!reuse) {
        if (c->fed.n > 0 || c->dirty) { res->reprefill = 1; c->reprefills++; }
        if (c->fed.n > 0 || c->dirty || be->pos(be->ctx) != 0) be->reset(be->ctx);
        c->fed.n = 0;
        free(c->fed_imgs);
        c->fed_imgs = NULL;
        c->dirty = 0;
    }
    if (be->save && c->fed.n == 0 && !has_images) {
        start = res->n_cached = restore_cache(c, &prompt);
        if (start > 0 && res->reprefill) { res->reprefill = 0; c->reprefills--; }
        if (save_at <= start) save_at = 0;
    }

    double t0 = now_s();
    const int chunk = be->prefill ? LM_CHAT_PREFILL_CHUNK : 1;
    int span_i = 0;
    while (span_i < n_spans && spans[span_i].id_start + spans[span_i].n_ids <= start) span_i++;
    for (int i = start; i < prompt.n; ) {
        if (opts->cancelled && opts->cancelled(opts->cancel_user)) {
            res->stop = LM_STOP_CANCEL;
            res->prefill_s = now_s() - t0;
            free(spans); free(timgs); free(img_fp); lm_ids_free(&prompt); pop_user(c);
            return LM_OK;
        }
        if (span_i < n_spans && spans[span_i].id_start == i) {
            const LmImgSpan *sp = &spans[span_i];
            if (sp->img_index < 0 || sp->img_index >= ntimgs) {
                set_err(err, errlen, "image span index out of range");
                c->dirty = 1;
                free(spans); free(timgs); free(img_fp); lm_ids_free(&prompt); pop_user(c);
                return -1;
            }
            const TurnImg *im = &timgs[sp->img_index];
            const int last = i + sp->n_ids == prompt.n;
            int frc = be->prefill_image(be->ctx, im->rows, im->n_rows, im->width, im->grid_h, im->grid_w,
                                        last ? c->logits : NULL);
            if (frc != LM_OK) {
                if (frc != LM_ERR_CTX) c->dirty = 1;
                else res->stop = LM_STOP_CTX;
                set_err(err, errlen, "backend image prefill failed with code %d at position %d", frc, i);
                free(spans); free(timgs); free(img_fp); lm_ids_free(&prompt); pop_user(c);
                return frc;
            }
            for (int k = 0; k < sp->n_ids; k++) {
                if (lm_ids_push(&c->fed, prompt.ids[i + k])) {
                    c->dirty = 1;
                    free(spans); free(timgs); free(img_fp); lm_ids_free(&prompt); pop_user(c);
                    set_err(err, errlen, "out of memory");
                    return -1;
                }
            }
            res->n_prefill += sp->n_ids;
            i += sp->n_ids;
            span_i++;
            continue;
        }
        int n = prompt.n - i < chunk ? prompt.n - i : chunk;
        if (span_i < n_spans && i + n > spans[span_i].id_start) n = spans[span_i].id_start - i;
        if (i < save_at && i + n > save_at) n = save_at - i;
        if (n < 1) n = 1;
        const int last = i + n == prompt.n;
        int frc = be->prefill ? be->prefill(be->ctx, prompt.ids + i, n, last ? c->logits : NULL)
                              : be->feed(be->ctx, prompt.ids[i], last ? c->logits : NULL);
        if (frc != LM_OK) {
            if (frc != LM_ERR_CTX) c->dirty = 1;
            else res->stop = LM_STOP_CTX;
            set_err(err, errlen, "backend feed failed with code %d while prefilling position %d of %d", frc, i, prompt.n);
            free(spans); free(timgs); free(img_fp); lm_ids_free(&prompt); pop_user(c);
            return frc;
        }
        for (int k = 0; k < n; k++) {
            if (lm_ids_push(&c->fed, prompt.ids[i + k])) {
                c->dirty = 1;
                free(spans); free(timgs); free(img_fp); lm_ids_free(&prompt); pop_user(c);
                set_err(err, errlen, "out of memory");
                return -1;
            }
        }
        res->n_prefill += n;
        i += n;
        if (save_at && c->fed.n == save_at) save_cache(c);
    }
    res->prefill_s = now_s() - t0;
    free(spans);
    free(timgs);
    free(c->fed_imgs);
    c->fed_imgs = img_fp;
    lm_ids_free(&prompt);

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
        lm_ids_free(&gen);
        free(col.reasoning.s);
        free(col.answer.s);
        pop_user(c);
        if (!internal && col.oom) set_err(err, errlen, "out of memory");
        return internal || col.oom ? -1 : result;
    }

    Hist a;
    memset(&a, 0, sizeof a);
    a.role = LM_ROLE_ASSISTANT;
    a.n_gen = gen.n;
    a.gen_thinking = c->cfg.thinking;
    a.content = col.answer.s ? col.answer.s : dup_str("");
    a.reasoning = col.reasoning.s ? col.reasoning.s : dup_str("");
    a.gen_ids = gen.ids ? gen.ids : (int *)malloc(sizeof(int));
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

int lm_chat_turn(LmChat *c, const char *user_text, const LmTurnOpts *opts, LmTurn *res,
                 char *err, size_t errlen)
{
    LmTurn local;
    if (!res) res = &local;
    memset(res, 0, sizeof *res);
    static const LmTurnOpts none;
    if (!opts) opts = &none;

    Hist u;
    memset(&u, 0, sizeof u);
    u.role = LM_ROLE_USER;
    u.content = dup_str(user_text);
    if (!u.content || hist_push(c, &u)) { free(u.content); set_err(err, errlen, "out of memory"); return -1; }
    return turn_core(c, opts, res, err, errlen);
}

int lm_chat_turn_parts(LmChat *c, const LmMsgPart *parts, int n_parts, const LmTurnOpts *opts,
                       LmTurn *res, char *err, size_t errlen)
{
    LmTurn local;
    if (!res) res = &local;
    memset(res, 0, sizeof *res);
    static const LmTurnOpts none;
    if (!opts) opts = &none;
    if (!parts || n_parts <= 0) { set_err(err, errlen, "user message has no parts"); return -1; }

    Hist u;
    memset(&u, 0, sizeof u);
    u.role = LM_ROLE_USER;
    u.content = parts_fingerprint(parts, n_parts);
    if (!u.content || hist_own_parts(&u, parts, n_parts) || hist_push(c, &u)) {
        hist_free_one(&u);
        set_err(err, errlen, "out of memory");
        return -1;
    }
    return turn_core(c, opts, res, err, errlen);
}
