/* lm_stw.c - see lm_stw.h. Original code. */
#include "io/lm_portable_io.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "io/lm_io.h"
#include "io/lm_stw.h"

typedef struct {
    char    *name;
    LmDtype  dtype;
    int      ndim;
    int64_t  shape[4];
    int64_t  nbytes;
    int64_t  start;     /* relative to the end of the header */
} WTensor;

struct LmStWriter {
    char    *path, *tmp;
    FILE    *f;
    WTensor *t;
    int      nt, cap;
    char   **mk, **mv;
    int      nm;
    int      began;
    int      cur;       /* index of the tensor being written, or -1 */
    int      next;      /* next tensor index lm_stw_tensor_begin accepts */
    int64_t  written;   /* bytes of the current tensor so far */
    int      failed;    /* a write failed: finish refuses */
};

LmStWriter *lm_stw_create(const char *path, char *err, size_t errlen)
{
    LmStWriter *w = (LmStWriter *)calloc(1, sizeof *w);
    if (!w) { lm_fail(err, errlen, "out of memory"); return NULL; }
    const size_t n = strlen(path);
    w->path = strdup(path);
    w->tmp = (char *)malloc(n + 5);
    if (!w->path || !w->tmp) { lm_fail(err, errlen, "out of memory"); lm_stw_abort(w); return NULL; }
    memcpy(w->tmp, path, n);
    memcpy(w->tmp + n, ".tmp", 5);
    w->f = fopen(w->tmp, "wb");
    if (!w->f) {
        lm_fail(err, errlen, "cannot create %s: %s", w->tmp, strerror(errno));
        free(w->path); free(w->tmp); free(w);
        return NULL;
    }
    w->cur = -1;
    return w;
}

void lm_stw_abort(LmStWriter *w)
{
    if (!w) return;
    if (w->f) fclose(w->f);
    if (w->tmp && w->f) unlink(w->tmp);
    for (int i = 0; i < w->nt; i++) free(w->t[i].name);
    free(w->t);
    for (int i = 0; i < w->nm; i++) { free(w->mk[i]); free(w->mv[i]); }
    free(w->mk); free(w->mv);
    free(w->path); free(w->tmp);
    free(w);
}

static int printable(const char *s)
{
    if (!*s) return 0;
    for (; *s; s++) if ((unsigned char)*s < 0x20 || (unsigned char)*s > 0x7e) return 0;
    return 1;
}

int lm_stw_meta(LmStWriter *w, const char *key, const char *value, char *err, size_t errlen)
{
    if (w->began) return lm_fail(err, errlen, "metadata after lm_stw_begin");
    if (!printable(key) || !(printable(value) || !*value))
        return lm_fail(err, errlen, "metadata %s: key and value must be printable ASCII", key);
    for (int i = 0; i < w->nm; i++)
        if (!strcmp(w->mk[i], key)) return lm_fail(err, errlen, "duplicate metadata key %s", key);
    char **nk = (char **)realloc(w->mk, (size_t)(w->nm + 1) * sizeof *nk);
    if (!nk) return lm_fail(err, errlen, "out of memory");
    w->mk = nk;
    char **nv = (char **)realloc(w->mv, (size_t)(w->nm + 1) * sizeof *nv);
    if (!nv) return lm_fail(err, errlen, "out of memory");
    w->mv = nv;
    w->mk[w->nm] = strdup(key);
    w->mv[w->nm] = strdup(value);
    if (!w->mk[w->nm] || !w->mv[w->nm]) return lm_fail(err, errlen, "out of memory");
    w->nm++;
    return 0;
}

int lm_stw_declare(LmStWriter *w, const char *name, LmDtype dtype, int ndim, const int64_t *shape,
                   char *err, size_t errlen)
{
    if (w->began) return lm_fail(err, errlen, "tensor %s declared after lm_stw_begin", name);
    if (!printable(name)) return lm_fail(err, errlen, "tensor name %s must be non-empty printable ASCII", name);
    if (lm_dtype_size(dtype) == 0) return lm_fail(err, errlen, "tensor %s has no dtype", name);
    if (ndim < 0 || ndim > 4) return lm_fail(err, errlen, "tensor %s has rank %d (0..4 supported)", name, ndim);
    for (int i = 0; i < w->nt; i++)
        if (!strcmp(w->t[i].name, name)) return lm_fail(err, errlen, "tensor %s declared twice", name);
    WTensor t;
    memset(&t, 0, sizeof t);
    t.dtype = dtype;
    t.ndim = ndim;
    for (int i = 0; i < ndim; i++) {
        if (shape[i] < 0) return lm_fail(err, errlen, "tensor %s has a negative dimension", name);
        t.shape[i] = shape[i];
    }
    LmTensor tmp;
    memset(&tmp, 0, sizeof tmp);
    tmp.ndim = ndim;
    memcpy(tmp.shape, t.shape, sizeof t.shape);
    const int64_t numel = lm_st_numel(&tmp);
    if (numel > INT64_MAX / lm_dtype_size(dtype))
        return lm_fail(err, errlen, "tensor %s is impossibly large", name);
    t.nbytes = numel * lm_dtype_size(dtype);
    if (w->nt == w->cap) {
        const int nc = w->cap ? w->cap * 2 : 256;
        WTensor *nt = (WTensor *)realloc(w->t, (size_t)nc * sizeof *nt);
        if (!nt) return lm_fail(err, errlen, "out of memory");
        w->t = nt;
        w->cap = nc;
    }
    t.name = strdup(name);
    if (!t.name) return lm_fail(err, errlen, "out of memory");
    w->t[w->nt++] = t;
    return 0;
}

static int cmp_t(const void *a, const void *b) { return strcmp(((const WTensor *)a)->name, ((const WTensor *)b)->name); }
typedef struct { const char *k, *v; } Kv;
static int cmp_kv(const void *a, const void *b) { return strcmp(((const Kv *)a)->k, ((const Kv *)b)->k); }

typedef struct { char *p; size_t n, cap; int oom; } Buf;

static void bput(Buf *b, const char *s, size_t n)
{
    if (b->oom) return;
    if (b->n + n + 1 > b->cap) {
        size_t nc = b->cap ? b->cap : 4096;
        while (b->n + n + 1 > nc) nc *= 2;
        char *np = (char *)realloc(b->p, nc);
        if (!np) { b->oom = 1; return; }
        b->p = np;
        b->cap = nc;
    }
    memcpy(b->p + b->n, s, n);
    b->n += n;
    b->p[b->n] = '\0';
}

static void bstr(Buf *b, const char *s) { bput(b, s, strlen(s)); }

static void bjson(Buf *b, const char *s)
{
    bput(b, "\"", 1);
    for (; *s; s++) {
        if (*s == '"' || *s == '\\') bput(b, "\\", 1);
        bput(b, s, 1);
    }
    bput(b, "\"", 1);
}

int lm_stw_begin(LmStWriter *w, char *err, size_t errlen)
{
    if (w->began) return lm_fail(err, errlen, "lm_stw_begin called twice");
    qsort(w->t, (size_t)w->nt, sizeof *w->t, cmp_t);
    int64_t off = 0;
    for (int i = 0; i < w->nt; i++) {
        if (off > INT64_MAX - w->t[i].nbytes) return lm_fail(err, errlen, "output file would exceed 2^63 bytes");
        w->t[i].start = off;
        off += w->t[i].nbytes;
    }

    Buf b = { 0 };
    char num[32];
    bput(&b, "{", 1);
    if (w->nm > 0) {
        Kv *kv = (Kv *)malloc((size_t)w->nm * sizeof *kv);
        if (!kv) return lm_fail(err, errlen, "out of memory");
        for (int i = 0; i < w->nm; i++) { kv[i].k = w->mk[i]; kv[i].v = w->mv[i]; }
        qsort(kv, (size_t)w->nm, sizeof *kv, cmp_kv);
        bstr(&b, "\"__metadata__\":{");
        for (int i = 0; i < w->nm; i++) {
            if (i) bput(&b, ",", 1);
            bjson(&b, kv[i].k); bput(&b, ":", 1); bjson(&b, kv[i].v);
        }
        bput(&b, "}", 1);
        free(kv);
    }
    for (int i = 0; i < w->nt; i++) {
        const WTensor *t = &w->t[i];
        if (i || w->nm > 0) bput(&b, ",", 1);
        bjson(&b, t->name);
        bstr(&b, ":{\"dtype\":\"");
        bstr(&b, lm_dtype_name(t->dtype));
        bstr(&b, "\",\"shape\":[");
        for (int d = 0; d < t->ndim; d++) {
            if (d) bput(&b, ",", 1);
            snprintf(num, sizeof num, "%lld", (long long)t->shape[d]);
            bstr(&b, num);
        }
        bstr(&b, "],\"data_offsets\":[");
        snprintf(num, sizeof num, "%lld", (long long)t->start);
        bstr(&b, num);
        bput(&b, ",", 1);
        snprintf(num, sizeof num, "%lld", (long long)(t->start + t->nbytes));
        bstr(&b, num);
        bstr(&b, "]}");
    }
    bput(&b, "}", 1);
    while (!b.oom && b.n % 8 != 0) bput(&b, " ", 1);
    if (b.oom) { free(b.p); return lm_fail(err, errlen, "out of memory building the header"); }
    if ((int64_t)b.n > LM_ST_MAX_HEADER) { free(b.p); return lm_fail(err, errlen, "header exceeds the 100 MB limit"); }

    unsigned char len[8];
    uint64_t n = b.n;
    for (int i = 0; i < 8; i++) { len[i] = (unsigned char)(n & 0xff); n >>= 8; }
    const int ok = fwrite(len, 1, 8, w->f) == 8 && fwrite(b.p, 1, b.n, w->f) == b.n;
    free(b.p);
    if (!ok) { w->failed = 1; return lm_fail(err, errlen, "write of %s failed: %s", w->tmp, strerror(errno)); }
    w->began = 1;
    return 0;
}

const char *lm_stw_next_name(const LmStWriter *w)
{
    return (w->began && w->next < w->nt) ? w->t[w->next].name : NULL;
}

int lm_stw_tensor_begin(LmStWriter *w, const char *name, char *err, size_t errlen)
{
    if (!w->began) return lm_fail(err, errlen, "lm_stw_tensor_begin before lm_stw_begin");
    if (w->cur >= 0) return lm_fail(err, errlen, "tensor %s begun while %s is unfinished", name, w->t[w->cur].name);
    if (w->next >= w->nt) return lm_fail(err, errlen, "tensor %s written but all %d declared tensors are done", name, w->nt);
    if (strcmp(w->t[w->next].name, name))
        return lm_fail(err, errlen, "tensor %s written out of order: layout order expects %s", name, w->t[w->next].name);
    w->cur = w->next++;
    w->written = 0;
    return 0;
}

int lm_stw_write(LmStWriter *w, const void *buf, int64_t nbytes, char *err, size_t errlen)
{
    if (w->cur < 0) return lm_fail(err, errlen, "write outside a tensor");
    const WTensor *t = &w->t[w->cur];
    if (nbytes < 0 || nbytes > t->nbytes - w->written)
        return lm_fail(err, errlen, "tensor %s: %lld more bytes would exceed its declared %lld",
                       t->name, (long long)nbytes, (long long)t->nbytes);
    if (nbytes && fwrite(buf, 1, (size_t)nbytes, w->f) != (size_t)nbytes) {
        w->failed = 1;
        return lm_fail(err, errlen, "write of %s failed: %s", w->tmp, strerror(errno));
    }
    w->written += nbytes;
    return 0;
}

int lm_stw_tensor_end(LmStWriter *w, char *err, size_t errlen)
{
    if (w->cur < 0) return lm_fail(err, errlen, "lm_stw_tensor_end outside a tensor");
    const WTensor *t = &w->t[w->cur];
    if (w->written != t->nbytes)
        return lm_fail(err, errlen, "tensor %s is short: %lld of %lld bytes written",
                       t->name, (long long)w->written, (long long)t->nbytes);
    w->cur = -1;
    return 0;
}

int lm_stw_finish(LmStWriter *w, char *err, size_t errlen)
{
    int rc = -1;
    if (!w->began) { lm_fail(err, errlen, "lm_stw_finish before lm_stw_begin"); goto out; }
    if (w->cur >= 0) { lm_fail(err, errlen, "tensor %s is unfinished", w->t[w->cur].name); goto out; }
    if (w->next < w->nt) {
        lm_fail(err, errlen, "%d of %d tensors never written (next: %s)", w->nt - w->next, w->nt, w->t[w->next].name);
        goto out;
    }
    if (w->failed) { lm_fail(err, errlen, "an earlier write failed"); goto out; }
    if (fflush(w->f) != 0 || fsync(fileno(w->f)) != 0) {
        lm_fail(err, errlen, "flush of %s failed: %s", w->tmp, strerror(errno));
        goto out;
    }
    const int cr = fclose(w->f);
    w->f = NULL;
    if (cr != 0) { lm_fail(err, errlen, "close of %s failed: %s", w->tmp, strerror(errno)); unlink(w->tmp); goto out; }
    if (rename(w->tmp, w->path) != 0) {
        lm_fail(err, errlen, "cannot rename %s to %s: %s", w->tmp, w->path, strerror(errno));
        unlink(w->tmp);
        goto out;
    }
    rc = 0;
out:
    if (rc == 0) {
        /* already closed and renamed: free without unlinking the finished file */
        for (int i = 0; i < w->nt; i++) free(w->t[i].name);
        free(w->t);
        for (int i = 0; i < w->nm; i++) { free(w->mk[i]); free(w->mv[i]); }
        free(w->mk); free(w->mv);
        free(w->path); free(w->tmp);
        free(w);
    } else {
        lm_stw_abort(w);
    }
    return rc;
}
