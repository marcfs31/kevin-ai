/* lm_gguf.c - see lm_gguf.h. Original code. Little-endian hosts only (x86-64, arm64). */
#include "io/lm_portable_io.h"

#include "io/lm_gguf.h"

#include <dirent.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "io/lm_io.h"

int lm_gguf_type_block(int type, int *be, size_t *bb)
{
    int e; size_t b;
    switch (type) {
    case LM_GG_F32:  e = 1;   b = 4;   break;
    case LM_GG_F16:  e = 1;   b = 2;   break;
    case LM_GG_Q8_0: e = 32;  b = 34;  break;
    case LM_GG_Q4_K: e = 256; b = 144; break;
    case LM_GG_Q5_K: e = 256; b = 176; break;
    case LM_GG_Q6_K: e = 256; b = 210; break;
    default: return -1;
    }
    if (be) *be = e;
    if (bb) *bb = b;
    return 0;
}

const char *lm_gguf_type_name(int type)
{
    static char buf[24];
    switch (type) {
    case LM_GG_F32:  return "F32";
    case LM_GG_F16:  return "F16";
    case LM_GG_Q8_0: return "Q8_0";
    case LM_GG_Q4_K: return "Q4_K";
    case LM_GG_Q5_K: return "Q5_K";
    case LM_GG_Q6_K: return "Q6_K";
    default: snprintf(buf, sizeof buf, "type %d", type); return buf;
    }
}

typedef struct { const unsigned char *p, *end; } Cur;

static int take(Cur *c, void *dst, size_t n)
{
    if ((size_t)(c->end - c->p) < n) return -1;
    if (dst) memcpy(dst, c->p, n);
    c->p += n;
    return 0;
}
static int rd_u32(Cur *c, uint32_t *v) { return take(c, v, 4); }
static int rd_u64(Cur *c, uint64_t *v) { return take(c, v, 8); }

static int rd_str(Cur *c, LmGgufStr *s)
{
    uint64_t n;
    if (rd_u64(c, &n) != 0 || n > (uint64_t)(c->end - c->p)) return -1;
    s->s = (const char *)c->p;
    s->n = (size_t)n;
    c->p += n;
    return 0;
}

static int scalar_size(int t)
{
    switch (t) {
    case LM_GGUF_U8: case LM_GGUF_I8: case LM_GGUF_BOOL: return 1;
    case LM_GGUF_U16: case LM_GGUF_I16: return 2;
    case LM_GGUF_U32: case LM_GGUF_I32: case LM_GGUF_F32: return 4;
    case LM_GGUF_U64: case LM_GGUF_I64: case LM_GGUF_F64: return 8;
    default: return 0;
    }
}

/* One scalar of type t at c, widened into kv->v. */
static int rd_scalar(Cur *c, int t, LmGgufKv *kv)
{
    unsigned char b[8];
    const int n = scalar_size(t);
    if (!n || take(c, b, (size_t)n) != 0) return -1;
    switch (t) {
    case LM_GGUF_U8: case LM_GGUF_BOOL: kv->v.u = b[0]; break;
    case LM_GGUF_I8: kv->v.i = (int8_t)b[0]; break;
    case LM_GGUF_U16: { uint16_t x; memcpy(&x, b, 2); kv->v.u = x; break; }
    case LM_GGUF_I16: { int16_t x; memcpy(&x, b, 2); kv->v.i = x; break; }
    case LM_GGUF_U32: { uint32_t x; memcpy(&x, b, 4); kv->v.u = x; break; }
    case LM_GGUF_I32: { int32_t x; memcpy(&x, b, 4); kv->v.i = x; break; }
    case LM_GGUF_F32: { float x; memcpy(&x, b, 4); kv->v.f = x; break; }
    case LM_GGUF_U64: memcpy(&kv->v.u, b, 8); break;
    case LM_GGUF_I64: memcpy(&kv->v.i, b, 8); break;
    case LM_GGUF_F64: memcpy(&kv->v.f, b, 8); break;
    default: return -1;
    }
    return 0;
}

static int read_kv(Cur *c, LmGgufKv *kv, char *err, size_t errlen)
{
    uint32_t t;
    memset(kv, 0, sizeof *kv);
    if (rd_str(c, &kv->key) != 0 || rd_u32(c, &t) != 0)
        return lm_fail(err, errlen, "truncated metadata (a key runs past the end of the file)");
    kv->vtype = (int)t;
    const int kl = (int)(kv->key.n > 200 ? 200 : kv->key.n);
    if (t == LM_GGUF_STR) {
        if (rd_str(c, &kv->v.s) != 0)
            return lm_fail(err, errlen, "metadata %.*s: string runs past the end of the file", kl, kv->key.s);
        return 0;
    }
    if (t == LM_GGUF_ARR) {
        uint32_t et;
        uint64_t n;
        if (rd_u32(c, &et) != 0 || rd_u64(c, &n) != 0)
            return lm_fail(err, errlen, "metadata %.*s: truncated array header", kl, kv->key.s);
        if (et == LM_GGUF_ARR) return lm_fail(err, errlen, "metadata %.*s: nested arrays are not supported", kl, kv->key.s);
        if (n > LM_GGUF_MAX_ARRAY) return lm_fail(err, errlen, "metadata %.*s: %llu array elements exceed the limit", kl, kv->key.s, (unsigned long long)n);
        kv->etype = (int)et;
        kv->n = n;
        kv->v.arr = c->p;
        if (et == LM_GGUF_STR) {
            for (uint64_t i = 0; i < n; i++) {
                LmGgufStr s;
                if (rd_str(c, &s) != 0)
                    return lm_fail(err, errlen, "metadata %.*s: string %llu runs past the end of the file", kl, kv->key.s, (unsigned long long)i);
            }
            return 0;
        }
        const int es = scalar_size((int)et);
        if (!es) return lm_fail(err, errlen, "metadata %.*s: unknown array element type %u", kl, kv->key.s, et);
        if (n > (uint64_t)(c->end - c->p) / (uint64_t)es)
            return lm_fail(err, errlen, "metadata %.*s: array runs past the end of the file", kl, kv->key.s);
        c->p += n * (uint64_t)es;
        return 0;
    }
    if (rd_scalar(c, (int)t, kv) != 0) {
        if (!scalar_size((int)t)) return lm_fail(err, errlen, "metadata %.*s: unknown value type %u", kl, kv->key.s, t);
        return lm_fail(err, errlen, "metadata %.*s: value runs past the end of the file", kl, kv->key.s);
    }
    return 0;
}

static int cmp_name(const void *a, const void *b)
{
    return strcmp(*(const char *const *)a, *(const char *const *)b);
}

void lm_gguf_close(LmGguf *g)
{
    if (!g) return;
    if (g->map) munmap((void *)g->map, (size_t)g->size);
    if (g->fd >= 0) close(g->fd);
    for (int i = 0; i < g->nt; i++) free(g->t[i].name);
    free(g->t);
    free(g->kv);
    free(g->path);
    memset(g, 0, sizeof *g);
    g->fd = -1;
}

int lm_gguf_open(LmGguf *g, const char *path, char *err, size_t errlen)
{
    memset(g, 0, sizeof *g);
    g->fd = -1;
    g->path = strdup(path);
    if (!g->path) return lm_fail(err, errlen, "out of memory");
    g->fd = open(path, O_RDONLY);
    if (g->fd < 0) { lm_fail(err, errlen, "cannot open %s: %s", path, strerror(errno)); lm_gguf_close(g); return -1; }
    struct stat st;
    if (fstat(g->fd, &st) != 0) { lm_fail(err, errlen, "cannot stat %s: %s", path, strerror(errno)); lm_gguf_close(g); return -1; }
    g->size = (uint64_t)st.st_size;
    if (g->size < 24) { lm_fail(err, errlen, "%s is %llu bytes, too short for a GGUF header", path, (unsigned long long)g->size); lm_gguf_close(g); return -1; }
    void *m = mmap(NULL, (size_t)g->size, PROT_READ, MAP_PRIVATE, g->fd, 0);
    if (m == MAP_FAILED) { const int e = errno; lm_fail(err, errlen, "cannot mmap %s: %s", path, strerror(e)); lm_gguf_close(g); return -1; }
    g->map = (const unsigned char *)m;

    Cur c = { g->map, g->map + g->size };
    if (memcmp(c.p, "GGUF", 4) != 0) { lm_fail(err, errlen, "%s is not a GGUF file (bad magic)", path); lm_gguf_close(g); return -1; }
    c.p += 4;
    uint64_t nt = 0, nkv = 0;
    rd_u32(&c, &g->version);
    if (g->version != 2 && g->version != 3) {
        const unsigned v = g->version;
        lm_fail(err, errlen, "%s: GGUF version %u is not supported (2 and 3 are)", path, v); lm_gguf_close(g); return -1;
    }
    rd_u64(&c, &nt);
    rd_u64(&c, &nkv);
    if (nt > LM_GGUF_MAX_TENSORS || nkv > LM_GGUF_MAX_KV) {
        lm_fail(err, errlen, "%s: %llu tensors and %llu metadata entries exceed the limits (%d, %d)", path,
                       (unsigned long long)nt, (unsigned long long)nkv, LM_GGUF_MAX_TENSORS, LM_GGUF_MAX_KV); lm_gguf_close(g); return -1;
    }
    g->kv = (LmGgufKv *)calloc(nkv ? (size_t)nkv : 1, sizeof *g->kv);
    g->t = (LmGgufTensor *)calloc(nt ? (size_t)nt : 1, sizeof *g->t);
    if (!g->kv || !g->t) { lm_fail(err, errlen, "out of memory"); lm_gguf_close(g); return -1; }
    for (uint64_t i = 0; i < nkv; i++) {
        if (read_kv(&c, &g->kv[i], err, errlen) != 0) { lm_gguf_close(g); return -1; }
        g->nkv++;
    }
    g->alignment = 32;
    {
        const LmGgufKv *a = lm_gguf_kv(g, "general.alignment");
        if (a) {
            uint64_t v;
            if (lm_gguf_get_u64(g, "general.alignment", &v) != 0 || v == 0 || (v & (v - 1))) {
                lm_fail(err, errlen, "%s: general.alignment must be a power of two", path); lm_gguf_close(g); return -1;
            }
            g->alignment = v;
        }
    }
    for (uint64_t i = 0; i < nt; i++) {
        LmGgufTensor *t = &g->t[i];
        LmGgufStr name;
        uint32_t nd, ty;
        if (rd_str(&c, &name) != 0 || rd_u32(&c, &nd) != 0) { lm_fail(err, errlen, "%s: truncated tensor table", path); lm_gguf_close(g); return -1; }
        t->name = (char *)malloc(name.n + 1);
        if (!t->name) { lm_fail(err, errlen, "out of memory"); lm_gguf_close(g); return -1; }
        memcpy(t->name, name.s, name.n);
        t->name[name.n] = 0;
        g->nt++;
        if (nd < 1 || nd > 4) { lm_fail(err, errlen, "%s: tensor %.100s has %u dims (1 to 4 are supported)", path, t->name, nd); lm_gguf_close(g); return -1; }
        t->ndim = (int)nd;
        t->numel = 1;
        for (uint32_t d = 0; d < nd; d++) {
            uint64_t e;
            if (rd_u64(&c, &e) != 0) { lm_fail(err, errlen, "%s: truncated tensor table", path); lm_gguf_close(g); return -1; }
            if (e == 0 || e > (uint64_t)1 << 40 || (uint64_t)t->numel > ((uint64_t)1 << 62) / e) {
                lm_fail(err, errlen, "%s: tensor %.100s has an empty or overflowing dimension", path, t->name); lm_gguf_close(g); return -1;
            }
            t->ne[d] = (int64_t)e;
            t->numel *= (int64_t)e;
        }
        for (uint32_t d = nd; d < 4; d++) t->ne[d] = 1;
        uint64_t off;
        if (rd_u32(&c, &ty) != 0 || rd_u64(&c, &off) != 0) { lm_fail(err, errlen, "%s: truncated tensor table", path); lm_gguf_close(g); return -1; }
        t->type = (int)ty;
        t->off = off;   /* relative for now */
    }
    g->data_start = ((uint64_t)(c.p - g->map) + g->alignment - 1) / g->alignment * g->alignment;
    if (g->data_start > g->size) { lm_fail(err, errlen, "%s: the tensor data starts past the end of the file (truncated)", path); lm_gguf_close(g); return -1; }

    char **names = (char **)malloc((nt ? (size_t)nt : 1) * sizeof *names);
    if (!names) { lm_fail(err, errlen, "out of memory"); lm_gguf_close(g); return -1; }
    for (int i = 0; i < g->nt; i++) names[i] = g->t[i].name;
    qsort(names, (size_t)g->nt, sizeof *names, cmp_name);
    for (int i = 1; i < g->nt; i++)
        if (!strcmp(names[i - 1], names[i])) {
            char nm[128];
            snprintf(nm, sizeof nm, "%.100s", names[i]);
            free(names);
            lm_fail(err, errlen, "%s: duplicate tensor %s", path, nm); lm_gguf_close(g); return -1;
        }
    free(names);

    for (int i = 0; i < g->nt; i++) {
        LmGgufTensor *t = &g->t[i];
        int be;
        size_t bb;
        const uint64_t rel = t->off;
        if (rel > g->size - g->data_start) { lm_fail(err, errlen, "%s: tensor %.100s starts past the end of the file (truncated)", path, t->name); lm_gguf_close(g); return -1; }
        t->off = g->data_start + rel;
        if (lm_gguf_type_block(t->type, &be, &bb) != 0) { t->nbytes = 0; continue; }
        if (t->ne[0] % be != 0) {
            lm_fail(err, errlen, "%s: tensor %.100s (%s) has a row of %lld elements, not a multiple of the %d-element block", path,
                           g->t[i].name, lm_gguf_type_name(g->t[i].type), (long long)g->t[i].ne[0], be); lm_gguf_close(g); return -1;
        }
        const uint64_t blocks = (uint64_t)t->numel / (uint64_t)be;
        if (blocks > ((uint64_t)1 << 62) / bb) { lm_fail(err, errlen, "%s: tensor %.100s is implausibly large", path, t->name); lm_gguf_close(g); return -1; }
        t->nbytes = blocks * bb;
        if (t->nbytes > g->size - t->off) {
            lm_fail(err, errlen, "%s: tensor %.100s runs past the end of the file (truncated)", path, g->t[i].name); lm_gguf_close(g); return -1;
        }
    }
    return 0;
}

const LmGgufKv *lm_gguf_kv(const LmGguf *g, const char *key)
{
    const size_t n = strlen(key);
    for (int i = 0; i < g->nkv; i++)
        if (g->kv[i].key.n == n && !memcmp(g->kv[i].key.s, key, n)) return &g->kv[i];
    return NULL;
}

int lm_gguf_get_u64(const LmGguf *g, const char *key, uint64_t *out)
{
    const LmGgufKv *kv = lm_gguf_kv(g, key);
    if (!kv) return -1;
    switch (kv->vtype) {
    case LM_GGUF_U8: case LM_GGUF_U16: case LM_GGUF_U32: case LM_GGUF_U64: case LM_GGUF_BOOL: *out = kv->v.u; return 0;
    case LM_GGUF_I8: case LM_GGUF_I16: case LM_GGUF_I32: case LM_GGUF_I64:
        if (kv->v.i < 0) return -1;
        *out = (uint64_t)kv->v.i;
        return 0;
    default: return -1;
    }
}

int lm_gguf_get_f64(const LmGguf *g, const char *key, double *out)
{
    const LmGgufKv *kv = lm_gguf_kv(g, key);
    if (!kv) return -1;
    switch (kv->vtype) {
    case LM_GGUF_F32: case LM_GGUF_F64: *out = kv->v.f; return 0;
    case LM_GGUF_U8: case LM_GGUF_U16: case LM_GGUF_U32: case LM_GGUF_U64: *out = (double)kv->v.u; return 0;
    case LM_GGUF_I8: case LM_GGUF_I16: case LM_GGUF_I32: case LM_GGUF_I64: *out = (double)kv->v.i; return 0;
    default: return -1;
    }
}

int lm_gguf_get_str(const LmGguf *g, const char *key, LmGgufStr *out)
{
    const LmGgufKv *kv = lm_gguf_kv(g, key);
    if (!kv || kv->vtype != LM_GGUF_STR) return -1;
    *out = kv->v.s;
    return 0;
}

LmGgufStr *lm_gguf_str_array(const LmGguf *g, const char *key, int *n, char *err, size_t errlen)
{
    const LmGgufKv *kv = lm_gguf_kv(g, key);
    if (!kv) { lm_fail(err, errlen, "%s: metadata %s is missing", g->path, key); return NULL; }
    if (kv->vtype != LM_GGUF_ARR || kv->etype != LM_GGUF_STR) { lm_fail(err, errlen, "%s: metadata %s is not a string array", g->path, key); return NULL; }
    LmGgufStr *out = (LmGgufStr *)malloc((kv->n ? (size_t)kv->n : 1) * sizeof *out);
    if (!out) { lm_fail(err, errlen, "out of memory"); return NULL; }
    Cur c = { kv->v.arr, g->map + g->size };
    for (uint64_t i = 0; i < kv->n; i++)
        if (rd_str(&c, &out[i]) != 0) { free(out); lm_fail(err, errlen, "%s: metadata %s is truncated", g->path, key); return NULL; }
    *n = (int)kv->n;
    return out;
}

int64_t *lm_gguf_int_array(const LmGguf *g, const char *key, int *n, char *err, size_t errlen)
{
    const LmGgufKv *kv = lm_gguf_kv(g, key);
    if (!kv) { lm_fail(err, errlen, "%s: metadata %s is missing", g->path, key); return NULL; }
    const int es = kv->vtype == LM_GGUF_ARR ? scalar_size(kv->etype) : 0;
    if (!es || kv->etype == LM_GGUF_F32 || kv->etype == LM_GGUF_F64) { lm_fail(err, errlen, "%s: metadata %s is not an integer array", g->path, key); return NULL; }
    int64_t *out = (int64_t *)malloc((kv->n ? (size_t)kv->n : 1) * sizeof *out);
    if (!out) { lm_fail(err, errlen, "out of memory"); return NULL; }
    for (uint64_t i = 0; i < kv->n; i++) {
        const unsigned char *p = kv->v.arr + i * (uint64_t)es;
        int64_t v = 0;
        switch (kv->etype) {
        case LM_GGUF_U8: case LM_GGUF_BOOL: v = p[0]; break;
        case LM_GGUF_I8: v = (int8_t)p[0]; break;
        case LM_GGUF_U16: { uint16_t x; memcpy(&x, p, 2); v = x; break; }
        case LM_GGUF_I16: { int16_t x; memcpy(&x, p, 2); v = x; break; }
        case LM_GGUF_U32: { uint32_t x; memcpy(&x, p, 4); v = x; break; }
        case LM_GGUF_I32: { int32_t x; memcpy(&x, p, 4); v = x; break; }
        default: memcpy(&v, p, 8); break;
        }
        out[i] = v;
    }
    *n = (int)kv->n;
    return out;
}

const LmGgufTensor *lm_gguf_find(const LmGguf *g, const char *name)
{
    for (int i = 0; i < g->nt; i++)
        if (!strcmp(g->t[i].name, name)) return &g->t[i];
    return NULL;
}

const void *lm_gguf_data(const LmGguf *g, const LmGgufTensor *t) { return g->map + t->off; }

static int ends_with(const char *s, const char *suf)
{
    const size_t n = strlen(s), m = strlen(suf);
    return n >= m && !strcmp(s + n - m, suf);
}

int lm_gguf_locate(const char *path, char *out, size_t outlen)
{
    struct stat st;
    if (stat(path, &st) != 0) return 0;
    if (S_ISREG(st.st_mode)) {
        if (!ends_with(path, ".gguf") || strlen(path) >= outlen) return 0;
        strcpy(out, path);
        return 1;
    }
    if (!S_ISDIR(st.st_mode)) return 0;
    char cfg[4096];
    snprintf(cfg, sizeof cfg, "%s/config.json", path);
    if (access(cfg, F_OK) == 0) return 0;
    DIR *d = opendir(path);
    if (!d) return 0;
    int found = 0;
    char hit[4096] = "";
    struct dirent *e;
    while ((e = readdir(d)) != NULL) {
        if (!ends_with(e->d_name, ".gguf")) continue;
        found++;
        snprintf(hit, sizeof hit, "%s/%s", path, e->d_name);
    }
    closedir(d);
    if (found != 1 || strlen(hit) >= outlen) return 0;
    strcpy(out, hit);
    return 1;
}
