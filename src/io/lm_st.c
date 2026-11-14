/* lm_st.c - safetensors reader. See lm_st.h for the change list against kimi-k3-in-c
 * src/io/k3_st.c (Apache-2.0), from which the header scanner (str_/i64_/skip_value), the
 * saturating integer parse, the span sweep and the FNV-1a hash index are ported.
 *
 * WHY A HAND-WRITTEN SCANNER FOR THE TENSOR HEADER: it is machine generated with a rigid
 * shape (one flat object of small objects with three known keys), so walking it once
 * straight into the index is simpler than building a DOM. It is not trusted blindly:
 * tests/unit/test_st.c feeds it every malformed shape listed in lm_st.h. The shard index
 * (model.safetensors.index.json) is the opposite case, an arbitrary-shaped document read
 * once, so it goes through third_party/json.h. */
#include "io/lm_portable_io.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "io/lm_io.h"
#include "io/lm_st.h"

#include "json.h"

int lm_dtype_size(LmDtype d)
{
    switch (d) {
    case LM_DT_U8:
    case LM_DT_I8:   return 1;
    case LM_DT_BF16:
    case LM_DT_F16:  return 2;
    case LM_DT_F32:  return 4;
    default:         return 0;
    }
}

const char *lm_dtype_name(LmDtype d)
{
    switch (d) {
    case LM_DT_U8:   return "U8";
    case LM_DT_I8:   return "I8";
    case LM_DT_BF16: return "BF16";
    case LM_DT_F16:  return "F16";
    case LM_DT_F32:  return "F32";
    default:         return "UNKNOWN";
    }
}

int64_t lm_st_numel(const LmTensor *t)
{
    return t->numel;
}

int64_t lm_shape_numel(int ndim, const int64_t *shape)
{
    int64_t n = 1;
    for (int i = 0; i < ndim; i++) {
        if (shape[i] < 0) return INT64_MAX;
        if (shape[i] == 0) return 0;
        if (n > INT64_MAX / shape[i]) return INT64_MAX;
        n *= shape[i];
    }
    return n;
}

static LmDtype dtype_of(const char *s)
{
    if (!strcmp(s, "U8"))   return LM_DT_U8;
    if (!strcmp(s, "I8"))   return LM_DT_I8;
    if (!strcmp(s, "BF16")) return LM_DT_BF16;
    if (!strcmp(s, "F16"))  return LM_DT_F16;
    if (!strcmp(s, "F32"))  return LM_DT_F32;
    return LM_DT_UNKNOWN;
}

static uint64_t fnv1a(const char *s)
{
    uint64_t h = 1469598103934665603ull;
    while (*s) { h ^= (unsigned char)*s++; h *= 1099511628211ull; }
    return h;
}

/* ------------------------------------------------------------------ scanner */

typedef struct { const char *p, *end; } Scan;

static void ws(Scan *s) { while (s->p < s->end && (unsigned char)*s->p <= ' ') s->p++; }

static int lit(Scan *s, char c)
{
    ws(s);
    if (s->p < s->end && *s->p == c) { s->p++; return 1; }
    return 0;
}

/* A JSON string into out (NUL terminated, cap bytes). Escapes are decoded so a mangled
 * name cannot turn a lookup into a spurious "missing weight". 0 on malformed or too long. */
static int str_(Scan *s, char *out, size_t cap)
{
    ws(s);
    if (s->p >= s->end || *s->p != '"') return 0;
    s->p++;
    size_t n = 0;
    while (s->p < s->end && *s->p != '"') {
        char c = *s->p++;
        if (c == '\\') {
            if (s->p >= s->end) return 0;
            const char e = *s->p++;
            switch (e) {
            case 'n': c = '\n'; break;  case 't': c = '\t'; break;
            case 'r': c = '\r'; break;  case 'b': c = '\b'; break;
            case 'f': c = '\f'; break;
            case 'u': {
                if (s->end - s->p < 4) return 0;
                unsigned v = 0;
                for (int i = 0; i < 4; i++) {
                    const char h = *s->p++;
                    if (h >= '0' && h <= '9') v = v * 16u + (unsigned)(h - '0');
                    else if ((h | 32) >= 'a' && (h | 32) <= 'f') v = v * 16u + (unsigned)((h | 32) - 'a' + 10);
                    else return 0;
                }
                c = (char)(v < 128 ? v : '?');
                break;
            }
            default: c = e;             /* \" \\ \/ */
            }
        }
        if (n + 1 >= cap) return 0;     /* too long: refuse, never truncate a name */
        out[n++] = c;
    }
    if (s->p >= s->end) return 0;
    s->p++;                              /* closing quote */
    out[n] = '\0';
    return 1;
}

/* Integer, saturating at the int64 limits so hostile offsets cannot wrap small and defeat
 * the range checks. */
static int i64_(Scan *s, int64_t *v)
{
    ws(s);
    int neg = 0;
    if (s->p < s->end && (*s->p == '-' || *s->p == '+')) neg = (*s->p++ == '-');
    if (s->p >= s->end || *s->p < '0' || *s->p > '9') return 0;
    int64_t a = 0;
    while (s->p < s->end && *s->p >= '0' && *s->p <= '9') {
        const int d = *s->p++ - '0';
        if (a > (INT64_MAX - d) / 10) a = INT64_MAX;
        else a = a * 10 + d;
    }
    if (neg) *v = (a == INT64_MAX) ? INT64_MIN : -a;
    else     *v = a;
    return 1;
}

/* Skip any value (unknown keys inside a tensor entry). */
static int skip_value(Scan *s)
{
    ws(s);
    if (s->p >= s->end) return 0;
    if (*s->p == '"') {
        s->p++;
        while (s->p < s->end && *s->p != '"') { if (*s->p == '\\') s->p++; s->p++; }
        if (s->p >= s->end) return 0;
        s->p++;
        return 1;
    }
    if (*s->p == '{' || *s->p == '[') {
        int depth = 0;
        do {
            if (s->p >= s->end) return 0;
            const char c = *s->p++;
            if (c == '"') {
                while (s->p < s->end && *s->p != '"') { if (*s->p == '\\') s->p++; s->p++; }
                if (s->p >= s->end) return 0;
                s->p++;
            } else if (c == '{' || c == '[') depth++;
            else if (c == '}' || c == ']') depth--;
        } while (depth > 0);
        return 1;
    }
    while (s->p < s->end && *s->p != ',' && *s->p != '}' && *s->p != ']') s->p++;
    return 1;
}

/* ------------------------------------------------------------------ span sweep */

typedef struct { int64_t o0, o1; int idx; } Span;

static int cmp_span(const void *a, const void *b)
{
    const Span *x = (const Span *)a, *y = (const Span *)b;
    if (x->o0 != y->o0) return x->o0 < y->o0 ? -1 : 1;
    if (x->o1 != y->o1) return x->o1 < y->o1 ? -1 : 1;
    return 0;
}

/* Tensors [first, s->nt) belong to one file. Sorted by start, a span that starts below the
 * furthest end seen so far overlaps an earlier one. Zero-length spans claim no bytes. */
static int sweep_spans(const LmSt *s, int first, char *err, size_t errlen)
{
    const int cnt = s->nt - first;
    Span *sp = (Span *)malloc((size_t)(cnt ? cnt : 1) * sizeof *sp);
    if (!sp) return lm_fail(err, errlen, "out of memory checking tensor spans");
    int n = 0;
    for (int i = first; i < s->nt; i++) {
        if (s->t[i].nbytes == 0) continue;
        sp[n].o0 = s->t[i].off;
        sp[n].o1 = s->t[i].off + s->t[i].nbytes;
        sp[n].idx = i;
        n++;
    }
    qsort(sp, (size_t)n, sizeof *sp, cmp_span);
    int64_t cursor = 0;
    int prev = -1;
    for (int i = 0; i < n; i++) {
        if (prev >= 0 && sp[i].o0 < cursor) {
            const LmTensor *a = &s->t[sp[i].idx], *b = &s->t[sp[prev].idx];
            lm_fail(err, errlen, "tensors overlap: %s [%lld,%lld) and %s [%lld,%lld)",
                    a->name, (long long)sp[i].o0, (long long)sp[i].o1,
                    b->name, (long long)sp[prev].o0, (long long)sp[prev].o1);
            free(sp);
            return -1;
        }
        if (sp[i].o1 > cursor) { cursor = sp[i].o1; prev = i; }
    }
    free(sp);
    return 0;
}

/* ------------------------------------------------------------------ close */

static void file_close(LmStFile *f)
{
    if (f->map) munmap((void *)f->map, (size_t)f->size);
    if (f->fd >= 0) close(f->fd);
    for (int i = 0; i < f->nmeta; i++) { free(f->meta_key[i]); free(f->meta_val[i]); }
    free(f->meta_key);
    free(f->meta_val);
    free(f->path);
    memset(f, 0, sizeof *f);
    f->fd = -1;
}

void lm_st_close(LmSt *s)
{
    if (s->files) for (int i = 0; i < s->nfiles; i++) file_close(&s->files[i]);
    free(s->files);
    if (s->t) for (int i = 0; i < s->nt; i++) free(s->t[i].name);
    free(s->t);
    free(s->bucket);
    memset(s, 0, sizeof *s);
}

/* ------------------------------------------------------------------ one file */

typedef struct { int cap; } Grow;

static int push_tensor(LmSt *s, Grow *g, const LmTensor *t)
{
    if (s->nt == g->cap) {
        const int nc = g->cap ? g->cap * 2 : 512;
        LmTensor *nt = (LmTensor *)realloc(s->t, (size_t)nc * sizeof *nt);
        if (!nt) return -1;
        s->t = nt;
        g->cap = nc;
    }
    s->t[s->nt++] = *t;
    return 0;
}

static int parse_entry(Scan *sc, LmTensor *t, int64_t *o0, int64_t *o1, const char *name,
                       char *err, size_t errlen, const char *hdr)
{
#define PFAIL(msg) return lm_fail(err, errlen, "malformed header: %s (tensor %s, near byte %ld)", \
                                  (msg), name, (long)(sc->p - hdr))
    int have_dt = 0, have_shape = 0, have_off = 0, first = 1;
    for (;;) {
        ws(sc);
        if (sc->p < sc->end && *sc->p == '}') { sc->p++; break; }
        if (!first && !lit(sc, ',')) PFAIL("expected ',' between tensor fields");
        first = 0;
        char key[64];
        if (!str_(sc, key, sizeof key)) PFAIL("bad tensor field name");
        if (!lit(sc, ':')) PFAIL("expected ':' after tensor field name");

        if (!strcmp(key, "dtype")) {
            char dv[32];
            if (have_dt) PFAIL("duplicate dtype");
            if (!str_(sc, dv, sizeof dv)) PFAIL("dtype is not a string");
            t->dtype = dtype_of(dv);
            if (t->dtype == LM_DT_UNKNOWN)
                return lm_fail(err, errlen,
                               "unsupported dtype '%s' on tensor %s (BF16, F16, F32, I8 and U8 only)",
                               dv, name);
            have_dt = 1;
        } else if (!strcmp(key, "shape")) {
            if (have_shape) PFAIL("duplicate shape");
            if (!lit(sc, '[')) PFAIL("shape is not an array");
            ws(sc);
            /* Any rank is accepted here: a tensor the engine never uses (a vision tower, say)
             * must not stop the file from opening. Users of a tensor check rank_overflow. */
            int rank = 0;
            int64_t nall = 1;
            if (sc->p < sc->end && *sc->p == ']') sc->p++;      /* scalar */
            else for (;;) {
                int64_t d;
                if (!i64_(sc, &d)) PFAIL("shape entry is not an integer");
                if (d < 0) return lm_fail(err, errlen, "tensor %s has a negative dimension %lld",
                                          name, (long long)d);
                if (rank < 4) t->shape[rank] = d;
                rank++;
                if (d == 0) nall = 0;                           /* zero wins over saturation */
                else if (nall != 0 && nall > INT64_MAX / d) nall = INT64_MAX;
                else if (nall != 0) nall *= d;
                if (lit(sc, ',')) continue;
                if (lit(sc, ']')) break;
                PFAIL("expected ',' or ']' in shape");
            }
            t->ndim = rank < 4 ? rank : 4;
            t->rank_overflow = rank > 4;
            t->numel = nall;
            have_shape = 1;
        } else if (!strcmp(key, "data_offsets")) {
            if (have_off) PFAIL("duplicate data_offsets");
            if (!lit(sc, '[') || !i64_(sc, o0) || !lit(sc, ',') || !i64_(sc, o1) || !lit(sc, ']'))
                PFAIL("data_offsets is not [start, end]");
            have_off = 1;
        } else if (!skip_value(sc)) PFAIL("unterminated value");
    }
    if (!have_dt || !have_shape || !have_off)
        return lm_fail(err, errlen, "malformed header: tensor %s is missing %s", name,
                       !have_dt ? "dtype" : !have_shape ? "shape" : "data_offsets");
    return 0;
#undef PFAIL
}

static int meta_push(LmStFile *f, const char *k, const char *v)
{
    char **nk = (char **)realloc(f->meta_key, (size_t)(f->nmeta + 1) * sizeof *nk);
    if (!nk) return -1;
    f->meta_key = nk;
    char **nv = (char **)realloc(f->meta_val, (size_t)(f->nmeta + 1) * sizeof *nv);
    if (!nv) return -1;
    f->meta_val = nv;
    char *kd = strdup(k), *vd = strdup(v);
    if (!kd || !vd) { free(kd); free(vd); return -1; }
    f->meta_key[f->nmeta] = kd;
    f->meta_val[f->nmeta] = vd;
    f->nmeta++;
    return 0;
}

/* __metadata__ is {"string": "string", ...} by the safetensors spec. */
static int parse_metadata(Scan *sc, LmStFile *f, char *err, size_t errlen)
{
    if (!lit(sc, '{')) return lm_fail(err, errlen, "malformed header: __metadata__ is not an object");
    int first = 1;
    for (;;) {
        ws(sc);
        if (sc->p < sc->end && *sc->p == '}') { sc->p++; return 0; }
        if (!first && !lit(sc, ',')) return lm_fail(err, errlen, "malformed header: __metadata__ expected ','");
        first = 0;
        char k[256], v[4096];
        if (!str_(sc, k, sizeof k) || !lit(sc, ':') || !str_(sc, v, sizeof v))
            return lm_fail(err, errlen, "malformed header: __metadata__ holds a non-string value or an oversized string");
        if (meta_push(f, k, v) != 0) return lm_fail(err, errlen, "out of memory");
    }
}

static int open_file(LmSt *s, Grow *g, const char *path, int use_mmap, char *err, size_t errlen)
{
    LmStFile *nf = (LmStFile *)realloc(s->files, (size_t)(s->nfiles + 1) * sizeof *nf);
    if (!nf) return lm_fail(err, errlen, "out of memory");
    s->files = nf;
    const int fi = s->nfiles;
    LmStFile *f = &s->files[fi];
    memset(f, 0, sizeof *f);
    f->fd = -1;
    s->nfiles++;                          /* counted now so lm_st_close releases it on failure */

    char *hdr = NULL;
    f->path = strdup(path);
    if (!f->path) return lm_fail(err, errlen, "out of memory");
    f->fd = open(path, O_RDONLY);
    if (f->fd < 0) return lm_fail(err, errlen, "cannot open %s: %s", path, strerror(errno));

    struct stat st;
    if (fstat(f->fd, &st) != 0) return lm_fail(err, errlen, "cannot stat %s: %s", path, strerror(errno));
    f->size = (int64_t)st.st_size;
    if (f->size < 8)
        return lm_fail(err, errlen, "%s is %lld bytes, too short for a safetensors header length (truncated?)",
                       path, (long long)f->size);

    unsigned char lenbuf[8];
    if (lm_pread_full(f->fd, lenbuf, 8, 0, err, errlen) != 0) return -1;
    uint64_t hlen = 0;
    for (int i = 7; i >= 0; i--) hlen = (hlen << 8) | lenbuf[i];   /* little endian */
    if (hlen == 0) return lm_fail(err, errlen, "%s: header length is zero", path);
    if (hlen > (uint64_t)LM_ST_MAX_HEADER)
        return lm_fail(err, errlen, "%s: header length %llu exceeds the %lld byte limit (oversized or corrupt)",
                       path, (unsigned long long)hlen, (long long)LM_ST_MAX_HEADER);
    /* Subtraction, not addition, so the comparison cannot wrap. */
    if (hlen > (uint64_t)(f->size - 8))
        return lm_fail(err, errlen, "%s: header length %llu runs past the end of a %lld byte file (truncated)",
                       path, (unsigned long long)hlen, (long long)f->size);
    f->data_base = 8 + (int64_t)hlen;

    hdr = (char *)malloc((size_t)hlen + 1);
    if (!hdr) return lm_fail(err, errlen, "out of memory reading a %llu byte header", (unsigned long long)hlen);
    if (lm_pread_full(f->fd, hdr, (size_t)hlen, 8, err, errlen) != 0) { free(hdr); return -1; }
    hdr[hlen] = '\0';

    const int first_t = s->nt;
    Scan sc = { hdr, hdr + hlen };
    if (!lit(&sc, '{')) {
        lm_fail(err, errlen, "%s: malformed header: not a JSON object", path);
        goto bad;
    }
    int first = 1, have_meta = 0;
    for (;;) {
        ws(&sc);
        if (sc.p < sc.end && *sc.p == '}') { sc.p++; break; }
        if (!first && !lit(&sc, ',')) {
            lm_fail(err, errlen, "%s: malformed header: expected ',' near byte %ld", path, (long)(sc.p - hdr));
            goto bad;
        }
        first = 0;
        char name[1024];
        if (!str_(&sc, name, sizeof name)) {
            lm_fail(err, errlen, "%s: malformed header: bad tensor name near byte %ld (or name over %zu bytes)",
                    path, (long)(sc.p - hdr), sizeof name - 1);
            goto bad;
        }
        if (!lit(&sc, ':')) {
            lm_fail(err, errlen, "%s: malformed header: expected ':' after %s", path, name);
            goto bad;
        }
        if (!strcmp(name, "__metadata__")) {
            if (have_meta) { lm_fail(err, errlen, "%s: malformed header: duplicate __metadata__", path); goto bad; }
            have_meta = 1;
            if (parse_metadata(&sc, f, err, errlen) != 0) goto bad;
            continue;
        }
        if (!lit(&sc, '{')) {
            lm_fail(err, errlen, "%s: malformed header: entry for %s is not an object", path, name);
            goto bad;
        }

        LmTensor t;
        memset(&t, 0, sizeof t);
        t.file = fi;
        int64_t o0 = 0, o1 = 0;
        if (parse_entry(&sc, &t, &o0, &o1, name, err, errlen, hdr) != 0) goto bad;

        /* Bound o0 and o1 BEFORE adding data_base so nothing can wrap. */
        const int64_t room = f->size - f->data_base;
        if (o0 < 0 || o1 < o0) {
            lm_fail(err, errlen, "%s: tensor %s has impossible data_offsets [%lld,%lld]",
                    path, name, (long long)o0, (long long)o1);
            goto bad;
        }
        if (o1 > room) {
            lm_fail(err, errlen,
                    "%s: tensor %s ends at data byte %lld but the file holds only %lld (truncated file or corrupt data_offsets)",
                    path, name, (long long)o1, (long long)room);
            goto bad;
        }
        t.off = f->data_base + o0;
        t.nbytes = o1 - o0;

        const int64_t numel = lm_st_numel(&t);
        const int64_t esz = lm_dtype_size(t.dtype);
        if (numel > INT64_MAX / esz) {
            lm_fail(err, errlen, "%s: tensor %s shape is impossibly large", path, name);
            goto bad;
        }
        if (t.nbytes != numel * esz) {
            lm_fail(err, errlen, "%s: tensor %s spans %lld bytes but its %s shape implies %lld",
                    path, name, (long long)t.nbytes, lm_dtype_name(t.dtype), (long long)(numel * esz));
            goto bad;
        }

        t.name = strdup(name);
        if (!t.name || push_tensor(s, g, &t) != 0) {
            free(t.name);
            lm_fail(err, errlen, "out of memory indexing tensors");
            goto bad;
        }
    }
    ws(&sc);
    if (sc.p != sc.end) {
        lm_fail(err, errlen, "%s: malformed header: unexpected data after the header object at byte %ld",
                path, (long)(sc.p - hdr));
        goto bad;
    }
    free(hdr);
    hdr = NULL;

    if (sweep_spans(s, first_t, err, errlen) != 0) return -1;

    if (use_mmap) {
        void *m = mmap(NULL, (size_t)f->size, PROT_READ, MAP_PRIVATE, f->fd, 0);
        if (m == MAP_FAILED) return lm_fail(err, errlen, "cannot mmap %s: %s", path, strerror(errno));
        f->map = (const unsigned char *)m;
    }
    return 0;
bad:
    free(hdr);
    return -1;
}

static int build_index(LmSt *s, char *err, size_t errlen)
{
    int nb = 1024;
    while (nb < s->nt * 2) nb <<= 1;
    s->bucket = (int32_t *)malloc((size_t)nb * sizeof(int32_t));
    if (!s->bucket) return lm_fail(err, errlen, "out of memory building the tensor index");
    memset(s->bucket, 0xFF, (size_t)nb * sizeof(int32_t));
    s->nbucket = nb;
    for (int i = 0; i < s->nt; i++) {
        int j = (int)(fnv1a(s->t[i].name) & (uint64_t)(nb - 1));
        while (s->bucket[j] >= 0) {
            const LmTensor *o = &s->t[s->bucket[j]];
            if (!strcmp(o->name, s->t[i].name)) {
                if (o->file == s->t[i].file)
                    return lm_fail(err, errlen, "duplicate tensor name %s in %s", o->name,
                                   s->files[o->file].path);
                return lm_fail(err, errlen, "tensor %s appears in both %s and %s", o->name,
                               s->files[o->file].path, s->files[s->t[i].file].path);
            }
            j = (j + 1) & (nb - 1);
        }
        s->bucket[j] = i;
    }
    return 0;
}

/* ------------------------------------------------------------------ index file */

static char *join_path(const char *dir, const char *leaf)
{
    const size_t dl = strlen(dir), ll = strlen(leaf);
    char *p = (char *)malloc(dl + ll + 2);
    if (!p) return NULL;
    memcpy(p, dir, dl);
    size_t n = dl;
    if (dl > 0 && dir[dl - 1] != '/') p[n++] = '/';
    memcpy(p + n, leaf, ll + 1);
    return p;
}

static char *dir_of(const char *path)
{
    const char *sl = strrchr(path, '/');
    if (!sl) return strdup(".");
    if (sl == path) return strdup("/");
    char *d = (char *)malloc((size_t)(sl - path) + 1);
    if (!d) return NULL;
    memcpy(d, path, (size_t)(sl - path));
    d[sl - path] = '\0';
    return d;
}

static int read_whole(const char *path, int64_t cap, char **out, char *err, size_t errlen)
{
    const int fd = open(path, O_RDONLY);
    if (fd < 0) return lm_fail(err, errlen, "cannot open %s: %s", path, strerror(errno));
    struct stat st;
    if (fstat(fd, &st) != 0) { close(fd); return lm_fail(err, errlen, "cannot stat %s: %s", path, strerror(errno)); }
    if ((int64_t)st.st_size > cap) {
        close(fd);
        return lm_fail(err, errlen, "%s is %lld bytes, over the %lld byte limit", path,
                       (long long)st.st_size, (long long)cap);
    }
    char *buf = (char *)malloc((size_t)st.st_size + 1);
    if (!buf) { close(fd); return lm_fail(err, errlen, "out of memory reading %s", path); }
    const int rc = lm_pread_full(fd, buf, (size_t)st.st_size, 0, err, errlen);
    close(fd);
    if (rc != 0) { free(buf); return -1; }
    buf[st.st_size] = '\0';
    *out = buf;
    return 0;
}

static int bare_filename(const char *n)
{
    if (!*n || !strcmp(n, ".") || !strcmp(n, "..")) return 0;
    for (const char *p = n; *p; p++) if (*p == '/' || *p == '\\') return 0;
    return 1;
}

static int open_indexed(LmSt *s, const char *index_path, int use_mmap, char *err, size_t errlen)
{
    char *text = NULL;
    if (read_whole(index_path, LM_ST_MAX_INDEX, &text, err, errlen) != 0) return -1;
    jval *root = json_parse(text, NULL);
    free(text);
    if (!root || root->t != J_OBJ) {
        json_free_tree(root);
        return lm_fail(err, errlen, "%s is not a valid JSON object", index_path);
    }
    jval *wm = json_get(root, "weight_map");
    if (!wm || wm->t != J_OBJ || wm->len == 0) {
        json_free_tree(root);
        return lm_fail(err, errlen, "%s has no non-empty \"weight_map\" object", index_path);
    }
    char *dir = dir_of(index_path);
    if (!dir) { json_free_tree(root); return lm_fail(err, errlen, "out of memory"); }

    /* Shard files in order of first appearance; fidx[i] = file of weight_map entry i. */
    int *fidx = (int *)malloc((size_t)wm->len * sizeof *fidx);
    char **names = (char **)calloc((size_t)wm->len, sizeof *names);   /* unique shard names */
    int nshard = 0, rc = -1;
    Grow g = { 0 };
    if (!fidx || !names) { lm_fail(err, errlen, "out of memory"); goto done; }
    for (int i = 0; i < wm->len; i++) {
        const jval *v = wm->kids[i];
        if (!v || v->t != J_STR || !bare_filename(v->str)) {
            lm_fail(err, errlen, "%s: weight_map entry %s is not a bare shard file name", index_path, wm->keys[i]);
            goto done;
        }
        int k = 0;
        while (k < nshard && strcmp(names[k], v->str)) k++;
        if (k == nshard) names[nshard++] = v->str;
        fidx[i] = k;
    }
    for (int k = 0; k < nshard; k++) {
        char *p = join_path(dir, names[k]);
        if (!p) { lm_fail(err, errlen, "out of memory"); goto done; }
        struct stat st;
        if (stat(p, &st) != 0) {
            lm_fail(err, errlen, "index %s names shard %s but %s does not exist", index_path, names[k], p);
            free(p);
            goto done;
        }
        const int orc = open_file(s, &g, p, use_mmap, err, errlen);
        free(p);
        if (orc != 0) goto done;
    }
    if (build_index(s, err, errlen) != 0) goto done;

    /* Index -> shards: every entry resolves in exactly the shard the index names, once.
     * Shards -> index: every tensor a shard holds is listed (count equality, given that
     * each entry resolves to a distinct tensor). */
    unsigned char *listed = (unsigned char *)calloc((size_t)(s->nt ? s->nt : 1), 1);
    if (!listed) { lm_fail(err, errlen, "out of memory"); goto done; }
    for (int i = 0; i < wm->len; i++) {
        const LmTensor *t = lm_st_find(s, wm->keys[i]);
        if (!t || t->file != fidx[i]) {
            if (!t)
                lm_fail(err, errlen, "index %s says %s is in %s, which has no such tensor",
                        index_path, wm->keys[i], names[fidx[i]]);
            else
                lm_fail(err, errlen, "index %s says %s is in %s, but it is in %s",
                        index_path, wm->keys[i], names[fidx[i]], s->files[t->file].path);
            free(listed);
            goto done;
        }
        const int ti = (int)(t - s->t);
        if (listed[ti]) {
            lm_fail(err, errlen, "index %s lists %s twice", index_path, wm->keys[i]);
            free(listed);
            goto done;
        }
        listed[ti] = 1;
    }
    for (int i = 0; i < s->nt; i++) {
        if (!listed[i]) {
            lm_fail(err, errlen, "shard %s holds tensor %s that index %s does not list",
                    s->files[s->t[i].file].path, s->t[i].name, index_path);
            free(listed);
            goto done;
        }
    }
    free(listed);
    rc = 0;
done:
    free(fidx);
    free(names);
    free(dir);
    json_free_tree(root);
    return rc;
}

/* ------------------------------------------------------------------ open */

int lm_st_open(LmSt *s, const char *path, int use_mmap, char *err, size_t errlen)
{
    memset(s, 0, sizeof *s);
    char *p = NULL;
    int rc = -1;
    struct stat st;
    if (stat(path, &st) != 0) return lm_fail(err, errlen, "cannot open %s: %s", path, strerror(errno));

    if (S_ISDIR(st.st_mode)) {
        char *ix = join_path(path, "model.safetensors.index.json");
        char *one = join_path(path, "model.safetensors");
        if (!ix || !one) { free(ix); free(one); return lm_fail(err, errlen, "out of memory"); }
        struct stat s2;
        if (stat(ix, &s2) == 0) { p = ix; free(one); }
        else if (stat(one, &s2) == 0) { p = one; free(ix); }
        else {
            lm_fail(err, errlen, "%s holds neither model.safetensors.index.json nor model.safetensors", path);
            free(ix); free(one);
            return -1;
        }
    } else {
        p = strdup(path);
        if (!p) return lm_fail(err, errlen, "out of memory");
    }

    const size_t n = strlen(p);
    const int is_index = n >= 5 && !strcmp(p + n - 5, ".json");
    if (is_index) {
        rc = open_indexed(s, p, use_mmap, err, errlen);
    } else {
        Grow g = { 0 };
        rc = open_file(s, &g, p, use_mmap, err, errlen);
        if (rc == 0) rc = build_index(s, err, errlen);
    }
    free(p);
    if (rc != 0) lm_st_close(s);
    return rc;
}

const LmTensor *lm_st_find(const LmSt *s, const char *name)
{
    if (!s->bucket) return NULL;
    int j = (int)(fnv1a(name) & (uint64_t)(s->nbucket - 1));
    while (s->bucket[j] >= 0) {
        const LmTensor *t = &s->t[s->bucket[j]];
        if (!strcmp(t->name, name)) return t;
        j = (j + 1) & (s->nbucket - 1);
    }
    return NULL;
}

const char *lm_st_meta(const LmSt *s, const char *key)
{
    for (int f = 0; f < s->nfiles; f++)
        for (int i = 0; i < s->files[f].nmeta; i++)
            if (!strcmp(s->files[f].meta_key[i], key)) return s->files[f].meta_val[i];
    return NULL;
}

/* ------------------------------------------------------------------ reading */

const void *lm_st_data(const LmSt *s, const LmTensor *t)
{
    const LmStFile *f = &s->files[t->file];
    if (!f->map) return NULL;
    const int64_t a = lm_dtype_size(t->dtype);
    if (a > 1 && (t->off % a) != 0) return NULL;
    return f->map + t->off;
}

int lm_st_read_slice(const LmSt *s, const LmTensor *t, int64_t byte_off, int64_t nbytes,
                     void *buf, char *err, size_t errlen)
{
    if (byte_off < 0 || nbytes < 0 || byte_off > t->nbytes || nbytes > t->nbytes - byte_off)
        return lm_fail(err, errlen, "slice [%lld,+%lld) is outside tensor %s (%lld bytes)",
                       (long long)byte_off, (long long)nbytes, t->name, (long long)t->nbytes);
    const LmStFile *f = &s->files[t->file];
    if (f->map) {
        if (nbytes) memcpy(buf, f->map + t->off + byte_off, (size_t)nbytes);
        return 0;
    }
    return lm_pread_full(f->fd, buf, (size_t)nbytes, t->off + byte_off, err, errlen);
}

int lm_st_read(const LmSt *s, const LmTensor *t, void *buf, char *err, size_t errlen)
{
    return lm_st_read_slice(s, t, 0, t->nbytes, buf, err, errlen);
}

#define LM_WIDEN_CHUNK (4 << 20)

int lm_st_read_f32_range(const LmSt *s, const LmTensor *t, int64_t first, int64_t count,
                         float *out, char *err, size_t errlen)
{
    const int64_t n = lm_st_numel(t);
    const int esz = lm_dtype_size(t->dtype);
    if (esz <= 0) return lm_fail(err, errlen, "tensor %s has no widenable dtype", t->name);
    if (first < 0 || count < 0 || first > n || count > n - first)
        return lm_fail(err, errlen, "element range [%lld,+%lld) is outside tensor %s (%lld elements)",
                       (long long)first, (long long)count, t->name, (long long)n);

    const int64_t chunk = LM_WIDEN_CHUNK / esz;
    void *raw = malloc((size_t)(chunk * esz));
    if (!raw) return lm_fail(err, errlen, "out of memory widening %s", t->name);
    for (int64_t done = 0; done < count;) {
        const int64_t take = (count - done < chunk) ? (count - done) : chunk;
        if (lm_st_read_slice(s, t, (first + done) * esz, take * esz, raw, err, errlen) != 0) {
            free(raw);
            return -1;
        }
        float *o = out + done;
        switch (t->dtype) {
        case LM_DT_F32: memcpy(o, raw, (size_t)take * 4); break;
        case LM_DT_BF16: { const uint16_t *p = (const uint16_t *)raw;
            for (int64_t i = 0; i < take; i++) o[i] = lm_bf16_to_f32(p[i]); } break;
        case LM_DT_F16: { const uint16_t *p = (const uint16_t *)raw;
            for (int64_t i = 0; i < take; i++) o[i] = lm_f16_to_f32(p[i]); } break;
        case LM_DT_I8: { const int8_t *p = (const int8_t *)raw;
            for (int64_t i = 0; i < take; i++) o[i] = (float)p[i]; } break;
        default: { const uint8_t *p = (const uint8_t *)raw;
            for (int64_t i = 0; i < take; i++) o[i] = (float)p[i]; } break;
        }
        done += take;
    }
    free(raw);
    return 0;
}

int lm_st_read_f32(const LmSt *s, const LmTensor *t, float *out, char *err, size_t errlen)
{
    return lm_st_read_f32_range(s, t, 0, lm_st_numel(t), out, err, errlen);
}
