/* lm_bind.c - see lm_bind.h. Original code (moved from lm_model.c). */
#include "io/lm_portable_io.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "core/lm_kernels.h"
#include "io/lm_io.h"
#include "model/lm_bind.h"

void *lm_bind_own(LmBind *b, void *p)
{
    if (!p) return NULL;
    if (b->nowned == b->capowned) {
        const int nc = b->capowned ? b->capowned * 2 : 256;
        void **np = (void **)realloc(b->owned, (size_t)nc * sizeof *np);
        if (!np) { free(p); return NULL; }
        b->owned = np;
        b->capowned = nc;
    }
    b->owned[b->nowned++] = p;
    return p;
}

const LmTensor *lm_bind_need(LmBind *b, const char *name, int ndim, int64_t d0, int64_t d1, char *err, size_t n)
{
    const LmTensor *t = lm_st_find(&b->st, name);
    if (!t) { lm_fail(err, n, "tensor %s is missing from the weights", name); return NULL; }
    if (t->rank_overflow) { lm_fail(err, n, "tensor %s has rank above 4; the engine supports at most 4", name); return NULL; }
    int ok = t->ndim == ndim;
    if (ok && ndim >= 1) ok = t->shape[0] == d0;
    if (ok && ndim >= 2) ok = t->shape[1] == d1;
    if (!ok) {
        lm_fail(err, n, "tensor %s has shape rank %d [%lld,%lld], the config needs rank %d [%lld,%lld]", name,
                t->ndim, (long long)t->shape[0], (long long)(t->ndim > 1 ? t->shape[1] : 0), ndim,
                (long long)d0, (long long)d1);
        return NULL;
    }
    return t;
}

/* A float vector of numel elements from any float dtype, widened into an owned buffer. */
const float *lm_bind_vec(LmBind *b, const char *name, int64_t numel, int ndim, int64_t d0, int64_t d1,
                         char *err, size_t n)
{
    const LmTensor *t = lm_bind_need(b, name, ndim, d0, d1, err, n);
    if (!t) return NULL;
    if (t->dtype != LM_DT_BF16 && t->dtype != LM_DT_F32 && t->dtype != LM_DT_F16) {
        lm_fail(err, n, "tensor %s has dtype %s; BF16, F16 or F32 expected", name, lm_dtype_name(t->dtype));
        return NULL;
    }
    float *buf = (float *)lm_bind_own(b, malloc((size_t)(numel ? numel : 1) * sizeof(float)));
    if (!buf) { lm_fail(err, n, "out of memory binding %s", name); return NULL; }
    if (lm_st_read_f32(&b->st, t, buf, err, n) != 0) return NULL;
    return buf;
}

int lm_bind_mat(LmBind *b, const char *name, int rows, int cols, LmMat *out, char *err, size_t n)
{
    const LmTensor *t = lm_bind_need(b, name, 2, rows, cols, err, n);
    if (!t) return -1;
    memset(out, 0, sizeof *out);
    out->rows = rows;
    out->cols = cols;
    if (b->q8) {
        if (t->dtype != LM_DT_I8) return lm_fail(err, n, "tensor %s has dtype %s; a q8 file stores matrices as I8", name, lm_dtype_name(t->dtype));
        if (cols % LM_Q8_GROUP) return lm_fail(err, n, "tensor %s: %d inputs is not a multiple of %d", name, cols, LM_Q8_GROUP);
        char sn[512];
        snprintf(sn, sizeof sn, "%s.q8scale", name);
        const LmTensor *s = lm_bind_need(b, sn, 2, rows, cols / LM_Q8_GROUP, err, n);
        if (!s) return -1;
        if (s->dtype != LM_DT_F32) return lm_fail(err, n, "tensor %s has dtype %s; F32 expected", sn, lm_dtype_name(s->dtype));
        float *sc = (float *)lm_bind_own(b, malloc((size_t)rows * (cols / LM_Q8_GROUP) * sizeof(float)));
        if (!sc) return lm_fail(err, n, "out of memory binding %s", sn);
        if (lm_st_read(&b->st, s, sc, err, n) != 0) return -1;
        const int8_t *q = (const int8_t *)lm_st_data(&b->st, t);
        if (!q) {
            int8_t *cp = (int8_t *)lm_bind_own(b, malloc((size_t)t->nbytes));
            if (!cp) return lm_fail(err, n, "out of memory binding %s", name);
            if (lm_st_read(&b->st, t, cp, err, n) != 0) return -1;
            q = cp;
        }
        out->kind = LM_MAT_Q8;
        out->q = q;
        out->sc = sc;
        return 0;
    }
    if (t->dtype == LM_DT_I8)
        return lm_fail(err, n, "tensor %s is I8: this is a q8 file, pass it as the weights file instead of a bf16 checkpoint", name);
    if (t->dtype != LM_DT_BF16 && t->dtype != LM_DT_F32)
        return lm_fail(err, n, "matrix %s has dtype %s; BF16 or F32 expected", name, lm_dtype_name(t->dtype));
    const void *p = lm_st_data(&b->st, t);
    if (!p) {                                 /* misaligned in the file: one owned copy */
        void *cp = lm_bind_own(b, malloc((size_t)(t->nbytes ? t->nbytes : 1)));
        if (!cp) return lm_fail(err, n, "out of memory binding %s", name);
        if (lm_st_read(&b->st, t, cp, err, n) != 0) return -1;
        p = cp;
    }
    if (t->dtype == LM_DT_BF16) { out->kind = LM_MAT_BF16; out->bf = (const uint16_t *)p; }
    else { out->kind = LM_MAT_F32; out->f = (const float *)p; }
    return 0;
}

int lm_bind_open(LmBind *b, const char *path, int q8, char *err, size_t n)
{
    if (lm_st_open(&b->st, path, 1, err, n) != 0) return -1;
    b->q8 = q8;
    if (q8) {
        const char *f = lm_st_meta(&b->st, "format"), *g = lm_st_meta(&b->st, "group");
        if (!f || strcmp(f, "lm-q8") || !g || strcmp(g, "32"))
            return lm_fail(err, n, "%s is not an lm-q8 group-32 file (metadata format=%s group=%s)", path,
                           f ? f : "(none)", g ? g : "(none)");
    }
    return 0;
}

void lm_bind_free(LmBind *b)
{
    for (int i = 0; i < b->nowned; i++) free(b->owned[i]);
    free(b->owned);
    b->owned = NULL;
    b->nowned = b->capowned = 0;
    lm_st_close(&b->st);
}
