/* lm_quantfile.c - see lm_quantfile.h. Original code. */
#include "io/lm_portable_io.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "core/lm_quant.h"
#include "io/lm_io.h"
#include "io/lm_quantfile.h"
#include "io/lm_sha256.h"
#include "io/lm_st.h"
#include "io/lm_stw.h"

#define BLOCK_ELEMS ((int64_t)4 << 20)

#define TEXT_PREFIX "model.language_model."

#define MTP_PREFIX "mtp."

static const char *const VISION_PREFIXES[] = { "model.visual.", "model.vision_tower.", "model.embed_vision." };

static int has_prefix(const char *name, const char *prefix)
{
    return strncmp(name, prefix, strlen(prefix)) == 0;
}

static int is_vision(const char *name)
{
    for (size_t i = 0; i < sizeof VISION_PREFIXES / sizeof *VISION_PREFIXES; i++)
        if (has_prefix(name, VISION_PREFIXES[i])) return 1;
    return 0;
}

/* The text model, its MTP head and the vision encoder are kept; anything else is dropped. */
static int ignored(const char *name)
{
    return !has_prefix(name, TEXT_PREFIX) && !has_prefix(name, MTP_PREFIX) && !is_vision(name);
}

static int ends_with(const char *s, const char *suffix)
{
    const size_t n = strlen(s), m = strlen(suffix);
    return n >= m && !strcmp(s + n - m, suffix);
}

/* Output shape. A vision ".weight" of rank 3 or more (the Conv3d/Conv2d patch embedding) is
 * written flattened to [d0, product of the rest], the matrix the encoder multiplies by. */
static int out_shape(const LmTensor *t, int64_t shape[4])
{
    if (is_vision(t->name) && ends_with(t->name, ".weight") && (t->ndim > 2 || t->rank_overflow)) {
        shape[0] = t->shape[0];
        shape[1] = t->shape[0] > 0 ? t->numel / t->shape[0] : 0;
        return 2;
    }
    for (int d = 0; d < t->ndim; d++) shape[d] = t->shape[d];
    return t->ndim;
}

/* A vision matrix whose width is not a whole number of q8 groups is copied as it is (the
 * engine binds either form); a text matrix like that is an error further down. */
static int quantised(const LmTensor *t)
{
    int64_t s[4];
    if (out_shape(t, s) != 2 || !ends_with(t->name, ".weight")) return 0;
    if (is_vision(t->name) && (s[1] == 0 || s[1] % LM_Q8_GROUP != 0)) return 0;
    return 1;
}

typedef struct { char *name; float *buf; } Pending;

static char *scale_name(const char *name)
{
    const size_t n = strlen(name);
    char *s = (char *)malloc(n + 9);
    if (!s) return NULL;
    memcpy(s, name, n);
    memcpy(s + n, ".q8scale", 9);
    return s;
}

/* Path of the file whose hash goes in the metadata. */
static int source_path(const char *model, char *out, size_t cap)
{
    struct stat st;
    if (stat(model, &st) != 0) return -1;
    if (!S_ISDIR(st.st_mode)) return snprintf(out, cap, "%s", model) < (int)cap ? 0 : -1;
    char ix[4096];
    if (snprintf(ix, sizeof ix, "%s/model.safetensors.index.json", model) >= (int)sizeof ix) return -1;
    if (stat(ix, &st) == 0) return snprintf(out, cap, "%s", ix) < (int)cap ? 0 : -1;
    return snprintf(out, cap, "%s/model.safetensors", model) < (int)cap ? 0 : -1;
}

int lm_quantize_checkpoint(const char *model, const char *outp, LmQuantStats *stats,
                           char *err, size_t errlen)
{
    LmQuantStats ls = { 0, 0, 0 };
    LmSt st;
    if (lm_st_open(&st, model, 0, err, errlen) != 0) return -1;

    char src[4096], hex[65];
    if (source_path(model, src, sizeof src) != 0) {
        lm_st_close(&st);
        return lm_fail(err, errlen, "cannot locate the source file to hash under %s", model);
    }
    if (lm_sha256_file(src, hex, err, errlen) != 0) { lm_st_close(&st); return -1; }

    LmStWriter *w = lm_stw_create(outp, err, errlen);
    if (!w) { lm_st_close(&st); return -1; }

    int rc = -1;
    Pending *pend = NULL;
    int npend = 0;
    int8_t *qbuf = NULL;
    float *fbuf = NULL;
    unsigned char *cbuf = NULL;
    const char *next;

    if (lm_stw_meta(w, "format", "lm-q8", err, errlen) != 0 ||
        lm_stw_meta(w, "group", "32", err, errlen) != 0 ||
        lm_stw_meta(w, "source", hex, err, errlen) != 0) goto out;

    for (int i = 0; i < st.nt; i++) {
        const LmTensor *t = &st.t[i];
        if (ignored(t->name)) { ls.n_ignored++; continue; }
        if (ends_with(t->name, ".q8scale")) {
            lm_fail(err, errlen, "source tensor %s collides with the .q8scale suffix of the output format", t->name);
            goto out;
        }
        int64_t oshape[4];
        const int ondim = out_shape(t, oshape);
        if (!quantised(t)) {
            if (t->rank_overflow && ondim != 2) {
                lm_fail(err, errlen, "tensor %s has rank above 4; the q8 format supports at most 4", t->name);
                goto out;
            }
            if (lm_stw_declare(w, t->name, t->dtype, ondim, oshape, err, errlen) != 0) goto out;
            ls.n_copied++;
            continue;
        }
        if (t->dtype != LM_DT_BF16 && t->dtype != LM_DT_F16 && t->dtype != LM_DT_F32) {
            lm_fail(err, errlen, "2-D weight %s has dtype %s; only BF16, F16 and F32 can be quantised",
                    t->name, lm_dtype_name(t->dtype));
            goto out;
        }
        if (oshape[1] == 0 || oshape[1] % LM_Q8_GROUP != 0) {
            lm_fail(err, errlen, "2-D weight %s has in = %lld, not a positive multiple of %d; refusing to guess",
                    t->name, (long long)oshape[1], LM_Q8_GROUP);
            goto out;
        }
        char *sn = scale_name(t->name);
        if (!sn) { lm_fail(err, errlen, "out of memory"); goto out; }
        const int64_t sshape[2] = { oshape[0], oshape[1] / LM_Q8_GROUP };
        const int r1 = lm_stw_declare(w, t->name, LM_DT_I8, 2, oshape, err, errlen);
        const int r2 = r1 == 0 ? lm_stw_declare(w, sn, LM_DT_F32, 2, sshape, err, errlen) : -1;
        free(sn);
        if (r1 != 0 || r2 != 0) goto out;
        ls.n_quantised++;
    }
    if (ls.n_quantised == 0) {
        lm_fail(err, errlen, "no 2-D .weight tensors found; this does not look like a model checkpoint");
        goto out;
    }
    if (lm_stw_begin(w, err, errlen) != 0) goto out;

    fbuf = (float *)malloc((size_t)BLOCK_ELEMS * sizeof *fbuf);
    qbuf = (int8_t *)malloc((size_t)BLOCK_ELEMS);
    cbuf = (unsigned char *)malloc((size_t)BLOCK_ELEMS);
    if (!fbuf || !qbuf || !cbuf) { lm_fail(err, errlen, "out of memory"); goto out; }

    while ((next = lm_stw_next_name(w)) != NULL) {
        char nm[1024];
        snprintf(nm, sizeof nm, "%s", next);
        if (lm_stw_tensor_begin(w, nm, err, errlen) != 0) goto out;

        const LmTensor *t = lm_st_find(&st, nm);
        if (!t) {
            /* a derived "<name>.q8scale": its table is waiting in pend */
            int k = 0;
            while (k < npend && strcmp(pend[k].name, nm)) k++;
            if (k == npend) { lm_fail(err, errlen, "internal: scale table missing for %s", nm); goto out; }
            nm[strlen(nm) - 8] = '\0';
            const LmTensor *base = lm_st_find(&st, nm);
            if (!base) { lm_fail(err, errlen, "internal: no source for scale table of %s", nm); goto out; }
            int64_t bs[4];
            out_shape(base, bs);
            const int64_t nbytes = bs[0] * (bs[1] / LM_Q8_GROUP) * 4;
            if (lm_stw_write(w, pend[k].buf, nbytes, err, errlen) != 0) goto out;
            free(pend[k].buf);
            free(pend[k].name);
            pend[k] = pend[--npend];
        } else if (!quantised(t)) {
            for (int64_t off = 0; off < t->nbytes; off += BLOCK_ELEMS) {
                const int64_t n = (t->nbytes - off < BLOCK_ELEMS) ? t->nbytes - off : BLOCK_ELEMS;
                if (lm_st_read_slice(&st, t, off, n, cbuf, err, errlen) != 0 ||
                    lm_stw_write(w, cbuf, n, err, errlen) != 0) goto out;
            }
        } else {
            int64_t ts[4];
            out_shape(t, ts);
            const int64_t rows = ts[0], in = ts[1], ng = in / LM_Q8_GROUP;
            if (in > BLOCK_ELEMS) { lm_fail(err, errlen, "%s: row of %lld inputs is wider than the block buffer", t->name, (long long)in); goto out; }
            const int64_t nscales = rows * ng;
            float *scales = (float *)malloc((size_t)(nscales > 0 ? nscales : 1) * sizeof(float));
            Pending *np = (Pending *)realloc(pend, (size_t)(npend + 1) * sizeof *np);
            char *sn = scale_name(t->name);
            if (np) pend = np;
            if (!scales || !np || !sn) { free(scales); free(sn); lm_fail(err, errlen, "out of memory"); goto out; }
            pend[npend].name = sn;
            pend[npend].buf = scales;
            npend++;
            const int64_t brows = BLOCK_ELEMS / in;
            for (int64_t r0 = 0; r0 < rows; r0 += brows) {
                const int64_t nr = (rows - r0 < brows) ? rows - r0 : brows;
                int64_t bad = 0;
                if (lm_st_read_f32_range(&st, t, r0 * in, nr * in, fbuf, err, errlen) != 0) goto out;
                const int qr = lm_q8_quantize(fbuf, nr, in, qbuf, scales + r0 * ng, &bad);
                if (qr == LM_Q8_ERR_NONFINITE) {
                    lm_fail(err, errlen, "%s has a non-finite value at element %lld (row %lld); refusing to quantise",
                            t->name, (long long)(r0 * in + bad), (long long)(r0 + bad / in));
                    goto out;
                }
                if (qr != LM_Q8_OK) { lm_fail(err, errlen, "quantiser rejected %s", t->name); goto out; }
                if (lm_stw_write(w, qbuf, nr * in, err, errlen) != 0) goto out;
            }
        }
        if (lm_stw_tensor_end(w, err, errlen) != 0) goto out;
    }
    if (npend != 0) { lm_fail(err, errlen, "internal: %d scale tables never written", npend); goto out; }
    {
        LmStWriter *wf = w;
        w = NULL;                       /* lm_stw_finish frees it, success or not */
        if (lm_stw_finish(wf, err, errlen) != 0) goto out;
    }
    rc = 0;
    if (stats) *stats = ls;
out:
    if (w) lm_stw_abort(w);
    for (int i = 0; i < npend; i++) { free(pend[i].buf); free(pend[i].name); }
    free(pend);
    free(fbuf); free(qbuf); free(cbuf);
    lm_st_close(&st);
    return rc;
}
