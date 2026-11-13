/* lm_model.c - include/lm/lm.h for Qwen3.5 text models: open (config + weight binding),
 * state, lm_feed. Original code.
 *
 * WEIGHTS. Two sources, chosen by LmOpts.weights:
 *   NULL  the bf16 checkpoint in model_dir (single file or sharded), read through mmap.
 *         Matrices may be BF16 or F32 and are used in place (no copy) whenever the mapping
 *         is aligned for the element type, else copied once into an owned buffer.
 *   FILE  a q8 file from lm-quant: every 2-D matrix is I8 [out,in] + F32 "<name>.q8scale"
 *         [out,in/32]. The int8 bytes are used in place; the scale tables are COPIED into
 *         owned float buffers at open because the file layout only guarantees 2-byte
 *         alignment for them.
 * Small vectors (norms, A_log, dt_bias, conv weights) are widened to owned f32 once.
 * Every tensor's shape is checked against the config at open; a missing tensor, a wrong
 * shape or an unsupported dtype fails lm_open with the tensor's name.
 *
 * STATE (all allocated in lm_open, nothing per token): per linear-attention layer a conv
 * state [conv_dim][k-1] and a delta-rule state S [n_v][dk][dv]; per full-attention layer an
 * f32 KV cache [ctx][kv_heads*head_dim] for K and for V. */
#include "io/lm_portable_io.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "lm/lm.h"
#include "core/lm_kernels.h"
#include "core/lm_ops.h"
#include "io/lm_io.h"
#include "io/lm_st.h"
#include "model/lm_config.h"
#include "model/lm_layers.h"

typedef struct {
    int          full;
    const float *in_norm, *post_norm;        /* [hidden] */
    LmMlpW       mlp;
    LmLinW       lin;
    LmAttnW      attn;
} Layer;

struct LmModel {
    LmCfg    cfg;
    int      ctx, pos;
    LmSt     st;
    int      q8;
    Layer   *layers;
    LmMat    embed, head;                    /* head == embed when tied */
    const float *final_norm;
    void   **owned;                          /* every buffer lm_open allocated */
    int      nowned, capowned;

    /* state */
    float  **conv;                           /* per layer, NULL for full layers */
    float  **S;
    float  **kc, **vc;                       /* per layer, NULL for linear layers */

    /* scratch */
    float *h, *n, *mix, *mlp_out, *scratch, *mlp_scratch;
    LmLinDims ld;
    LmAttnDims ad;
};

/* ------------------------------------------------------------------ binding helpers */

static void *own(LmModel *m, void *p)
{
    if (!p) return NULL;
    if (m->nowned == m->capowned) {
        const int nc = m->capowned ? m->capowned * 2 : 256;
        void **np = (void **)realloc(m->owned, (size_t)nc * sizeof *np);
        if (!np) { free(p); return NULL; }
        m->owned = np;
        m->capowned = nc;
    }
    m->owned[m->nowned++] = p;
    return p;
}

static const LmTensor *need(LmModel *m, const char *name, int ndim, int64_t d0, int64_t d1,
                            char *err, size_t n)
{
    const LmTensor *t = lm_st_find(&m->st, name);
    if (!t) { lm_fail(err, n, "tensor %s is missing from the weights", name); return NULL; }
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
static const float *bind_vec(LmModel *m, const char *name, int64_t numel, int ndim, int64_t d0, int64_t d1,
                             char *err, size_t n)
{
    const LmTensor *t = need(m, name, ndim, d0, d1, err, n);
    if (!t) return NULL;
    if (t->dtype != LM_DT_BF16 && t->dtype != LM_DT_F32 && t->dtype != LM_DT_F16) {
        lm_fail(err, n, "tensor %s has dtype %s; BF16, F16 or F32 expected", name, lm_dtype_name(t->dtype));
        return NULL;
    }
    float *buf = (float *)own(m, malloc((size_t)(numel ? numel : 1) * sizeof(float)));
    if (!buf) { lm_fail(err, n, "out of memory binding %s", name); return NULL; }
    if (lm_st_read_f32(&m->st, t, buf, err, n) != 0) return NULL;
    return buf;
}

static int bind_mat(LmModel *m, const char *name, int rows, int cols, LmMat *out, char *err, size_t n)
{
    const LmTensor *t = need(m, name, 2, rows, cols, err, n);
    if (!t) return -1;
    memset(out, 0, sizeof *out);
    out->rows = rows;
    out->cols = cols;
    if (m->q8) {
        if (t->dtype != LM_DT_I8) return lm_fail(err, n, "tensor %s has dtype %s; a q8 file stores matrices as I8", name, lm_dtype_name(t->dtype));
        if (cols % LM_Q8_GROUP) return lm_fail(err, n, "tensor %s: %d inputs is not a multiple of %d", name, cols, LM_Q8_GROUP);
        char sn[512];
        snprintf(sn, sizeof sn, "%s.q8scale", name);
        const LmTensor *s = need(m, sn, 2, rows, cols / LM_Q8_GROUP, err, n);
        if (!s) return -1;
        if (s->dtype != LM_DT_F32) return lm_fail(err, n, "tensor %s has dtype %s; F32 expected", sn, lm_dtype_name(s->dtype));
        float *sc = (float *)own(m, malloc((size_t)rows * (cols / LM_Q8_GROUP) * sizeof(float)));
        if (!sc) return lm_fail(err, n, "out of memory binding %s", sn);
        if (lm_st_read(&m->st, s, sc, err, n) != 0) return -1;
        const int8_t *q = (const int8_t *)lm_st_data(&m->st, t);
        if (!q) {
            int8_t *cp = (int8_t *)own(m, malloc((size_t)t->nbytes));
            if (!cp) return lm_fail(err, n, "out of memory binding %s", name);
            if (lm_st_read(&m->st, t, cp, err, n) != 0) return -1;
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
    const void *p = lm_st_data(&m->st, t);
    if (!p) {                                 /* misaligned in the file: one owned copy */
        void *cp = own(m, malloc((size_t)(t->nbytes ? t->nbytes : 1)));
        if (!cp) return lm_fail(err, n, "out of memory binding %s", name);
        if (lm_st_read(&m->st, t, cp, err, n) != 0) return -1;
        p = cp;
    }
    if (t->dtype == LM_DT_BF16) { out->kind = LM_MAT_BF16; out->bf = (const uint16_t *)p; }
    else { out->kind = LM_MAT_F32; out->f = (const float *)p; }
    return 0;
}

#define BIND_MAT(var, rows, cols, ...) do { char nm_[512]; snprintf(nm_, sizeof nm_, __VA_ARGS__); \
    if (bind_mat(m, nm_, (rows), (cols), &(var), err, errlen) != 0) goto fail; } while (0)
#define BIND_VEC(var, numel, ndim, d0, d1, ...) do { char nm_[512]; snprintf(nm_, sizeof nm_, __VA_ARGS__); \
    (var) = bind_vec(m, nm_, (numel), (ndim), (d0), (d1), err, errlen); if (!(var)) goto fail; } while (0)

#define PFX "model.language_model."

void lm_close(LmModel *m)
{
    if (!m) return;
    const int L = m->cfg.layers;
    for (int i = 0; i < L && m->conv; i++) { free(m->conv[i]); free(m->S[i]); free(m->kc[i]); free(m->vc[i]); }
    free(m->conv); free(m->S); free(m->kc); free(m->vc);
    for (int i = 0; i < m->nowned; i++) free(m->owned[i]);
    free(m->owned);
    free(m->layers);
    free(m->h); free(m->n); free(m->mix); free(m->mlp_out); free(m->scratch); free(m->mlp_scratch);
    lm_st_close(&m->st);
    lm_config_free(&m->cfg);
    free(m);
}

LmModel *lm_open(const char *model_dir, const LmOpts *opts, char *err, size_t errlen)
{
    LmModel *m = (LmModel *)calloc(1, sizeof *m);
    if (!m) { lm_fail(err, errlen, "out of memory"); return NULL; }
    const char *weights = opts ? opts->weights : NULL;
    int ctx = opts ? opts->ctx : 0;
    if (ctx < 0) { lm_fail(err, errlen, "ctx must not be negative"); free(m); return NULL; }
    if (ctx == 0) ctx = 4096;
    m->ctx = ctx;
#ifdef _OPENMP
    if (opts && opts->threads > 0) omp_set_num_threads(opts->threads);
#endif

    if (lm_config_load(model_dir, &m->cfg, err, errlen) != 0) { free(m); return NULL; }
    const LmCfg *c = &m->cfg;

    if (lm_st_open(&m->st, weights ? weights : model_dir, 1, err, errlen) != 0) goto fail;
    m->q8 = weights != NULL;
    if (m->q8) {
        const char *f = lm_st_meta(&m->st, "format"), *g = lm_st_meta(&m->st, "group");
        if (!f || strcmp(f, "lm-q8") || !g || strcmp(g, "32")) {
            lm_fail(err, errlen, "%s is not an lm-q8 group-32 file (metadata format=%s group=%s)", weights,
                    f ? f : "(none)", g ? g : "(none)");
            goto fail;
        }
    }

    m->layers = (Layer *)calloc((size_t)c->layers, sizeof(Layer));
    if (!m->layers) { lm_fail(err, errlen, "out of memory"); goto fail; }

    BIND_MAT(m->embed, c->vocab, c->hidden, PFX "embed_tokens.weight");
    if (c->tied) m->head = m->embed;
    else BIND_MAT(m->head, c->vocab, c->hidden, "lm_head.weight");
    BIND_VEC(m->final_norm, c->hidden, 1, c->hidden, 0, PFX "norm.weight");

    for (int i = 0; i < c->layers; i++) {
        Layer *L = &m->layers[i];
        L->full = c->layer_type[i] == LM_LAYER_FULL;
        BIND_VEC(L->in_norm, c->hidden, 1, c->hidden, 0, PFX "layers.%d.input_layernorm.weight", i);
        BIND_VEC(L->post_norm, c->hidden, 1, c->hidden, 0, PFX "layers.%d.post_attention_layernorm.weight", i);
        L->mlp.inter = c->inter;
        BIND_MAT(L->mlp.gate, c->inter, c->hidden, PFX "layers.%d.mlp.gate_proj.weight", i);
        BIND_MAT(L->mlp.up, c->inter, c->hidden, PFX "layers.%d.mlp.up_proj.weight", i);
        BIND_MAT(L->mlp.down, c->hidden, c->inter, PFX "layers.%d.mlp.down_proj.weight", i);
        if (L->full) {
            const int qd = c->heads * c->head_dim, kvd = c->kv_heads * c->head_dim;
            BIND_MAT(L->attn.q, 2 * qd, c->hidden, PFX "layers.%d.self_attn.q_proj.weight", i);
            BIND_MAT(L->attn.k, kvd, c->hidden, PFX "layers.%d.self_attn.k_proj.weight", i);
            BIND_MAT(L->attn.v, kvd, c->hidden, PFX "layers.%d.self_attn.v_proj.weight", i);
            BIND_MAT(L->attn.o, c->hidden, qd, PFX "layers.%d.self_attn.o_proj.weight", i);
            BIND_VEC(L->attn.q_norm, c->head_dim, 1, c->head_dim, 0, PFX "layers.%d.self_attn.q_norm.weight", i);
            BIND_VEC(L->attn.k_norm, c->head_dim, 1, c->head_dim, 0, PFX "layers.%d.self_attn.k_norm.weight", i);
        } else {
            BIND_MAT(L->lin.qkv, c->conv_dim, c->hidden, PFX "layers.%d.linear_attn.in_proj_qkv.weight", i);
            BIND_MAT(L->lin.z, c->value_dim, c->hidden, PFX "layers.%d.linear_attn.in_proj_z.weight", i);
            BIND_MAT(L->lin.b, c->lv_heads, c->hidden, PFX "layers.%d.linear_attn.in_proj_b.weight", i);
            BIND_MAT(L->lin.a, c->lv_heads, c->hidden, PFX "layers.%d.linear_attn.in_proj_a.weight", i);
            BIND_MAT(L->lin.out, c->hidden, c->value_dim, PFX "layers.%d.linear_attn.out_proj.weight", i);
            /* conv1d.weight is [conv_dim, 1, k] */
            {
                char nm[512];
                snprintf(nm, sizeof nm, PFX "layers.%d.linear_attn.conv1d.weight", i);
                const LmTensor *t = lm_st_find(&m->st, nm);
                if (!t) { lm_fail(err, errlen, "tensor %s is missing from the weights", nm); goto fail; }
                if (t->ndim != 3 || t->shape[0] != c->conv_dim || t->shape[1] != 1 || t->shape[2] != c->conv_k) {
                    lm_fail(err, errlen, "tensor %s must have shape [%d,1,%d]", nm, c->conv_dim, c->conv_k);
                    goto fail;
                }
                L->lin.conv_w = bind_vec(m, nm, (int64_t)c->conv_dim * c->conv_k, 3, c->conv_dim, 1, err, errlen);
                if (!L->lin.conv_w) goto fail;
            }
            BIND_VEC(L->lin.dt_bias, c->lv_heads, 1, c->lv_heads, 0, PFX "layers.%d.linear_attn.dt_bias", i);
            BIND_VEC(L->lin.A_log, c->lv_heads, 1, c->lv_heads, 0, PFX "layers.%d.linear_attn.A_log", i);
            BIND_VEC(L->lin.norm_w, c->lv_dim, 1, c->lv_dim, 0, PFX "layers.%d.linear_attn.norm.weight", i);
        }
    }

    /* state and scratch */
    m->ld = (LmLinDims){ c->lk_heads, c->lv_heads, c->lk_dim, c->lv_dim, c->conv_k, c->eps };
    m->ad = (LmAttnDims){ c->heads, c->kv_heads, c->head_dim, c->rot, c->theta, c->eps };
    const size_t kvd = (size_t)c->kv_heads * c->head_dim;
    m->conv = (float **)calloc((size_t)c->layers, sizeof(float *));
    m->S = (float **)calloc((size_t)c->layers, sizeof(float *));
    m->kc = (float **)calloc((size_t)c->layers, sizeof(float *));
    m->vc = (float **)calloc((size_t)c->layers, sizeof(float *));
    if (!m->conv || !m->S || !m->kc || !m->vc) { lm_fail(err, errlen, "out of memory"); goto fail; }
    for (int i = 0; i < c->layers; i++) {
        if (m->layers[i].full) {
            m->kc[i] = (float *)calloc((size_t)ctx * kvd, sizeof(float));
            m->vc[i] = (float *)calloc((size_t)ctx * kvd, sizeof(float));
            if (!m->kc[i] || !m->vc[i]) { lm_fail(err, errlen, "out of memory allocating the KV cache for ctx %d", ctx); goto fail; }
        } else {
            m->conv[i] = (float *)calloc((size_t)c->conv_dim * (size_t)(c->conv_k - 1) + 1, sizeof(float));
            m->S[i] = (float *)calloc((size_t)c->lv_heads * c->lk_dim * c->lv_dim, sizeof(float));
            if (!m->conv[i] || !m->S[i]) { lm_fail(err, errlen, "out of memory allocating layer state"); goto fail; }
        }
    }
    size_t sc = lm_linear_scratch(&m->ld);
    const size_t sa = lm_attn_scratch(&m->ad, ctx);
    if (sa > sc) sc = sa;
    m->h = (float *)calloc((size_t)c->hidden, sizeof(float));
    m->n = (float *)calloc((size_t)c->hidden, sizeof(float));
    m->mix = (float *)calloc((size_t)c->hidden, sizeof(float));
    m->mlp_out = (float *)calloc((size_t)c->hidden, sizeof(float));
    m->scratch = (float *)calloc(sc, sizeof(float));
    m->mlp_scratch = (float *)calloc(lm_mlp_scratch(c->inter), sizeof(float));
    if (!m->h || !m->n || !m->mix || !m->mlp_out || !m->scratch || !m->mlp_scratch) {
        lm_fail(err, errlen, "out of memory allocating scratch");
        goto fail;
    }
    return m;
fail:
    lm_close(m);
    return NULL;
}

int  lm_n_vocab(const LmModel *m) { return m->cfg.vocab; }
int  lm_n_ctx(const LmModel *m)   { return m->ctx; }
int  lm_pos(const LmModel *m)     { return m->pos; }

void lm_reset(LmModel *m)
{
    const LmCfg *c = &m->cfg;
    const size_t kvd = (size_t)c->kv_heads * c->head_dim;
    for (int i = 0; i < c->layers; i++) {
        if (m->conv[i]) memset(m->conv[i], 0, ((size_t)c->conv_dim * (size_t)(c->conv_k - 1) + 1) * sizeof(float));
        if (m->S[i]) memset(m->S[i], 0, (size_t)c->lv_heads * c->lk_dim * c->lv_dim * sizeof(float));
        if (m->kc[i]) memset(m->kc[i], 0, (size_t)m->ctx * kvd * sizeof(float));
        if (m->vc[i]) memset(m->vc[i], 0, (size_t)m->ctx * kvd * sizeof(float));
    }
    m->pos = 0;
}

int lm_feed(LmModel *m, int token, float *logits)
{
    const LmCfg *c = &m->cfg;
    if (token < 0 || token >= c->vocab) return LM_ERR_TOKEN;
    if (m->pos >= m->ctx) return LM_ERR_CTX;

    lm_mat_row(&m->embed, token, m->h);
    for (int i = 0; i < c->layers; i++) {
        const Layer *L = &m->layers[i];
        lm_rmsnorm(m->n, m->h, L->in_norm, c->hidden, c->eps);
        if (L->full)
            lm_attn_mixer(m->mix, m->n, &L->attn, &m->ad, m->kc[i], m->vc[i], m->pos, m->ctx, m->scratch);
        else
            lm_linear_mixer(m->mix, m->n, &L->lin, &m->ld, m->conv[i], m->S[i], m->scratch);
        for (int j = 0; j < c->hidden; j++) m->h[j] += m->mix[j];
        lm_rmsnorm(m->n, m->h, L->post_norm, c->hidden, c->eps);
        lm_mlp(m->mlp_out, m->n, &L->mlp, m->mlp_scratch);
        for (int j = 0; j < c->hidden; j++) m->h[j] += m->mlp_out[j];
    }
    m->pos++;
    if (logits) {
        lm_rmsnorm(m->n, m->h, m->final_norm, c->hidden, c->eps);
        lm_mat_mul(&m->head, logits, m->n);
    }
    return LM_OK;
}
