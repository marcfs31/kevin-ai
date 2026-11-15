/* lm_model.c - include/lm/lm.h: the family dispatch (config.json model_type: qwen3_5 here,
 * gemma4 in lm_gemma.c) and the Qwen3.5 text model: open (config + weight binding), state,
 * lm_feed. Original code.
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
#include "model/lm_bind.h"
#include "model/lm_config.h"
#include "model/lm_gemma.h"
#include "model/lm_layers.h"

typedef struct {
    int          full;
    const float *in_norm, *post_norm;        /* [hidden] */
    LmMlpW       mlp;
    LmLinW       lin;
    LmAttnW      attn;
} Layer;

struct LmModel {
    LmGemma *gemma;                          /* non-NULL: a Gemma 4 model, every other field unused */
    LmCfg    cfg;
    int      ctx, pos;
    LmBind   b;                              /* the weight set and every buffer lm_open allocated */
    Layer   *layers;
    LmMat    embed, head;                    /* head == embed when tied */
    const float *final_norm;

    /* state */
    float  **conv;                           /* per layer, NULL for full layers */
    float  **S;
    float  **kc, **vc;                       /* per layer, NULL for linear layers */

    /* scratch */
    float *h, *n, *mix, *mlp_out, *scratch, *mlp_scratch;
    LmLinDims ld;
    LmAttnDims ad;
};

#define BIND_MAT(var, rows, cols, ...) do { char nm_[512]; snprintf(nm_, sizeof nm_, __VA_ARGS__); \
    if (lm_bind_mat(&m->b, nm_, (rows), (cols), &(var), err, errlen) != 0) goto fail; } while (0)
#define BIND_VEC(var, numel, ndim, d0, d1, ...) do { char nm_[512]; snprintf(nm_, sizeof nm_, __VA_ARGS__); \
    (var) = lm_bind_vec(&m->b, nm_, (numel), (ndim), (d0), (d1), err, errlen); if (!(var)) goto fail; } while (0)

#define PFX "model.language_model."

void lm_close(LmModel *m)
{
    if (!m) return;
    if (m->gemma) { lm_gemma_close(m->gemma); free(m); return; }
    const int L = m->cfg.layers;
    for (int i = 0; i < L && m->conv; i++) { free(m->conv[i]); free(m->S[i]); free(m->kc[i]); free(m->vc[i]); }
    free(m->conv); free(m->S); free(m->kc); free(m->vc);
    free(m->layers);
    free(m->h); free(m->n); free(m->mix); free(m->mlp_out); free(m->scratch); free(m->mlp_scratch);
    lm_bind_free(&m->b);
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
    const int fast = opts ? opts->q8_int8_acts : 0;
    if (fast < 0 || fast > 1) { lm_fail(err, errlen, "q8_int8_acts must be 0 or 1"); free(m); return NULL; }
    if (fast && !weights) { lm_fail(err, errlen, "the int8 fast path needs a q8 weights file (--weights)"); free(m); return NULL; }
#ifdef _OPENMP
    if (opts && opts->threads > 0) omp_set_num_threads(opts->threads);
#endif

    /* family dispatch on config.json model_type; anything that is not a string falls
     * through to the Qwen reader, which reports the missing or malformed field */
    {
        char mt[64] = "";
        if (lm_config_model_type(model_dir, mt, sizeof mt) == 0) {
            if (!strcmp(mt, "gemma4")) {
                m->gemma = lm_gemma_open(model_dir, weights, ctx, fast, err, errlen);
                if (!m->gemma) { free(m); return NULL; }
                m->ctx = ctx;
                return m;
            }
            if (strcmp(mt, "qwen3_5") != 0) {
                lm_fail(err, errlen, "model_type \"%s\" is not supported (supported: qwen3_5, gemma4)", mt);
                free(m);
                return NULL;
            }
        }
    }

    if (lm_config_load(model_dir, &m->cfg, err, errlen) != 0) { free(m); return NULL; }
    const LmCfg *c = &m->cfg;

    if (lm_bind_open(&m->b, weights ? weights : model_dir, weights != NULL, err, errlen) != 0) goto fail;
    m->b.fast = fast;

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
                const LmTensor *t = lm_st_find(&m->b.st, nm);
                if (!t) { lm_fail(err, errlen, "tensor %s is missing from the weights", nm); goto fail; }
                if (t->ndim != 3 || t->shape[0] != c->conv_dim || t->shape[1] != 1 || t->shape[2] != c->conv_k) {
                    lm_fail(err, errlen, "tensor %s must have shape [%d,1,%d]", nm, c->conv_dim, c->conv_k);
                    goto fail;
                }
                L->lin.conv_w = lm_bind_vec(&m->b, nm, (int64_t)c->conv_dim * c->conv_k, 3, c->conv_dim, 1, err, errlen);
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

int  lm_n_vocab(const LmModel *m) { return m->gemma ? lm_gemma_n_vocab(m->gemma) : m->cfg.vocab; }
int  lm_n_ctx(const LmModel *m)   { return m->gemma ? lm_gemma_n_ctx(m->gemma) : m->ctx; }
int  lm_pos(const LmModel *m)     { return m->gemma ? lm_gemma_pos(m->gemma) : m->pos; }

void lm_reset(LmModel *m)
{
    if (m->gemma) { lm_gemma_reset(m->gemma); return; }
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
    if (m->gemma) return lm_gemma_feed(m->gemma, token, logits);
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
