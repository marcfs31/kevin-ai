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
 * f32 KV cache [ctx][kv_heads*head_dim] for K and for V.
 *
 * MTP HEAD (LmOpts.mtp; the maths in tools/ref_qwen35.py mtp_step). Its entry at position p is
 * computed from the final normed hidden h_p and the token at p+1, so after the token at pos-1
 * is fed the head's KV cache holds entries 0..pos-2 and hlast holds h_{pos-1}: every fed chunk
 * also runs the head over the entries its tokens complete. lm_draft computes entry pos-1 from
 * the next token (writing a slot the next feed rewrites with the same values) and its logits.
 * lm_verify runs a chunk with logits for every token and keeps, per linear layer, the state
 * after each token but the last (the attention caches need no copy: a rolled-back position is
 * just rewritten later), so lm_accept can roll a rejected draft back. */
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
#include "model/lm_ctx.h"
#include "model/lm_gemma.h"
#include "model/lm_layers.h"
#include "model/lm_times.h"

typedef struct {
    int          full;
    const float *in_norm, *post_norm;        /* [hidden] */
    LmMlpW       mlp;
    LmLinW       lin;
    LmAttnW      attn;
} Layer;

struct LmModel {
    unsigned long id;                        /* unique per lm_open: snapshots name their model by it */
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

    /* MTP head */
    int      mtp;
    Layer    mtpL;
    LmMat    mtp_fc;                         /* [hidden][2*hidden] */
    const float *mtp_pre_e, *mtp_pre_h, *mtp_norm;
    float   *mkc, *mvc;                      /* its KV cache [ctx][kvd] */
    float   *hlast;                          /* final normed hidden of the last fed position */
    float   *mx, *mn, *min2;                 /* head batch: x, normed x [CHUNK][hidden], fc input [CHUNK][2*hidden] */
    float  **ck;                             /* per linear layer: lm_verify's state after each token but the last */
    float   *vh;                             /* [LM_VERIFY_MAX][hidden]: lm_verify's normed hidden per token */
    int      vstart, vn;                     /* the last lm_verify: start position and tokens (0 = none) */

    /* scratch */
    float *h, *n, *mix, *mlp_out, *scratch, *mlp_scratch;
    float *bh, *bn, *bmix, *bscratch;        /* prefill: LM_PREFILL_CHUNK tokens of hidden, normed, mixer output; layer scratch */
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
    free(m->bh); free(m->bn); free(m->bmix); free(m->bscratch);
    for (int i = 0; i < L && m->ck; i++) free(m->ck[i]);
    free(m->ck); free(m->mkc); free(m->mvc); free(m->hlast); free(m->mx); free(m->mn); free(m->min2); free(m->vh);
    lm_bind_free(&m->b);
    lm_config_free(&m->cfg);
    free(m);
}

/* The MTP head's weights (shaped like a full-attention layer plus fc and three norms) and its
 * buffers. */
static int bind_mtp(LmModel *m, int ctx, char *err, size_t errlen)
{
    const LmCfg *c = &m->cfg;
    if (!lm_st_find(&m->b.st, "mtp.fc.weight")) {
        lm_fail(err, errlen, "the weights have no MTP head (mtp.*): a q8 file needs re-quantising with this lm-quant");
        return -1;
    }
    Layer *L = &m->mtpL;
    const int qd = c->heads * c->head_dim, kvd = c->kv_heads * c->head_dim, H = c->hidden;
    L->full = 1;
    BIND_MAT(m->mtp_fc, H, 2 * H, "mtp.fc.weight");
    BIND_VEC(m->mtp_pre_e, H, 1, H, 0, "mtp.pre_fc_norm_embedding.weight");
    BIND_VEC(m->mtp_pre_h, H, 1, H, 0, "mtp.pre_fc_norm_hidden.weight");
    BIND_VEC(m->mtp_norm, H, 1, H, 0, "mtp.norm.weight");
    BIND_VEC(L->in_norm, H, 1, H, 0, "mtp.layers.0.input_layernorm.weight");
    BIND_VEC(L->post_norm, H, 1, H, 0, "mtp.layers.0.post_attention_layernorm.weight");
    L->mlp.inter = c->inter;
    BIND_MAT(L->mlp.gate, c->inter, H, "mtp.layers.0.mlp.gate_proj.weight");
    BIND_MAT(L->mlp.up, c->inter, H, "mtp.layers.0.mlp.up_proj.weight");
    BIND_MAT(L->mlp.down, H, c->inter, "mtp.layers.0.mlp.down_proj.weight");
    BIND_MAT(L->attn.q, 2 * qd, H, "mtp.layers.0.self_attn.q_proj.weight");
    BIND_MAT(L->attn.k, kvd, H, "mtp.layers.0.self_attn.k_proj.weight");
    BIND_MAT(L->attn.v, kvd, H, "mtp.layers.0.self_attn.v_proj.weight");
    BIND_MAT(L->attn.o, H, qd, "mtp.layers.0.self_attn.o_proj.weight");
    BIND_VEC(L->attn.q_norm, c->head_dim, 1, c->head_dim, 0, "mtp.layers.0.self_attn.q_norm.weight");
    BIND_VEC(L->attn.k_norm, c->head_dim, 1, c->head_dim, 0, "mtp.layers.0.self_attn.k_norm.weight");

    m->mkc = (float *)calloc((size_t)ctx * kvd, sizeof(float));
    m->mvc = (float *)calloc((size_t)ctx * kvd, sizeof(float));
    m->hlast = (float *)calloc((size_t)H, sizeof(float));
    m->mx = (float *)calloc((size_t)LM_PREFILL_CHUNK * H, sizeof(float));
    m->mn = (float *)calloc((size_t)LM_PREFILL_CHUNK * H, sizeof(float));
    m->min2 = (float *)calloc((size_t)LM_PREFILL_CHUNK * 2 * H, sizeof(float));
    m->vh = (float *)calloc((size_t)LM_VERIFY_MAX * H, sizeof(float));
    m->ck = (float **)calloc((size_t)c->layers, sizeof(float *));
    if (!m->mkc || !m->mvc || !m->hlast || !m->mx || !m->mn || !m->min2 || !m->vh || !m->ck) {
        lm_fail(err, errlen, "out of memory allocating the MTP head's state");
        return -1;
    }
    for (int i = 0; i < c->layers; i++) {
        if (m->layers[i].full) continue;
        m->ck[i] = (float *)malloc((size_t)(LM_VERIFY_MAX - 1) * lm_linear_state_floats(&m->ld) * sizeof(float));
        if (!m->ck[i]) { lm_fail(err, errlen, "out of memory allocating the verify checkpoints"); return -1; }
    }
    m->mtp = 1;
    return 0;
fail:
    return -1;
}

/* Bytes lm_open allocates for a window of ctx positions: it mirrors the allocations below and
 * in bind_mtp, so a change there changes this too. */
typedef struct { const LmCfg *c; const LmLinDims *ld; const LmAttnDims *ad; int mtp; } QwenState;

static size_t qwen_state_bytes(const void *arg, int ctx)
{
    const QwenState *q = (const QwenState *)arg;
    const LmCfg *c = q->c;
    const size_t kvd = (size_t)c->kv_heads * c->head_dim, H = (size_t)c->hidden, T = LM_PREFILL_CHUNK;
    size_t nfull = 0;
    for (int i = 0; i < c->layers; i++) nfull += c->layer_type[i] == LM_LAYER_FULL;
    const size_t nlin = (size_t)c->layers - nfull;
    size_t f = nfull * 2 * (size_t)ctx * kvd
             + nlin * ((size_t)c->conv_dim * (size_t)(c->conv_k - 1) + 1 + (size_t)c->lv_heads * c->lk_dim * c->lv_dim);
    size_t sc = lm_linear_scratch(q->ld);
    const size_t sa = lm_attn_scratch(q->ad, ctx);
    f += (sa > sc ? sa : sc) + 4 * H + lm_mlp_scratch(c->inter);
    size_t bs = lm_linear_batch_scratch(q->ld, LM_PREFILL_CHUNK);
    const size_t ba = lm_attn_batch_scratch(q->ad, ctx, LM_PREFILL_CHUNK), bm = lm_mlp_batch_scratch(c->inter, LM_PREFILL_CHUNK);
    if (ba > bs) bs = ba;
    if (bm > bs) bs = bm;
    f += 3 * T * H + bs;
    if (q->mtp)
        f += 2 * (size_t)ctx * kvd + H + 4 * T * H + (size_t)LM_VERIFY_MAX * H
           + nlin * (size_t)(LM_VERIFY_MAX - 1) * lm_linear_state_floats(q->ld);
    return f * sizeof(float);
}

LmModel *lm_open(const char *model_dir, const LmOpts *opts, char *err, size_t errlen)
{
    LmModel *m = (LmModel *)calloc(1, sizeof *m);
    static unsigned long next_id;
    if (m) m->id = ++next_id;
    if (!m) { lm_fail(err, errlen, "out of memory"); return NULL; }
    const char *weights = opts ? opts->weights : NULL;
    int ctx = opts ? opts->ctx : 0;
    if (ctx < 0) { lm_fail(err, errlen, "ctx must not be negative"); free(m); return NULL; }
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
                if (opts && opts->mtp) { lm_fail(err, errlen, "Gemma 4 has no MTP head; mtp is for Qwen3.5"); free(m); return NULL; }
                m->gemma = lm_gemma_open(model_dir, weights, ctx, fast, err, errlen);
                if (!m->gemma) { free(m); return NULL; }
                m->ctx = lm_gemma_n_ctx(m->gemma);
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
    {
        const QwenState qs = { c, &m->ld, &m->ad, opts && opts->mtp };
        uint64_t weights = m->b.bytes;
        if (qs.mtp)   /* bind_mtp binds them after the state is sized */
            for (int i = 0; i < m->b.st.nt; i++)
                if (!strncmp(m->b.st.t[i].name, "mtp.", 4)) weights += (uint64_t)m->b.st.t[i].nbytes;
        if (lm_ctx_choose_here(ctx, c->max_pos, weights, qwen_state_bytes, &qs, &ctx, err, errlen) != 0) goto fail;
        m->ctx = ctx;
    }
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
    if (opts && opts->mtp && bind_mtp(m, ctx, err, errlen) != 0) goto fail;
    {   /* prefill buffers: one chunk of tokens, sized for the largest of the three sub-layers */
        size_t bs = lm_linear_batch_scratch(&m->ld, LM_PREFILL_CHUNK);
        const size_t ba = lm_attn_batch_scratch(&m->ad, ctx, LM_PREFILL_CHUNK);
        const size_t bm = lm_mlp_batch_scratch(c->inter, LM_PREFILL_CHUNK);
        if (ba > bs) bs = ba;
        if (bm > bs) bs = bm;
        const size_t bt = (size_t)LM_PREFILL_CHUNK * c->hidden;
        m->bh = (float *)calloc(bt, sizeof(float));
        m->bn = (float *)calloc(bt, sizeof(float));
        m->bmix = (float *)calloc(bt, sizeof(float));
        m->bscratch = (float *)calloc(bs, sizeof(float));
        if (!m->bh || !m->bn || !m->bmix || !m->bscratch) {
            lm_fail(err, errlen, "out of memory allocating prefill buffers");
            goto fail;
        }
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
    if (m->mtp) {
        memset(m->mkc, 0, (size_t)m->ctx * kvd * sizeof(float));
        memset(m->mvc, 0, (size_t)m->ctx * kvd * sizeof(float));
        memset(m->hlast, 0, (size_t)c->hidden * sizeof(float));
    }
    m->pos = 0;
    m->vn = 0;
}

/* ---- snapshots ------------------------------------------------------------------- */

struct LmSnapshot { unsigned long model; int pos; size_t n; float *data; };

/* The state regions at position pos, in a fixed order (lm_gemma_state_regions for Gemma). */
static void state_regions(LmModel *m, int pos, LmRegionFn fn, void *user)
{
    if (m->gemma) { lm_gemma_state_regions(m->gemma, pos, fn, user); return; }
    const LmCfg *c = &m->cfg;
    const size_t kvd = (size_t)c->kv_heads * c->head_dim;
    for (int i = 0; i < c->layers; i++) {
        if (m->conv[i]) fn(user, m->conv[i], (size_t)c->conv_dim * (size_t)(c->conv_k - 1) + 1);
        if (m->S[i]) fn(user, m->S[i], (size_t)c->lv_heads * c->lk_dim * c->lv_dim);
        if (m->kc[i]) fn(user, m->kc[i], (size_t)pos * kvd);
        if (m->vc[i]) fn(user, m->vc[i], (size_t)pos * kvd);
    }
    if (m->mtp) {
        fn(user, m->mkc, (size_t)pos * kvd);
        fn(user, m->mvc, (size_t)pos * kvd);
        fn(user, m->hlast, (size_t)c->hidden);
    }
}

typedef struct { float *buf; size_t off; int mode; } Walk; /* mode 0 count, 1 save, 2 restore */

static void walk(void *u, float *p, size_t n)
{
    Walk *w = (Walk *)u;
    if (w->mode == 1) memcpy(w->buf + w->off, p, n * sizeof(float));
    else if (w->mode == 2) memcpy(p, w->buf + w->off, n * sizeof(float));
    w->off += n;
}

LmSnapshot *lm_snapshot(const LmModel *cm)
{
    LmModel *m = (LmModel *)cm; /* the walk only reads when saving */
    const int pos = lm_pos(m);
    Walk w = { NULL, 0, 0 };
    state_regions(m, pos, walk, &w);
    LmSnapshot *s = (LmSnapshot *)calloc(1, sizeof *s);
    if (!s) return NULL;
    s->data = (float *)malloc((w.off ? w.off : 1) * sizeof(float));
    if (!s->data) { free(s); return NULL; }
    s->model = m->id;
    s->pos = pos;
    s->n = w.off;
    Walk save = { s->data, 0, 1 };
    state_regions(m, pos, walk, &save);
    return s;
}

int lm_restore(LmModel *m, const LmSnapshot *s)
{
    if (!s || s->model != m->id) return -1;
    Walk w = { s->data, 0, 2 };
    state_regions(m, s->pos, walk, &w);
    if (m->gemma) lm_gemma_set_pos(m->gemma, s->pos);
    else { m->pos = s->pos; m->vn = 0; }
    return LM_OK;
}

int    lm_snapshot_pos(const LmSnapshot *s)   { return s->pos; }
size_t lm_snapshot_bytes(const LmSnapshot *s) { return s->n * sizeof(float); }
void   lm_snapshot_free(LmSnapshot *s)        { if (s) { free(s->data); free(s); } }

/* The MTP head over n entries starting at position q0: entry j from hidden hid[j] and token
 * tok[j]. Writes the head's KV slots q0..q0+n-1; logits (may be NULL) after the last entry. */
static void mtp_run(LmModel *m, const float *const *hid, const int *tok, int q0, int n, float *logits)
{
    const LmCfg *c = &m->cfg;
    const int H = c->hidden;
    const Layer *L = &m->mtpL;
    for (int j = 0; j < n; j++) {
        float *e = m->min2 + (size_t)j * 2 * H;
        lm_mat_row(&m->embed, tok[j], e);
        lm_rmsnorm(e, e, m->mtp_pre_e, H, c->eps);
        lm_rmsnorm(e + H, hid[j], m->mtp_pre_h, H, c->eps);
    }
    lm_mat_mul_batch(&m->mtp_fc, m->mx, m->min2, n);
    for (int j = 0; j < n; j++) lm_rmsnorm(m->mn + (size_t)j * H, m->mx + (size_t)j * H, L->in_norm, H, c->eps);
    lm_attn_mixer_batch(m->bmix, m->mn, &L->attn, &m->ad, m->mkc, m->mvc, q0, m->ctx, m->bscratch, n);
    for (size_t j = 0; j < (size_t)n * H; j++) m->mx[j] += m->bmix[j];
    for (int j = 0; j < n; j++) lm_rmsnorm(m->mn + (size_t)j * H, m->mx + (size_t)j * H, L->post_norm, H, c->eps);
    lm_mlp_batch(m->bmix, m->mn, &L->mlp, m->bscratch, n);
    for (size_t j = 0; j < (size_t)n * H; j++) m->mx[j] += m->bmix[j];
    if (logits) {
        lm_rmsnorm(m->n, m->mx + (size_t)(n - 1) * H, m->mtp_norm, H, c->eps);
        lm_mat_mul(&m->head, logits, m->n);
    }
}

static int feed(LmModel *m, int token, float *logits)
{
    if (m->gemma) return lm_gemma_feed(m->gemma, token, logits);
    const LmCfg *c = &m->cfg;
    if (token < 0 || token >= c->vocab) return LM_ERR_TOKEN;
    if (m->pos >= m->ctx) return LM_ERR_CTX;
    m->vn = 0;

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
    if (m->mtp && m->pos > 0) {   /* the head's entry pos-1 is complete now that its next token is known */
        const float *hid = m->hlast;
        mtp_run(m, &hid, &token, m->pos - 1, 1, NULL);
    }
    m->pos++;
    if (logits || m->mtp) lm_rmsnorm(m->n, m->h, m->final_norm, c->hidden, c->eps);
    if (m->mtp) memcpy(m->hlast, m->n, (size_t)c->hidden * sizeof(float));
    if (logits) lm_mat_mul(&m->head, logits, m->n);
    return LM_OK;
}

/* The checks every batched entry point shares: LM_OK when the n tokens can be fed. */
static int batch_ok(const LmModel *m, const int *tokens, int n)
{
    if (n < 0 || (n > 0 && !tokens)) return LM_ERR_TOKEN;
    const int V = lm_n_vocab(m);
    for (int i = 0; i < n; i++)
        if (tokens[i] < 0 || tokens[i] >= V) return LM_ERR_TOKEN;
    if (n > lm_n_ctx(m) - lm_pos(m)) return LM_ERR_CTX;
    return LM_OK;
}

/* Feed n checked tokens in chunks. rows: logits for every token (n rows) when all_rows, else
 * after the last one only (may be NULL). ckpt: lm_verify's state checkpoints (one chunk). */
static void run_chunks(LmModel *m, const int *tokens, int n, float *rows, int all_rows, int ckpt)
{
    const LmCfg *c = &m->cfg;
    const int H = c->hidden, V = c->vocab;
    for (int s = 0; s < n; s += LM_PREFILL_CHUNK) {
        const int T = n - s < LM_PREFILL_CHUNK ? n - s : LM_PREFILL_CHUNK;
        const int pos0 = m->pos, last = s + T == n;
        for (int t = 0; t < T; t++) lm_mat_row(&m->embed, tokens[s + t], m->bh + (size_t)t * H);
        for (int i = 0; i < c->layers; i++) {
            const Layer *L = &m->layers[i];
            for (int t = 0; t < T; t++) lm_rmsnorm(m->bn + (size_t)t * H, m->bh + (size_t)t * H, L->in_norm, H, c->eps);
            if (L->full)
                lm_attn_mixer_batch(m->bmix, m->bn, &L->attn, &m->ad, m->kc[i], m->vc[i], m->pos, m->ctx, m->bscratch, T);
            else
                lm_linear_mixer_batch(m->bmix, m->bn, &L->lin, &m->ld, m->conv[i], m->S[i], m->bscratch, T,
                                      ckpt ? m->ck[i] : NULL);
            for (size_t j = 0; j < (size_t)T * H; j++) m->bh[j] += m->bmix[j];
            for (int t = 0; t < T; t++) lm_rmsnorm(m->bn + (size_t)t * H, m->bh + (size_t)t * H, L->post_norm, H, c->eps);
            lm_mlp_batch(m->bmix, m->bn, &L->mlp, m->bscratch, T);
            for (size_t j = 0; j < (size_t)T * H; j++) m->bh[j] += m->bmix[j];
        }
        m->pos += T;
        /* final normed hidden, into bn: every token for the head or all_rows, else the last */
        const int from = m->mtp || all_rows ? 0 : T - 1;
        if (m->mtp || all_rows || (last && rows))
            for (int t = from; t < T; t++) lm_rmsnorm(m->bn + (size_t)t * H, m->bh + (size_t)t * H, m->final_norm, H, c->eps);
        if (all_rows)
            for (int t = 0; t < T; t++) lm_mat_mul(&m->head, rows + (size_t)(s + t) * V, m->bn + (size_t)t * H);
        else if (last && rows)
            lm_mat_mul(&m->head, rows, m->bn + (size_t)(T - 1) * H);
        if (m->mtp) {
            /* entries pos0-1 .. pos0+T-2: hidden of the previous position, token of this one */
            const float *hid[LM_PREFILL_CHUNK];
            const int q0 = pos0 > 0 ? pos0 - 1 : 0, ne = pos0 + T - 1 - q0;
            for (int j = 0; j < ne; j++) {
                const int q = q0 + j;
                hid[j] = q == pos0 - 1 ? m->hlast : m->bn + (size_t)(q - pos0) * H;
            }
            if (ne > 0) mtp_run(m, hid, tokens + s + (q0 + 1 - pos0), q0, ne, NULL);
            memcpy(m->hlast, m->bn + (size_t)(T - 1) * H, (size_t)H * sizeof(float));
            if (ckpt) memcpy(m->vh, m->bn, (size_t)T * H * sizeof(float));
        }
    }
}

static int prefill(LmModel *m, const int *tokens, int n, float *logits)
{
    const int rc = batch_ok(m, tokens, n);
    if (rc != LM_OK || n == 0) return rc;
    if (m->gemma) return lm_gemma_prefill(m->gemma, tokens, n, logits);
    m->vn = 0;
    run_chunks(m, tokens, n, logits, 0, 0);
    return LM_OK;
}

int lm_draft(LmModel *m, int token, float *logits)
{
    if (m->gemma || !m->mtp || m->pos < 1) return LM_ERR_STATE;
    if (token < 0 || token >= m->cfg.vocab) return LM_ERR_TOKEN;
    LM_TIMES_T0(t0);
    const float *hid = m->hlast;
    mtp_run(m, &hid, &token, m->pos - 1, 1, logits);
    LM_TIMES_ADD(total, t0);
    return LM_OK;
}

int lm_verify(LmModel *m, const int *tokens, int n, float *rows)
{
    if (m->gemma || !m->mtp || n < 1 || n > LM_VERIFY_MAX || !rows) return LM_ERR_STATE;
    const int rc = batch_ok(m, tokens, n);
    if (rc != LM_OK) return rc;
    LM_TIMES_T0(t0);
    m->vstart = m->pos;
    run_chunks(m, tokens, n, rows, 1, 1);
    m->vn = n;
    LM_TIMES_ADD(total, t0);
    return LM_OK;
}

int lm_accept(LmModel *m, int k)
{
    if (m->gemma || !m->mtp || m->vn < 1 || k < 1 || k > m->vn) return LM_ERR_STATE;
    const LmCfg *c = &m->cfg;
    if (k < m->vn) {
        const size_t nst = lm_linear_state_floats(&m->ld);
        const size_t nc = (size_t)c->conv_dim * (size_t)(c->conv_k - 1);
        for (int i = 0; i < c->layers; i++) {
            if (!m->ck[i]) continue;
            const float *ck = m->ck[i] + (size_t)(k - 1) * nst;
            memcpy(m->conv[i], ck, nc * sizeof(float));
            memcpy(m->S[i], ck + nc, (nst - nc) * sizeof(float));
        }
        memcpy(m->hlast, m->vh + (size_t)(k - 1) * c->hidden, (size_t)c->hidden * sizeof(float));
        m->pos = m->vstart + k;
    }
    m->vn = 0;
    return LM_OK;
}

int lm_greedy_check(LmModel *m, int token, int draft, float *rows, float *logits, int *kept)
{
    *kept = 0;
    const int pair[2] = { token, draft };
    int r = lm_verify(m, pair, 2, rows);
    if (r != LM_OK) return r;
    const int V = m->cfg.vocab;
    int best = 0;
    for (int i = 1; i < V; i++) if (rows[i] > rows[best]) best = i;
    const int k = best == draft ? 2 : 1;
    r = lm_accept(m, k);
    if (r != LM_OK) return r;
    memcpy(logits, rows + (size_t)(k - 1) * V, (size_t)V * sizeof(float));
    *kept = k;
    return LM_OK;
}

int lm_feed(LmModel *m, int token, float *logits)
{
    LM_TIMES_T0(t0);
    const int r = feed(m, token, logits);
    LM_TIMES_ADD(total, t0);
    return r;
}

int lm_prefill(LmModel *m, const int *tokens, int n, float *logits)
{
    LM_TIMES_T0(t0);
    const int r = prefill(m, tokens, n, logits);
    LM_TIMES_ADD(total, t0);
    return r;
}
