/* lm_gemma.c - see lm_gemma.h. Original code. */
#include "io/lm_portable_io.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "lm/lm.h"
#include "core/lm_bf16.h"
#include "core/lm_kernels.h"
#include "core/lm_ops.h"
#include "io/lm_io.h"
#include "model/lm_bind.h"
#include "model/lm_gemma.h"
#include "model/lm_layers.h"

#define PFX "model.language_model."

typedef struct {
    int          full, reuse, producer, hd;
    const float *in_norm, *post_attn_norm, *pre_ff_norm, *post_ff_norm, *post_ple_norm;   /* [hidden] */
    const float *q_norm, *k_norm;            /* [hd]; k_norm NULL on a reuse layer */
    float        layer_scalar;
    LmMat        q, k, v, o;                 /* k, v unbound on a reuse layer */
    LmMat        gate, up, down;
    LmMat        ple_gate, ple_proj;         /* [P][hidden], [hidden][P] */
} GLayer;

struct LmGemma {
    LmGemmaCfg cfg;
    int        ctx, pos;
    LmBind     b;
    GLayer    *layers;
    LmMat      embed, ple_model;             /* ple_model: per_layer_model_projection [L*P][hidden] */
    const float *final_norm, *ple_norm;      /* [hidden], [P] */
    const LmTensor *ple_t, *ple_s;           /* embed_tokens_per_layer (bf16 or I8) and its .q8scale (q8 only) */
    double    *freqs[2];                     /* [0] sliding, [1] full: head_dim/2 RoPE frequencies */
    float    **kc, **vc;                     /* per layer; NULL for reuse layers */
    float     *arena;                        /* all scratch below, one allocation */
    unsigned char *rowbuf;                   /* raw PLE row: L*P*2 bytes */
    float *h, *n, *mix, *mlp_out, *mlp_s, *q, *k, *v, *attn, *scores;
    float *ple, *ple_ctx, *ple_scales, *ple_g, *ple_d;
    float     *barena;                       /* prefill buffers for LM_PREFILL_CHUNK tokens, one allocation */
    float *bh, *bn, *bmix, *bmlp, *bq, *bk, *bv, *battn, *bple, *bple_ctx, *bple_g, *bple_d;
};

/* ------------------------------------------------------------------ sizes */

static int slots_of(const LmGemmaCfg *c, int full, int ctx)
{
    if (full) return ctx;
    return c->window < ctx ? c->window : ctx;
}

static int hd_of(const LmGemmaCfg *c, int full) { return full ? c->global_head_dim : c->head_dim; }

typedef struct { size_t kv_floats, scratch_floats, batch_floats, rowbuf_bytes; } Sizes;

static Sizes sizes(const LmGemmaCfg *c, int ctx)
{
    Sizes s = { 0, 0, 0, 0 };
    int any_full = 0, hdmax = c->head_dim;
    for (int i = 0; i < c->layers; i++) {
        if (c->full[i]) any_full = 1;
        if (hd_of(c, c->full[i]) > hdmax) hdmax = hd_of(c, c->full[i]);
        if (i < c->first_shared)
            s.kv_floats += 2 * (size_t)slots_of(c, c->full[i], ctx) * c->kv_heads * hd_of(c, c->full[i]);
    }
    const size_t qd = (size_t)c->heads * hdmax, kvd = (size_t)c->kv_heads * hdmax;
    const size_t LP = (size_t)c->layers * c->ple_dim;
    const size_t nvis = (size_t)(any_full ? ctx : slots_of(c, 0, ctx));
    s.scratch_floats = 4 * (size_t)c->hidden + 2 * (size_t)c->inter           /* h n mix mlp_out, gate+up */
                     + qd + 2 * kvd + qd + (size_t)c->heads * nvis             /* q k v attn scores */
                     + 2 * LP + LP / LM_Q8_GROUP + 1                           /* ple, ple_ctx, scales */
                     + (size_t)c->ple_dim + (size_t)c->hidden;                 /* ple_g, ple_d */
    /* one chunk of: h n mix, gate+up, q, attn, k, v, the per-layer inputs and their projection, ple_g, ple_d */
    s.batch_floats = (size_t)LM_PREFILL_CHUNK * (4 * (size_t)c->hidden + 2 * (size_t)c->inter + 2 * qd + 2 * kvd
                                                 + 2 * LP + (size_t)c->ple_dim);
    s.rowbuf_bytes = LP * 2;
    return s;
}

size_t lm_gemma_state_bytes(const LmGemmaCfg *c, int ctx)
{
    const Sizes s = sizes(c, ctx);
    return (s.kv_floats + s.scratch_floats + s.batch_floats) * sizeof(float) + s.rowbuf_bytes;
}

/* ------------------------------------------------------------------ open / close */

#define BIND_MAT(var, rows, cols, ...) do { char nm_[512]; snprintf(nm_, sizeof nm_, __VA_ARGS__); \
    if (lm_bind_mat(&g->b, nm_, (rows), (cols), &(var), err, errlen) != 0) goto fail; } while (0)
#define BIND_VEC(var, numel, ...) do { char nm_[512]; snprintf(nm_, sizeof nm_, __VA_ARGS__); \
    (var) = lm_bind_vec(&g->b, nm_, (numel), 1, (numel), 0, err, errlen); if (!(var)) goto fail; } while (0)

void lm_gemma_close(LmGemma *g)
{
    if (!g) return;
    for (int i = 0; i < g->cfg.layers && g->kc; i++) { free(g->kc[i]); free(g->vc[i]); }
    free(g->kc); free(g->vc);
    free(g->layers);
    free(g->freqs[0]); free(g->freqs[1]);
    free(g->arena); free(g->barena); free(g->rowbuf);
    lm_bind_free(&g->b);
    lm_gemma_config_free(&g->cfg);
    free(g);
}

static int bind_ple_table(LmGemma *g, char *err, size_t errlen)
{
    const LmGemmaCfg *c = &g->cfg;
    const int LP = c->layers * c->ple_dim;
    const char *name = PFX "embed_tokens_per_layer.weight";
    g->ple_t = lm_bind_need(&g->b, name, 2, c->vocab, LP, err, errlen);
    if (!g->ple_t) return -1;
    if (g->b.q8) {
        if (g->ple_t->dtype != LM_DT_I8)
            return lm_fail(err, errlen, "tensor %s has dtype %s; a q8 file stores matrices as I8", name, lm_dtype_name(g->ple_t->dtype));
        if (LP % LM_Q8_GROUP) return lm_fail(err, errlen, "tensor %s: %d inputs is not a multiple of %d", name, LP, LM_Q8_GROUP);
        g->ple_s = lm_bind_need(&g->b, PFX "embed_tokens_per_layer.weight.q8scale", 2, c->vocab, LP / LM_Q8_GROUP, err, errlen);
        if (!g->ple_s) return -1;
        if (g->ple_s->dtype != LM_DT_F32)
            return lm_fail(err, errlen, "tensor %s.q8scale has dtype %s; F32 expected", name, lm_dtype_name(g->ple_s->dtype));
    } else if (g->ple_t->dtype != LM_DT_BF16) {
        if (g->ple_t->dtype == LM_DT_I8)
            return lm_fail(err, errlen, "tensor %s is I8: this is a q8 file, pass it as the weights file instead of a bf16 checkpoint", name);
        return lm_fail(err, errlen, "tensor %s has dtype %s; BF16 expected", name, lm_dtype_name(g->ple_t->dtype));
    }
    return 0;
}

LmGemma *lm_gemma_open(const char *model_dir, const char *weights, int ctx, int fast, char *err, size_t errlen)
{
    LmGemma *g = (LmGemma *)calloc(1, sizeof *g);
    if (!g) { lm_fail(err, errlen, "out of memory"); return NULL; }
    g->ctx = ctx;
    if (lm_gemma_config_load(model_dir, &g->cfg, err, errlen) != 0) { free(g); return NULL; }
    const LmGemmaCfg *c = &g->cfg;
    const int P = c->ple_dim, LP = c->layers * P;

    if (lm_bind_open(&g->b, weights ? weights : model_dir, weights != NULL, err, errlen) != 0) goto fail;
    g->b.fast = fast;

    g->layers = (GLayer *)calloc((size_t)c->layers, sizeof(GLayer));
    if (!g->layers) { lm_fail(err, errlen, "out of memory"); goto fail; }

    BIND_MAT(g->embed, c->vocab, c->hidden, PFX "embed_tokens.weight");
    BIND_MAT(g->ple_model, LP, c->hidden, PFX "per_layer_model_projection.weight");
    BIND_VEC(g->ple_norm, P, PFX "per_layer_projection_norm.weight");
    BIND_VEC(g->final_norm, c->hidden, PFX "norm.weight");
    if (bind_ple_table(g, err, errlen) != 0) goto fail;

    for (int i = 0; i < c->layers; i++) {
        GLayer *L = &g->layers[i];
        L->full = c->full[i];
        L->producer = c->producer[i];
        L->reuse = i >= c->first_shared;
        L->hd = hd_of(c, L->full);
        const int hd = L->hd, qd = c->heads * hd, kvd = c->kv_heads * hd;
        BIND_VEC(L->in_norm, c->hidden, PFX "layers.%d.input_layernorm.weight", i);
        BIND_VEC(L->post_attn_norm, c->hidden, PFX "layers.%d.post_attention_layernorm.weight", i);
        BIND_VEC(L->pre_ff_norm, c->hidden, PFX "layers.%d.pre_feedforward_layernorm.weight", i);
        BIND_VEC(L->post_ff_norm, c->hidden, PFX "layers.%d.post_feedforward_layernorm.weight", i);
        BIND_VEC(L->post_ple_norm, c->hidden, PFX "layers.%d.post_per_layer_input_norm.weight", i);
        {
            const float *ls;
            BIND_VEC(ls, 1, PFX "layers.%d.layer_scalar", i);
            L->layer_scalar = ls[0];
        }
        BIND_MAT(L->gate, c->inter, c->hidden, PFX "layers.%d.mlp.gate_proj.weight", i);
        BIND_MAT(L->up, c->inter, c->hidden, PFX "layers.%d.mlp.up_proj.weight", i);
        BIND_MAT(L->down, c->hidden, c->inter, PFX "layers.%d.mlp.down_proj.weight", i);
        BIND_MAT(L->ple_gate, P, c->hidden, PFX "layers.%d.per_layer_input_gate.weight", i);
        BIND_MAT(L->ple_proj, c->hidden, P, PFX "layers.%d.per_layer_projection.weight", i);
        BIND_MAT(L->q, qd, c->hidden, PFX "layers.%d.self_attn.q_proj.weight", i);
        BIND_MAT(L->o, c->hidden, qd, PFX "layers.%d.self_attn.o_proj.weight", i);
        BIND_VEC(L->q_norm, hd, PFX "layers.%d.self_attn.q_norm.weight", i);
        if (!L->reuse) {   /* a reuse layer's k_proj, v_proj and k_norm exist in the file and are ignored */
            BIND_MAT(L->k, kvd, c->hidden, PFX "layers.%d.self_attn.k_proj.weight", i);
            BIND_MAT(L->v, kvd, c->hidden, PFX "layers.%d.self_attn.v_proj.weight", i);
            BIND_VEC(L->k_norm, hd, PFX "layers.%d.self_attn.k_norm.weight", i);
        }
    }

    /* RoPE frequencies per layer type */
    for (int s = 0; s < 2; s++) {
        if (!(c->theta[s] > 0)) continue;                     /* the type does not occur */
        const int hd = hd_of(c, s);
        g->freqs[s] = (double *)malloc((size_t)(hd / 2) * sizeof(double));
        if (!g->freqs[s]) { lm_fail(err, errlen, "out of memory"); goto fail; }
        if (lm_rope_freqs(g->freqs[s], hd, c->theta[s], c->partial[s]) != 0) {
            lm_fail(err, errlen, "rope parameters of %s attention do not fit head size %d", s ? "full" : "sliding", hd);
            goto fail;
        }
        for (int i = 0; i < hd / 2; i++) g->freqs[s][i] /= c->factor[s];
    }

    /* state and scratch */
    const Sizes sz = sizes(c, ctx);
    g->kc = (float **)calloc((size_t)c->layers, sizeof(float *));
    g->vc = (float **)calloc((size_t)c->layers, sizeof(float *));
    if (!g->kc || !g->vc) { lm_fail(err, errlen, "out of memory"); goto fail; }
    for (int i = 0; i < c->first_shared; i++) {
        const size_t n = (size_t)slots_of(c, c->full[i], ctx) * c->kv_heads * hd_of(c, c->full[i]);
        g->kc[i] = (float *)calloc(n, sizeof(float));
        g->vc[i] = (float *)calloc(n, sizeof(float));
        if (!g->kc[i] || !g->vc[i]) { lm_fail(err, errlen, "out of memory allocating the KV cache for ctx %d", ctx); goto fail; }
    }
    g->arena = (float *)calloc(sz.scratch_floats, sizeof(float));
    g->rowbuf = (unsigned char *)calloc(sz.rowbuf_bytes, 1);
    if (!g->arena || !g->rowbuf) { lm_fail(err, errlen, "out of memory allocating scratch"); goto fail; }
    {
        int hdmax = c->head_dim, any_full = 0;
        for (int i = 0; i < c->layers; i++) { if (c->full[i]) any_full = 1; if (hd_of(c, c->full[i]) > hdmax) hdmax = hd_of(c, c->full[i]); }
        const size_t qd = (size_t)c->heads * hdmax, kvd = (size_t)c->kv_heads * hdmax;
        float *p = g->arena;
        g->h = p;       p += c->hidden;
        g->n = p;       p += c->hidden;
        g->mix = p;     p += c->hidden;
        g->mlp_out = p; p += c->hidden;
        g->mlp_s = p;   p += 2 * (size_t)c->inter;
        g->q = p;       p += qd;
        g->k = p;       p += kvd;
        g->v = p;       p += kvd;
        g->attn = p;    p += qd;
        g->scores = p;  p += (size_t)c->heads * (size_t)(any_full ? ctx : slots_of(c, 0, ctx));
        g->ple = p;     p += LP;
        g->ple_ctx = p; p += LP;
        g->ple_scales = p; p += LP / LM_Q8_GROUP + 1;
        g->ple_g = p;   p += P;
        g->ple_d = p;   p += c->hidden;
        if ((size_t)(p - g->arena) != sz.scratch_floats) { lm_fail(err, errlen, "internal error: scratch layout"); goto fail; }

        const size_t T = LM_PREFILL_CHUNK;
        g->barena = (float *)calloc(sz.batch_floats, sizeof(float));
        if (!g->barena) { lm_fail(err, errlen, "out of memory allocating prefill buffers"); goto fail; }
        p = g->barena;
        g->bh = p;      p += T * c->hidden;
        g->bn = p;      p += T * c->hidden;
        g->bmix = p;    p += T * c->hidden;
        g->bmlp = p;    p += T * 2 * (size_t)c->inter;
        g->bq = p;      p += T * qd;
        g->battn = p;   p += T * qd;
        g->bk = p;      p += T * kvd;
        g->bv = p;      p += T * kvd;
        g->bple = p;    p += T * LP;
        g->bple_ctx = p; p += T * LP;
        g->bple_g = p;  p += T * P;
        g->bple_d = p;  p += T * c->hidden;
        if ((size_t)(p - g->barena) != sz.batch_floats) { lm_fail(err, errlen, "internal error: prefill buffer layout"); goto fail; }
    }
    return g;
fail:
    lm_gemma_close(g);
    return NULL;
}

int  lm_gemma_n_vocab(const LmGemma *g) { return g->cfg.vocab; }
int  lm_gemma_n_ctx(const LmGemma *g)   { return g->ctx; }
int  lm_gemma_pos(const LmGemma *g)     { return g->pos; }

void lm_gemma_reset(LmGemma *g)
{
    const LmGemmaCfg *c = &g->cfg;
    for (int i = 0; i < c->first_shared; i++) {
        const size_t n = (size_t)slots_of(c, c->full[i], g->ctx) * c->kv_heads * hd_of(c, c->full[i]);
        memset(g->kc[i], 0, n * sizeof(float));
        memset(g->vc[i], 0, n * sizeof(float));
    }
    g->pos = 0;
}

/* ------------------------------------------------------------------ forward */

/* ple[l*P + i] for the token whose scaled embedding is e:
 * (rms(proj(e) * hidden^-1/2, norm) + table[token] * bf16(sqrt(P))) * 2^-1/2.
 * The table row is read from the mapped file, one row, never the table. */
/* tok[LP] = the per-layer table row of `token` as floats (bf16 widened, or q8 dequantised). */
static int ple_read_row(LmGemma *g, int token, float *tok, char *err, size_t errlen)
{
    const LmGemmaCfg *c = &g->cfg;
    const int P = c->ple_dim, LP = c->layers * P;
    if (!g->b.q8) {
        if (lm_st_read_slice(&g->b.st, g->ple_t, (int64_t)token * LP * 2, (int64_t)LP * 2, g->rowbuf, err, errlen) != 0) return -1;
        const uint16_t *r = (const uint16_t *)g->rowbuf;     /* rowbuf is malloc-aligned */
        for (int i = 0; i < LP; i++) tok[i] = lm_bf16_to_f32(r[i]);
    } else {
        const int ng = LP / LM_Q8_GROUP;
        if (lm_st_read_slice(&g->b.st, g->ple_t, (int64_t)token * LP, (int64_t)LP, g->rowbuf, err, errlen) != 0) return -1;
        if (lm_st_read_slice(&g->b.st, g->ple_s, (int64_t)token * ng * 4, (int64_t)ng * 4, g->ple_scales, err, errlen) != 0) return -1;
        const int8_t *q = (const int8_t *)g->rowbuf;
        for (int i = 0; i < LP; i++) tok[i] = (float)q[i] * g->ple_scales[i / LM_Q8_GROUP];
    }
    return 0;
}

/* tok[LP] (the table row) becomes the per-layer inputs, from cx[LP] = per_layer_model_projection * h. */
static void ple_combine(const LmGemma *g, float *tok, float *cx_all)
{
    const LmGemmaCfg *c = &g->cfg;
    const int P = c->ple_dim;
    const float cscale = (float)(1.0 / sqrt((double)c->hidden));
    const float half = (float)(1.0 / sqrt(2.0));
    for (int l = 0; l < c->layers; l++) {
        float *cx = cx_all + (size_t)l * P, *tk = tok + (size_t)l * P;
        for (int i = 0; i < P; i++) cx[i] *= cscale;
        lm_rmsnorm_plain(cx, cx, g->ple_norm, P, c->eps);
        for (int i = 0; i < P; i++) tk[i] = (cx[i] + tk[i] * c->ple_scale) * half;
    }
}

static int per_layer_inputs(LmGemma *g, int token, char *err, size_t errlen)
{
    if (ple_read_row(g, token, g->ple, err, errlen) != 0) return -1;
    lm_mat_mul(&g->ple_model, g->ple_ctx, g->h);
    ple_combine(g, g->ple, g->ple_ctx);
    return 0;
}

static void attention(LmGemma *g, int l, const float *x, float *out)
{
    const LmGemmaCfg *c = &g->cfg;
    const GLayer *L = &g->layers[l];
    const int hd = L->hd, kvd = c->kv_heads * hd;
    const double *freqs = g->freqs[L->full];
    float *q = g->q, *k = g->k, *v = g->v;

    lm_mat_mul(&L->q, q, x);
    for (int h = 0; h < c->heads; h++) lm_rmsnorm_plain(q + (size_t)h * hd, q + (size_t)h * hd, L->q_norm, hd, c->eps);
    (void)lm_rope_freqs_apply(q, c->heads, hd, freqs, g->pos);
    if (!L->reuse) {
        lm_mat_mul(&L->k, k, x);
        lm_mat_mul(&L->v, v, x);
        for (int h = 0; h < c->kv_heads; h++) {
            lm_rmsnorm_plain(k + (size_t)h * hd, k + (size_t)h * hd, L->k_norm, hd, c->eps);
            lm_rmsnorm_plain(v + (size_t)h * hd, v + (size_t)h * hd, NULL, hd, c->eps);
        }
        (void)lm_rope_freqs_apply(k, c->kv_heads, hd, freqs, g->pos);
        const size_t slot = (size_t)(g->pos % slots_of(c, L->full, g->ctx));
        memcpy(g->kc[l] + slot * kvd, k, (size_t)kvd * sizeof(float));
        memcpy(g->vc[l] + slot * kvd, v, (size_t)kvd * sizeof(float));
    }
    const int p = L->producer;
    (void)lm_attn_window(g->attn, q, g->kc[p], g->vc[p], g->pos, L->full ? 0 : c->window,
                         slots_of(c, L->full, g->ctx), c->heads, c->kv_heads, hd, 1.0f, g->scores);
    lm_mat_mul(&L->o, out, g->attn);
}

int lm_gemma_feed(LmGemma *g, int token, float *logits)
{
    const LmGemmaCfg *c = &g->cfg;
    const int H = c->hidden, P = c->ple_dim;
    if (token < 0 || token >= c->vocab) return LM_ERR_TOKEN;
    if (g->pos >= g->ctx) return LM_ERR_CTX;

    lm_mat_row(&g->embed, token, g->h);
    for (int j = 0; j < H; j++) g->h[j] *= c->embed_scale;
    {
        char err[256];
        if (per_layer_inputs(g, token, err, sizeof err) != 0) return LM_ERR_IO;
    }
    for (int i = 0; i < c->layers; i++) {
        const GLayer *L = &g->layers[i];
        lm_rmsnorm_plain(g->n, g->h, L->in_norm, H, c->eps);
        attention(g, i, g->n, g->mix);
        lm_rmsnorm_plain(g->mix, g->mix, L->post_attn_norm, H, c->eps);
        for (int j = 0; j < H; j++) g->h[j] += g->mix[j];

        lm_rmsnorm_plain(g->n, g->h, L->pre_ff_norm, H, c->eps);
        lm_mat_mul(&L->gate, g->mlp_s, g->n);
        lm_mat_mul(&L->up, g->mlp_s + c->inter, g->n);
        lm_gelu_tanh_mul(g->mlp_s, g->mlp_s, g->mlp_s + c->inter, c->inter);
        lm_mat_mul(&L->down, g->mlp_out, g->mlp_s);
        lm_rmsnorm_plain(g->mlp_out, g->mlp_out, L->post_ff_norm, H, c->eps);
        for (int j = 0; j < H; j++) g->h[j] += g->mlp_out[j];

        const float *pl = g->ple + (size_t)i * P;
        lm_mat_mul(&L->ple_gate, g->ple_g, g->h);
        for (int j = 0; j < P; j++) g->ple_g[j] = lm_gelu_tanh(g->ple_g[j]) * pl[j];
        lm_mat_mul(&L->ple_proj, g->ple_d, g->ple_g);
        lm_rmsnorm_plain(g->ple_d, g->ple_d, L->post_ple_norm, H, c->eps);
        for (int j = 0; j < H; j++) g->h[j] = (g->h[j] + g->ple_d[j]) * L->layer_scalar;
    }
    g->pos++;
    if (logits) {
        lm_rmsnorm_plain(g->n, g->h, g->final_norm, H, c->eps);
        lm_mat_mul(&g->embed, logits, g->n);
        lm_logit_softcap(logits, c->vocab, c->cap);
    }
    return LM_OK;
}

/* ------------------------------------------------------------------ prefill */

/* The batched twin of attention(): the projections run over the T tokens at once, the rest (norms,
 * RoPE, the cache write, the windowed attention) token by token in position order. */
static void attention_batch(LmGemma *g, int l, const float *x, float *out, int T, int pos0)
{
    const LmGemmaCfg *c = &g->cfg;
    const GLayer *L = &g->layers[l];
    const int hd = L->hd, kvd = c->kv_heads * hd, qd = c->heads * hd;
    const double *freqs = g->freqs[L->full];
    const int slots = slots_of(c, L->full, g->ctx), p = L->producer;

    lm_mat_mul_batch(&L->q, g->bq, x, T);
    if (!L->reuse) {
        lm_mat_mul_batch(&L->k, g->bk, x, T);
        lm_mat_mul_batch(&L->v, g->bv, x, T);
    }
    for (int t = 0; t < T; t++) {
        const int pos = pos0 + t;
        float *q = g->bq + (size_t)t * qd;
        for (int h = 0; h < c->heads; h++) lm_rmsnorm_plain(q + (size_t)h * hd, q + (size_t)h * hd, L->q_norm, hd, c->eps);
        (void)lm_rope_freqs_apply(q, c->heads, hd, freqs, pos);
        if (!L->reuse) {
            float *k = g->bk + (size_t)t * kvd, *v = g->bv + (size_t)t * kvd;
            for (int h = 0; h < c->kv_heads; h++) {
                lm_rmsnorm_plain(k + (size_t)h * hd, k + (size_t)h * hd, L->k_norm, hd, c->eps);
                lm_rmsnorm_plain(v + (size_t)h * hd, v + (size_t)h * hd, NULL, hd, c->eps);
            }
            (void)lm_rope_freqs_apply(k, c->kv_heads, hd, freqs, pos);
            const size_t slot = (size_t)(pos % slots);
            memcpy(g->kc[l] + slot * kvd, k, (size_t)kvd * sizeof(float));
            memcpy(g->vc[l] + slot * kvd, v, (size_t)kvd * sizeof(float));
        }
        (void)lm_attn_window(g->battn + (size_t)t * qd, q, g->kc[p], g->vc[p], pos, L->full ? 0 : c->window, slots,
                             c->heads, c->kv_heads, hd, 1.0f, g->scores);
    }
    lm_mat_mul_batch(&L->o, out, g->battn, T);
}

/* Layers [from, to) over rows 0..T-1 of bh, the token of row t being at position pos0 + t. */
static void layers_batch(LmGemma *g, int from, int to, int T, int pos0)
{
    const LmGemmaCfg *c = &g->cfg;
    const int H = c->hidden, P = c->ple_dim, LP = c->layers * P;
    for (int i = from; i < to; i++) {
        const GLayer *L = &g->layers[i];
        for (int t = 0; t < T; t++) lm_rmsnorm_plain(g->bn + (size_t)t * H, g->bh + (size_t)t * H, L->in_norm, H, c->eps);
        attention_batch(g, i, g->bn, g->bmix, T, pos0);
        for (int t = 0; t < T; t++) {
            float *mix = g->bmix + (size_t)t * H, *h = g->bh + (size_t)t * H;
            lm_rmsnorm_plain(mix, mix, L->post_attn_norm, H, c->eps);
            for (int j = 0; j < H; j++) h[j] += mix[j];
            lm_rmsnorm_plain(g->bn + (size_t)t * H, h, L->pre_ff_norm, H, c->eps);
        }
        float *gate = g->bmlp, *up = g->bmlp + (size_t)T * c->inter;
        lm_mat_mul_batch(&L->gate, gate, g->bn, T);
        lm_mat_mul_batch(&L->up, up, g->bn, T);
        for (int t = 0; t < T; t++)
            lm_gelu_tanh_mul(gate + (size_t)t * c->inter, gate + (size_t)t * c->inter, up + (size_t)t * c->inter, c->inter);
        lm_mat_mul_batch(&L->down, g->bmix, gate, T);
        for (int t = 0; t < T; t++) {
            float *mo = g->bmix + (size_t)t * H, *h = g->bh + (size_t)t * H;
            lm_rmsnorm_plain(mo, mo, L->post_ff_norm, H, c->eps);
            for (int j = 0; j < H; j++) h[j] += mo[j];
        }
        lm_mat_mul_batch(&L->ple_gate, g->bple_g, g->bh, T);
        for (int t = 0; t < T; t++) {
            const float *pl = g->bple + (size_t)t * LP + (size_t)i * P;
            float *pg = g->bple_g + (size_t)t * P;
            for (int j = 0; j < P; j++) pg[j] = lm_gelu_tanh(pg[j]) * pl[j];
        }
        lm_mat_mul_batch(&L->ple_proj, g->bple_d, g->bple_g, T);
        for (int t = 0; t < T; t++) {
            float *pd = g->bple_d + (size_t)t * H, *h = g->bh + (size_t)t * H;
            lm_rmsnorm_plain(pd, pd, L->post_ple_norm, H, c->eps);
            for (int j = 0; j < H; j++) h[j] = (h[j] + pd[j]) * L->layer_scalar;
        }
    }
}

int lm_gemma_prefill(LmGemma *g, const int *tokens, int n, float *logits)
{
    const LmGemmaCfg *c = &g->cfg;
    const int H = c->hidden, P = c->ple_dim, LP = c->layers * P;
    int last_T = 0;
    for (int s = 0; s < n; s += LM_PREFILL_CHUNK) {
        const int T = n - s < LM_PREFILL_CHUNK ? n - s : LM_PREFILL_CHUNK;
        last_T = T;
        for (int t = 0; t < T; t++) {
            float *h = g->bh + (size_t)t * H;
            lm_mat_row(&g->embed, tokens[s + t], h);
            for (int j = 0; j < H; j++) h[j] *= c->embed_scale;
        }
        lm_mat_mul_batch(&g->ple_model, g->bple_ctx, g->bh, T);
        for (int t = 0; t < T; t++) {
            char err[256];
            if (ple_read_row(g, tokens[s + t], g->bple + (size_t)t * LP, err, sizeof err) != 0) return LM_ERR_IO;
            ple_combine(g, g->bple + (size_t)t * LP, g->bple_ctx + (size_t)t * LP);
        }
        /* Only the layers that own a KV cache. The last num_kv_shared_layers layers hold no state: they
         * read the caches of the producing layers and their output is a token's own logits and nothing
         * else, so for every token whose logits are not asked for they are skipped (they are 18 of the
         * 42 layers of the real E4B). The state after the call is exactly that of lm_gemma_feed. */
        layers_batch(g, 0, c->first_shared, T, g->pos);
        g->pos += T;
    }
    if (logits) {   /* the last token also runs the layers that read the shared caches */
        if (last_T > 1) {
            memcpy(g->bh, g->bh + (size_t)(last_T - 1) * H, (size_t)H * sizeof(float));
            memcpy(g->bple, g->bple + (size_t)(last_T - 1) * LP, (size_t)LP * sizeof(float));
        }
        layers_batch(g, c->first_shared, c->layers, 1, g->pos - 1);
        lm_rmsnorm_plain(g->n, g->bh, g->final_norm, H, c->eps);
        lm_mat_mul(&g->embed, logits, g->n);
        lm_logit_softcap(logits, c->vocab, c->cap);
    }
    return LM_OK;
}
