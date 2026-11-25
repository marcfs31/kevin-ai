/* lm_qwen2.c - see lm_qwen2.h. Original code. */
#include "io/lm_portable_io.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "lm/lm.h"
#include "core/lm_bf16.h"
#include "core/lm_gguf_quant.h"
#include "core/lm_ops.h"
#include "io/lm_gguf.h"
#include "io/lm_io.h"
#include "model/lm_ctx.h"
#include "model/lm_layers.h"
#include "chat/lm_catalog.h"
#include "model/lm_qwen2.h"

typedef struct {
    const float *attn_norm, *ffn_norm;      /* [hidden] */
    const float *bq, *bk, *bv;              /* [heads*hd], [kvd], [kvd] */
    LmMat        q, k, v, o;
    LmMlpW       mlp;
} Layer;

struct LmQwen2 {
    LmGguf    g;
    LmQwen2Cfg cfg;
    int       ctx, pos;
    uint64_t  bytes;                         /* file bytes of every tensor bound */
    void    **owned;
    int       nowned, capowned;
    Layer    *layers;
    LmMat     embed, head;                   /* head == embed when tied */
    const float *final_norm;
    double   *freqs;                         /* [hd/2] */
    float   **kc, **vc;                      /* per layer [ctx][kvd] */
    float    *h, *n, *q, *k, *v, *attn, *o, *mlp_out, *mlp_scratch, *scores;
    float    *bh, *bn, *bq, *bk, *bv, *battn, *bo, *bmlp, *bmlp_scratch;   /* prefill chunk */
};

static void *own(LmQwen2 *q, void *p)
{
    if (!p) return NULL;
    if (q->nowned == q->capowned) {
        const int nc = q->capowned ? q->capowned * 2 : 128;
        void **np = (void **)realloc(q->owned, (size_t)nc * sizeof *np);
        if (!np) { free(p); return NULL; }
        q->owned = np;
        q->capowned = nc;
    }
    q->owned[q->nowned++] = p;
    return p;
}

#define CHUNK LM_PREFILL_CHUNK

size_t lm_qwen2_state_bytes(const LmQwen2Cfg *c, int ctx)
{
    const size_t kvd = (size_t)c->kv_heads * c->head_dim, qd = (size_t)c->heads * c->head_dim;
    const size_t kv = 2 * (size_t)c->layers * (size_t)ctx * kvd * sizeof(float);
    const size_t row = (size_t)c->hidden * 3 + qd * 2 + kvd * 2 + 2 * (size_t)c->inter + (size_t)c->heads * (size_t)ctx;
    const size_t batch = CHUNK * ((size_t)c->hidden * 3 + qd * 2 + kvd * 2 + 2 * (size_t)c->inter);
    return kv + (row + batch) * sizeof(float);
}

static size_t state_bytes_cb(const void *arg, int ctx) { return lm_qwen2_state_bytes((const LmQwen2Cfg *)arg, ctx); }

void lm_qwen2_close(LmQwen2 *q)
{
    if (!q) return;
    for (int i = 0; i < q->cfg.layers && q->kc; i++) { free(q->kc[i]); free(q->vc[i]); }
    free(q->kc); free(q->vc);
    free(q->layers);
    free(q->freqs);
    free(q->h); free(q->n); free(q->q); free(q->k); free(q->v); free(q->attn); free(q->o);
    free(q->mlp_out); free(q->mlp_scratch); free(q->scores);
    free(q->bh); free(q->bn); free(q->bq); free(q->bk); free(q->bv); free(q->battn); free(q->bo);
    free(q->bmlp); free(q->bmlp_scratch);
    for (int i = 0; i < q->nowned; i++) free(q->owned[i]);
    free(q->owned);
    lm_gguf_close(&q->g);
    free(q);
}

/* ------------------------------------------------------------------ binding */

static const LmGgufTensor *need(LmQwen2 *q, const char *name, int ndim, int64_t ne0, int64_t ne1, char *err, size_t n)
{
    const LmGgufTensor *t = lm_gguf_find(&q->g, name);
    if (!t) { lm_fail(err, n, "tensor %s is missing from the GGUF file", name); return NULL; }
    int ok = t->ndim == ndim && t->ne[0] == ne0 && (ndim < 2 || t->ne[1] == ne1);
    if (!ok) {
        lm_fail(err, n, "tensor %s has shape [%lld,%lld] (rank %d), the config needs [%lld,%lld] (rank %d)", name,
                (long long)t->ne[0], (long long)(t->ndim > 1 ? t->ne[1] : 0), t->ndim, (long long)ne0, (long long)(ndim > 1 ? ne1 : 0), ndim);
        return NULL;
    }
    q->bytes += t->nbytes ? t->nbytes : (uint64_t)t->numel * 4;
    return t;
}

/* A float vector (F32 or F16 in the file) copied into an owned f32 buffer. */
static const float *bind_vec(LmQwen2 *q, const char *name, int64_t n, char *err, size_t errlen)
{
    const LmGgufTensor *t = need(q, name, 1, n, 0, err, errlen);
    if (!t) return NULL;
    if (t->type != LM_GG_F32 && t->type != LM_GG_F16) { lm_fail(err, errlen, "tensor %s is %s; F32 or F16 expected", name, lm_gguf_type_name(t->type)); return NULL; }
    float *b = (float *)own(q, malloc((size_t)n * sizeof(float)));
    if (!b) { lm_fail(err, errlen, "out of memory binding %s", name); return NULL; }
    const unsigned char *src = (const unsigned char *)lm_gguf_data(&q->g, t);
    if (t->type == LM_GG_F32) memcpy(b, src, (size_t)n * sizeof(float));
    else for (int64_t i = 0; i < n; i++) { uint16_t h; memcpy(&h, src + 2 * i, 2); b[i] = lm_f16_to_f32(h); }
    for (int64_t i = 0; i < n; i++) if (!isfinite(b[i])) { lm_fail(err, errlen, "tensor %s holds a NaN or infinity", name); return NULL; }
    return b;
}

static int bind_mat(LmQwen2 *q, const char *name, int rows, int cols, LmMat *out, char *err, size_t errlen)
{
    const LmGgufTensor *t = need(q, name, 2, cols, rows, err, errlen);
    if (!t) return -1;
    memset(out, 0, sizeof *out);
    out->rows = rows;
    out->cols = cols;
    if (t->type == LM_GG_F32) {
        out->kind = LM_MAT_F32;
        out->f = (const float *)lm_gguf_data(&q->g, t);
        return 0;
    }
    if (t->type == LM_GG_F16) return lm_fail(err, errlen, "matrix %s is F16, which the engine does not multiply; use a quantised or F32 file", name);
    if (lm_gguf_type_block(t->type, NULL, NULL) != 0)
        return lm_fail(err, errlen, "matrix %s is %s, which this engine does not read (supported: Q4_K, Q5_K, Q6_K, Q8_0, F32)", name, lm_gguf_type_name(t->type));
    out->kind = LM_MAT_GG;
    out->gtype = t->type;
    out->g = lm_gguf_data(&q->g, t);
    return 0;
}

static int need_u64(const LmGguf *g, const char *key, int *out, char *err, size_t errlen)
{
    uint64_t v;
    if (lm_gguf_get_u64(g, key, &v) != 0) return lm_fail(err, errlen, "%s: metadata %s is missing or not an integer", g->path, key);
    if (v == 0 || v > (1u << 30)) return lm_fail(err, errlen, "%s: metadata %s = %llu is out of range", g->path, key, (unsigned long long)v);
    *out = (int)v;
    return 0;
}

static int read_cfg(LmQwen2 *q, char *err, size_t errlen)
{
    const LmGguf *g = &q->g;
    LmQwen2Cfg *c = &q->cfg;
    LmGgufStr arch;
    if (lm_gguf_get_str(g, "general.architecture", &arch) != 0) return lm_fail(err, errlen, "%s: general.architecture is missing", g->path);
    if (arch.n != 5 || memcmp(arch.s, "qwen2", 5) != 0) {
        const char *hint = lm_catalog_arch_hint(arch.s, arch.n);
        if (hint)
            return lm_fail(err, errlen, "%s: architecture \"%.*s\" is not the native qwen2 loader; %s",
                           g->path, (int)arch.n, arch.s, hint);
        return lm_fail(err, errlen, "%s: architecture \"%.40s\" is not supported (supported: qwen2)", g->path, arch.s);
    }
    if (need_u64(g, "qwen2.block_count", &c->layers, err, errlen) || need_u64(g, "qwen2.embedding_length", &c->hidden, err, errlen) ||
        need_u64(g, "qwen2.feed_forward_length", &c->inter, err, errlen) || need_u64(g, "qwen2.attention.head_count", &c->heads, err, errlen) ||
        need_u64(g, "qwen2.attention.head_count_kv", &c->kv_heads, err, errlen) || need_u64(g, "qwen2.context_length", &c->max_pos, err, errlen))
        return -1;
    double eps, theta;
    if (lm_gguf_get_f64(g, "qwen2.attention.layer_norm_rms_epsilon", &eps) != 0 || !(eps > 0 && eps < 1))
        return lm_fail(err, errlen, "%s: qwen2.attention.layer_norm_rms_epsilon is missing or not in (0, 1)", g->path);
    if (lm_gguf_get_f64(g, "qwen2.rope.freq_base", &theta) != 0 || !(theta > 0 && theta < 1e12))
        return lm_fail(err, errlen, "%s: qwen2.rope.freq_base is missing or not a positive number", g->path);
    c->eps = (float)eps;
    c->theta = (float)theta;
    if (c->hidden % c->heads != 0 || c->heads % c->kv_heads != 0)
        return lm_fail(err, errlen, "%s: %d heads and %d kv heads do not divide the hidden size %d", g->path, c->heads, c->kv_heads, c->hidden);
    c->head_dim = c->hidden / c->heads;
    if (c->head_dim % 2) return lm_fail(err, errlen, "%s: head size %d is odd; RoPE needs an even one", g->path, c->head_dim);
    uint64_t v;
    if (lm_gguf_get_u64(g, "qwen2.rope.dimension_count", &v) == 0 && (int)v != c->head_dim)
        return lm_fail(err, errlen, "%s: rope.dimension_count %llu is not the head size %d (partial rotary is not implemented)", g->path, (unsigned long long)v, c->head_dim);
    if (lm_gguf_get_u64(g, "qwen2.attention.key_length", &v) == 0 && (int)v != c->head_dim)
        return lm_fail(err, errlen, "%s: attention.key_length %llu is not hidden / heads = %d", g->path, (unsigned long long)v, c->head_dim);
    LmGgufStr st;
    if (lm_gguf_get_str(g, "qwen2.rope.scaling.type", &st) == 0 && !(st.n == 4 && !memcmp(st.s, "none", 4)))
        return lm_fail(err, errlen, "%s: rope scaling \"%.20s\" is not implemented", g->path, st.s);
    const LmGgufTensor *te = lm_gguf_find(g, "token_embd.weight");
    if (!te || te->ndim != 2 || te->ne[0] != c->hidden) return lm_fail(err, errlen, "%s: token_embd.weight is missing or not [vocab][%d]", g->path, c->hidden);
    if (te->ne[1] > (1 << 22)) return lm_fail(err, errlen, "%s: implausible vocabulary size %lld", g->path, (long long)te->ne[1]);
    c->vocab = (int)te->ne[1];
    return 0;
}

LmQwen2 *lm_qwen2_open(const char *path, int ctx, char *err, size_t errlen)
{
    LmQwen2 *q = (LmQwen2 *)calloc(1, sizeof *q);
    if (!q) { lm_fail(err, errlen, "out of memory"); return NULL; }
    q->g.fd = -1;
    if (ctx < 0) { lm_fail(err, errlen, "ctx must not be negative"); goto fail; }
    if (lm_gguf_open(&q->g, path, err, errlen) != 0) goto fail;
    if (read_cfg(q, err, errlen) != 0) goto fail;
    const LmQwen2Cfg *c = &q->cfg;
    const int H = c->hidden, qd = c->heads * c->head_dim, kvd = c->kv_heads * c->head_dim;

    q->layers = (Layer *)calloc((size_t)c->layers, sizeof(Layer));
    q->freqs = (double *)malloc((size_t)(c->head_dim / 2) * sizeof(double));
    if (!q->layers || !q->freqs) { lm_fail(err, errlen, "out of memory"); goto fail; }
    if (lm_rope_freqs(q->freqs, c->head_dim, (double)c->theta, 1.0) != 0) { lm_fail(err, errlen, "bad RoPE parameters"); goto fail; }

#define NAME(...) do { snprintf(nm, sizeof nm, __VA_ARGS__); } while (0)
    char nm[128];
    if (bind_mat(q, "token_embd.weight", c->vocab, H, &q->embed, err, errlen) != 0) goto fail;
    if (!(q->final_norm = bind_vec(q, "output_norm.weight", H, err, errlen))) goto fail;
    if (lm_gguf_find(&q->g, "output.weight")) { if (bind_mat(q, "output.weight", c->vocab, H, &q->head, err, errlen) != 0) goto fail; }
    else q->head = q->embed;
    for (int i = 0; i < c->layers; i++) {
        Layer *L = &q->layers[i];
        NAME("blk.%d.attn_norm.weight", i); if (!(L->attn_norm = bind_vec(q, nm, H, err, errlen))) goto fail;
        NAME("blk.%d.ffn_norm.weight", i);  if (!(L->ffn_norm = bind_vec(q, nm, H, err, errlen))) goto fail;
        NAME("blk.%d.attn_q.bias", i); if (!(L->bq = bind_vec(q, nm, qd, err, errlen))) goto fail;
        NAME("blk.%d.attn_k.bias", i); if (!(L->bk = bind_vec(q, nm, kvd, err, errlen))) goto fail;
        NAME("blk.%d.attn_v.bias", i); if (!(L->bv = bind_vec(q, nm, kvd, err, errlen))) goto fail;
        NAME("blk.%d.attn_q.weight", i); if (bind_mat(q, nm, qd, H, &L->q, err, errlen)) goto fail;
        NAME("blk.%d.attn_k.weight", i); if (bind_mat(q, nm, kvd, H, &L->k, err, errlen)) goto fail;
        NAME("blk.%d.attn_v.weight", i); if (bind_mat(q, nm, kvd, H, &L->v, err, errlen)) goto fail;
        NAME("blk.%d.attn_output.weight", i); if (bind_mat(q, nm, H, qd, &L->o, err, errlen)) goto fail;
        L->mlp.inter = c->inter;
        NAME("blk.%d.ffn_gate.weight", i); if (bind_mat(q, nm, c->inter, H, &L->mlp.gate, err, errlen)) goto fail;
        NAME("blk.%d.ffn_up.weight", i); if (bind_mat(q, nm, c->inter, H, &L->mlp.up, err, errlen)) goto fail;
        NAME("blk.%d.ffn_down.weight", i); if (bind_mat(q, nm, H, c->inter, &L->mlp.down, err, errlen)) goto fail;
    }
#undef NAME
    if (lm_ctx_choose_here(ctx, c->max_pos, q->bytes, state_bytes_cb, c, &ctx, err, errlen) != 0) goto fail;
    q->ctx = ctx;
    q->kc = (float **)calloc((size_t)c->layers, sizeof(float *));
    q->vc = (float **)calloc((size_t)c->layers, sizeof(float *));
    if (!q->kc || !q->vc) { lm_fail(err, errlen, "out of memory"); goto fail; }
    for (int i = 0; i < c->layers; i++) {
        q->kc[i] = (float *)calloc((size_t)ctx * (size_t)kvd, sizeof(float));
        q->vc[i] = (float *)calloc((size_t)ctx * (size_t)kvd, sizeof(float));
        if (!q->kc[i] || !q->vc[i]) { lm_fail(err, errlen, "out of memory allocating the KV cache for ctx %d", ctx); goto fail; }
    }
#define ALLOC(p, n) do { (p) = (float *)calloc((size_t)(n), sizeof(float)); if (!(p)) { lm_fail(err, errlen, "out of memory"); goto fail; } } while (0)
    ALLOC(q->h, H); ALLOC(q->n, H); ALLOC(q->q, qd); ALLOC(q->k, kvd); ALLOC(q->v, kvd); ALLOC(q->attn, qd); ALLOC(q->o, H);
    ALLOC(q->mlp_out, H); ALLOC(q->mlp_scratch, lm_mlp_scratch(c->inter)); ALLOC(q->scores, (size_t)c->heads * (size_t)ctx);
    ALLOC(q->bh, (size_t)CHUNK * H); ALLOC(q->bn, (size_t)CHUNK * H); ALLOC(q->bq, (size_t)CHUNK * qd);
    ALLOC(q->bk, (size_t)CHUNK * kvd); ALLOC(q->bv, (size_t)CHUNK * kvd); ALLOC(q->battn, (size_t)CHUNK * qd);
    ALLOC(q->bo, (size_t)CHUNK * H); ALLOC(q->bmlp, (size_t)CHUNK * H); ALLOC(q->bmlp_scratch, lm_mlp_batch_scratch(c->inter, CHUNK));
#undef ALLOC
    return q;
fail:
    lm_qwen2_close(q);
    return NULL;
}

const LmQwen2Cfg *lm_qwen2_cfg(const LmQwen2 *q) { return &q->cfg; }
int  lm_qwen2_n_vocab(const LmQwen2 *q) { return q->cfg.vocab; }
int  lm_qwen2_n_ctx(const LmQwen2 *q) { return q->ctx; }
int  lm_qwen2_pos(const LmQwen2 *q) { return q->pos; }
void lm_qwen2_reset(LmQwen2 *q) { q->pos = 0; }   /* the KV rows below pos are rewritten before they are read */
void lm_qwen2_set_pos(LmQwen2 *q, int pos) { q->pos = pos; }

void lm_qwen2_state_regions(LmQwen2 *q, int pos, LmQwen2RegionFn fn, void *user)
{
    const size_t kvd = (size_t)q->cfg.kv_heads * q->cfg.head_dim;
    for (int i = 0; i < q->cfg.layers; i++) {
        fn(user, q->kc[i], (size_t)pos * kvd);
        fn(user, q->vc[i], (size_t)pos * kvd);
    }
}

/* ------------------------------------------------------------------ forward */

static void add_bias(float *y, const float *b, int n) { for (int i = 0; i < n; i++) y[i] += b[i]; }

/* One token's K and V rows into the caches and its attention over positions 0..pos. q is the
 * token's rotated query, k and v its (rotated) key and value. */
static void attend(LmQwen2 *m, int l, int pos, float *q, const float *k, const float *v, float *out)
{
    const LmQwen2Cfg *c = &m->cfg;
    const size_t kvd = (size_t)c->kv_heads * c->head_dim;
    memcpy(m->kc[l] + (size_t)pos * kvd, k, kvd * sizeof(float));
    memcpy(m->vc[l] + (size_t)pos * kvd, v, kvd * sizeof(float));
    (void)lm_attn_window(out, q, m->kc[l], m->vc[l], pos, 0, m->ctx, c->heads, c->kv_heads, c->head_dim,
                         (float)(1.0 / sqrt((double)c->head_dim)), m->scores);
}

static void qkv_post(LmQwen2 *m, const Layer *L, int pos, float *q, float *k, float *v)
{
    const LmQwen2Cfg *c = &m->cfg;
    add_bias(q, L->bq, c->heads * c->head_dim);
    add_bias(k, L->bk, c->kv_heads * c->head_dim);
    add_bias(v, L->bv, c->kv_heads * c->head_dim);
    (void)lm_rope_freqs_apply(q, c->heads, c->head_dim, m->freqs, pos);
    (void)lm_rope_freqs_apply(k, c->kv_heads, c->head_dim, m->freqs, pos);
}

static void logits_of(LmQwen2 *m, const float *h, float *logits)
{
    lm_rmsnorm_plain(m->n, h, m->final_norm, m->cfg.hidden, m->cfg.eps);
    lm_mat_mul(&m->head, logits, m->n);
}

int lm_qwen2_feed(LmQwen2 *m, int token, float *logits)
{
    const LmQwen2Cfg *c = &m->cfg;
    if (token < 0 || token >= c->vocab) return LM_ERR_TOKEN;
    if (m->pos >= m->ctx) return LM_ERR_CTX;
    const int H = c->hidden;
    lm_mat_row(&m->embed, token, m->h);
    for (int l = 0; l < c->layers; l++) {
        const Layer *L = &m->layers[l];
        lm_rmsnorm_plain(m->n, m->h, L->attn_norm, H, c->eps);
        lm_mat_mul(&L->q, m->q, m->n);
        lm_mat_mul(&L->k, m->k, m->n);
        lm_mat_mul(&L->v, m->v, m->n);
        qkv_post(m, L, m->pos, m->q, m->k, m->v);
        attend(m, l, m->pos, m->q, m->k, m->v, m->attn);
        lm_mat_mul(&L->o, m->o, m->attn);
        for (int i = 0; i < H; i++) m->h[i] += m->o[i];
        lm_rmsnorm_plain(m->n, m->h, L->ffn_norm, H, c->eps);
        lm_mlp(m->mlp_out, m->n, &L->mlp, m->mlp_scratch);
        for (int i = 0; i < H; i++) m->h[i] += m->mlp_out[i];
    }
    m->pos++;
    if (logits) logits_of(m, m->h, logits);
    return LM_OK;
}

int lm_qwen2_prefill(LmQwen2 *m, const int *tokens, int n, float *logits)
{
    const LmQwen2Cfg *c = &m->cfg;
    const int H = c->hidden, qd = c->heads * c->head_dim, kvd = c->kv_heads * c->head_dim;
    for (int s = 0; s < n; s += CHUNK) {
        const int T = n - s < CHUNK ? n - s : CHUNK;
        const int pos0 = m->pos;
        for (int t = 0; t < T; t++) lm_mat_row(&m->embed, tokens[s + t], m->bh + (size_t)t * H);
        for (int l = 0; l < c->layers; l++) {
            const Layer *L = &m->layers[l];
            for (int t = 0; t < T; t++) lm_rmsnorm_plain(m->bn + (size_t)t * H, m->bh + (size_t)t * H, L->attn_norm, H, c->eps);
            lm_mat_mul_batch(&L->q, m->bq, m->bn, T);
            lm_mat_mul_batch(&L->k, m->bk, m->bn, T);
            lm_mat_mul_batch(&L->v, m->bv, m->bn, T);
            for (int t = 0; t < T; t++) {
                float *q = m->bq + (size_t)t * qd, *k = m->bk + (size_t)t * kvd, *v = m->bv + (size_t)t * kvd;
                qkv_post(m, L, pos0 + t, q, k, v);
                attend(m, l, pos0 + t, q, k, v, m->battn + (size_t)t * qd);
            }
            lm_mat_mul_batch(&L->o, m->bo, m->battn, T);
            for (size_t i = 0; i < (size_t)T * H; i++) m->bh[i] += m->bo[i];
            for (int t = 0; t < T; t++) lm_rmsnorm_plain(m->bn + (size_t)t * H, m->bh + (size_t)t * H, L->ffn_norm, H, c->eps);
            lm_mlp_batch(m->bmlp, m->bn, &L->mlp, m->bmlp_scratch, T);
            for (size_t i = 0; i < (size_t)T * H; i++) m->bh[i] += m->bmlp[i];
        }
        m->pos += T;
        if (logits && s + T == n) logits_of(m, m->bh + (size_t)(T - 1) * H, logits);
    }
    return LM_OK;
}
