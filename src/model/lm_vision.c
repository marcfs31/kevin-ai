/* lm_vision.c - see lm_vision.h. Original code; maths per docs/PLAN.md "Phase 3" and
 * transformers modeling_qwen3_5.py (Qwen3_5VisionModel) and modeling_gemma4.py
 * (Gemma4VisionModel, Gemma4MultimodalEmbedder); oracles tools/ref_vision_qwen35.py and
 * tools/ref_vision_gemma4.py. Activations are f32; norms and softmax sums accumulate in
 * double. */
#include "io/lm_portable_io.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "io/lm_io.h"
#include "json.h"
#include "model/lm_bind.h"
#include "model/lm_config.h"
#include "model/lm_vision.h"

#define DIM_MAX (1 << 20)

typedef struct {
    LmMat qkv, proj, fc1, fc2;
    const float *n1w, *n1b, *n2w, *n2b, *qkv_b, *proj_b, *fc1_b, *fc2_b;
} QBlock;

/* Gemma4ClippableLinear: clamp the input, multiply, clamp the output (lo/hi unused unless clipped). */
typedef struct { LmMat m; float in_lo, in_hi, out_lo, out_hi; } GLin;

typedef struct {
    GLin q, k, v, o, gate, up, down;
    const float *in_n, *post_attn_n, *pre_ff_n, *post_ff_n, *q_n, *k_n;
} GBlock;

struct LmVision {
    LmVisionInfo info;
    LmBind b;
    int depth, hidden, heads, kv_heads, head_dim, inter, patch;
    double theta;
    /* Qwen */
    int merge, temporal, side;
    LmMat pe;
    const float *pe_b, *pos;          /* pos: [side*side][hidden] */
    QBlock *blk;
    const float *mn_w, *mn_b, *mf1_b, *mf2_b;
    LmMat mf1, mf2;
    /* Gemma */
    int pool, pos_size, clipped;
    double eps;
    LmMat in_proj, emb;
    const float *ptab;                /* [2][pos_size][hidden]: x table, then y table */
    GBlock *gblk;
};

/* ------------------------------------------------------------------------- config */

static int req_int(jval *o, const char *key, int *out, char *err, size_t n)
{
    jval *v = o ? json_get(o, key) : NULL;
    if (!v) return lm_fail(err, n, "config.json is missing \"%s\"", key);
    if (v->t != J_NUM || v->num != floor(v->num) || v->num < 0 || v->num > 2147483647.0)
        return lm_fail(err, n, "config.json: \"%s\" must be a non-negative integer", key);
    *out = (int)v->num;
    return 0;
}

static int qwen_config(LmVision *v, jval *root, char *err, size_t n)
{
    jval *vc = json_get(root, "vision_config");
    if (!vc || vc->t != J_OBJ) return lm_fail(err, n, "config.json has no vision_config object");
    int in_ch, n_pos;
    if (req_int(vc, "depth", &v->depth, err, n) || req_int(vc, "hidden_size", &v->hidden, err, n) ||
        req_int(vc, "num_heads", &v->heads, err, n) || req_int(vc, "intermediate_size", &v->inter, err, n) ||
        req_int(vc, "patch_size", &v->patch, err, n) || req_int(vc, "spatial_merge_size", &v->merge, err, n) ||
        req_int(vc, "temporal_patch_size", &v->temporal, err, n) || req_int(vc, "in_channels", &in_ch, err, n) ||
        req_int(vc, "out_hidden_size", &v->info.out_dim, err, n) ||
        req_int(vc, "num_position_embeddings", &n_pos, err, n) ||
        req_int(root, "image_token_id", &v->info.image_token, err, n) ||
        req_int(root, "vision_start_token_id", &v->info.start_token, err, n) ||
        req_int(root, "vision_end_token_id", &v->info.end_token, err, n)) return -1;
    jval *act = json_get(vc, "hidden_act");
    if (!act || act->t != J_STR || strcmp(act->str, "gelu_pytorch_tanh"))
        return lm_fail(err, n, "vision_config.hidden_act must be \"gelu_pytorch_tanh\"");
    jval *ds = json_get(vc, "deepstack_visual_indexes");
    if (ds && (ds->t != J_ARR || ds->len != 0))
        return lm_fail(err, n, "vision_config.deepstack_visual_indexes is not supported");
    v->theta = 10000.0; /* Qwen3_5VisionRotaryEmbedding's own default, not a guess */
    jval *rp = json_get(vc, "rope_parameters");
    jval *th = rp && rp->t == J_OBJ ? json_get(rp, "rope_theta") : NULL;
    if (th) {
        if (th->t != J_NUM || !(th->num > 0)) return lm_fail(err, n, "vision rope_theta must be a positive number");
        v->theta = th->num;
    }
    if (in_ch != 3) return lm_fail(err, n, "vision in_channels %d; only 3 is supported", in_ch);
    if (v->depth < 1 || v->hidden < 1 || v->heads < 1 || v->hidden % v->heads || v->inter < 1 || v->patch < 1 ||
        v->merge < 1 || v->temporal < 1 || v->info.out_dim < 1 || v->hidden > DIM_MAX || v->inter > DIM_MAX ||
        v->patch > 64 || v->merge > 8 || v->temporal > 8 || v->depth > 256)
        return lm_fail(err, n, "vision_config has an unsupported shape");
    v->head_dim = v->hidden / v->heads;
    if (v->head_dim % 4) return lm_fail(err, n, "vision head_dim %d is not a multiple of 4", v->head_dim);
    v->side = (int)lround(sqrt((double)n_pos));
    if (n_pos < 1 || v->side * v->side != n_pos)
        return lm_fail(err, n, "vision num_position_embeddings %d is not a square", n_pos);
    v->info.family = LM_VISION_QWEN;
    v->info.unit = v->patch * v->merge;
    v->kv_heads = v->heads;
    return 0;
}

static int req_bool(jval *o, const char *key, int *out, char *err, size_t n)
{
    jval *v = json_get(o, key);
    if (!v || v->t != J_BOOL) return lm_fail(err, n, "config.json: \"%s\" must be true or false", key);
    *out = v->boolean;
    return 0;
}

static int gemma_config(LmVision *v, jval *root, char *err, size_t n)
{
    jval *vc = json_get(root, "vision_config"), *tc = json_get(root, "text_config");
    if (!vc || vc->t != J_OBJ || !tc || tc->t != J_OBJ)
        return lm_fail(err, n, "config.json has no vision_config or text_config object");
    int standardize;
    if (req_int(vc, "hidden_size", &v->hidden, err, n) || req_int(vc, "intermediate_size", &v->inter, err, n) ||
        req_int(vc, "num_hidden_layers", &v->depth, err, n) || req_int(vc, "num_attention_heads", &v->heads, err, n) ||
        req_int(vc, "num_key_value_heads", &v->kv_heads, err, n) || req_int(vc, "head_dim", &v->head_dim, err, n) ||
        req_int(vc, "patch_size", &v->patch, err, n) || req_int(vc, "pooling_kernel_size", &v->pool, err, n) ||
        req_int(vc, "position_embedding_size", &v->pos_size, err, n) ||
        req_bool(vc, "use_clipped_linears", &v->clipped, err, n) || req_bool(vc, "standardize", &standardize, err, n) ||
        req_int(tc, "hidden_size", &v->info.out_dim, err, n) ||
        req_int(root, "image_token_id", &v->info.image_token, err, n) ||
        req_int(root, "boi_token_id", &v->info.start_token, err, n) ||
        req_int(root, "eoi_token_id", &v->info.end_token, err, n)) return -1;
    jval *act = json_get(vc, "hidden_activation");
    if (!act || act->t != J_STR || strcmp(act->str, "gelu_pytorch_tanh"))
        return lm_fail(err, n, "vision_config.hidden_activation must be \"gelu_pytorch_tanh\"");
    if (standardize) return lm_fail(err, n, "vision_config.standardize is not supported");
    jval *eps = json_get(vc, "rms_norm_eps"), *rp = json_get(vc, "rope_parameters");
    jval *th = rp && rp->t == J_OBJ ? json_get(rp, "rope_theta") : NULL;
    if (!eps || eps->t != J_NUM || !(eps->num > 0)) return lm_fail(err, n, "vision rms_norm_eps must be a positive number");
    if (!th || th->t != J_NUM || !(th->num > 0)) return lm_fail(err, n, "vision rope_parameters.rope_theta must be a positive number");
    v->eps = eps->num;
    v->theta = th->num;
    if (v->depth < 1 || v->depth > 256 || v->hidden < 1 || v->hidden > DIM_MAX || v->inter < 1 || v->inter > DIM_MAX ||
        v->heads < 1 || v->kv_heads < 1 || v->heads % v->kv_heads || v->head_dim < 4 || v->head_dim % 4 ||
        v->head_dim > 1024 || v->patch < 1 || v->patch > 64 || v->pool < 1 || v->pool > 8 || v->pos_size < 1 ||
        v->info.out_dim < 1)
        return lm_fail(err, n, "vision_config has an unsupported shape");
    v->info.family = LM_VISION_GEMMA;
    v->info.unit = v->patch * v->pool;
    return 0;
}

/* ------------------------------------------------------------------------- binding */

static int qwen_bind(LmVision *v, char *err, size_t n)
{
    LmBind *b = &v->b;
    const int D = v->hidden, pin = 3 * v->temporal * v->patch * v->patch, cat = D * v->merge * v->merge;
    char nm[256];
#define VEC(dst, name, len) do { snprintf(nm, sizeof nm, "model.visual.%s", name); \
        if (!((dst) = lm_bind_vec(b, nm, (len), 1, (len), 0, err, n))) return -1; } while (0)
#define MAT(dst, name, r, c, flat) do { snprintf(nm, sizeof nm, "model.visual.%s", name); \
        if (lm_bind_mat_any(b, nm, (r), (c), (flat), &(dst), err, n)) return -1; } while (0)
    MAT(v->pe, "patch_embed.proj.weight", D, pin, 1);
    VEC(v->pe_b, "patch_embed.proj.bias", D);
    {   /* a 2-D .weight, so q8 in a q8 file: widened once, it is read per patch */
        const int np = v->side * v->side;
        LmMat pm;
        MAT(pm, "pos_embed.weight", np, D, 0);
        float *pos = (float *)lm_bind_own(b, malloc((size_t)np * D * sizeof(float)));
        if (!pos) return lm_fail(err, n, "out of memory");
        for (int r = 0; r < np; r++) lm_mat_row(&pm, r, pos + (size_t)r * D);
        v->pos = pos;
    }
    v->blk = (QBlock *)lm_bind_own(b, calloc((size_t)v->depth, sizeof *v->blk));
    if (!v->blk) return lm_fail(err, n, "out of memory");
    for (int i = 0; i < v->depth; i++) {
        QBlock *q = &v->blk[i];
        char p[96];
#define BV(dst, s, len) do { snprintf(p, sizeof p, "blocks.%d.%s", i, s); VEC(dst, p, len); } while (0)
#define BM(dst, s, r, c) do { snprintf(p, sizeof p, "blocks.%d.%s", i, s); MAT(dst, p, r, c, 0); } while (0)
        BV(q->n1w, "norm1.weight", D);
        BV(q->n1b, "norm1.bias", D);
        BV(q->n2w, "norm2.weight", D);
        BV(q->n2b, "norm2.bias", D);
        BM(q->qkv, "attn.qkv.weight", 3 * D, D);
        BV(q->qkv_b, "attn.qkv.bias", 3 * D);
        BM(q->proj, "attn.proj.weight", D, D);
        BV(q->proj_b, "attn.proj.bias", D);
        BM(q->fc1, "mlp.linear_fc1.weight", v->inter, D);
        BV(q->fc1_b, "mlp.linear_fc1.bias", v->inter);
        BM(q->fc2, "mlp.linear_fc2.weight", D, v->inter);
        BV(q->fc2_b, "mlp.linear_fc2.bias", D);
#undef BV
#undef BM
    }
    VEC(v->mn_w, "merger.norm.weight", D);
    VEC(v->mn_b, "merger.norm.bias", D);
    MAT(v->mf1, "merger.linear_fc1.weight", cat, cat, 0);
    VEC(v->mf1_b, "merger.linear_fc1.bias", cat);
    MAT(v->mf2, "merger.linear_fc2.weight", v->info.out_dim, cat, 0);
    VEC(v->mf2_b, "merger.linear_fc2.bias", v->info.out_dim);
#undef VEC
#undef MAT
    return 0;
}

static int gemma_lin(LmVision *v, const char *p, int rows, int cols, GLin *out, char *err, size_t n)
{
    char nm[256];
    snprintf(nm, sizeof nm, "%slinear.weight", p);
    if (lm_bind_mat_any(&v->b, nm, rows, cols, 0, &out->m, err, n)) return -1;
    if (!v->clipped) return 0;
    static const char *const k[4] = { "input_min", "input_max", "output_min", "output_max" };
    float *dst[4] = { &out->in_lo, &out->in_hi, &out->out_lo, &out->out_hi };
    for (int i = 0; i < 4; i++) {
        snprintf(nm, sizeof nm, "%s%s", p, k[i]);
        const float *s = lm_bind_vec(&v->b, nm, 1, 0, 0, 0, err, n);
        if (!s) return -1;
        *dst[i] = *s;
    }
    if (!(out->in_lo <= out->in_hi) || !(out->out_lo <= out->out_hi))
        return lm_fail(err, n, "%s has clamp bounds out of order", p);
    return 0;
}

static int gemma_bind(LmVision *v, char *err, size_t n)
{
    LmBind *b = &v->b;
    const int D = v->hidden, hd = v->head_dim, pin = 3 * v->patch * v->patch;
    if (lm_bind_mat_any(b, "model.vision_tower.patch_embedder.input_proj.weight", D, pin, 0, &v->in_proj, err, n) ||
        lm_bind_mat_any(b, "model.embed_vision.embedding_projection.weight", v->info.out_dim, D, 0, &v->emb, err, n))
        return -1;
    if (!(v->ptab = lm_bind_vec(b, "model.vision_tower.patch_embedder.position_embedding_table",
                                (int64_t)2 * v->pos_size * D, 3, 2, v->pos_size, err, n))) return -1;
    v->gblk = (GBlock *)lm_bind_own(b, calloc((size_t)v->depth, sizeof *v->gblk));
    if (!v->gblk) return lm_fail(err, n, "out of memory");
    for (int i = 0; i < v->depth; i++) {
        GBlock *g = &v->gblk[i];
        char p[160], nm[224];
#define GV(dst, s, len) do { snprintf(nm, sizeof nm, "%s%s", p, s); \
        if (!((dst) = lm_bind_vec(b, nm, (len), 1, (len), 0, err, n))) return -1; } while (0)
#define GL(dst, s, r, c) do { snprintf(nm, sizeof nm, "%s%s", p, s); \
        if (gemma_lin(v, nm, (r), (c), &(dst), err, n)) return -1; } while (0)
        snprintf(p, sizeof p, "model.vision_tower.encoder.layers.%d.", i);
        GV(g->in_n, "input_layernorm.weight", D);
        GV(g->post_attn_n, "post_attention_layernorm.weight", D);
        GV(g->pre_ff_n, "pre_feedforward_layernorm.weight", D);
        GV(g->post_ff_n, "post_feedforward_layernorm.weight", D);
        GV(g->q_n, "self_attn.q_norm.weight", hd);
        GV(g->k_n, "self_attn.k_norm.weight", hd);
        GL(g->q, "self_attn.q_proj.", v->heads * hd, D);
        GL(g->k, "self_attn.k_proj.", v->kv_heads * hd, D);
        GL(g->v, "self_attn.v_proj.", v->kv_heads * hd, D);
        GL(g->o, "self_attn.o_proj.", D, v->heads * hd);
        GL(g->gate, "mlp.gate_proj.", v->inter, D);
        GL(g->up, "mlp.up_proj.", v->inter, D);
        GL(g->down, "mlp.down_proj.", D, v->inter);
#undef GV
#undef GL
    }
    return 0;
}

LmVision *lm_vision_open(const char *model_dir, const char *weights, char *err, size_t errlen)
{
    jval *root = lm_config_read_json(model_dir, err, errlen);
    if (!root) return NULL;
    LmVision *v = (LmVision *)calloc(1, sizeof *v);
    if (!v) { json_free_tree(root); lm_fail(err, errlen, "out of memory"); return NULL; }
    jval *mt = json_get(root, "model_type");
    int rc;
    if (mt && mt->t == J_STR && !strcmp(mt->str, "qwen3_5")) rc = qwen_config(v, root, err, errlen);
    else if (mt && mt->t == J_STR && !strcmp(mt->str, "gemma4")) rc = gemma_config(v, root, err, errlen);
    else rc = lm_fail(err, errlen, "model_type %s has no supported vision encoder",
                      mt && mt->t == J_STR ? mt->str : "(missing)");
    json_free_tree(root);
    if (rc != 0) { free(v); return NULL; }
    if (lm_bind_open(&v->b, weights ? weights : model_dir, weights != NULL, err, errlen) != 0 ||
        (v->info.family == LM_VISION_QWEN ? qwen_bind(v, err, errlen) : gemma_bind(v, err, errlen)) != 0) {
        lm_vision_close(v);
        return NULL;
    }
    return v;
}

void lm_vision_close(LmVision *v)
{
    if (!v) return;
    lm_bind_free(&v->b);
    free(v);
}

const LmVisionInfo *lm_vision_info(const LmVision *v) { return &v->info; }

int lm_vision_tokens(const LmVision *v, int h, int w)
{
    const int u = v->info.unit;
    if (h < u || w < u || h % u || w % u) return -1;
    const long long np = (long long)(h / v->patch) * (w / v->patch);
    if (np > LM_VISION_MAX_PATCHES) return -1;
    if (v->info.family == LM_VISION_GEMMA) {
        if (h / v->patch > v->pos_size || w / v->patch > v->pos_size) return -1;   /* past the position table */
        return (int)(np / (v->pool * v->pool));
    }
    return (int)(np / (v->merge * v->merge));
}

/* ------------------------------------------------------------------------- ops */

/* LayerNorm with bias over each of T rows of n. y may alias x. */
static void layernorm(float *y, const float *x, const float *w, const float *b, int n, int T, double eps)
{
    for (int t = 0; t < T; t++) {
        const float *xr = x + (size_t)t * n;
        float *yr = y + (size_t)t * n;
        double s = 0, ss = 0;
        for (int i = 0; i < n; i++) s += xr[i];
        const double mean = s / n;
        for (int i = 0; i < n; i++) { const double d = xr[i] - mean; ss += d * d; }
        const double r = 1.0 / sqrt(ss / n + eps);
        for (int i = 0; i < n; i++) yr[i] = (float)((xr[i] - mean) * r * w[i] + b[i]);
    }
}

static void add_bias(float *y, const float *b, int n, int T)
{
    for (int t = 0; t < T; t++)
        for (int i = 0; i < n; i++) y[(size_t)t * n + i] += b[i];
}

static float gelu_tanh(float x)
{
    const double v = x;
    return (float)(0.5 * v * (1.0 + tanh(0.7978845608028654 * (v + 0.044715 * v * v * v))));
}

static float gelu_erf(float x)
{
    const double v = x;
    return (float)(0.5 * v * (1.0 + erf(v * 0.7071067811865476)));
}

/* Bilinear tap pair of index on a grid of size along a table side (align_corners, clamped). */
static void taps(int index, int size, int side, int t[2], double wt[2])
{
    const double src = size > 1 ? (double)index * (side - 1) / (size - 1) : 0.0;
    const double f = floor(src);
    for (int o = 0; o < 2; o++) {
        int k = (int)f + o;
        if (k < 0) k = 0;
        if (k > side - 1) k = side - 1;
        const double d = 1.0 - fabs(src - f - o);
        t[o] = k;
        wt[o] = d > 0 ? d : 0;
    }
}

/* Bidirectional attention over T positions. q: [T][heads*hd], k and v: [T][kv_heads*hd] (q, k
 * after rope); query head h reads kv head h / (heads / kv_heads); out: [T][heads*hd];
 * scores: nthreads * T floats. */
static void attention(float *out, const float *q, const float *k, const float *v, int T, int heads,
                      int kv_heads, int hd, float scale, float *scores)
{
    const int D = heads * hd, Dk = kv_heads * hd, group = heads / kv_heads;
#pragma omp parallel for schedule(dynamic, 4)
    for (int i = 0; i < T; i++) {
#ifdef _OPENMP
        float *s = scores + (size_t)omp_get_thread_num() * T;
#else
        float *s = scores;
#endif
        for (int h = 0; h < heads; h++) {
            const float *qi = q + (size_t)i * D + h * hd;
            const int kh = h / group;
            float mx = -INFINITY;
            for (int j = 0; j < T; j++) {
                const float *kj = k + (size_t)j * Dk + kh * hd;
                float d = 0;
                for (int e = 0; e < hd; e++) d += qi[e] * kj[e];
                s[j] = d * scale;
                if (s[j] > mx) mx = s[j];
            }
            double sum = 0;
            for (int j = 0; j < T; j++) { s[j] = expf(s[j] - mx); sum += s[j]; }
            const float inv = (float)(1.0 / sum);
            float *o = out + (size_t)i * D + h * hd;
            for (int e = 0; e < hd; e++) o[e] = 0;
            for (int j = 0; j < T; j++) {
                const float p = s[j] * inv;
                const float *vj = v + (size_t)j * Dk + kh * hd;
                for (int e = 0; e < hd; e++) o[e] += p * vj[e];
            }
        }
    }
}

/* ------------------------------------------------------------------------- encode */

/* Gemma RMSNorm over each of T rows of n: x * (mean(x^2) + eps)^-0.5 * w (w NULL: no scale). */
static void rmsnorm(float *y, const float *x, const float *w, int n, int T, double eps)
{
    for (int t = 0; t < T; t++) {
        const float *xr = x + (size_t)t * n;
        float *yr = y + (size_t)t * n;
        double ss = 0;
        for (int i = 0; i < n; i++) ss += (double)xr[i] * xr[i];
        const double r = 1.0 / sqrt(ss / n + eps);
        for (int i = 0; i < n; i++) yr[i] = (float)(xr[i] * r * (w ? w[i] : 1.0f));
    }
}

/* Y[T][rows] = clip(M clip(X)); tmp holds T * cols floats when the linear is clipped. */
static void glin(const LmVision *v, const GLin *g, float *Y, const float *X, int T, float *tmp)
{
    const int cols = g->m.cols, rows = g->m.rows;
    if (v->clipped) {
        for (size_t i = 0; i < (size_t)T * cols; i++)
            tmp[i] = X[i] < g->in_lo ? g->in_lo : X[i] > g->in_hi ? g->in_hi : X[i];
        X = tmp;
    }
    lm_mat_mul_batch(&g->m, Y, X, T);
    if (v->clipped)
        for (size_t i = 0; i < (size_t)T * rows; i++)
            Y[i] = Y[i] < g->out_lo ? g->out_lo : Y[i] > g->out_hi ? g->out_hi : Y[i];
}

/* Gemma rope on n_heads vectors of one position: channels [0, hd/2) rotate by x, [hd/2, hd)
 * by y, each half as rotate-half with angle p * theta^(-2i/(hd/2)), i < hd/4. */
static void gemma_rope(float *x, int n_heads, int hd, const float *cs, const float *sn)
{
    const int half = hd / 2, q = hd / 4;
    for (int h = 0; h < n_heads; h++)
        for (int part = 0; part < 2; part++) {
            float *p = x + (size_t)h * hd + part * half;
            const float *c = cs + part * q, *s = sn + part * q;
            for (int i = 0; i < q; i++) {
                const float a = p[i], b = p[i + q];
                p[i] = a * c[i] - b * s[i];
                p[i + q] = b * c[i] + a * s[i];
            }
        }
}

static int gemma_encode(LmVision *v, const uint8_t *rgb, int h, int w, int ntok, float *rows, char *err,
                        size_t errlen)
{
    const int P = v->patch, K = v->pool, ph = h / P, pw = w / P, N = ph * pw, D = v->hidden, hd = v->head_dim;
    const int pin = 3 * P * P, Dq = v->heads * hd, Dk = v->kv_heads * hd, half = hd / 2, q4 = hd / 4;
    const int wide = pin > v->inter ? (pin > Dq ? pin : Dq) : (v->inter > Dq ? v->inter : Dq);
    int nthr = 1;
#ifdef _OPENMP
    nthr = omp_get_max_threads();
#endif
    float *X = (float *)malloc((size_t)N * (wide > D ? wide : D) * sizeof(float));
    float *T = (float *)malloc((size_t)N * (wide > D ? wide : D) * sizeof(float));
    float *H = (float *)malloc((size_t)N * D * sizeof(float));
    float *Nn = (float *)malloc((size_t)N * D * sizeof(float));
    float *Q = (float *)malloc((size_t)N * Dq * sizeof(float));
    float *Kb = (float *)malloc((size_t)N * Dk * sizeof(float));
    float *Vb = (float *)malloc((size_t)N * Dk * sizeof(float));
    float *G = (float *)malloc((size_t)N * v->inter * sizeof(float));
    float *cs = (float *)malloc((size_t)N * hd * sizeof(float));      /* per patch: cos[x q4][y q4], sin the same */
    float *sc = (float *)malloc((size_t)nthr * N * sizeof(float));
    int rc = -1;
    if (!X || !T || !H || !Nn || !Q || !Kb || !Vb || !G || !cs || !sc) { lm_fail(err, errlen, "out of memory encoding an image"); goto out; }

    for (int py = 0; py < ph; py++)        /* raster order; each patch vector is [y][x][c] */
        for (int px = 0; px < pw; px++) {
            float *x = X + ((size_t)py * pw + px) * pin;
            for (int y = 0; y < P; y++)
                for (int xx = 0; xx < P; xx++)
                    for (int c = 0; c < 3; c++)
                        *x++ = (float)(2.0 * (rgb[((size_t)(py * P + y) * w + px * P + xx) * 3 + c] / 255.0 - 0.5));
        }
    lm_mat_mul_batch(&v->in_proj, H, X, N);
    for (int py = 0; py < ph; py++)
        for (int px = 0; px < pw; px++) {
            const size_t k = (size_t)py * pw + px;
            const float *tx = v->ptab + (size_t)px * D, *ty = v->ptab + ((size_t)v->pos_size + py) * D;
            for (int i = 0; i < D; i++) H[k * D + i] += tx[i] + ty[i];
            for (int part = 0; part < 2; part++)
                for (int i = 0; i < q4; i++) {
                    const double ang = (part ? py : px) * pow(v->theta, -(2.0 * i) / half);
                    cs[k * hd + part * q4 + i] = (float)cos(ang);
                    cs[k * hd + half + part * q4 + i] = (float)sin(ang);
                }
        }

    for (int l = 0; l < v->depth; l++) {
        const GBlock *g = &v->gblk[l];
        rmsnorm(Nn, H, g->in_n, D, N, v->eps);
        glin(v, &g->q, Q, Nn, N, T);
        glin(v, &g->k, Kb, Nn, N, T);
        glin(v, &g->v, Vb, Nn, N, T);
        rmsnorm(Q, Q, g->q_n, hd, N * v->heads, v->eps);
        rmsnorm(Kb, Kb, g->k_n, hd, N * v->kv_heads, v->eps);
        rmsnorm(Vb, Vb, NULL, hd, N * v->kv_heads, v->eps);
        for (int k = 0; k < N; k++) {
            const float *c = cs + (size_t)k * hd;
            gemma_rope(Q + (size_t)k * Dq, v->heads, hd, c, c + half);
            gemma_rope(Kb + (size_t)k * Dk, v->kv_heads, hd, c, c + half);
        }
        attention(X, Q, Kb, Vb, N, v->heads, v->kv_heads, hd, 1.0f, sc);
        glin(v, &g->o, Nn, X, N, T);
        rmsnorm(Nn, Nn, g->post_attn_n, D, N, v->eps);
        for (size_t i = 0; i < (size_t)N * D; i++) H[i] += Nn[i];
        rmsnorm(Nn, H, g->pre_ff_n, D, N, v->eps);
        glin(v, &g->gate, G, Nn, N, T);
        glin(v, &g->up, X, Nn, N, T);
        for (size_t i = 0; i < (size_t)N * v->inter; i++) {
            const double a = G[i];
            G[i] = (float)(0.5 * a * (1.0 + tanh(0.7978845608028654 * (a + 0.044715 * a * a * a)))) * X[i];
        }
        glin(v, &g->down, Nn, G, N, T);
        rmsnorm(Nn, Nn, g->post_ff_n, D, N, v->eps);
        for (size_t i = 0; i < (size_t)N * D; i++) H[i] += Nn[i];
    }
    /* K x K average pooling, pooled grid row-major, then * sqrt(hidden); embed_vision */
    {
        const int gw = pw / K;
        const double root = sqrt((double)D), inv = 1.0 / (K * K);
        for (int t = 0; t < ntok; t++) {
            const int by = t / gw, bx = t % gw;
            for (int i = 0; i < D; i++) {
                double acc = 0;
                for (int y = 0; y < K; y++)
                    for (int x = 0; x < K; x++) acc += H[((size_t)(by * K + y) * pw + bx * K + x) * D + i];
                Nn[(size_t)t * D + i] = (float)(acc * inv * root);
            }
        }
        rmsnorm(Nn, Nn, NULL, D, ntok, v->eps);
        lm_mat_mul_batch(&v->emb, rows, Nn, ntok);
    }
    rc = 0;
out:
    free(X); free(T); free(H); free(Nn); free(Q); free(Kb); free(Vb); free(G); free(cs); free(sc);
    return rc;
}

int lm_vision_encode(LmVision *v, const uint8_t *rgb, int h, int w, float *rows, char *err, size_t errlen)
{
    const int ntok = lm_vision_tokens(v, h, w);
    if (ntok < 1)
        return lm_fail(err, errlen, "image %dx%d: sides must be multiples of %d and at most %d patches",
                       w, h, v->info.unit, LM_VISION_MAX_PATCHES);
    if (v->info.family == LM_VISION_GEMMA) return gemma_encode(v, rgb, h, w, ntok, rows, err, errlen);
    const int P = v->patch, M = v->merge, gh = h / P, gw = w / P, N = gh * gw, D = v->hidden;
    const int hd = v->head_dim, pin = 3 * v->temporal * P * P, cat = D * M * M;
    int nthr = 1;
#ifdef _OPENMP
    nthr = omp_get_max_threads();
#endif
    const size_t big = (size_t)N * (size_t)(pin > 3 * D ? pin : 3 * D);
    float *X = (float *)malloc(big * sizeof(float));            /* patches, then qkv */
    float *H = (float *)malloc((size_t)N * D * sizeof(float));
    float *Nn = (float *)malloc((size_t)N * D * sizeof(float));
    float *O = (float *)malloc((size_t)N * D * sizeof(float));
    float *F = (float *)malloc((size_t)N * (v->inter > D ? v->inter : D) * sizeof(float));
    float *cs = (float *)malloc((size_t)N * hd * sizeof(float));      /* [N][hd/2] cos, then sin */
    float *sc = (float *)malloc((size_t)nthr * N * sizeof(float));
    int *row = (int *)malloc((size_t)N * 2 * sizeof(int));
    int rc = -1;
    if (!X || !H || !Nn || !O || !F || !cs || !sc || !row) { lm_fail(err, errlen, "out of memory encoding an image"); goto out; }

    /* patch order: merge x merge blocks, row-major inside each block */
    {
        int k = 0;
        for (int bh = 0; bh < gh / M; bh++)
            for (int bw = 0; bw < gw / M; bw++)
                for (int mh = 0; mh < M; mh++)
                    for (int mw = 0; mw < M; mw++, k++) {
                        row[2 * k] = bh * M + mh;
                        row[2 * k + 1] = bw * M + mw;
                    }
    }
    for (int k = 0; k < N; k++) {        /* each patch vector is [c][t][py][px], the frame repeated */
        float *x = X + (size_t)k * pin;
        const int r0 = row[2 * k] * P, c0 = row[2 * k + 1] * P;
        for (int c = 0; c < 3; c++)
            for (int t = 0; t < v->temporal; t++)
                for (int y = 0; y < P; y++)
                    for (int xx = 0; xx < P; xx++)
                        *x++ = (float)((rgb[((size_t)(r0 + y) * w + c0 + xx) * 3 + c] / 255.0 - 0.5) / 0.5);
    }
    lm_mat_mul_batch(&v->pe, H, X, N);
    for (int k = 0; k < N; k++) {
        int th[2], tw[2];
        double wh[2], ww[2];
        taps(row[2 * k], gh, v->side, th, wh);
        taps(row[2 * k + 1], gw, v->side, tw, ww);
        float *hr = H + (size_t)k * D;
        for (int i = 0; i < D; i++) {
            double acc = 0;
            for (int a = 0; a < 2; a++)
                for (int b2 = 0; b2 < 2; b2++)
                    acc += wh[a] * ww[b2] * v->pos[(size_t)(th[a] * v->side + tw[b2]) * D + i];
            hr[i] += v->pe_b[i] + (float)acc;
        }
    }
    /* 2-D rope: pair i of hd/2 uses the row for i < hd/4, the column after */
    {
        const int half = hd / 2, quarter = hd / 4;
        for (int k = 0; k < N; k++)
            for (int i = 0; i < half; i++) {
                const int fi = i < quarter ? i : i - quarter;
                const double inv = pow(v->theta, -(2.0 * fi) / half);
                const double ang = (i < quarter ? row[2 * k] : row[2 * k + 1]) * inv;
                cs[(size_t)k * hd + i] = (float)cos(ang);
                cs[(size_t)k * hd + half + i] = (float)sin(ang);
            }
    }

    for (int l = 0; l < v->depth; l++) {
        const QBlock *q = &v->blk[l];
        layernorm(Nn, H, q->n1w, q->n1b, D, N, 1e-6);
        lm_mat_mul_batch(&q->qkv, X, Nn, N);
        add_bias(X, q->qkv_b, 3 * D, N);
        const int half = hd / 2;
        for (int k = 0; k < N; k++) {
            const float *c = cs + (size_t)k * hd, *s = c + half;
            for (int part = 0; part < 2; part++)            /* q then k */
                for (int hh = 0; hh < v->heads; hh++) {
                    float *x = X + (size_t)k * 3 * D + part * D + hh * hd;
                    for (int i = 0; i < half; i++) {
                        const float a = x[i], b2 = x[i + half];
                        x[i] = a * c[i] - b2 * s[i];
                        x[i + half] = b2 * c[i] + a * s[i];
                    }
                }
        }
        /* split q, k, v out of the [N][3D] rows: q into Nn, k into O, v into F */
        for (int k = 0; k < N; k++) {
            memcpy(Nn + (size_t)k * D, X + (size_t)k * 3 * D, (size_t)D * sizeof(float));
            memcpy(O + (size_t)k * D, X + (size_t)k * 3 * D + D, (size_t)D * sizeof(float));
            memcpy(F + (size_t)k * D, X + (size_t)k * 3 * D + 2 * D, (size_t)D * sizeof(float));
        }
        attention(X, Nn, O, F, N, v->heads, v->heads, hd, (float)(1.0 / sqrt((double)hd)), sc);
        lm_mat_mul_batch(&q->proj, O, X, N);
        add_bias(O, q->proj_b, D, N);
        for (size_t i = 0; i < (size_t)N * D; i++) H[i] += O[i];
        layernorm(Nn, H, q->n2w, q->n2b, D, N, 1e-6);
        lm_mat_mul_batch(&q->fc1, F, Nn, N);
        add_bias(F, q->fc1_b, v->inter, N);
        for (size_t i = 0; i < (size_t)N * v->inter; i++) F[i] = gelu_tanh(F[i]);
        lm_mat_mul_batch(&q->fc2, O, F, N);
        add_bias(O, q->fc2_b, D, N);
        for (size_t i = 0; i < (size_t)N * D; i++) H[i] += O[i];
    }
    /* merger: the M*M patches of a block are consecutive, so [N][D] is [ntok][cat] */
    layernorm(Nn, H, v->mn_w, v->mn_b, D, N, 1e-6);
    lm_mat_mul_batch(&v->mf1, F, Nn, ntok);
    add_bias(F, v->mf1_b, cat, ntok);
    for (size_t i = 0; i < (size_t)ntok * cat; i++) F[i] = gelu_erf(F[i]);
    lm_mat_mul_batch(&v->mf2, rows, F, ntok);
    add_bias(rows, v->mf2_b, v->info.out_dim, ntok);
    rc = 0;
out:
    free(X); free(H); free(Nn); free(O); free(F); free(cs); free(sc); free(row);
    return rc;
}
