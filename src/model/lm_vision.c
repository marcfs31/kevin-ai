/* lm_vision.c - see lm_vision.h. Original code; Qwen3.5 maths per docs/PLAN.md "Phase 3"
 * and transformers modeling_qwen3_5.py (Qwen3_5VisionModel), oracle
 * tools/ref_vision_qwen35.py. Activations are f32; norms and softmax sums accumulate in
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

struct LmVision {
    LmVisionInfo info;
    LmBind b;
    int depth, hidden, heads, head_dim, inter, patch, merge, temporal, side;
    double theta;
    LmMat pe;
    const float *pe_b, *pos;          /* pos: [side*side][hidden] */
    QBlock *blk;
    const float *mn_w, *mn_b, *mf1_b, *mf2_b;
    LmMat mf1, mf2;
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
    v->theta = 10000.0;
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

LmVision *lm_vision_open(const char *model_dir, const char *weights, char *err, size_t errlen)
{
    jval *root = lm_config_read_json(model_dir, err, errlen);
    if (!root) return NULL;
    LmVision *v = (LmVision *)calloc(1, sizeof *v);
    if (!v) { json_free_tree(root); lm_fail(err, errlen, "out of memory"); return NULL; }
    jval *mt = json_get(root, "model_type");
    int rc;
    if (mt && mt->t == J_STR && !strcmp(mt->str, "qwen3_5")) rc = qwen_config(v, root, err, errlen);
    else rc = lm_fail(err, errlen, "model_type %s has no supported vision encoder",
                      mt && mt->t == J_STR ? mt->str : "(missing)");
    json_free_tree(root);
    if (rc != 0) { free(v); return NULL; }
    if (lm_bind_open(&v->b, weights ? weights : model_dir, weights != NULL, err, errlen) != 0 ||
        qwen_bind(v, err, errlen) != 0) {
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

/* Bidirectional attention over T positions. q, k, v: [T][heads*hd] (q, k after rope);
 * out: [T][heads*hd]; scores: nthreads * T floats. */
static void attention(float *out, const float *q, const float *k, const float *v, int T, int heads, int hd,
                      float *scores)
{
    const int D = heads * hd;
    const float scale = (float)(1.0 / sqrt((double)hd));
#pragma omp parallel for schedule(dynamic, 4)
    for (int i = 0; i < T; i++) {
#ifdef _OPENMP
        float *s = scores + (size_t)omp_get_thread_num() * T;
#else
        float *s = scores;
#endif
        for (int h = 0; h < heads; h++) {
            const float *qi = q + (size_t)i * D + h * hd;
            float mx = -INFINITY;
            for (int j = 0; j < T; j++) {
                const float *kj = k + (size_t)j * D + h * hd;
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
                const float *vj = v + (size_t)j * D + h * hd;
                for (int e = 0; e < hd; e++) o[e] += p * vj[e];
            }
        }
    }
}

/* ------------------------------------------------------------------------- encode */

int lm_vision_encode(LmVision *v, const uint8_t *rgb, int h, int w, float *rows, char *err, size_t errlen)
{
    const int ntok = lm_vision_tokens(v, h, w);
    if (ntok < 1)
        return lm_fail(err, errlen, "image %dx%d: sides must be multiples of %d and at most %d patches",
                       w, h, v->info.unit, LM_VISION_MAX_PATCHES);
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
        attention(X, Nn, O, F, N, v->heads, hd, sc);
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
