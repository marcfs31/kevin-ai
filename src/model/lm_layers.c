/* lm_layers.c - see lm_layers.h. Original code. */
#include <math.h>
#include <string.h>

#include "core/lm_bf16.h"
#include "core/lm_kernels.h"
#include "core/lm_ops.h"
#include "model/lm_layers.h"
#include "model/lm_times.h"

static void mat_mul(const LmMat *m, float *y, const float *x)
{
    switch (m->kind) {
    case LM_MAT_F32:  lm_matvec_f32(y, x, m->f, m->cols, m->rows); break;
    case LM_MAT_BF16: lm_matvec_bf16(y, x, m->bf, m->cols, m->rows); break;
    case LM_MAT_Q8:
        if (m->int8_acts) {
            int8_t qx[LM_Q8I_MAX_COLS];
            float sx[LM_Q8I_MAX_COLS / LM_Q8_GROUP];
            (void)lm_q8_quantize_acts(qx, sx, x, m->cols);
            (void)lm_matvec_q8i(y, qx, sx, m->q, m->sc, m->cols, m->rows);
        } else {
            (void)lm_matvec_q8(y, x, m->q, m->sc, m->cols, m->rows);
        }
        break;
    default: break;
    }
}

void lm_mat_mul(const LmMat *m, float *y, const float *x)
{
    LM_TIMES_T0(t0);
    mat_mul(m, y, x);
    LM_TIMES_ADD(matmul, t0);
}

void lm_mat_row(const LmMat *m, int r, float *out)
{
    const int c = m->cols;
    switch (m->kind) {
    case LM_MAT_F32:  memcpy(out, m->f + (size_t)r * c, (size_t)c * sizeof(float)); break;
    case LM_MAT_BF16: for (int i = 0; i < c; i++) out[i] = lm_bf16_to_f32(m->bf[(size_t)r * c + i]); break;
    case LM_MAT_Q8: {
        const int ng = c / LM_Q8_GROUP;
        for (int i = 0; i < c; i++)
            out[i] = (float)m->q[(size_t)r * c + i] * m->sc[(size_t)r * ng + i / LM_Q8_GROUP];
    } break;
    default: break;
    }
}

/* ------------------------------------------------------------------ MLP */

size_t lm_mlp_scratch(int inter) { return 2 * (size_t)inter; }

void lm_mlp(float *out, const float *x, const LmMlpW *w, float *scratch)
{
    float *g = scratch, *u = scratch + w->inter;
    lm_mat_mul(&w->gate, g, x);
    lm_mat_mul(&w->up, u, x);
    lm_silu_mul(g, g, u, w->inter);
    lm_mat_mul(&w->down, out, g);
}

/* ------------------------------------------------------------------ linear attention */

size_t lm_linear_scratch(const LmLinDims *d)
{
    const size_t conv_dim = (size_t)(2 * d->n_k * d->dk + d->n_v * d->dv);
    return conv_dim + 2 * (size_t)d->n_v * d->dv + 4 * (size_t)d->n_v;
}

void lm_linear_mixer(float *out, const float *x, const LmLinW *w, const LmLinDims *d,
                     float *conv_state, float *S, float *scratch)
{
    const int kd = d->n_k * d->dk, vd = d->n_v * d->dv, conv_dim = 2 * kd + vd;
    float *qkv = scratch, *z = qkv + conv_dim, *o = z + vd;
    float *b = o + vd, *a = b + d->n_v, *g = a + d->n_v, *beta = g + d->n_v;
    LM_TIMES_T0(t0);
    const double mm0 = lm_times_acc.matmul;

    lm_mat_mul(&w->qkv, qkv, x);
    lm_mat_mul(&w->z, z, x);
    lm_mat_mul(&w->b, b, x);
    lm_mat_mul(&w->a, a, x);

    lm_conv1d_step(qkv, conv_state, qkv, w->conv_w, conv_dim, d->conv_k);

    float *q = qkv, *k = qkv + kd, *v = qkv + 2 * kd;
    const float qscale = (float)(1.0 / sqrt((double)d->dk));
    for (int h = 0; h < d->n_k; h++) {
        lm_l2norm(q + h * d->dk, q + h * d->dk, d->dk);
        lm_l2norm(k + h * d->dk, k + h * d->dk, d->dk);
        for (int i = 0; i < d->dk; i++) q[h * d->dk + i] *= qscale;
    }
    lm_gdn_gates(g, beta, a, b, w->A_log, w->dt_bias, d->n_v);
    (void)lm_gated_delta_step(S, o, q, k, v, g, beta, d->n_k, d->n_v, d->dk, d->dv);

    for (int h = 0; h < d->n_v; h++)
        lm_rmsnorm_gated(o + h * d->dv, o + h * d->dv, z + h * d->dv, w->norm_w, d->dv, d->eps);
    lm_mat_mul(&w->out, out, o);
    LM_TIMES_ADD_OWN(linear, t0, mm0);
}

/* ------------------------------------------------------------------ full attention */

size_t lm_attn_scratch(const LmAttnDims *d, int ctx)
{
    const size_t qd = (size_t)d->heads * d->head_dim, kvd = (size_t)d->kv_heads * d->head_dim;
    return 2 * qd /* qg */ + qd /* q */ + qd /* gate */ + kvd /* k */ + kvd /* v */ + qd /* attn */
         + (size_t)d->heads * (size_t)ctx;
}

void lm_attn_mixer(float *out, const float *x, const LmAttnW *w, const LmAttnDims *d,
                   float *k_cache, float *v_cache, int pos, int ctx, float *scratch)
{
    (void)ctx;
    const int hd = d->head_dim, qd = d->heads * hd, kvd = d->kv_heads * hd;
    float *qg = scratch, *q = qg + 2 * qd, *gate = q + qd, *k = gate + qd, *v = k + kvd;
    float *attn = v + kvd, *scores = attn + qd;
    LM_TIMES_T0(t0);
    const double mm0 = lm_times_acc.matmul;

    lm_mat_mul(&w->q, qg, x);
    lm_mat_mul(&w->k, k, x);
    lm_mat_mul(&w->v, v, x);

    /* q_proj rows are per head [query hd | gate hd] */
    for (int h = 0; h < d->heads; h++) {
        lm_rmsnorm(q + h * hd, qg + (size_t)h * 2 * hd, w->q_norm, hd, d->eps);
        memcpy(gate + h * hd, qg + (size_t)h * 2 * hd + hd, (size_t)hd * sizeof(float));
    }
    for (int h = 0; h < d->kv_heads; h++) lm_rmsnorm(k + h * hd, k + h * hd, w->k_norm, hd, d->eps);
    (void)lm_rope(q, d->heads, hd, d->rot, pos, d->theta);
    (void)lm_rope(k, d->kv_heads, hd, d->rot, pos, d->theta);

    memcpy(k_cache + (size_t)pos * kvd, k, (size_t)kvd * sizeof(float));
    memcpy(v_cache + (size_t)pos * kvd, v, (size_t)kvd * sizeof(float));
    (void)lm_attn_step(attn, q, k_cache, v_cache, pos + 1, d->heads, d->kv_heads, hd, scores);
    lm_mul_sigmoid(attn, gate, qd);
    lm_mat_mul(&w->o, out, attn);
    LM_TIMES_ADD_OWN(attention, t0, mm0);
}

/* ------------------------------------------------------------------ batched (prefill) */

void lm_mat_mul_batch(const LmMat *m, float *Y, const float *X, int T)
{
    LM_TIMES_T0(t0);
    if (m->kind == LM_MAT_Q8 && !m->int8_acts)
        (void)lm_matmul_q8(Y, X, T, m->q, m->sc, m->cols, m->rows);
    else
        for (int t = 0; t < T; t++) mat_mul(m, Y + (size_t)t * m->rows, X + (size_t)t * m->cols);
    LM_TIMES_ADD(matmul, t0);
}

size_t lm_mlp_batch_scratch(int inter, int T) { return 2 * (size_t)inter * (size_t)T; }

void lm_mlp_batch(float *out, const float *x, const LmMlpW *w, float *scratch, int T)
{
    float *g = scratch, *u = scratch + (size_t)w->inter * T;
    lm_mat_mul_batch(&w->gate, g, x, T);
    lm_mat_mul_batch(&w->up, u, x, T);
    for (int t = 0; t < T; t++)
        lm_silu_mul(g + (size_t)t * w->inter, g + (size_t)t * w->inter, u + (size_t)t * w->inter, w->inter);
    lm_mat_mul_batch(&w->down, out, g, T);
}

size_t lm_linear_batch_scratch(const LmLinDims *d, int T)
{
    const size_t conv_dim = (size_t)(2 * d->n_k * d->dk + d->n_v * d->dv), vd = (size_t)d->n_v * d->dv;
    return (size_t)T * (conv_dim + 2 * vd + 2 * (size_t)d->n_v) + 2 * (size_t)d->n_v;
}

void lm_linear_mixer_batch(float *out, const float *x, const LmLinW *w, const LmLinDims *d,
                           float *conv_state, float *S, float *scratch, int T)
{
    const int kd = d->n_k * d->dk, vd = d->n_v * d->dv, conv_dim = 2 * kd + vd;
    float *qkv = scratch, *z = qkv + (size_t)T * conv_dim, *o = z + (size_t)T * vd;
    float *b = o + (size_t)T * vd, *a = b + (size_t)T * d->n_v, *g = a + (size_t)T * d->n_v, *beta = g + d->n_v;
    LM_TIMES_T0(t0);
    const double mm0 = lm_times_acc.matmul;

    lm_mat_mul_batch(&w->qkv, qkv, x, T);
    lm_mat_mul_batch(&w->z, z, x, T);
    lm_mat_mul_batch(&w->b, b, x, T);
    lm_mat_mul_batch(&w->a, a, x, T);

    const float qscale = (float)(1.0 / sqrt((double)d->dk));
    for (int t = 0; t < T; t++) {   /* the state-carrying part, token by token, as lm_linear_mixer */
        float *qkv_t = qkv + (size_t)t * conv_dim, *o_t = o + (size_t)t * vd, *z_t = z + (size_t)t * vd;
        lm_conv1d_step(qkv_t, conv_state, qkv_t, w->conv_w, conv_dim, d->conv_k);
        float *q = qkv_t, *k = qkv_t + kd, *v = qkv_t + 2 * kd;
        for (int h = 0; h < d->n_k; h++) {
            lm_l2norm(q + h * d->dk, q + h * d->dk, d->dk);
            lm_l2norm(k + h * d->dk, k + h * d->dk, d->dk);
            for (int i = 0; i < d->dk; i++) q[h * d->dk + i] *= qscale;
        }
        lm_gdn_gates(g, beta, a + (size_t)t * d->n_v, b + (size_t)t * d->n_v, w->A_log, w->dt_bias, d->n_v);
        (void)lm_gated_delta_step(S, o_t, q, k, v, g, beta, d->n_k, d->n_v, d->dk, d->dv);
        for (int h = 0; h < d->n_v; h++)
            lm_rmsnorm_gated(o_t + h * d->dv, o_t + h * d->dv, z_t + h * d->dv, w->norm_w, d->dv, d->eps);
    }
    lm_mat_mul_batch(&w->out, out, o, T);
    LM_TIMES_ADD_OWN(linear, t0, mm0);
}

size_t lm_attn_batch_scratch(const LmAttnDims *d, int ctx, int T)
{
    const size_t qd = (size_t)d->heads * d->head_dim, kvd = (size_t)d->kv_heads * d->head_dim;
    return (size_t)T * (2 * qd + 2 * kvd + qd) + 2 * qd + (size_t)d->heads * (size_t)ctx;
}

void lm_attn_mixer_batch(float *out, const float *x, const LmAttnW *w, const LmAttnDims *d,
                         float *k_cache, float *v_cache, int pos0, int ctx, float *scratch, int T)
{
    (void)ctx;
    const int hd = d->head_dim, qd = d->heads * hd, kvd = d->kv_heads * hd;
    float *qgb = scratch, *kb = qgb + (size_t)T * 2 * qd, *vb = kb + (size_t)T * kvd;
    float *attnb = vb + (size_t)T * kvd, *q = attnb + (size_t)T * qd, *gate = q + qd, *scores = gate + qd;
    LM_TIMES_T0(t0);
    const double mm0 = lm_times_acc.matmul;

    lm_mat_mul_batch(&w->q, qgb, x, T);
    lm_mat_mul_batch(&w->k, kb, x, T);
    lm_mat_mul_batch(&w->v, vb, x, T);

    for (int t = 0; t < T; t++) {   /* norm, RoPE, cache write and attention, in position order */
        const int pos = pos0 + t;
        float *qg = qgb + (size_t)t * 2 * qd, *k = kb + (size_t)t * kvd, *v = vb + (size_t)t * kvd;
        float *attn = attnb + (size_t)t * qd;
        for (int h = 0; h < d->heads; h++) {
            lm_rmsnorm(q + h * hd, qg + (size_t)h * 2 * hd, w->q_norm, hd, d->eps);
            memcpy(gate + h * hd, qg + (size_t)h * 2 * hd + hd, (size_t)hd * sizeof(float));
        }
        for (int h = 0; h < d->kv_heads; h++) lm_rmsnorm(k + h * hd, k + h * hd, w->k_norm, hd, d->eps);
        (void)lm_rope(q, d->heads, hd, d->rot, pos, d->theta);
        (void)lm_rope(k, d->kv_heads, hd, d->rot, pos, d->theta);
        memcpy(k_cache + (size_t)pos * kvd, k, (size_t)kvd * sizeof(float));
        memcpy(v_cache + (size_t)pos * kvd, v, (size_t)kvd * sizeof(float));
        (void)lm_attn_step(attn, q, k_cache, v_cache, pos + 1, d->heads, d->kv_heads, hd, scores);
        lm_mul_sigmoid(attn, gate, qd);
    }
    lm_mat_mul_batch(&w->o, out, attnb, T);
    LM_TIMES_ADD_OWN(attention, t0, mm0);
}
