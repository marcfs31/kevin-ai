/* lm_ops.c - see lm_ops.h for the contract and for what was ported from
 * kimi-k3-in-c src/core/k3_ops.c (Apache-2.0). */
#include <math.h>
#include <stddef.h>

#include "core/lm_ops.h"

void lm_rmsnorm(float *y, const float *x, const float *w, int n, float eps)
{
    /* double accumulator: 2560 squared terms in float32 loses precision the oracle sees */
    double ss = 0.0;
    for (int i = 0; i < n; i++) ss += (double)x[i] * (double)x[i];
    const float inv = (float)(1.0 / sqrt(ss / (double)n + (double)eps));
    for (int i = 0; i < n; i++) y[i] = (x[i] * inv) * (1.0f + w[i]);
}

void lm_rmsnorm_gated(float *y, const float *x, const float *z, const float *w, int n, float eps)
{
    double ss = 0.0;
    for (int i = 0; i < n; i++) ss += (double)x[i] * (double)x[i];
    const float inv = (float)(1.0 / sqrt(ss / (double)n + (double)eps));
    for (int i = 0; i < n; i++) y[i] = (w[i] * (x[i] * inv)) * lm_silu(z[i]);
}

void lm_l2norm(float *y, const float *x, int n)
{
    double ss = 0.0;
    for (int i = 0; i < n; i++) ss += (double)x[i] * (double)x[i];
    const float inv = (float)(1.0 / sqrt(ss + 1e-6));
    for (int i = 0; i < n; i++) y[i] = x[i] * inv;
}

int lm_rope(float *x, int n_heads, int head_dim, int rot_dim, int pos, float theta)
{
    const int p[3] = { pos, pos, pos };
    return lm_mrope(x, n_heads, head_dim, rot_dim, p, 0, 0, theta);
}

int lm_mrope(float *x, int n_heads, int head_dim, int rot_dim, const int pos[3], int sec_h, int sec_w, float theta)
{
    if (rot_dim < 0 || (rot_dim & 1) || rot_dim > head_dim || n_heads < 0 || head_dim < 0 || sec_h < 0 || sec_w < 0)
        return -1;
    const int half = rot_dim / 2;
    for (int i = 0; i < half; i++) {
        const int axis = (i % 3 == 1 && i < 3 * sec_h) ? 1 : (i % 3 == 2 && i < 3 * sec_w) ? 2 : 0;
        /* double: the angle reaches pos * 1 radians at i = 0, where float loses digits */
        const double inv_freq = pow((double)theta, -2.0 * (double)i / (double)rot_dim);
        const double ang = (double)pos[axis] * inv_freq;
        const float c = (float)cos(ang), s = (float)sin(ang);
        for (int h = 0; h < n_heads; h++) {
            float *hv = x + (size_t)h * head_dim;
            const float a = hv[i], b = hv[i + half];
            hv[i] = a * c - b * s;
            hv[i + half] = b * c + a * s;
        }
    }
    return 0;
}

void lm_conv1d_step(float *y, float *state, const float *x, const float *w, int channels, int k)
{
    const int hist = k - 1;
    for (int c = 0; c < channels; c++) {
        const float cur = x[c];
        float *st = state + (size_t)c * hist;
        const float *wc = w + (size_t)c * k;
        /* taps are ordered oldest..newest: wc[k-1] multiplies the CURRENT input */
        float acc = wc[hist] * cur;
        for (int j = 0; j < hist; j++) acc += wc[j] * st[j];
        for (int j = 0; j + 1 < hist; j++) st[j] = st[j + 1];
        if (hist > 0) st[hist - 1] = cur;
        y[c] = lm_silu(acc);
    }
}

void lm_gdn_gates(float *g, float *beta, const float *a, const float *b,
                  const float *A_log, const float *dt_bias, int n_v_heads)
{
    for (int h = 0; h < n_v_heads; h++) {
        g[h] = -expf(A_log[h]) * lm_softplus(a[h] + dt_bias[h]);
        beta[h] = lm_sigmoid(b[h]);
    }
}

int lm_gated_delta_step(float *S, float *o, const float *q, const float *k, const float *v,
                        const float *g, const float *beta,
                        int n_k_heads, int n_v_heads, int dk, int dv)
{
    if (n_k_heads < 1 || n_v_heads < n_k_heads || n_v_heads % n_k_heads != 0) return -1;
    const int group = n_v_heads / n_k_heads;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (n_v_heads > 1)
#endif
    for (int h = 0; h < n_v_heads; h++) {
        const int kh = h / group;
        const float *qh = q + (size_t)kh * dk;
        const float *kk = k + (size_t)kh * dk;
        const float *vh = v + (size_t)h * dv;
        float *Sh = S + (size_t)h * dk * dv;
        float *oh = o + (size_t)h * dv;
        const float decay = expf(g[h]);
        const float bt = beta[h];

        /* 1. decay the whole state */
        for (size_t i = 0; i < (size_t)dk * dv; i++) Sh[i] *= decay;
        /* 2. kv = S^T k, accumulated in oh (free until step 5) */
        for (int j = 0; j < dv; j++) oh[j] = 0.0f;
        for (int i = 0; i < dk; i++) {
            const float ki = kk[i];
            const float *row = Sh + (size_t)i * dv;
            for (int j = 0; j < dv; j++) oh[j] += ki * row[j];
        }
        /* 3. delta = (v - kv) * beta, in place over kv */
        for (int j = 0; j < dv; j++) oh[j] = (vh[j] - oh[j]) * bt;
        /* 4. S += k outer delta */
        for (int i = 0; i < dk; i++) {
            const float ki = kk[i];
            float *row = Sh + (size_t)i * dv;
            for (int j = 0; j < dv; j++) row[j] += ki * oh[j];
        }
        /* 5. o = S^T q, from the UPDATED state */
        for (int j = 0; j < dv; j++) oh[j] = 0.0f;
        for (int i = 0; i < dk; i++) {
            const float qi = qh[i];
            const float *row = Sh + (size_t)i * dv;
            for (int j = 0; j < dv; j++) oh[j] += qi * row[j];
        }
    }
    return 0;
}

int lm_attn_step(float *out, const float *q, const float *k_cache, const float *v_cache,
                 int n_pos, int n_heads, int n_kv_heads, int head_dim, float *scores)
{
    if (n_pos < 1 || n_kv_heads < 1 || n_heads < n_kv_heads || n_heads % n_kv_heads != 0) return -1;
    const int group = n_heads / n_kv_heads;
    const size_t stride = (size_t)n_kv_heads * head_dim;     /* floats per cached position */
    const float scale = (float)(1.0 / sqrt((double)head_dim));
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (n_heads > 1 && n_pos > 64)
#endif
    for (int h = 0; h < n_heads; h++) {
        const int kvh = h / group;
        const float *qh = q + (size_t)h * head_dim;
        float *sc = scores + (size_t)h * n_pos;
        float mx = -INFINITY;
        for (int t = 0; t < n_pos; t++) {
            const float *kt = k_cache + (size_t)t * stride + (size_t)kvh * head_dim;
            float d = 0.0f;
            for (int i = 0; i < head_dim; i++) d += qh[i] * kt[i];
            d *= scale;
            sc[t] = d;
            if (d > mx) mx = d;
        }
        double sum = 0.0;
        for (int t = 0; t < n_pos; t++) {
            const float e = expf(sc[t] - mx);
            sc[t] = e;
            sum += (double)e;
        }
        const float inv = (float)(1.0 / sum);
        float *oh = out + (size_t)h * head_dim;
        for (int i = 0; i < head_dim; i++) oh[i] = 0.0f;
        for (int t = 0; t < n_pos; t++) {
            const float *vt = v_cache + (size_t)t * stride + (size_t)kvh * head_dim;
            const float p = sc[t] * inv;
            for (int i = 0; i < head_dim; i++) oh[i] += p * vt[i];
        }
    }
    return 0;
}

void lm_mul_sigmoid(float *y, const float *gate, int n)
{
    for (int i = 0; i < n; i++) y[i] *= lm_sigmoid(gate[i]);
}

void lm_silu_mul(float *out, const float *gate, const float *up, int n)
{
    for (int i = 0; i < n; i++) out[i] = lm_silu(gate[i]) * up[i];
}

/* ------------------------------------------------------------------ Gemma 4 */

void lm_rmsnorm_plain(float *y, const float *x, const float *w, int n, float eps)
{
    double ss = 0.0;
    for (int i = 0; i < n; i++) ss += (double)x[i] * (double)x[i];
    const float inv = (float)(1.0 / sqrt(ss / (double)n + (double)eps));
    if (w) for (int i = 0; i < n; i++) y[i] = (x[i] * inv) * w[i];
    else for (int i = 0; i < n; i++) y[i] = x[i] * inv;
}

void lm_gelu_tanh_mul(float *out, const float *gate, const float *up, int n)
{
    for (int i = 0; i < n; i++) out[i] = lm_gelu_tanh(gate[i]) * up[i];
}

int lm_rope_freqs(double *freqs, int head_dim, double theta, double partial)
{
    if (head_dim < 2 || (head_dim & 1) || !(theta > 0.0) || !(partial > 0.0 && partial <= 1.0)) return -1;
    const int half = head_dim / 2;
    const int r = (int)(partial * (double)head_dim / 2.0);
    for (int i = 0; i < half; i++)
        freqs[i] = i < r ? pow(theta, -2.0 * (double)i / (double)head_dim) : 0.0;
    return 0;
}

int lm_rope_freqs_apply(float *x, int n_heads, int head_dim, const double *freqs, int pos)
{
    if (head_dim < 2 || (head_dim & 1) || n_heads < 0 || pos < 0) return -1;
    const int half = head_dim / 2;
    for (int i = 0; i < half; i++) {
        const double ang = (double)pos * freqs[i];
        const float c = (float)cos(ang), s = (float)sin(ang);
        for (int h = 0; h < n_heads; h++) {
            float *hv = x + (size_t)h * head_dim;
            const float a = hv[i], b = hv[i + half];
            hv[i] = a * c - b * s;
            hv[i + half] = b * c + a * s;
        }
    }
    return 0;
}

void lm_logit_softcap(float *x, int n, float cap)
{
    const float inv = 1.0f / cap;
    for (int i = 0; i < n; i++) x[i] = cap * tanhf(x[i] * inv);
}

int lm_attn_window(float *out, const float *q, const float *k_cache, const float *v_cache,
                   int pos, int window, int slots, int n_heads, int n_kv_heads, int head_dim,
                   float scaling, float *scores)
{
    if (pos < 0 || slots < 1 || n_kv_heads < 1 || n_heads < n_kv_heads || n_heads % n_kv_heads != 0) return -1;
    const int first = (window > 0 && pos - window + 1 > 0) ? pos - window + 1 : 0;
    const int nvis = pos - first + 1;
    if (nvis > slots) return -1;
    const int group = n_heads / n_kv_heads;
    const size_t stride = (size_t)n_kv_heads * head_dim;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (n_heads > 1 && nvis > 64)
#endif
    for (int h = 0; h < n_heads; h++) {
        const int kvh = h / group;
        const float *qh = q + (size_t)h * head_dim;
        float *sc = scores + (size_t)h * nvis;
        float mx = -INFINITY;
        for (int t = 0; t < nvis; t++) {
            const float *kt = k_cache + (size_t)((first + t) % slots) * stride + (size_t)kvh * head_dim;
            float d = 0.0f;
            for (int i = 0; i < head_dim; i++) d += qh[i] * kt[i];
            d *= scaling;
            sc[t] = d;
            if (d > mx) mx = d;
        }
        double sum = 0.0;
        for (int t = 0; t < nvis; t++) {
            const float e = expf(sc[t] - mx);
            sc[t] = e;
            sum += (double)e;
        }
        const float inv = (float)(1.0 / sum);
        float *oh = out + (size_t)h * head_dim;
        for (int i = 0; i < head_dim; i++) oh[i] = 0.0f;
        for (int t = 0; t < nvis; t++) {
            const float *vt = v_cache + (size_t)((first + t) % slots) * stride + (size_t)kvh * head_dim;
            const float p = sc[t] * inv;
            for (int i = 0; i < head_dim; i++) oh[i] += p * vt[i];
        }
    }
    return 0;
}
