/* lm_ops.h - the non-matmul numeric ops of the Qwen3.5 text model. Pure functions over
 * caller-owned buffers: no I/O, no allocation, no global state. Semantics follow
 * transformers modeling_qwen3_5.py (docs/PLAN.md names the classes).
 *
 * Ported from kimi-k3-in-c src/core/k3_ops.c (Apache-2.0): k3_rmsnorm -> lm_rmsnorm,
 * k3_shortconv -> lm_conv1d_step, k3_kda_step -> lm_gated_delta_step. The ports are
 * rewritten, not renamed: RMSNorm is zero-centred (1+w) here and has a gated variant;
 * the conv is one step with a carried state (the original ran T steps); the KDA step has
 * a per-channel decay and a delta write that skips zeros, while the Gated DeltaNet step
 * has one scalar decay per value head, GQA over key heads, and a different write order.
 * RoPE, attention, the gates and the helpers are original. */
#ifndef LM_OPS_H
#define LM_OPS_H

#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- scalar helpers, safe for any finite |x| (no inf/inf, no overflow into NaN) ---- */

/* 1/(1+e^-x) without evaluating expf of a large positive argument. */
static inline float lm_sigmoid(float x)
{
    if (x >= 0.0f) return 1.0f / (1.0f + expf(-x));
    const float e = expf(x);
    return e / (1.0f + e);
}

static inline float lm_silu(float x) { return x * lm_sigmoid(x); }

/* log(1+e^x). For x > 20 the result is x to within float precision (torch's threshold),
 * which also avoids expf overflow; for very negative x, log1pf keeps full precision. */
static inline float lm_softplus(float x)
{
    if (x > 20.0f) return x;
    return log1pf(expf(x));
}

/* ---- norms ---- */

/* Zero-centred RMSNorm: y = x * rsqrt(mean(x^2) + eps) * (1 + w). Used for the layer
 * norms, q_norm, k_norm and the final norm. y may alias x. */
void lm_rmsnorm(float *y, const float *x, const float *w, int n, float eps);

/* Gated RMSNorm (linear_attn.norm): y = x * rsqrt(mean(x^2) + eps) * w * silu(z), with
 * PLAIN w (not 1 + w). y may alias x. */
void lm_rmsnorm_gated(float *y, const float *x, const float *z, const float *w, int n, float eps);

/* l2norm: y = x * rsqrt(sum(x^2) + 1e-6). y may alias x. */
void lm_l2norm(float *y, const float *x, int n);

/* ---- rotary embedding ---- */

/* Partial rotate-half RoPE for the heads of one position, in place. x is n_heads vectors
 * of head_dim floats. Only the first rot_dim dims of each head rotate (rot_dim even, <=
 * head_dim): for i < rot_dim/2, angle = pos * theta^(-2i/rot_dim),
 * o[i] = x[i]*cos - x[i+rot/2]*sin, o[i+rot/2] = x[i+rot/2]*cos + x[i]*sin; dims >=
 * rot_dim pass through. Returns -1 on a bad shape (rot_dim odd or out of range). */
int lm_rope(float *x, int n_heads, int head_dim, int rot_dim, int pos, float theta);

/* Interleaved M-RoPE (Qwen3.5). Same pairing as lm_rope. Frequency i (0 .. rot_dim/2) uses
 * position h when i % 3 == 1 and i < h_lim, w when i % 3 == 2 and i < w_lim, otherwise t.
 * h_lim and w_lim are mrope_section[1] * 3 and mrope_section[2] * 3. When t == h == w the
 * result is bit-identical to lm_rope at that position. Returns -1 on a bad shape or a
 * negative position. */
int lm_mrope(float *x, int n_heads, int head_dim, int rot_dim,
             int t, int h, int w, float theta, int h_lim, int w_lim);

/* ---- linear attention (Gated DeltaNet) ---- */

/* One causal depthwise conv1d step + SiLU. w is [channels][k] (the checkpoint's
 * [C,1,k] weight, no bias); w[c*k + k-1] multiplies the CURRENT input. state holds the
 * previous k-1 inputs per channel, oldest first: state[c*(k-1) + j]; it is updated in
 * place, so a decode loop carries it forward (zero it at position 0). y may alias x. */
void lm_conv1d_step(float *y, float *state, const float *x, const float *w, int channels, int k);

/* Per value head: g = -exp(A_log) * softplus(a + dt_bias), beta = sigmoid(b). */
void lm_gdn_gates(float *g, float *beta, const float *a, const float *b,
                  const float *A_log, const float *dt_bias, int n_v_heads);

/* One gated-delta-rule step for all value heads.
 *   S     [n_v_heads][dk][dv] row-major, updated in place (zero at position 0)
 *   o     [n_v_heads*dv] output
 *   q, k  [n_k_heads*dk], already l2-normalised; q ALSO pre-scaled by 1/sqrt(dk)
 *   v     [n_v_heads*dv]
 *   g     [n_v_heads] log decay (<= 0), beta [n_v_heads] in (0,1)  (see lm_gdn_gates)
 * Value head h reads key head h / (n_v_heads / n_k_heads): each key head serves that many
 * CONSECUTIVE value heads (repeat_interleave). n_v_heads must be a multiple of n_k_heads.
 * Per head, in this order: S *= exp(g); kv = S^T k; delta = (v - kv) * beta;
 * S += k outer delta; o = S^T q. Returns -1 on a bad head ratio. */
int lm_gated_delta_step(float *S, float *o, const float *q, const float *k, const float *v,
                        const float *g, const float *beta,
                        int n_k_heads, int n_v_heads, int dk, int dv);

/* ---- full attention ---- */

/* One attention step for the newest position over a KV cache that already holds it.
 *   out      [n_heads*head_dim]
 *   q        [n_heads*head_dim], after q_norm and RoPE
 *   k_cache  [n_pos][n_kv_heads][head_dim], v_cache the same; position n_pos-1 is the
 *            new token (the caller appends before calling), so the step is causal by
 *            construction: it sees exactly positions 0..n_pos-1
 *   scores   scratch of n_heads*n_pos floats
 * Query head h uses KV head h / (n_heads / n_kv_heads); softmax of q.k / sqrt(head_dim).
 * Returns -1 when n_heads is not a multiple of n_kv_heads or n_pos < 1. */
int lm_attn_step(float *out, const float *q, const float *k_cache, const float *v_cache,
                 int n_pos, int n_heads, int n_kv_heads, int head_dim, float *scores);

/* ---- elementwise ---- */

/* y[i] *= sigmoid(gate[i]): the attention output gate. */
void lm_mul_sigmoid(float *y, const float *gate, int n);

/* out[i] = silu(gate[i]) * up[i]: the SiLU-gated MLP combine. out may alias gate or up. */
void lm_silu_mul(float *out, const float *gate, const float *up, int n);


/* ======================================================================== Gemma 4 ops
 * Semantics per docs/PLAN.md "Phase 2" and transformers modeling_gemma4.py; the oracle is
 * tools/ref_gemma4.py, the values tests/fixtures/ops/{gemma_rmsnorm,gelu_tanh,gelu_tanh_mul,
 * gemma_rope,logit_softcap,sliding_attention}.json. */

/* Gemma RMSNorm: y = x * rsqrt(mean(x^2) + eps) * w with PLAIN w (NOT 1 + w). w == NULL is
 * the weightless norm (v_norm, w = 1). y may alias x. */
void lm_rmsnorm_plain(float *y, const float *x, const float *w, int n, float eps);

/* gelu_pytorch_tanh: 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3))). */
static inline float lm_gelu_tanh(float x)
{
    const float k = 0.7978845608028654f;       /* sqrt(2/pi) */
    return 0.5f * x * (1.0f + tanhf(k * (x + 0.044715f * x * x * x)));
}

/* out[i] = gelu_tanh(gate[i]) * up[i]: the Gemma MLP combine. out may alias gate or up. */
void lm_gelu_tanh_mul(float *out, const float *gate, const float *up, int n);

/* Per-pair RoPE frequencies for rotate-half over the WHOLE head (pairs i, i + head_dim/2):
 * freqs[head_dim/2]. r = (int)(partial * head_dim / 2); freqs[i] = theta^(-2i/head_dim) for
 * i < r, 0 for i >= r (those dims pass through). partial = 1 is the default rope. The
 * exponent denominator is head_dim, not the rotated width (differs from lm_rope). Returns -1
 * for an odd or non-positive head_dim, theta <= 0 or partial outside (0, 1]. */
int lm_rope_freqs(double *freqs, int head_dim, double theta, double partial);

/* Rotate-half RoPE with explicit frequencies for the n_heads vectors of one position, in
 * place: angle = pos * freqs[i] computed in double, o[i] = a cos - b sin, o[i+d/2] = b cos + a sin.
 * A zero frequency leaves the pair unchanged exactly. Returns -1 on a bad shape. */
int lm_rope_freqs_apply(float *x, int n_heads, int head_dim, const double *freqs, int pos);

/* x[i] = cap * tanh(x[i] / cap), in place (the final logit softcap). cap > 0. */
void lm_logit_softcap(float *x, int n, float cap);

/* One attention step for the query at position pos over a KV cache of `slots` positions
 * (position j lives in slot j % slots, so a ring of the last `window` positions and a plain
 * linear cache are the same call). Keys j with first <= j <= pos are visible, where first =
 * 0 when window <= 0 and max(0, pos - window + 1) otherwise; the cache must hold all of them
 * (slots >= pos - first + 1). Scores are q.k * scaling (Gemma: 1.0, no 1/sqrt(d)), softmax,
 * then the weighted sum of v. Query head h uses KV head h / (n_heads / n_kv_heads).
 *   out [n_heads*head_dim]; q [n_heads*head_dim]; k_cache, v_cache [slots][n_kv_heads*head_dim];
 *   scores scratch of n_heads * min(pos+1, window > 0 ? window : pos+1) floats.
 * Returns -1 on a bad shape or a cache too small. */
int lm_attn_window(float *out, const float *q, const float *k_cache, const float *v_cache,
                   int pos, int window, int slots, int n_heads, int n_kv_heads, int head_dim,
                   float scaling, float *scores);

#ifdef __cplusplus
}
#endif
#endif /* LM_OPS_H */
