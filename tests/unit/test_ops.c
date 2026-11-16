/* test_ops.c - the model ops against hand-computed values, each with a negative case
 * showing the wrong variant (plain w instead of 1+w, no conv carry, wrong GQA mapping,
 * wrong delta-rule order, non-causal attention) does NOT produce the same numbers. The
 * oracle fixtures under tests/fixtures/ops/ are wired in by the model tests, not here.
 * Style follows kimi-k3-in-c tests/unit/test_ops.c (Apache-2.0). */
#include "lm_test.h"

#include "core/lm_ops.h"

#define LN2 0.69314718055994530942

static double dsilu(double x) { return x / (1.0 + exp(-x)); }

static void test_scalar_helpers(void)
{
    CHECK(lm_sigmoid(0.0f) == 0.5f);
    CHECK_NEAR(lm_sigmoid(2.0f), 1.0 / (1.0 + exp(-2.0)), 1e-7);
    CHECK_NEAR(lm_sigmoid(-2.0f), 1.0 / (1.0 + exp(2.0)), 1e-7);
    /* large |x|: saturates, never NaN or Inf (expf(1000) would overflow in the naive form) */
    CHECK(lm_sigmoid(1000.0f) == 1.0f);
    CHECK(lm_sigmoid(-1000.0f) == 0.0f);
    CHECK(lm_sigmoid(-3.0e38f) == 0.0f && lm_sigmoid(3.0e38f) == 1.0f);
    CHECK(!isnan(lm_silu(-1000.0f)) && fabsf(lm_silu(-1000.0f)) < 1e-30f);
    CHECK(lm_silu(1000.0f) == 1000.0f);
    CHECK_NEAR(lm_silu(1.0f), dsilu(1.0), 1e-7);
    CHECK_NEAR(lm_softplus(0.0f), LN2, 1e-7);
    CHECK(lm_softplus(1000.0f) == 1000.0f);                       /* expf(1000) would be Inf */
    CHECK(lm_softplus(-1000.0f) == 0.0f);
    CHECK_NEAR(lm_softplus(-20.0f), 2.0611536e-9, 1e-14);          /* log1p keeps the tail, log(1+e) would give 0 */
    CHECK_NEAR(lm_softplus(19.0f), log1p(exp(19.0)), 1e-5);
    CHECK_NEAR(lm_softplus(25.0f), 25.0, 1e-6);
}

static void test_rmsnorm(void)
{
    const float x[2] = { 3, 4 }, w[2] = { 0, 1 };
    float y[2];
    const double inv = 1.0 / sqrt(12.5);
    lm_rmsnorm(y, x, w, 2, 0.0f);
    CHECK_NEAR(y[0], 3 * inv * 1.0, 1e-6);          /* w = 0 means scale 1 (zero-centred) */
    CHECK_NEAR(y[1], 4 * inv * 2.0, 1e-6);
    CHECK(fabsf(y[0]) > 0.5f);                      /* NEGATIVE: plain w would zero this */
    /* eps enters under the root */
    lm_rmsnorm(y, x, w, 2, 1.0f);
    CHECK_NEAR(y[0], 3.0 / sqrt(13.5), 1e-6);
    /* in place */
    float z[2] = { 3, 4 };
    lm_rmsnorm(z, z, w, 2, 0.0f);
    CHECK_NEAR(z[1], 4 * inv * 2.0, 1e-6);
    /* all-zero input stays finite thanks to eps */
    const float zero[2] = { 0, 0 };
    lm_rmsnorm(y, zero, w, 2, 1e-6f);
    CHECK(y[0] == 0.0f && y[1] == 0.0f);
}

static void test_rmsnorm_gated(void)
{
    const float x[2] = { 3, 4 }, z[2] = { 0, 1 }, w[2] = { 1, 2 };
    float y[2];
    const double inv = 1.0 / sqrt(12.5);
    lm_rmsnorm_gated(y, x, z, w, 2, 0.0f);
    CHECK_NEAR(y[0], 0.0, 1e-7);                                   /* silu(0) = 0 */
    CHECK_NEAR(y[1], 2.0 * 4 * inv * dsilu(1.0), 1e-6);            /* plain w = 2, not 1 + 2 */
    CHECK(fabsf(y[1] - (float)(3.0 * 4 * inv * dsilu(1.0))) > 0.1f);   /* NEGATIVE: (1+w) differs */
    const float zneg[2] = { 5, -5 };
    lm_rmsnorm_gated(y, x, zneg, w, 2, 0.0f);
    CHECK_NEAR(y[0], 1.0 * 3 * inv * dsilu(5.0), 1e-6);
    CHECK_NEAR(y[1], 2.0 * 4 * inv * dsilu(-5.0), 1e-6);
    float xi[2] = { 3, 4 };
    lm_rmsnorm_gated(xi, xi, zneg, w, 2, 0.0f);                    /* aliasing */
    CHECK_NEAR(xi[0], 1.0 * 3 * inv * dsilu(5.0), 1e-6);
}

static void test_l2norm(void)
{
    const float x[2] = { 3, 4 };
    float y[2];
    lm_l2norm(y, x, 2);
    CHECK_NEAR(y[0], 0.6, 2e-6);
    CHECK_NEAR(y[1], 0.8, 2e-6);
    const float tiny[2] = { 1e-4f, 0 };
    lm_l2norm(y, tiny, 2);
    CHECK_NEAR(y[0], 1e-4 / sqrt(1e-8 + 1e-6), 1e-8);              /* eps dominates: not unit length */
    const float zero[2] = { 0, 0 };
    lm_l2norm(y, zero, 2);
    CHECK(y[0] == 0.0f && y[1] == 0.0f);
}

static void test_rope(void)
{
    /* head_dim 8, rot 4, theta 100, pos 1: angles 1*100^0 = 1 and 1*100^(-2/4) = 0.1 */
    float x[16];
    for (int i = 0; i < 16; i++) x[i] = (float)(i % 8 + 1);        /* two identical heads 1..8 */
    float ref[16];
    memcpy(ref, x, sizeof x);
    CHECK(lm_rope(x, 2, 8, 4, 1, 100.0f) == 0);
    const double c0 = cos(1.0), s0 = sin(1.0), c1 = cos(0.1), s1 = sin(0.1);
    for (int h = 0; h < 2; h++) {
        const float *o = x + h * 8;
        CHECK_NEAR(o[0], 1 * c0 - 3 * s0, 1e-6);
        CHECK_NEAR(o[2], 3 * c0 + 1 * s0, 1e-6);
        CHECK_NEAR(o[1], 2 * c1 - 4 * s1, 1e-6);
        CHECK_NEAR(o[3], 4 * c1 + 2 * s1, 1e-6);
        for (int i = 4; i < 8; i++) CHECK(o[i] == ref[h * 8 + i]);  /* pass-through tail, bit exact */
    }
    /* NEGATIVE: interleaved pairing (x0,x1),(x2,x3) would give a different o[1] */
    CHECK(fabs((double)x[1] - (2 * c0 + 1 * s0)) > 0.1);
    /* position 0 is the identity; rotation preserves the norm of the rotated part */
    memcpy(x, ref, sizeof x);
    CHECK(lm_rope(x, 2, 8, 4, 0, 100.0f) == 0 && !memcmp(x, ref, sizeof x));
    memcpy(x, ref, sizeof x);
    lm_rope(x, 1, 8, 4, 37, 1e7f);
    CHECK_NEAR(x[0] * x[0] + x[2] * x[2], 1 + 9, 1e-4);
    CHECK_NEAR(x[1] * x[1] + x[3] * x[3], 4 + 16, 1e-4);
    /* full rotation (rot == head_dim) and the Qwen shape: 64 of 256 dims */
    float full[4] = { 1, 0, 0, 0 };
    lm_rope(full, 1, 4, 4, 2, 10.0f);                              /* i=0: angle 2 */
    CHECK_NEAR(full[0], cos(2.0), 1e-6);
    CHECK_NEAR(full[2], sin(2.0), 1e-6);
    float big[256];
    for (int i = 0; i < 256; i++) big[i] = 1.0f;
    CHECK(lm_rope(big, 1, 256, 64, 1000, 1e7f) == 0);
    for (int i = 64; i < 256; i++) CHECK(big[i] == 1.0f);
    CHECK_NEAR(big[0], cos(1000.0) - sin(1000.0), 1e-5);
    CHECK_NEAR(big[32], cos(1000.0) + sin(1000.0), 1e-5);
    /* bad shapes */
    CHECK(lm_rope(x, 1, 8, 3, 1, 100.0f) == -1);
    CHECK(lm_rope(x, 1, 8, 10, 1, 100.0f) == -1);
}

static void test_conv1d(void)
{
    /* channel 0: w = [1,2,3,4] (4 multiplies the CURRENT input); channel 1: w = [0,0,0,1] */
    const float w[8] = { 1, 2, 3, 4, 0, 0, 0, 1 };
    float state[6] = { 0 };
    float y[2];
    float in[2] = { 1, 5 };
    lm_conv1d_step(y, state, in, w, 2, 4);
    CHECK_NEAR(y[0], dsilu(4.0), 1e-6);
    CHECK_NEAR(y[1], dsilu(5.0), 1e-6);
    CHECK(state[2] == 1.0f && state[5] == 5.0f && state[0] == 0.0f);   /* newest input last */
    /* next input 0: the earlier 1 now sits at tap w[2] = 3. Without the carried state this is silu(0) = 0. */
    in[0] = 0; in[1] = 0;
    lm_conv1d_step(y, state, in, w, 2, 4);
    CHECK_NEAR(y[0], dsilu(3.0), 1e-6);
    CHECK(fabsf(y[0]) > 1.0f);                                          /* NEGATIVE: not the stateless answer */
    CHECK_NEAR(y[1], 0.0, 1e-7);                                        /* channel 1 only sees the current input */
    lm_conv1d_step(y, state, in, w, 2, 4);
    CHECK_NEAR(y[0], dsilu(2.0), 1e-6);
    lm_conv1d_step(y, state, in, w, 2, 4);
    CHECK_NEAR(y[0], dsilu(1.0), 1e-6);
    lm_conv1d_step(y, state, in, w, 2, 4);
    CHECK_NEAR(y[0], 0.0, 1e-7);                                        /* the 1 has left the window */
    /* in place: y may alias x */
    float st2[3] = { 0, 0, 0 }, io[1] = { 2.0f }, w1[4] = { 1, 1, 1, 1 };
    lm_conv1d_step(io, st2, io, w1, 1, 4);
    CHECK_NEAR(io[0], dsilu(2.0), 1e-6);
    /* k = 1: no history at all */
    float y1, x1 = -3.0f, w11 = 2.0f;
    lm_conv1d_step(&y1, NULL, &x1, &w11, 1, 1);
    CHECK_NEAR(y1, dsilu(-6.0), 1e-6);
}

static void test_gates(void)
{
    float g[2], b[2];
    const float a[2] = { 0, 1000 }, bb[2] = { 0, -1000 }, A[2] = { 0, 0 }, dt[2] = { 0, 0 };
    lm_gdn_gates(g, b, a, bb, A, dt, 2);
    CHECK_NEAR(g[0], -LN2, 1e-6);
    CHECK_NEAR(b[0], 0.5, 1e-7);
    CHECK(g[1] == -1000.0f && b[1] == 0.0f);                           /* large |x| stays finite */
    const float a2[1] = { 0.5f }, b2[1] = { 2.0f }, A2[1] = { (float)log(3.0) }, dt2[1] = { 0.25f };
    lm_gdn_gates(g, b, a2, b2, A2, dt2, 1);
    CHECK_NEAR(g[0], -3.0 * log1p(exp(0.75)), 1e-6);                   /* a and dt_bias are ADDED inside softplus */
    CHECK_NEAR(b[0], 1.0 / (1.0 + exp(-2.0)), 1e-7);
}

static void test_gated_delta(void)
{
    /* one key head, two value heads, dk = dv = 2 */
    float S[8] = { 0 }, o[4];
    const float q[2] = { 1, 0 }, k[2] = { 1, 0 }, v[4] = { 2, 4, 2, 4 };
    float g[2] = { 0, 0 }, beta[2] = { 0.5f, 0.5f };
    CHECK(lm_gated_delta_step(S, o, q, k, v, g, beta, 1, 2, 2, 2) == 0);
    /* kv = 0, delta = (v - 0) * 0.5 = [1,2], S row0 = [1,2], o = row0 */
    CHECK(S[0] == 1.0f && S[1] == 2.0f && S[2] == 0.0f && S[3] == 0.0f);
    CHECK(o[0] == 1.0f && o[1] == 2.0f && o[2] == 1.0f && o[3] == 2.0f);
    /* second step with decay exp(g) = 0.5: S row0 = [.5,1]; kv = [.5,1]; delta = (v-kv)*.5 = [.75,1.5];
     * S row0 = [1.25,2.5]; o is read from the UPDATED state */
    g[0] = g[1] = -(float)LN2;
    lm_gated_delta_step(S, o, q, k, v, g, beta, 1, 2, 2, 2);
    CHECK_NEAR(S[0], 1.25, 1e-6);
    CHECK_NEAR(S[1], 2.5, 1e-6);
    CHECK_NEAR(o[0], 1.25, 1e-6);
    CHECK_NEAR(o[1], 2.5, 1e-6);
    /* NEGATIVE (order): reading before the write would give the decayed state [.5,1];
     * skipping the decay before the kv read would give delta = [.5, 1] and S = [1.5, 3] */
    CHECK(fabsf(o[0] - 0.5f) > 0.5f);
    CHECK(fabsf(S[0] - 1.5f) > 0.1f);
    /* the key is used on both reads: k = [0,1] touches row 1 only */
    float S2[4] = { 0 }, o2[2];
    const float q2[2] = { 0, 1 }, k2[2] = { 0, 1 }, v2[2] = { 3, 6 }, g2[1] = { 0 }, b2[1] = { 1.0f };
    lm_gated_delta_step(S2, o2, q2, k2, v2, g2, b2, 1, 1, 2, 2);
    CHECK(S2[0] == 0 && S2[1] == 0 && S2[2] == 3.0f && S2[3] == 6.0f && o2[0] == 3.0f && o2[1] == 6.0f);
    /* beta = 0 writes nothing */
    float S3[4] = { 1, 2, 3, 4 }, o3[2];
    const float b0[1] = { 0.0f };
    lm_gated_delta_step(S3, o3, q2, k2, v2, g2, b0, 1, 1, 2, 2);
    CHECK(S3[2] == 3.0f && S3[3] == 4.0f && o3[0] == 3.0f && o3[1] == 4.0f);
}

static void test_gated_delta_gqa(void)
{
    /* 2 key heads serve 4 value heads: value head h reads key head h / 2 (consecutive).
     * dk = 2, dv = 1. beta = 0 and g = 0 freeze the state, so o_h = S_h[:,0] . q_kh. */
    float S[8], o[4];
    for (int h = 0; h < 4; h++) { S[h * 2] = (float)(10 * h + 1); S[h * 2 + 1] = (float)(10 * h + 2); }
    const float q[4] = { 1, 0, 0, 1 }, k[4] = { 0, 0, 0, 0 }, v[4] = { 0, 0, 0, 0 };
    const float g[4] = { 0, 0, 0, 0 }, beta[4] = { 0, 0, 0, 0 };
    CHECK(lm_gated_delta_step(S, o, q, k, v, g, beta, 2, 4, 2, 1) == 0);
    /* heads 0,1 -> q head 0 = [1,0]: S[0]; heads 2,3 -> q head 1 = [0,1]: S[1] */
    CHECK(o[0] == 1.0f && o[1] == 11.0f && o[2] == 22.0f && o[3] == 32.0f);
    /* NEGATIVE: the wrong (h % n_k_heads) mapping would give 1, 12, 21, 32 */
    CHECK(o[1] != 12.0f && o[2] != 21.0f);
    /* the same key head feeds two value heads with their own state and value */
    float S2[8] = { 0 }, o2[2];
    const float q2[2] = { 1, 0 }, k2[2] = { 1, 0 }, v2[2] = { 5, 7 }, g2[2] = { 0, 0 }, b2[2] = { 1, 1 };
    lm_gated_delta_step(S2, o2, q2, k2, v2, g2, b2, 1, 2, 2, 1);
    CHECK(o2[0] == 5.0f && o2[1] == 7.0f && S2[0] == 5.0f && S2[2] == 7.0f);
    /* a head count that is not a multiple is refused */
    CHECK(lm_gated_delta_step(S, o, q, k, v, g, beta, 2, 3, 2, 1) == -1);
    CHECK(lm_gated_delta_step(S, o, q, k, v, g, beta, 4, 2, 2, 1) == -1);
}

/* The five-pass delta rule as first written, kept as a bit-exact oracle for the fused kernel. */
static void delta_ref(float *S, float *o, const float *q, const float *k, const float *v,
                      const float *g, const float *beta, int n_k, int n_v, int dk, int dv)
{
    const int group = n_v / n_k;
    for (int h = 0; h < n_v; h++) {
        const float *qh = q + (size_t)(h / group) * dk, *kk = k + (size_t)(h / group) * dk;
        const float *vh = v + (size_t)h * dv;
        float *Sh = S + (size_t)h * dk * dv, *oh = o + (size_t)h * dv;
        const float decay = expf(g[h]), bt = beta[h];
        for (size_t i = 0; i < (size_t)dk * dv; i++) Sh[i] *= decay;
        for (int j = 0; j < dv; j++) oh[j] = 0.0f;
        for (int i = 0; i < dk; i++)
            for (int j = 0; j < dv; j++) oh[j] += kk[i] * Sh[(size_t)i * dv + j];
        for (int j = 0; j < dv; j++) oh[j] = (vh[j] - oh[j]) * bt;
        for (int i = 0; i < dk; i++)
            for (int j = 0; j < dv; j++) Sh[(size_t)i * dv + j] += kk[i] * oh[j];
        for (int j = 0; j < dv; j++) oh[j] = 0.0f;
        for (int i = 0; i < dk; i++)
            for (int j = 0; j < dv; j++) oh[j] += qh[i] * Sh[(size_t)i * dv + j];
    }
}

static float urand(unsigned *s)
{
    *s = *s * 1664525u + 1013904223u;
    return (float)(*s >> 8) / 16777216.0f * 2.0f - 1.0f;
}

/* Qwen3.5-4B shape (dk = dv = 128, GQA 2) with fewer heads, over several steps: the
 * fused kernel must match the five-pass order bit for bit, state and output. */
static void test_gated_delta_bitexact(void)
{
    enum { NK = 2, NV = 4, DK = 128, DV = 128, STEPS = 5 };
    static float S[NV * DK * DV], Sr[NV * DK * DV], o[NV * DV], orf[NV * DV];
    float q[NK * DK], k[NK * DK], v[NV * DV], g[NV], beta[NV];
    unsigned seed = 12345u;
    for (int i = 0; i < NV * DK * DV; i++) S[i] = Sr[i] = urand(&seed) * 0.1f;
    int ok = 1;
    for (int t = 0; t < STEPS; t++) {
        for (int i = 0; i < NK * DK; i++) { q[i] = urand(&seed) * 0.09f; k[i] = urand(&seed) * 0.09f; }
        for (int i = 0; i < NV * DV; i++) v[i] = urand(&seed);
        for (int h = 0; h < NV; h++) { g[h] = -0.5f + 0.4f * urand(&seed); beta[h] = 0.5f + 0.4f * urand(&seed); }
        CHECK(lm_gated_delta_step(S, o, q, k, v, g, beta, NK, NV, DK, DV) == 0);
        delta_ref(Sr, orf, q, k, v, g, beta, NK, NV, DK, DV);
        ok &= memcmp(S, Sr, sizeof S) == 0 && memcmp(o, orf, sizeof o) == 0;
    }
    CHECK(ok);
}

static void test_attention(void)
{
    /* head_dim 2, 4 query heads over 2 KV heads (group 2), 2 cached positions */
    const float K[8] = { 1, 0, 0, 1,    0, 1, 1, 0 };     /* [pos][kv][d]: p0 kv0 [1,0] kv1 [0,1]; p1 kv0 [0,1] kv1 [1,0] */
    const float V[8] = { 10, 0, 0, 20,  0, 30, 40, 0 };
    float scores[8], out[8];
    const double s = 1.0 / sqrt(2.0);
    const float a = 3.0f;
    /* heads 0,1 -> kv 0; heads 2,3 -> kv 1. q0, q2 = 0 (uniform); q1, q3 = [a, 0] */
    const float q[8] = { 0, 0,  a, 0,  0, 0,  a, 0 };
    CHECK(lm_attn_step(out, q, K, V, 2, 4, 2, 2, scores) == 0);
    CHECK_NEAR(out[0], 5.0, 1e-6);  CHECK_NEAR(out[1], 15.0, 1e-6);          /* mean of kv0 values */
    const double e = exp((double)a * s);
    double w0 = e / (e + 1.0), w1 = 1.0 / (e + 1.0);                          /* head 1: scores a*s on p0, 0 on p1 */
    CHECK_NEAR(out[2], w0 * 10 + w1 * 0, 1e-5);
    CHECK_NEAR(out[3], w0 * 0 + w1 * 30, 1e-5);
    CHECK_NEAR(out[4], 20.0, 1e-6); CHECK_NEAR(out[5], 10.0, 1e-6);          /* head 2 uses kv 1, NOT kv 0 */
    w1 = e / (e + 1.0); w0 = 1.0 / (e + 1.0);                                 /* head 3: kv1 keys p0 [0,1] -> 0, p1 [1,0] -> a*s */
    CHECK_NEAR(out[6], w0 * 0 + w1 * 40, 1e-5);
    CHECK_NEAR(out[7], w0 * 20 + w1 * 0, 1e-5);
    /* NEGATIVE (GQA): the wrong mapping h % n_kv would give head 2 the kv0 mean [5,15] */
    CHECK(fabsf(out[4] - 5.0f) > 1.0f);
    /* NEGATIVE (causal): with n_pos = 1 the second cached position must be invisible */
    CHECK(lm_attn_step(out, q, K, V, 1, 4, 2, 2, scores) == 0);
    CHECK_NEAR(out[0], 10.0, 1e-6); CHECK_NEAR(out[1], 0.0, 1e-6);
    CHECK_NEAR(out[2], 10.0, 1e-6);
    CHECK_NEAR(out[4], 0.0, 1e-6); CHECK_NEAR(out[5], 20.0, 1e-6);
    /* scale: a sharper score gives a larger weight, and exactly 1/sqrt(head_dim) is applied */
    const float q1[2] = { 1, 0 };
    float o1[2], sc1[2];
    lm_attn_step(o1, q1, K, V, 2, 1, 1, 2, sc1);                              /* 1 head, 1 kv head (kv0 keys) */
    const double ee = exp(s);
    CHECK_NEAR(o1[0], ee / (ee + 1.0) * 10.0, 1e-5);
    /* large scores do not overflow the softmax */
    const float qb[2] = { 1000, 0 };
    lm_attn_step(o1, qb, K, V, 2, 1, 1, 2, sc1);
    CHECK_NEAR(o1[0], 10.0, 1e-4);
    CHECK(!isnan(o1[1]));
    /* bad arguments */
    CHECK(lm_attn_step(out, q, K, V, 0, 4, 2, 2, scores) == -1);
    CHECK(lm_attn_step(out, q, K, V, 2, 3, 2, 2, scores) == -1);
}

static void test_elementwise(void)
{
    float y[3] = { 2, 4, -6 };
    const float gate[3] = { 0, 1000, -1000 };
    lm_mul_sigmoid(y, gate, 3);
    CHECK(y[0] == 1.0f && y[1] == 4.0f && y[2] == 0.0f);
    float gt[3] = { 0, 2, -1000 }, up[3] = { 7, 3, 5 }, out[3];
    lm_silu_mul(out, gt, up, 3);
    CHECK(out[0] == 0.0f);
    CHECK_NEAR(out[1], dsilu(2.0) * 3, 1e-6);
    CHECK(!isnan(out[2]) && fabsf(out[2]) < 1e-30f);
    lm_silu_mul(gt, gt, up, 3);                                               /* aliasing */
    CHECK_NEAR(gt[1], dsilu(2.0) * 3, 1e-6);
    /* NEGATIVE: the product is silu(gate) * up, not gate * silu(up) */
    CHECK(fabsf(out[1] - (float)(2.0 * dsilu(3.0))) > 0.1f);
}

int main(void)
{
    test_scalar_helpers();
    test_rmsnorm();
    test_rmsnorm_gated();
    test_l2norm();
    test_rope();
    test_conv1d();
    test_gates();
    test_gated_delta();
    test_gated_delta_gqa();
    test_gated_delta_bitexact();
    test_attention();
    test_elementwise();
    return lt_finish("test_ops");
}
