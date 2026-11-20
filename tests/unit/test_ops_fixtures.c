/* test_ops_fixtures.c - every op and layer against the float64 oracle values in
 * tests/fixtures/ops/ (schemas in tests/fixtures/README.md). Run from the repository root.
 * A missing fixture directory is a FAILURE, not a skip.
 *
 * TOLERANCE. The oracle computes in float64; the engine in float32 (double accumulators in
 * the norms and matvecs). The fixtures' own README suggests 1e-5 * max(1, max|y|) for f32
 * math. Measured worst error over every case is 4.9e-7 * max(1,max|ref|) (the deepest chain,
 * the linear-attention layer: 4 projections, conv, l2norm, delta rule, gated norm, out_proj,
 * a few dozen roundings of 6e-8 each), so the bound used is TOL(ref) = 2e-6 * max(1,
 * max|ref|) over the case's expected outputs: 5x below the README's suggestion and about 4x
 * above the worst observed error. The test prints "worst err / tol" so drift toward the
 * bound is visible before it fails. The q8 quantiser is checked EXACTLY (integers and f32
 * bit patterns), as the format demands. */
#include "lm_test.h"

#include "core/lm_bf16.h"
#include "core/lm_ops.h"
#include "core/lm_quant.h"
#include "io/lm_io.h"
#include "json.h"
#include "model/lm_layers.h"

#define DIR "tests/fixtures/ops/"

static jval *load(const char *file)
{
    char path[256];
    snprintf(path, sizeof path, DIR "%s", file);
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "  FAIL cannot open %s (run from the repository root)\n", path); lt_fails++; return NULL; }
    fseek(f, 0, SEEK_END);
    const long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *t = (char *)malloc((size_t)n + 1);
    const size_t got = fread(t, 1, (size_t)n, f);
    fclose(f);
    t[got] = '\0';
    jval *v = json_parse(t, NULL);
    free(t);
    if (!v) { fprintf(stderr, "  FAIL %s does not parse\n", path); lt_fails++; }
    return v;
}

static jval *K(jval *o, const char *key)
{
    jval *v = json_get(o, key);
    if (!v) { fprintf(stderr, "  FAIL fixture key \"%s\" missing\n", key); lt_fails++; exit(1); }
    return v;
}

/* flatten a (possibly nested) numeric array into n floats; returns count */
static int flat_rec(const jval *v, float *out, int cap, int n)
{
    if (v->t == J_NUM) { if (n < cap) out[n] = (float)v->num; return n + 1; }
    for (int i = 0; i < v->len; i++) n = flat_rec(v->kids[i], out, cap, n);
    return n;
}

static float *flat(const jval *v, int *count)
{
    int n = flat_rec(v, NULL, 0, 0);
    float *p = (float *)malloc((size_t)(n ? n : 1) * sizeof(float));
    flat_rec(v, p, n, 0);
    if (count) *count = n;
    return p;
}

static double dnum(jval *o, const char *key) { return K(o, key)->num; }

static double worst_ratio;     /* max over cases of err / tol, for the report */

/* compare got[n] with the oracle's ref[n] at 1e-5 * max(1, max|ref|) */
static void cmp(const char *what, const float *got, const float *ref, int n)
{
    double mx = 1.0, worst = 0.0;
    for (int i = 0; i < n; i++) if (fabs((double)ref[i]) > mx) mx = fabs((double)ref[i]);
    const double tol = 2e-6 * mx;
    int bad = -1;
    for (int i = 0; i < n; i++) {
        const double e = fabs((double)got[i] - (double)ref[i]);
        if (e > worst) worst = e;
        if (!(e <= tol) && bad < 0) bad = i;
    }
    if (worst / tol > worst_ratio) worst_ratio = worst / tol;
    CHECK_MSG(bad < 0, "%s: element %d got %.9g want %.9g (tol %.3g)", what, bad, bad >= 0 ? got[bad] : 0.0,
              bad >= 0 ? ref[bad] : 0.0, tol);
}

/* ---- element-wise ops ---- */

static void test_norms(void)
{
    jval *r = load("rmsnorm.json");
    if (!r) return;
    const float eps = (float)dnum(r, "eps");
    jval *cs = K(r, "cases");
    for (int i = 0; i < cs->len; i++) {
        jval *c = cs->kids[i];
        int n;
        float *x = flat(K(c, "x"), &n), *w = flat(K(c, "w"), NULL), *y = flat(K(c, "y"), NULL), *o = (float *)malloc((size_t)n * 4);
        lm_rmsnorm(o, x, w, n, eps);
        cmp("rmsnorm", o, y, n);
        lm_rmsnorm(x, x, w, n, eps);                        /* in place gives the same */
        CHECK(!memcmp(x, o, (size_t)n * 4));
        free(x); free(w); free(y); free(o);
    }
    json_free_tree(r);

    r = load("rmsnorm_gated.json");
    if (!r) return;
    const float eps2 = (float)dnum(r, "eps");
    cs = K(r, "cases");
    for (int i = 0; i < cs->len; i++) {
        jval *c = cs->kids[i];
        int n;
        float *x = flat(K(c, "x"), &n), *z = flat(K(c, "z"), NULL), *w = flat(K(c, "w"), NULL), *y = flat(K(c, "y"), NULL);
        float *o = (float *)malloc((size_t)n * 4);
        lm_rmsnorm_gated(o, x, z, w, n, eps2);
        cmp("rmsnorm_gated", o, y, n);
        free(x); free(z); free(w); free(y); free(o);
    }
    json_free_tree(r);

    r = load("l2norm.json");
    if (!r) return;
    cs = K(r, "cases");
    for (int i = 0; i < cs->len; i++) {
        jval *c = cs->kids[i];
        int n;
        float *x = flat(K(c, "x"), &n), *y = flat(K(c, "y"), NULL), *o = (float *)malloc((size_t)n * 4);
        lm_l2norm(o, x, n);
        cmp("l2norm", o, y, n);
        free(x); free(y); free(o);
    }
    json_free_tree(r);
}

static void test_rope(void)
{
    jval *r = load("rope.json");
    if (!r) return;
    jval *gs = K(r, "groups");
    int ncase = 0;
    for (int gi = 0; gi < gs->len; gi++) {
        jval *g = gs->kids[gi];
        const int hd = (int)dnum(g, "head_dim"), rot = (int)dnum(g, "rot_dim");
        const float theta = (float)dnum(g, "theta");
        jval *cs = K(g, "cases");
        for (int i = 0; i < cs->len; i++) {
            jval *c = cs->kids[i];
            float *x = flat(K(c, "x"), NULL), *y = flat(K(c, "y"), NULL);
            CHECK(lm_rope(x, 1, hd, rot, (int)dnum(c, "pos"), theta) == 0);
            cmp("rope", x, y, hd);
            ncase++;
        }
    }
    CHECK(ncase == 16);
    json_free_tree(r);
}

static void test_conv(void)
{
    jval *r = load("conv1d_silu.json");
    if (!r) return;
    jval *cs = K(r, "cases");
    for (int i = 0; i < cs->len; i++) {
        jval *c = cs->kids[i];
        const int C = (int)dnum(c, "channels"), k = (int)dnum(c, "kernel"), split = (int)dnum(c, "split");
        float *w = flat(K(c, "weight"), NULL), *x = flat(K(c, "x"), NULL), *y = flat(K(c, "y"), NULL);
        float *st = flat(K(c, "state_at_split"), NULL);       /* [k-1][C], oldest first */
        const int T = K(c, "x")->len;
        float *state = (float *)calloc((size_t)C * (k - 1) + 1, 4), *o = (float *)malloc((size_t)C * 4);
        for (int t = 0; t < T; t++) {
            lm_conv1d_step(o, state, x + (size_t)t * C, w, C, k);
            cmp("conv1d from zero", o, y + (size_t)t * C, C);
        }
        /* resuming from the carried state at `split` reproduces the tail (state is really carried) */
        for (int ch = 0; ch < C; ch++)
            for (int j = 0; j < k - 1; j++) state[ch * (k - 1) + j] = st[j * C + ch];
        for (int t = split; t < T; t++) {
            lm_conv1d_step(o, state, x + (size_t)t * C, w, C, k);
            cmp("conv1d resumed", o, y + (size_t)t * C, C);
        }
        free(w); free(x); free(y); free(st); free(state); free(o);
    }
    json_free_tree(r);
}

static void test_delta_rule(void)
{
    jval *r = load("gated_delta_rule.json");
    if (!r) return;
    jval *cs = K(r, "cases");
    for (int i = 0; i < cs->len; i++) {
        jval *c = cs->kids[i];
        const int nk = (int)dnum(c, "num_k_heads"), nv = (int)dnum(c, "num_v_heads");
        const int dk = (int)dnum(c, "dk"), dv = (int)dnum(c, "dv");
        float *S = (float *)calloc((size_t)nv * dk * dv, 4), *o = (float *)malloc((size_t)nv * dv * 4);
        float *q = (float *)malloc((size_t)nk * dk * 4), *k = (float *)malloc((size_t)nk * dk * 4);
        jval *steps = K(c, "steps");
        for (int s = 0; s < steps->len; s++) {
            jval *st = steps->kids[s];
            float *qr = flat(K(st, "q"), NULL), *kr = flat(K(st, "k"), NULL), *v = flat(K(st, "v"), NULL);
            float *g = flat(K(st, "g"), NULL), *beta = flat(K(st, "beta"), NULL), *oref = flat(K(st, "o"), NULL);
            for (int h = 0; h < nk; h++) {
                lm_l2norm(q + h * dk, qr + h * dk, dk);
                lm_l2norm(k + h * dk, kr + h * dk, dk);
                for (int j = 0; j < dk; j++) q[h * dk + j] *= (float)(1.0 / sqrt((double)dk));
            }
            CHECK(lm_gated_delta_step(S, o, q, k, v, g, beta, nk, nv, dk, dv) == 0);
            cmp("delta rule output", o, oref, nv * dv);
            free(qr); free(kr); free(v); free(g); free(beta); free(oref);
        }
        float *fs = flat(K(c, "final_state"), NULL);
        cmp("delta rule final state", S, fs, nv * dk * dv);
        free(fs); free(S); free(o); free(q); free(k);
    }
    json_free_tree(r);
}

/* ---- layers ---- */

/* weight matrix from a fixture: as F32 or as exact BF16 (the fixtures' weights are bf16-exact) */
static LmMat mk_mat(jval *wts, const char *key, int rows, int cols, int as_bf16)
{
    int n;
    float *f = flat(K(wts, key), &n);
    CHECK_MSG(n == rows * cols, "%s: %d values, expected %d", key, n, rows * cols);
    LmMat m;
    memset(&m, 0, sizeof m);
    m.rows = rows; m.cols = cols;
    if (as_bf16) {
        uint16_t *b = (uint16_t *)malloc((size_t)n * 2);
        for (int i = 0; i < n; i++) { b[i] = lm_f32_to_bf16(f[i]); CHECK(lm_bf16_to_f32(b[i]) == f[i]); }
        free(f);
        m.kind = LM_MAT_BF16; m.bf = b;
    } else { m.kind = LM_MAT_F32; m.f = f; }
    return m;
}

static void free_mat(LmMat *m) { free((void *)m->f); free((void *)m->bf); }

static void test_mlp(int as_bf16)
{
    jval *r = load("mlp.json");
    if (!r) return;
    const int H = (int)dnum(r, "hidden"), I = (int)dnum(r, "intermediate");
    jval *w = K(r, "weights");
    LmMlpW mw = { mk_mat(w, "gate_proj.weight", I, H, as_bf16), mk_mat(w, "up_proj.weight", I, H, as_bf16),
                  mk_mat(w, "down_proj.weight", H, I, as_bf16), I };
    float *x = flat(K(r, "x"), NULL), *y = flat(K(r, "y"), NULL);
    float *sc = (float *)malloc(lm_mlp_scratch(I) * 4), *o = (float *)malloc((size_t)H * 4);
    const int T = K(r, "x")->len;
    for (int t = 0; t < T; t++) {
        lm_mlp(o, x + t * H, &mw, sc);
        cmp(as_bf16 ? "mlp (bf16 weights)" : "mlp", o, y + t * H, H);
    }
    free(x); free(y); free(sc); free(o);
    free_mat(&mw.gate); free_mat(&mw.up); free_mat(&mw.down);
    json_free_tree(r);
}

static void test_full_attention(int as_bf16)
{
    jval *r = load("full_attention_layer.json");
    if (!r) return;
    const int H = (int)dnum(r, "hidden"), nh = (int)dnum(r, "heads"), nkv = (int)dnum(r, "kv_heads");
    const int hd = (int)dnum(r, "head_dim");
    jval *w = K(r, "weights");
    LmAttnW aw;
    memset(&aw, 0, sizeof aw);
    aw.q = mk_mat(w, "q_proj.weight", nh * hd * 2, H, as_bf16);
    aw.k = mk_mat(w, "k_proj.weight", nkv * hd, H, as_bf16);
    aw.v = mk_mat(w, "v_proj.weight", nkv * hd, H, as_bf16);
    aw.o = mk_mat(w, "o_proj.weight", H, nh * hd, as_bf16);
    aw.q_norm = flat(K(w, "q_norm.weight"), NULL);
    aw.k_norm = flat(K(w, "k_norm.weight"), NULL);
    LmAttnDims d = { nh, nkv, hd, (int)dnum(r, "rot_dim"), (float)dnum(r, "theta"), (float)dnum(r, "eps") };
    const int T = K(r, "x")->len;
    float *x = flat(K(r, "x"), NULL), *y = flat(K(r, "y"), NULL), *o = (float *)malloc((size_t)H * 4);
    float *kc = (float *)calloc((size_t)T * nkv * hd, 4), *vc = (float *)calloc((size_t)T * nkv * hd, 4);
    float *sc = (float *)malloc(lm_attn_scratch(&d, T) * 4);
    for (int t = 0; t < T; t++) {
        lm_attn_mixer(o, x + t * H, &aw, &d, kc, vc, t, t, t, t, T, sc);
        cmp(as_bf16 ? "full attention (bf16 weights)" : "full attention", o, y + t * H, H);
    }
    free(x); free(y); free(o); free(kc); free(vc); free(sc);
    free((void *)aw.q_norm); free((void *)aw.k_norm);
    free_mat(&aw.q); free_mat(&aw.k); free_mat(&aw.v); free_mat(&aw.o);
    json_free_tree(r);
}

static void test_linear_attention(int as_bf16)
{
    jval *r = load("linear_attention_layer.json");
    if (!r) return;
    const int H = (int)dnum(r, "hidden"), nk = (int)dnum(r, "num_k_heads"), nv = (int)dnum(r, "num_v_heads");
    const int dk = (int)dnum(r, "key_head_dim"), dv = (int)dnum(r, "value_head_dim"), ck = (int)dnum(r, "conv_kernel");
    const int conv_dim = 2 * nk * dk + nv * dv;
    jval *w = K(r, "weights");
    LmLinW lw;
    memset(&lw, 0, sizeof lw);
    lw.qkv = mk_mat(w, "in_proj_qkv.weight", conv_dim, H, as_bf16);
    lw.z = mk_mat(w, "in_proj_z.weight", nv * dv, H, as_bf16);
    lw.b = mk_mat(w, "in_proj_b.weight", nv, H, as_bf16);
    lw.a = mk_mat(w, "in_proj_a.weight", nv, H, as_bf16);
    lw.out = mk_mat(w, "out_proj.weight", H, nv * dv, as_bf16);
    lw.conv_w = flat(K(w, "conv1d.weight"), NULL);
    lw.dt_bias = flat(K(w, "dt_bias"), NULL);
    lw.A_log = flat(K(w, "A_log"), NULL);
    lw.norm_w = flat(K(w, "norm.weight"), NULL);
    LmLinDims d = { nk, nv, dk, dv, ck, (float)dnum(r, "eps") };
    const int T = K(r, "x")->len;
    float *x = flat(K(r, "x"), NULL), *y = flat(K(r, "y"), NULL), *o = (float *)malloc((size_t)H * 4);
    float *cs = (float *)calloc((size_t)conv_dim * (ck - 1), 4), *S = (float *)calloc((size_t)nv * dk * dv, 4);
    float *sc = (float *)malloc(lm_linear_scratch(&d) * 4);
    for (int t = 0; t < T; t++) {
        lm_linear_mixer(o, x + t * H, &lw, &d, cs, S, sc);
        cmp(as_bf16 ? "linear attention (bf16 weights)" : "linear attention", o, y + t * H, H);
    }
    free(x); free(y); free(o); free(cs); free(S); free(sc);
    free((void *)lw.conv_w); free((void *)lw.dt_bias); free((void *)lw.A_log); free((void *)lw.norm_w);
    free_mat(&lw.qkv); free_mat(&lw.z); free_mat(&lw.b); free_mat(&lw.a); free_mat(&lw.out);
    json_free_tree(r);
}

/* ---- Gemma 4 ops ---- */

static void test_gemma_norm_gelu_softcap(void)
{
    jval *r = load("gemma_rmsnorm.json");
    if (r) {
        const float eps = (float)dnum(r, "eps");
        jval *cs = K(r, "cases");
        for (int i = 0; i < cs->len; i++) {
            jval *c = cs->kids[i];
            int n;
            float *x = flat(K(c, "x"), &n), *w = flat(K(c, "w"), NULL), *y = flat(K(c, "y"), NULL), *o = (float *)malloc((size_t)n * 4);
            lm_rmsnorm_plain(o, x, w, n, eps);
            cmp("gemma rmsnorm", o, y, n);
            lm_rmsnorm_plain(x, x, w, n, eps);              /* in place gives the same */
            CHECK(!memcmp(x, o, (size_t)n * 4));
            free(x); free(w); free(y); free(o);
        }
        CHECK(cs->len >= 8);
        cs = K(r, "cases_noweight");
        for (int i = 0; i < cs->len; i++) {
            jval *c = cs->kids[i];
            int n;
            float *x = flat(K(c, "x"), &n), *y = flat(K(c, "y"), NULL), *o = (float *)malloc((size_t)n * 4);
            lm_rmsnorm_plain(o, x, NULL, n, eps);
            cmp("gemma rmsnorm without weight", o, y, n);
            free(x); free(y); free(o);
        }
        CHECK(cs->len >= 5);
        json_free_tree(r);
    }

    r = load("gelu_tanh.json");
    if (r) {
        jval *cs = K(r, "cases");
        for (int i = 0; i < cs->len; i++) {
            int n;
            float *x = flat(K(cs->kids[i], "x"), &n), *y = flat(K(cs->kids[i], "y"), NULL), *o = (float *)malloc((size_t)n * 4);
            for (int j = 0; j < n; j++) o[j] = lm_gelu_tanh(x[j]);
            cmp("gelu_tanh", o, y, n);
            free(x); free(y); free(o);
        }
        json_free_tree(r);
    }

    r = load("gelu_tanh_mul.json");
    if (r) {
        jval *cs = K(r, "cases");
        for (int i = 0; i < cs->len; i++) {
            int n;
            float *g = flat(K(cs->kids[i], "gate"), &n), *u = flat(K(cs->kids[i], "up"), NULL), *y = flat(K(cs->kids[i], "y"), NULL);
            float *o = (float *)malloc((size_t)n * 4);
            lm_gelu_tanh_mul(o, g, u, n);
            cmp("gelu_tanh_mul", o, y, n);
            lm_gelu_tanh_mul(g, g, u, n);                   /* out aliasing gate gives the same */
            CHECK(!memcmp(g, o, (size_t)n * 4));
            free(g); free(u); free(y); free(o);
        }
        json_free_tree(r);
    }

    r = load("logit_softcap.json");
    if (r) {
        jval *cs = K(r, "cases");
        for (int i = 0; i < cs->len; i++) {
            int n;
            float *x = flat(K(cs->kids[i], "x"), &n), *y = flat(K(cs->kids[i], "y"), NULL);
            lm_logit_softcap(x, n, (float)dnum(cs->kids[i], "cap"));
            cmp("logit softcap", x, y, n);
            free(x); free(y);
        }
        json_free_tree(r);
    }
}

static void test_gemma_rope(void)
{
    jval *r = load("gemma_rope.json");
    if (!r) return;
    jval *gs = K(r, "groups");
    int ncase = 0, nprop = 0;
    for (int gi = 0; gi < gs->len; gi++) {
        jval *g = gs->kids[gi];
        const int hd = (int)dnum(g, "head_dim");
        const double theta = dnum(g, "theta"), partial = dnum(g, "partial_rotary_factor");
        const int prop = !strcmp(K(g, "rope_type")->str, "proportional");
        double *f = (double *)malloc((size_t)(hd / 2) * sizeof(double));
        CHECK(lm_rope_freqs(f, hd, theta, partial) == 0);
        jval *fr = K(g, "freqs");
        CHECK(fr->len == hd / 2);
        for (int i = 0; i < hd / 2; i++) CHECK_NEAR(f[i], fr->kids[i]->num, 1e-15 * fabs(fr->kids[i]->num));
        if (prop) nprop++;
        jval *cs = K(g, "cases");
        for (int i = 0; i < cs->len; i++) {
            jval *c = cs->kids[i];
            float *x = flat(K(c, "x"), NULL), *y = flat(K(c, "y"), NULL);
            CHECK(lm_rope_freqs_apply(x, 1, hd, f, (int)dnum(c, "pos")) == 0);
            cmp("gemma rope", x, y, hd);
            ncase++;
            free(x); free(y);
        }
        free(f);
    }
    CHECK(gs->len >= 5 && nprop >= 3 && ncase >= 20);
    /* negatives: an odd head, a bad partial factor, a bad theta */
    double f[8];
    CHECK(lm_rope_freqs(f, 15, 1e4, 1.0) == -1);
    CHECK(lm_rope_freqs(f, 16, 1e4, 0.0) == -1);
    CHECK(lm_rope_freqs(f, 16, 1e4, 1.5) == -1);
    CHECK(lm_rope_freqs(f, 16, 0.0, 1.0) == -1);
    json_free_tree(r);
}

static void test_sliding_attention(void)
{
    jval *r = load("sliding_attention.json");
    if (!r) return;
    const float scaling = (float)dnum(r, "scaling");
    jval *cs = K(r, "cases");
    int ring_cases = 0, wrapped = 0;
    for (int i = 0; i < cs->len; i++) {
        jval *c = cs->kids[i];
        const int H = (int)dnum(c, "heads"), KV = (int)dnum(c, "kv_heads"), D = (int)dnum(c, "head_dim");
        const int pos = (int)dnum(c, "pos"), first_ref = (int)dnum(c, "first_visible");
        jval *wj = K(c, "window");
        const int window = wj->t == J_NUM ? (int)wj->num : 0;
        const int first = window > 0 && pos - window + 1 > 0 ? pos - window + 1 : 0;
        CHECK_MSG(first == first_ref, "case %d: first_visible %d, formula %d", i, first_ref, first);
        const size_t kvd = (size_t)KV * D;
        float *q = flat(K(c, "q"), NULL), *k = flat(K(c, "k"), NULL), *v = flat(K(c, "v"), NULL), *y = flat(K(c, "y"), NULL);
        float *o = (float *)malloc((size_t)H * D * 4), *sc = (float *)malloc((size_t)H * (pos + 1) * 4);

        /* a linear cache of every position, window applied by the mask */
        CHECK(lm_attn_window(o, q, k, v, pos, window, pos + 1, H, KV, D, scaling, sc) == 0);
        cmp("sliding attention (linear cache)", o, y, H * D);

        /* a ring cache of min(window, pos+1) slots, filled exactly as lm_feed fills it */
        const int slots = window > 0 && window < pos + 1 ? window : pos + 1;
        float *rk = (float *)calloc((size_t)slots * kvd, 4), *rv = (float *)calloc((size_t)slots * kvd, 4);
        for (int j = 0; j <= pos; j++) {
            memcpy(rk + (size_t)(j % slots) * kvd, k + (size_t)j * kvd, kvd * 4);
            memcpy(rv + (size_t)(j % slots) * kvd, v + (size_t)j * kvd, kvd * 4);
        }
        memset(o, 0, (size_t)H * D * 4);
        CHECK(lm_attn_window(o, q, rk, rv, pos, window, slots, H, KV, D, scaling, sc) == 0);
        cmp("sliding attention (ring cache)", o, y, H * D);
        if (window > 0) { ring_cases++; if (pos >= window) wrapped++; }

        /* NEGATIVE: a cache one slot too small must be refused, not read wrongly */
        if (slots > 1) {
            const int nvis = pos - first + 1;
            CHECK(lm_attn_window(o, q, rk, rv, pos, window, nvis - 1, H, KV, D, scaling, sc) == -1);
        }
        free(q); free(k); free(v); free(y); free(o); free(sc); free(rk); free(rv);
    }
    CHECK_MSG(ring_cases >= 7 && wrapped >= 4, "fixture has %d windowed cases, %d beyond the window", ring_cases, wrapped);
    /* NEGATIVE: a different window must change the answer on a case beyond the window (the mask is live) */
    {
        jval *c = cs->kids[4];                              /* window 8 at pos 20 */
        const int H = (int)dnum(c, "heads"), KV = (int)dnum(c, "kv_heads"), D = (int)dnum(c, "head_dim"), pos = (int)dnum(c, "pos");
        float *q = flat(K(c, "q"), NULL), *k = flat(K(c, "k"), NULL), *v = flat(K(c, "v"), NULL), *y = flat(K(c, "y"), NULL);
        float *o = (float *)malloc((size_t)H * D * 4), *sc = (float *)malloc((size_t)H * (pos + 1) * 4);
        CHECK(pos == 20 && (int)dnum(c, "window") == 8);
        CHECK(lm_attn_window(o, q, k, v, pos, 0, pos + 1, H, KV, D, 1.0f, sc) == 0);
        double d = 0;
        for (int i = 0; i < H * D; i++) d = fmax(d, fabs((double)o[i] - (double)y[i]));
        CHECK_MSG(d > 1e-3, "unwindowed attention equals the windowed answer (%g): the fixture does not exercise the mask", d);
        free(q); free(k); free(v); free(y); free(o); free(sc);
    }
    json_free_tree(r);
}

/* ---- q8 quantiser: exact ---- */

static uint32_t fbits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }

static void test_q8(void)
{
    jval *r = load("q8_quantize.json");
    if (!r) return;
    jval *gs = K(r, "groups");
    for (int i = 0; i < gs->len; i++) {
        jval *g = gs->kids[i];
        /* the input is the bf16 bit patterns widened, which is what lm-quant feeds the quantiser */
        jval *bits = K(g, "x_bf16_bits");
        float x[32];
        int8_t q[32];
        float sc;
        CHECK(bits->len == 32);
        for (int j = 0; j < 32; j++) x[j] = lm_bf16_to_f32((uint16_t)bits->kids[j]->num);
        CHECK(lm_q8_quantize(x, 1, 32, q, &sc, NULL) == LM_Q8_OK);
        CHECK_MSG(fbits(sc) == (uint32_t)dnum(g, "scale_f32_bits"), "%s: scale bits %08x want %08x",
                  K(g, "name")->str, fbits(sc), (unsigned)dnum(g, "scale_f32_bits"));
        jval *qq = K(g, "q");
        int same = 1;
        for (int j = 0; j < 32; j++) if (q[j] != (int)qq->kids[j]->num) same = 0;
        CHECK_MSG(same, "%s: q differs", K(g, "name")->str);
    }
    CHECK(gs->len >= 14);
    jval *m = K(r, "matrix");
    const int rows = (int)dnum(m, "rows"), cols = (int)dnum(m, "cols");
    float *x = flat(K(m, "x"), NULL), *sref = flat(K(m, "scale"), NULL);
    int8_t *q = (int8_t *)malloc((size_t)rows * cols);
    float *sc = (float *)malloc((size_t)rows * (cols / 32) * 4);
    CHECK(lm_q8_quantize(x, rows, cols, q, sc, NULL) == LM_Q8_OK);
    jval *qq = K(m, "q"), *sb = K(m, "scale_f32_bits");
    int qsame = 1, ssame = 1;
    for (int i = 0; i < rows * cols; i++) if (q[i] != (int)qq->kids[i]->num) qsame = 0;
    for (int i = 0; i < rows * (cols / 32); i++) if (fbits(sc[i]) != (uint32_t)sb->kids[i]->num) ssame = 0;
    CHECK(qsame && ssame);
    free(x); free(sref); free(q); free(sc);
    json_free_tree(r);
}

int main(void)
{
    test_norms();
    test_rope();
    test_conv();
    test_delta_rule();
    for (int bf = 0; bf <= 1; bf++) {
        test_mlp(bf);
        test_full_attention(bf);
        test_linear_attention(bf);
    }
    test_gemma_norm_gelu_softcap();
    test_gemma_rope();
    test_sliding_attention();
    test_q8();
    printf("test_ops_fixtures: worst err / tolerance = %.3f (tolerance 2e-6 * max(1,max|ref|))\n", worst_ratio);
    return lt_finish("test_ops_fixtures");
}
