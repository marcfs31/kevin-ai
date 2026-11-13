/* lm_layers.h - the three sub-layers of a Qwen3.5 decoder layer (linear-attention mixer,
 * full-attention mixer, SiLU MLP) over caller-owned weights, state and scratch, for one
 * token. Original code; semantics per docs/PLAN.md and transformers modeling_qwen3_5.py.
 * They are what lm_feed runs and what tests/unit/test_ops_fixtures.c checks against the
 * oracle, so the fixtures exercise the real code path. No allocation, no I/O. */
#ifndef LM_LAYERS_H
#define LM_LAYERS_H

#include <stddef.h>
#include <stdint.h>

/* A row-major [rows][cols] weight matrix in one of three storage forms. */
typedef enum { LM_MAT_NONE = 0, LM_MAT_F32, LM_MAT_BF16, LM_MAT_Q8 } LmMatKind;

typedef struct {
    LmMatKind       kind;
    int             rows, cols;
    const float    *f;       /* LM_MAT_F32 */
    const uint16_t *bf;      /* LM_MAT_BF16 */
    const int8_t   *q;       /* LM_MAT_Q8: int8 [rows][cols] */
    const float    *sc;      /* LM_MAT_Q8: f32 [rows][cols/32]; must be float-aligned */
} LmMat;

/* y[rows] = M x[cols]. */
void lm_mat_mul(const LmMat *m, float *y, const float *x);

/* out[cols] = row r of M widened to f32 (a q8 row is dequantised). */
void lm_mat_row(const LmMat *m, int r, float *out);

/* ---- MLP: down(silu(gate(x)) * up(x)) ---- */
typedef struct { LmMat gate, up, down; int inter; } LmMlpW;
size_t lm_mlp_scratch(int inter);                       /* floats */
void   lm_mlp(float *out, const float *x, const LmMlpW *w, float *scratch);

/* ---- Gated DeltaNet mixer ---- */
typedef struct {
    LmMat qkv, z, b, a, out;
    const float *conv_w;       /* [conv_dim][conv_k] */
    const float *dt_bias, *A_log;   /* [n_v] */
    const float *norm_w;       /* [dv], plain weight */
} LmLinW;
typedef struct { int n_k, n_v, dk, dv, conv_k; float eps; } LmLinDims;
size_t lm_linear_scratch(const LmLinDims *d);           /* floats */
/* x: normed hidden [hidden]; out: mixer output [hidden]. conv_state [conv_dim][conv_k-1]
 * and S [n_v][dk][dv] are carried and updated in place (zero at position 0). */
void   lm_linear_mixer(float *out, const float *x, const LmLinW *w, const LmLinDims *d,
                       float *conv_state, float *S, float *scratch);

/* ---- full attention mixer ---- */
typedef struct {
    LmMat q, k, v, o;
    const float *q_norm, *k_norm;   /* [head_dim], zero-centred */
} LmAttnW;
typedef struct { int heads, kv_heads, head_dim, rot; float theta, eps; } LmAttnDims;
size_t lm_attn_scratch(const LmAttnDims *d, int ctx);   /* floats */
/* Writes K and V of position pos into the caches ([ctx][kv_heads*head_dim]) and attends
 * over positions 0..pos. */
void   lm_attn_mixer(float *out, const float *x, const LmAttnW *w, const LmAttnDims *d,
                     float *k_cache, float *v_cache, int pos, int ctx, float *scratch);

#endif /* LM_LAYERS_H */
