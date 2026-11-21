/* lm_qwen2.h - the Qwen2 / Qwen2.5 text model from a GGUF file, behind include/lm/lm.h
 * (lm_model.c dispatches here when the model path is a .gguf, or a directory holding exactly
 * one and no config.json). Original code; the oracle is tools/ref_qwen2.py.
 *
 * ARCHITECTURE (general.architecture "qwen2"): pre-norm decoder, RMSNorm with a plain weight
 * (not 1 + w), grouped-query attention with a bias on q, k and v (not on the output
 * projection), rotate-half RoPE over the whole head with theta from qwen2.rope.freq_base, a
 * SwiGLU MLP, a final norm, and an output matrix (tied to the embedding when the file has no
 * output.weight). No qk-norm, no gate, no linear attention.
 *
 * REFUSED, loudly: another architecture, a missing or non-positive hyper-parameter, a head
 * count that does not divide the hidden size or the kv-head count, a rope dimension count
 * other than the head size, any rope scaling (YaRN and friends are not implemented), a matrix
 * of a type the kernels lack, a tensor of the wrong shape, a missing tensor.
 *
 * WEIGHTS stay in the mapped file (resident through the page cache); only the norms and
 * biases are copied into owned float vectors. STATE (all allocated in lm_qwen2_open): an f32
 * KV cache [ctx][kv_heads * head_dim] per layer for K and V, plus scratch. */
#ifndef LM_QWEN2_H
#define LM_QWEN2_H

#include <stddef.h>

typedef struct LmQwen2 LmQwen2;

typedef struct {
    int    hidden, inter, layers, vocab, heads, kv_heads, head_dim, max_pos;
    float  eps, theta;
} LmQwen2Cfg;

/* gguf_path: the file. ctx: 0 = automatic (lm_ctx.h). NULL on failure with a message in err. */
LmQwen2 *lm_qwen2_open(const char *gguf_path, int ctx, char *err, size_t errlen);
void     lm_qwen2_close(LmQwen2 *q);
const LmQwen2Cfg *lm_qwen2_cfg(const LmQwen2 *q);
int      lm_qwen2_n_vocab(const LmQwen2 *q);
int      lm_qwen2_n_ctx(const LmQwen2 *q);
int      lm_qwen2_pos(const LmQwen2 *q);
void     lm_qwen2_reset(LmQwen2 *q);
int      lm_qwen2_feed(LmQwen2 *q, int token, float *logits);              /* LM_* of lm/lm.h */
/* n tokens, bit-identical to n lm_qwen2_feed calls; logits after the last one only. The caller
 * (lm_prefill) has validated ids and room. */
int      lm_qwen2_prefill(LmQwen2 *q, const int *tokens, int n, float *logits);

/* Bytes of KV cache and scratch at this context length (the weights are separate). */
size_t   lm_qwen2_state_bytes(const LmQwen2Cfg *c, int ctx);

/* The state regions at position pos (per layer, K then V rows 0..pos-1) and the position
 * setter: what lm_snapshot / lm_restore copy. */
typedef void (*LmQwen2RegionFn)(void *user, float *p, size_t n);
void     lm_qwen2_state_regions(LmQwen2 *q, int pos, LmQwen2RegionFn fn, void *user);
void     lm_qwen2_set_pos(LmQwen2 *q, int pos);

#endif /* LM_QWEN2_H */
