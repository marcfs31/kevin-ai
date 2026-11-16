/* lm_gemma.h - the Gemma 4 text model behind include/lm/lm.h (lm_model.c dispatches here
 * when config.json model_type is "gemma4"). Original code; semantics of docs/PLAN.md
 * "Phase 2", tools/ref_gemma4.py is the oracle.
 *
 * WEIGHTS: same two sources as the Qwen path (bf16 checkpoint, or a q8 file from lm-quant).
 * Matrices bind through lm_bind.c. The per-layer embedding table embed_tokens_per_layer
 * ([vocab, layers*P], 5.6 GB bf16 / 3.2 GB q8 on the real E4B) is NEVER bound as a matrix and
 * never copied: lm_feed reads the one row of the current token straight from the mapped
 * file (lm_st_read_slice; for q8 the int8 row plus its scale row), widens it into a
 * layers*P float scratch and is done. The k_proj, v_proj and k_norm tensors of the KV-sharing
 * layers are never opened.
 *
 * STATE (all allocated in lm_open, nothing per token): an f32 KV cache for every layer below
 * first_shared = layers - num_kv_shared_layers: full_attention layers hold ctx positions,
 * sliding_attention layers a ring of min(sliding_window, ctx) positions (position j lives in
 * slot j % slots). Layers at or above first_shared own no cache and attend, with their own
 * query and their own type's mask, to the cache of the last producing layer of their type. */
#ifndef LM_GEMMA_H
#define LM_GEMMA_H

#include <stddef.h>

#include "model/lm_gemma_config.h"

typedef struct LmGemma LmGemma;

/* weights: a q8 file or NULL for the bf16 checkpoint in model_dir. ctx: 0 = automatic,
 * see lm_ctx.h. NULL on failure
 * with a message in err. */
LmGemma *lm_gemma_open(const char *model_dir, const char *weights, int ctx, int fast, char *err, size_t errlen);
void     lm_gemma_close(LmGemma *g);
int      lm_gemma_n_vocab(const LmGemma *g);
int      lm_gemma_n_ctx(const LmGemma *g);
int      lm_gemma_pos(const LmGemma *g);
void     lm_gemma_reset(LmGemma *g);
int      lm_gemma_feed(LmGemma *g, int token, float *logits);   /* LM_* code of lm/lm.h */

/* Bytes lm_gemma_open allocates for KV caches, scratch and the per-token row buffers at this
 * context length (weights and their widened norm vectors / q8 scale tables are separate). */
size_t   lm_gemma_state_bytes(const LmGemmaCfg *c, int ctx);

/* n tokens as n lm_gemma_feed calls: the model state and the logits after the last token are
 * BIT-IDENTICAL. Chunks of LM_PREFILL_CHUNK tokens: the projections and MLPs run over the chunk
 * (each weight read once per chunk), norms, RoPE, the cache write and the attention run token by
 * token in position order. The layers that only read the shared KV caches hold no state, so they
 * run only for the last token and only when logits are asked for. The caller (lm_prefill) has
 * validated the ids and that they fit. logits (may be NULL): after the last token only.
 * Returns LM_OK, or LM_ERR_IO if a per-layer table row cannot be read (the chunk in progress is
 * then not applied, as with lm_gemma_feed, but earlier chunks are). */
int      lm_gemma_prefill(LmGemma *g, const int *tokens, int n, float *logits);

/* The state regions at position pos, in a fixed order: for every cache-owning layer its K and
 * V rows in use (min(pos, slots) of them; a ring once it wrapped is all of its slots). With
 * lm_gemma_set_pos this is what lm_snapshot / lm_restore copy. */
typedef void (*LmRegionFn)(void *user, float *p, size_t n);
void     lm_gemma_state_regions(LmGemma *g, int pos, LmRegionFn fn, void *user);
void     lm_gemma_set_pos(LmGemma *g, int pos);

#endif /* LM_GEMMA_H */
