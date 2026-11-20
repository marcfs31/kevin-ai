/* lm_gemma_config.h - config.json -> LmGemmaCfg for Gemma 4 text models. Original code;
 * semantics of docs/PLAN.md "Phase 2" and tools/ref_gemma4.py (Config).
 *
 * Top-level model_type "gemma4" with the numbers in text_config (model_type "gemma4_text").
 * Every field is REQUIRED: a missing one is an error naming it, never a default. Refused
 * loudly, each with its own message: enable_moe_block, attention_k_eq_v, use_double_wide_mlp
 * or attention_bias true; a non-null use_bidirectional_attention, num_global_key_value_heads
 * or per_layer_config; hidden_activation other than gelu_pytorch_tanh; untied embeddings;
 * final_logit_softcapping missing or <= 0; hidden_size_per_layer_input 0; vocab_size
 * different from vocab_size_per_layer_input; a rope type other than default / proportional;
 * layer types other than sliding_attention / full_attention; a last layer that is not
 * full_attention; a layer type that KV-shared layers need but no earlier layer produces.
 * pad_token_id, boi_token_id and eoi_token_id may be absent (stored as -1; image prefill then
 * returns LM_ERR_STATE). A present value that is not an integer is an error. They are not
 * required to lie inside vocab_size: the tiny text fixture keeps the real ids, which do not. */
#ifndef LM_GEMMA_CONFIG_H
#define LM_GEMMA_CONFIG_H

#include <stddef.h>

typedef struct {
    int    hidden, inter, layers, vocab;
    int    heads, kv_heads;
    int    head_dim, global_head_dim;   /* sliding / full attention head size */
    int    window;                      /* sliding_window */
    int    max_pos;                     /* max_position_embeddings */
    int    ple_dim;                     /* hidden_size_per_layer_input (P) */
    int    pad_id, boi, eoi;            /* text pad and the image-block ends; -1 when config omits the id */
    int    kv_shared, first_shared;     /* num_kv_shared_layers; first reuse layer = layers - kv_shared */
    float  eps, cap;                    /* rms_norm_eps, final_logit_softcapping */
    float  embed_scale, ple_scale;      /* bf16(sqrt(hidden)), bf16(sqrt(P)) */
    int   *full;                        /* [layers] 1 = full_attention; owned */
    int   *producer;                    /* [layers] layer whose KV cache this layer reads (itself when it owns one); owned */
    double theta[2], partial[2], factor[2];   /* rope per type: [0] sliding, [1] full; partial 1 = default rope */
} LmGemmaCfg;

/* path: the model directory (config.json inside it) or the config file itself. Returns 0 or
 * -1 with a message in err. On failure *c holds nothing to free. */
int  lm_gemma_config_load(const char *path, LmGemmaCfg *c, char *err, size_t errlen);
void lm_gemma_config_free(LmGemmaCfg *c);

#endif /* LM_GEMMA_CONFIG_H */
