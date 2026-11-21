/* lm_config.h - config.json -> LmCfg for Qwen3.5 text models. Original code.
 *
 * Reads config.json's text_config (a flat config is accepted too, as the oracle does).
 * Every field is REQUIRED: a missing one is an error naming it, never a default. Refused
 * loudly: model_type other than "qwen3_5", rope_type other than "default", hidden_act other
 * than "silu", attn_output_gate false, layer types other than linear_attention /
 * full_attention, non-divisible head counts, an odd or empty rotary dimension. */
#ifndef LM_CONFIG_H
#define LM_CONFIG_H

#include <stddef.h>

enum { LM_LAYER_LINEAR = 0, LM_LAYER_FULL = 1 };

typedef struct {
    int    hidden, inter, layers, vocab;
    int   *layer_type;            /* [layers], LM_LAYER_*; owned, freed by lm_config_free */
    float  eps;
    int    heads, kv_heads, head_dim;
    float  theta;
    int    rot;                   /* rotary dims per head = (int)(head_dim * partial_rotary_factor) */
    int    sec_h, sec_w;          /* mrope_section[1], [2] (interleaved M-RoPE; [t,h,w] sums to rot/2) */
    int    lk_heads, lv_heads, lk_dim, lv_dim, conv_k;
    int    tied;                  /* tie_word_embeddings */
    int    max_pos;               /* max_position_embeddings */
    int    key_dim, value_dim, conv_dim;   /* derived */
} LmCfg;

/* config.json of a model directory (or the file itself) parsed into a JSON object tree the
 * caller frees with json_free_tree (json.h). NULL with a message in err when the file is
 * missing, implausibly sized or not a JSON object. Shared by every model family. */
struct jval *lm_config_read_json(const char *path, char *err, size_t errlen);

/* The string config.json's top-level "model_type" into out (n bytes). 0, or -1 when the file
 * cannot be read, the key is missing, is not a string, or does not fit. */
int lm_config_model_type(const char *path, char *out, size_t n);

/* path: the model directory (config.json inside it) or the config file itself. Returns 0
 * or -1 with a message in err. On failure *c holds nothing to free. */
int  lm_config_load(const char *path, LmCfg *c, char *err, size_t errlen);
void lm_config_free(LmCfg *c);

#endif /* LM_CONFIG_H */
