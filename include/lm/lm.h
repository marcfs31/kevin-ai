/* lm.h -- the engine's public API. This is the contract between the engine and
 * everything above it (CLI, chat, tests). Token ids in, logits out; the engine knows
 * nothing about text.
 *
 * Supported model families (read from config.json): Qwen3.5 text (qwen3_5), with
 * Gemma 4 text (gemma4) planned. Weights are either the original bf16 safetensors
 * (exact, streamed through the page cache) or a group-quantised q8 file written by
 * bin/lm-quant (resident on an 8 GB machine). */
#ifndef LM_H
#define LM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct LmModel LmModel;

enum {
    LM_OK        = 0,
    LM_ERR_CTX   = 1, /* context window full; nothing was fed */
    LM_ERR_TOKEN = 2, /* token id outside [0, vocab) */
    LM_ERR_IO    = 3  /* a weight read failed; the model state is no longer valid */
};

typedef struct {
    const char *weights; /* path to the weights: a q8 file from lm-quant, or NULL for the
                            bf16 safetensors in model_dir (sharded, with index) */
    int ctx;             /* positions the KV cache can hold. 0 = 4096 */
    int threads;         /* 0 = the OpenMP default */
} LmOpts;

/* model_dir holds config.json in the Hugging Face layout. Every architecture number
 * is read from it; a missing field is an error, never a default.
 * Returns NULL on failure with a message in err. */
LmModel *lm_open(const char *model_dir, const LmOpts *opts, char *err, size_t errlen);
void     lm_close(LmModel *m);

int  lm_n_vocab(const LmModel *m);
int  lm_n_ctx(const LmModel *m);
int  lm_pos(const LmModel *m);      /* tokens fed since the last reset */

/* Drop the KV cache and every recurrent/conv state, back to position 0. Linear-
 * attention state cannot be rewound, so there is no truncate: a caller whose new
 * prompt does not extend what was fed resets and re-prefills. */
void lm_reset(LmModel *m);

/* Append one token at the next position and run all layers. If logits is non-NULL it
 * receives lm_n_vocab() floats for the next-token distribution; NULL skips the final
 * norm and unembedding, which is what prompt prefill wants. Returns an LM_* code. */
int  lm_feed(LmModel *m, int token, float *logits);

#ifdef __cplusplus
}
#endif
#endif
