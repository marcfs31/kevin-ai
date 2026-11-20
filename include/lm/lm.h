/* lm.h -- the engine's public API. This is the contract between the engine and
 * everything above it (CLI, chat, tests). Token ids in, logits out; the engine knows
 * nothing about text.
 *
 * Supported model families (read from config.json model_type): Qwen3.5 text (qwen3_5)
 * and Gemma 4 text (gemma4). Weights are either the original bf16 safetensors
 * (exact, streamed through the page cache) or a group-quantised q8 file written by
 * bin/lm-quant (resident; under 6 GiB for either model). */
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
    LM_ERR_IO    = 3, /* a weight read failed; the model state is no longer valid */
    LM_ERR_STATE = 4  /* the call does not apply now (no MTP head, nothing fed, no lm_verify
                         to accept); nothing changed */
};

typedef struct {
    const char *weights; /* path to the weights: a q8 file from lm-quant, or NULL for the
                            bf16 safetensors in model_dir (sharded, with index) */
    int ctx;             /* positions the KV cache can hold. 0 = the largest window that keeps
                            the weights and the state in free memory, up to the model's
                            max_position_embeddings (4096 when free memory is unknown or the
                            weights do not fit). A window whose state does not fit is refused.
                            lm_n_ctx gives the window chosen. */
    int threads;         /* 0 = the OpenMP default */
    int q8_int8_acts;    /* 1 = fast path: round the activations of every q8 matrix multiply to
                            int8 (per group of 32). Faster, NOT bit-identical to 0 and slightly
                            less exact; requires `weights` (a q8 file). 0 = exact, the default */
    int mtp;             /* 1 = also load Qwen3.5's multi-token-prediction head (mtp.* in the
                            weights; q8 files from an lm-quant older than the head lack it) for
                            lm_draft / lm_verify / lm_accept. Every fed token then also runs the
                            head's one layer to keep its cache. lm_open fails if there is none */
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

/* Append n tokens, exactly as n calls of lm_feed would (the model state and the logits are
 * BIT-IDENTICAL), but faster for a prompt: they run in chunks so each weight
 * is read once per chunk instead of once per token. logits (may be NULL) receives the
 * next-token distribution after the LAST token only. n == 0 is a no-op returning LM_OK.
 * Nothing is fed and the model is unchanged when it returns LM_ERR_TOKEN (an id outside
 * [0, lm_n_vocab()), n < 0, or tokens NULL with n > 0) or LM_ERR_CTX (the tokens do not all
 * fit in the context window). Both model families batch the prompt in chunks; Gemma 4 can also
 * return LM_ERR_IO if a per-layer embedding row cannot be read. */
int  lm_prefill(LmModel *m, const int *tokens, int n, float *logits);

/* Feed one image block: the configured start token, n vision rows, then the end token.
 * rows is n * width floats in encoder order; width is the text hidden size. grid_h * grid_w
 * must equal n (the encoder's pooled grid, image sides divided by its unit). logits (nullable)
 * is the next-token distribution after the end token. pos_thw (nullable) receives (n + 2) * 3
 * ints, the (t, h, w) of the start token, each row and the end token.
 *
 * Qwen3.5 uses the grid as M-RoPE. Start and end are text positions (p, p, p). Merged cell
 * (i, j) is (t = p, h = p + i, w = p + j), where p is the text position after the start token;
 * the text position then advances by max(grid_h, grid_w). The cache index still advances once
 * per position. Each row replaces that position's token embedding. Snapshots store the text
 * position and lm_reset zeroes it.
 *
 * Gemma 4 checks the same grid and then ignores it for positions. Token k of the block is
 * (p + k, p + k, p + k) at cache index p + k, and p advances by one each time. Attention stays
 * causal. A vision row replaces that position's token embedding and is not multiplied by the
 * text embedding scale. The per-layer table lookup there is pad_token_id; the projection is of
 * the vision row (Gemma4Model.forward scatters the vision features before project_per_layer_inputs).
 *
 * Nothing is fed on LM_ERR_TOKEN (bad rows, n, width or grid, or a start, end or pad id outside
 * the vocabulary) or LM_ERR_CTX (the block does not fit). LM_ERR_STATE when this model's config
 * has no pad, start or end id. LM_ERR_IO: Qwen could not allocate the block scratch (nothing
 * fed); Gemma could not read a per-layer row (that token is not applied, earlier tokens of this
 * call stay fed, and pos_thw is left untouched). */
int lm_prefill_image(LmModel *m, const float *rows, int n, int width, int grid_h, int grid_w,
                     float *logits, int *pos_thw);

/* ---- speculative decoding with the MTP head (LmOpts.mtp) ----
 * The head drafts the token after next; one batched forward checks the draft, and a rejected
 * draft is rolled back without feeding again. Greedy decoding with them gives exactly the ids
 * plain lm_feed decoding gives (the logits are BIT-IDENTICAL), in fewer weight passes when
 * drafts are accepted:
 *     t = argmax(last logits); lm_draft(m, t, dl); d = argmax(dl);
 *     lm_verify(m, (int[]){t, d}, 2, rows);       rows[0] decides d, rows[1] follows it
 *     lm_accept(m, argmax(rows[0]) == d ? 2 : 1);
 * lm_draft: token is the id about to be fed (sampled from the last logits); logits receives the
 * head's lm_n_vocab() floats for the id AFTER it. Needs lm_pos >= 1. The model is unchanged. */
int lm_draft(LmModel *m, int token, float *logits);
/* lm_prefill of n tokens (1 <= n <= LM_VERIFY_MAX) with logits for EVERY one of them: rows is
 * n * lm_n_vocab() floats. Errors as lm_prefill, plus LM_ERR_STATE without the MTP head, for n
 * out of range or for rows NULL. */
#define LM_VERIFY_MAX 2
int lm_verify(LmModel *m, const int *tokens, int n, float *rows);
/* Right after lm_verify: keep its first k tokens (1 <= k <= n); the model is then exactly as if
 * only those k had been fed. LM_ERR_STATE if the last call was not lm_verify. */
int lm_accept(LmModel *m, int k);
/* The greedy step above after lm_draft: verify {token, draft}, keep the draft if the argmax of
 * rows[0] (lowest id wins a tie) is draft. *kept is then 2 or 1, and logits gets the row after
 * the last kept token. rows: LM_VERIFY_MAX * lm_n_vocab() floats of scratch. Errors as
 * lm_verify; on an error nothing is kept. */
int lm_greedy_check(LmModel *m, int token, int draft, float *rows, float *logits, int *kept);

/* A copy of the model state at the current position: every recurrent and conv state, the KV
 * caches up to lm_pos (whole sliding-window rings) and the position. lm_restore puts the model
 * back exactly there -- the state, and so every later logit, BIT-IDENTICAL to the moment it was
 * taken -- whatever was fed or reset in between, as often as needed. A shared prompt prefix
 * (a system prompt) is then fed once and restored instead of fed again. Size: lm_snapshot_bytes;
 * for Qwen3.5-4B about 50 MiB (conv and delta-rule state) plus 64 KiB per position, 72 KiB
 * with the MTP head. */

typedef struct LmSnapshot LmSnapshot;
LmSnapshot *lm_snapshot(const LmModel *m);                 /* NULL on allocation failure */
/* LM_OK, or -1 (nothing changed) if s was taken from another LmModel. */
int         lm_restore(LmModel *m, const LmSnapshot *s);
int         lm_snapshot_pos(const LmSnapshot *s);
size_t      lm_snapshot_bytes(const LmSnapshot *s);
void        lm_snapshot_free(LmSnapshot *s);               /* NULL is a no-op */

/* Where the time goes (lm-run --profile). Process-wide and OFF by default; while on, every
 * lm_feed and lm_prefill adds wall-clock seconds to these counters. Results are unchanged by
 * it. Not thread-safe: profile one model at a time, from one thread. total minus the other
 * three is the rest: embeddings, layer norms, residual adds, the MLP activation and Gemma's
 * per-layer inputs. */
typedef struct {
    double total;     /* inside lm_feed and lm_prefill */
    double matmul;    /* weight matrix multiplies, the unembedding included */
    double linear;    /* Qwen linear-attention mixers outside their multiplies: conv, q/k norms,
                         gates, the delta rule and the gated norm */
    double attention; /* attention mixers outside their multiplies: q/k norms, RoPE, the cache
                         write, scores, softmax and the weighted sum of values */
} LmTimes;
void lm_times_enable(int on);    /* 1: zero the counters and start; 0: stop, keeping them */
void lm_times_get(LmTimes *out);

#ifdef __cplusplus
}
#endif
#endif
