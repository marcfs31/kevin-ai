/* lm_ctx.h - the context window from the memory this machine can hand out.
 *
 * A model needs its weights resident (they are memory-mapped, so the page cache holds them)
 * plus its state: KV caches and scratch that grow with the context window. With ctx 0 the
 * window is the largest that keeps both in memory, up to the model's max_position_embeddings;
 * an explicit ctx is refused when its state would push resident weights out of memory or
 * would not fit at all. Weights larger than memory (bf16 streamed from disk) stay allowed:
 * then only the state must fit, and ctx 0 means LM_CTX_DEFAULT.
 *
 * Pure: the caller passes the memory reading, so every case is testable. */
#ifndef LM_CTX_H
#define LM_CTX_H

#include <stddef.h>
#include <stdint.h>

#include "io/lm_sysmem.h"

#define LM_CTX_DEFAULT 4096                     /* ctx 0 when memory is unknown or weights stream */
#define LM_CTX_MARGIN  ((uint64_t)512 << 20)    /* left free for the OS and other programs */
#define LM_CTX_ALIGN   256                      /* an automatic window is a multiple of this */

typedef size_t (*LmStateBytesFn)(const void *arg, int ctx);   /* state bytes for a ctx, monotone */

/* requested: 0 = automatic, > 0 = this window. max_pos: the model's limit (> 0). weights: bytes
 * of every tensor the model reads. st/avail: lm_mem_available's answer. On success *ctx is set
 * and 0 returned; -1 with a message in err (sizes in GiB, and what to lower) otherwise. With
 * memory unknown nothing is refused and ctx 0 means min(LM_CTX_DEFAULT, max_pos). */
int lm_ctx_choose(int requested, int max_pos, uint64_t weights, LmStateBytesFn fn, const void *arg,
                  LmMemStatus st, uint64_t avail, int *ctx, char *err, size_t errlen);

/* lm_ctx_choose with this machine's lm_mem_available. */
int lm_ctx_choose_here(int requested, int max_pos, uint64_t weights, LmStateBytesFn fn, const void *arg,
                       int *ctx, char *err, size_t errlen);

#endif /* LM_CTX_H */
