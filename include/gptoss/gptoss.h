/* gptoss.h -- the engine's public API. This is the contract between the engine and
 * everything above it (CLI, chat, tests). Token ids in, logits out; the engine knows
 * nothing about text. */
#ifndef GPTOSS_H
#define GPTOSS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct OssModel OssModel;

enum {
    OSS_OK        = 0,
    OSS_ERR_CTX   = 1, /* context window full; nothing was fed */
    OSS_ERR_TOKEN = 2, /* token id outside [0, vocab) */
    OSS_ERR_IO    = 3  /* a weight read failed; the model state is no longer valid */
};

typedef struct {
    double ram_gb;      /* resident budget for weights + expert cache.
                           0 = the machine's available memory minus 1 GiB headroom */
    int    ctx;         /* positions the KV cache can hold. 0 = 4096 */
    int    threads;     /* 0 = the OpenMP default */
    int    cache_slots; /* expert-cache slots. 0 = derive from ram_gb. Any value must
                           produce identical logits; only speed may change */
} OssOpts;

typedef struct {
    uint64_t expert_hits;
    uint64_t expert_misses;
    uint64_t bytes_read;   /* expert bytes read from disk since open */
} OssStats;

/* model_dir holds config.json and model.safetensors in the layout of the
 * `original/` folder of openai/gpt-oss-20b. Every architecture number is read from
 * config.json; a missing field is an error, never a default.
 * Returns NULL on failure with a message in err. */
OssModel *oss_open(const char *model_dir, const OssOpts *opts, char *err, size_t errlen);
void      oss_close(OssModel *m);

int  oss_n_vocab(const OssModel *m);
int  oss_n_ctx(const OssModel *m);
int  oss_pos(const OssModel *m);    /* tokens fed since the last reset */
void oss_reset(OssModel *m);        /* drop the KV cache, back to position 0 */

/* Rewind to position pos (0 <= pos <= oss_pos()), keeping the KV cache below it, so a
 * conversation that shares a prefix with what was already fed re-feeds only the tail.
 * Returns OSS_ERR_CTX and changes nothing when pos is out of range. */
int  oss_truncate(OssModel *m, int pos);

/* Append one token at the next position and run all layers. If logits is non-NULL it
 * receives oss_n_vocab() floats for the next-token distribution; NULL skips the final
 * norm and unembedding, which is what prompt prefill wants. Returns an OSS_* code. */
int  oss_feed(OssModel *m, int token, float *logits);

void oss_stats(const OssModel *m, OssStats *out);

#ifdef __cplusplus
}
#endif
#endif
