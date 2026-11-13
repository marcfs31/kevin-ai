/* lm_sampler.h -- deterministic temperature / top-k / top-p sampling.
 *
 * Adapted from the unreviewed gpt-oss sampler src/chat/oss_sampler.[ch] of an earlier
 * worktree, itself a port of kimi-k3-in-c src/chat/k3_sampler.[ch] (Apache-2.0).
 * Changes against that: renamed oss_ -> lm_; top-k added (applied before top-p, on the
 * renormalised survivors, as Hugging Face's warpers chain them) and done by selection so
 * a 248k-vocab step does not sort the vocabulary; temperature 0 is greedy argmax (lowest
 * id wins a tie, random stream not advanced); NaN logits are refused instead of compared.
 *
 * One PCG32 stream per sampler, seeded once; the same seed replays the same draws on any
 * platform (no libc rand, 53-bit doubles from two 32-bit outputs). */
#ifndef LM_SAMPLER_H
#define LM_SAMPLER_H

#include <stdint.h>

typedef struct { double l; int id; } LmCand; /* l = logit */

typedef struct {
    double temperature; /* >= 0; 0 = greedy */
    double top_p;       /* (0, 1]; 1 = off */
    int top_k;          /* >= 0; 0 = off */
    uint64_t state, inc;
    LmCand *cand;
    double *prob;
    int cap;
} LmSampler;

/* Arguments are checked by lm_sampler_next (it returns -1 for an invalid combination);
 * init cannot fail. */
void lm_sampler_init(LmSampler *s, double temperature, double top_p, int top_k, uint64_t seed);
void lm_sampler_free(LmSampler *s);

/* Pick a token id from logits[0..n). Returns 0 and *out, or -1 for bad arguments, a NaN
 * logit, or allocation failure (nothing is drawn). */
int lm_sampler_next(LmSampler *s, const float *logits, int n, int *out);

#endif
