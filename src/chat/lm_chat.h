/* lm_chat.h -- the generation loop: transcript -> template -> ids -> backend -> sampler ->
 * parser, one turn at a time. The model is reached only through LmBackend, so the loop is
 * testable with a scripted fake and knows nothing about LmModel or any chat format.
 *
 * KV / recurrent-state reuse. The engine cannot rewind (linear attention state has no
 * truncate), so a turn feeds only the tail of the new prompt when the ids already fed are
 * an exact prefix of it, and otherwise resets and re-feeds everything. Feeding is one
 * token per forward pass, as slow as decoding, so a re-feed is the expensive event and
 * the template choice is made to avoid it: by default earlier assistant turns are rendered
 * exactly as they were generated (LmRenderOpts.verbatim_history), which makes every
 * prompt an extension of the previous state. cfg.strict_history selects the model
 * template's own history rendering instead (reasoning dropped), which costs a full
 * re-feed on every turn after the first. lm_chat_reprefills() counts them.
 *
 * Tokens that were sampled but not fed (the last one when max_tokens or the context
 * ends the turn, a stop id) are simply part of the next prompt's tail. */
#ifndef LM_CHAT_H
#define LM_CHAT_H

#include <stddef.h>
#include <stdint.h>

#include "chat/lm_sampler.h"
#include "chat/lm_template.h"
#include "tokenizer/lm_tok.h"

/* The engine as the loop sees it. feed/reset/pos mirror lm_feed/lm_reset/lm_pos and
 * return LM_* codes from lm/lm.h (feed). logits is n_vocab floats or NULL (prefill). */
typedef struct {
    void *ctx;
    int n_vocab;
    int n_ctx;
    int  (*feed)(void *ctx, int token, float *logits);
    void (*reset)(void *ctx);
    int  (*pos)(void *ctx);
    /* Optional (NULL = feed token by token). Appends n ids as n feeds would, logits (may be NULL)
     * after the last one only, like lm_prefill. The loop hands it chunks of at most
     * LM_CHAT_PREFILL_CHUNK ids and polls cancellation between them; an error code other than
     * LM_ERR_CTX leaves the backend state unknown (the next turn re-feeds from scratch). */
    int  (*prefill)(void *ctx, const int *ids, int n, float *logits);
} LmBackend;

#define LM_CHAT_PREFILL_CHUNK 16   /* ids per prefill call: most of the batching gain, Ctrl-C answers in a chunk */

typedef struct {
    const LmTok *tok;           /* borrowed; must outlive the chat */
    const LmTemplate *tpl;      /* borrowed */
    LmBackend backend;
    double temperature, top_p;  /* temperature 0 = greedy */
    int top_k;                  /* 0 = off */
    uint64_t seed;
    int thinking;               /* generation prompt opens a think block */
    int strict_history;         /* 1 = the model template's history rendering (re-feeds every turn) */
    const char *system;         /* NULL = no system message */
} LmChatConfig;

typedef struct LmChat LmChat;

/* NULL with a message in err on an unusable backend or allocation failure. */
LmChat *lm_chat_new(const LmChatConfig *cfg, char *err, size_t errlen);
void    lm_chat_free(LmChat *c);

/* Forget the conversation (the system message stays) and reset the backend. */
void lm_chat_reset(LmChat *c);

/* How many turns had to reset the backend and re-feed a non-empty state. */
int lm_chat_reprefills(const LmChat *c);
/* Ids the backend currently holds, i.e. the prompt prefix known to be fed. */
int lm_chat_fed(const LmChat *c);

enum {
    LM_STOP_EOS = 1,    /* the model ended its turn */
    LM_STOP_MAX,        /* max_tokens reached */
    LM_STOP_CANCEL,     /* cancelled by the caller */
    LM_STOP_CTX,        /* context window full (the call returns LM_ERR_CTX) */
    LM_STOP_TOOLCALL    /* the model attempted a tool call and generation stopped after it */
};

typedef struct {
    int max_tokens;                 /* <= 0 = until stop or context */
    LmSink sink;                    /* may be NULL */
    void *user;
    int (*cancelled)(void *user);   /* may be NULL; polled before each token */
    void *cancel_user;
} LmTurnOpts;

typedef struct {
    int stop;               /* LM_STOP_* */
    int n_prompt;           /* ids in this turn's rendered prompt */
    int n_prefill;          /* ids actually fed for it */
    int reprefill;          /* 1 if the backend was reset and the state re-fed */
    int n_gen;              /* generated ids, stop id excluded */
    int tool_calls;
    int thinking_open;      /* the turn ended inside an unclosed think block */
    double prefill_s, decode_s;
    const char *reasoning;  /* the parsed text, owned by the chat, valid until the next turn */
    const char *answer;
} LmTurn;

/* Run one turn: append user_text, render, prefill, generate. Returns LM_OK (the turn
 * ended by stop, max_tokens, cancellation or a tool call), LM_ERR_CTX (the prompt did not
 * fit, or the window filled while generating: res->stop == LM_STOP_CTX, and whatever was
 * generated is kept in the history), another LM_ERR_* from the backend (propagated; the
 * state is then treated as invalid and the next turn re-feeds from scratch), or -1 for
 * an internal error (template, allocation, non-finite logits) with a message in err. A
 * turn that fails before generating anything leaves the history as it was. */
int lm_chat_turn(LmChat *c, const char *user_text, const LmTurnOpts *opts, LmTurn *res,
                 char *err, size_t errlen);

#endif
