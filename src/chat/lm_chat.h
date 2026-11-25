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
 * Prefix cache. A backend with save/restore/drop/snap_bytes keeps snapshots of the state, each
 * taken at the end of a system message. When a turn cannot extend the fed ids (another conversation, a
 * /reset, an edited history) and the snapshot's ids are a prefix of the new prompt, the state
 * is restored from it and only the rest is fed. When the old ids (fed, or fed before the last
 * lm_chat_reset) and the new prompt share the system message and the cache does not cover it,
 * the turn snapshots the state as its prefill passes that point, replacing the cache: the
 * second conversation that starts with the same system prompt pays for it once, every later
 * one restores it. Snapshots are kept most recently used first while their bytes fit
 * cfg.cache_bytes (the newest always stays, so the peak is the budget plus one snapshot);
 * with a budget of 0 there is one slot, and clients alternating between two system prompts
 * replace it each time (a copy of the state, never a wrong answer). No system message, no cache. A shared prefix shorter than
 * LM_CHAT_CACHE_MIN ids (a template's opening tokens) does not replace the cache: it would
 * evict a snapshot that saves more than it does. Restoring is bit-identical to
 * feeding (lm_restore), so the cache changes speed, never output.
 *
 * Rewind marks. A backend with mark/rewind also keeps up to LM_CHAT_MARKS marks (lm_mark: the
 * state that later positions overwrite, a fixed size whatever the position) at the start of
 * each turn's new tail and one id before the end of its prompt. When a prompt does not extend
 * the fed ids but shares a prefix with them (the last reply regenerated, the last user message
 * edited, a reply stopped and retried), the loop rewinds to the latest mark inside that prefix
 * and feeds only the rest: one id for a regenerate. Marks share cfg.cache_bytes with the prefix
 * cache, oldest dropped first (the newest always stays); a reset, a cache restore or a prompt
 * with images drops them. Rewinding is bit-identical to feeding, so marks change speed, never
 * output.
 *
 * Tokens that were sampled but not fed (the last one when max_tokens or the context
 * ends the turn, a stop id) are simply part of the next prompt's tail. */
#ifndef LM_CHAT_H
#define LM_CHAT_H

#include <stddef.h>
#include <stdint.h>

#include "json.h"
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
    /* Optional, all four or none (the prefix cache): a copy of the current state or NULL on
     * failure (the turn then goes on without caching); put a copy back, LM_OK or an error
     * (after an error the loop resets the backend); free a copy; the bytes a copy holds. */
    void *(*save)(void *ctx);
    int   (*restore)(void *ctx, void *snap);
    void  (*drop)(void *ctx, void *snap);
    size_t (*snap_bytes)(void *ctx, void *snap);
    /* Optional: feed one image block (lm_prefill_image). Required when a turn's prompt has
     * image spans; without it those turns fail. */
    int (*prefill_image)(void *ctx, const float *rows, int n, int width, int grid_h, int grid_w,
                         float *logits);
    /* Optional, both or none, and only with the four above (rewind marks): a mark of the
     * current state or NULL (the turn goes on without it); rewind to a mark taken since the
     * last reset or restore, at or behind the current position, LM_OK or an error (nothing
     * changed: the loop resets). drop and snap_bytes free and size marks too. */
    void *(*mark)(void *ctx);
    int   (*rewind)(void *ctx, void *mark);
} LmBackend;

#define LM_CHAT_MARKS 4            /* rewind marks kept at most */
#define LM_CHAT_CACHE_MIN 16       /* the shortest shared prefix worth a snapshot */
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
    size_t cache_bytes;         /* prefix-cache budget: 0 = one slot */
} LmChatConfig;

typedef struct LmChat LmChat;

/* NULL with a message in err on an unusable backend or allocation failure. */
LmChat *lm_chat_new(const LmChatConfig *cfg, char *err, size_t errlen);
void    lm_chat_free(LmChat *c);

/* Forget the conversation (the system message stays) and reset the backend. The prefix
 * cache stays, and what was fed is remembered for its next snapshot. */
void lm_chat_reset(LmChat *c);
void lm_chat_purge(LmChat *c);
int lm_chat_set_system(LmChat *c, const char *text);

/* How many turns had to reset the backend and re-feed a non-empty state from scratch (a turn
 * restored from the prefix cache is not counted). */
int lm_chat_reprefills(const LmChat *c);
/* Ids the backend currently holds, i.e. the prompt prefix known to be fed. */
int lm_chat_fed(const LmChat *c);
/* Ids the most recently used prefix-cache slot covers (0 = empty or no cache), the number of
 * slots, and the bytes they hold. */
int lm_chat_cached(const LmChat *c);
int lm_chat_cache_slots(const LmChat *c);
size_t lm_chat_cache_bytes(const LmChat *c);
/* Rewind marks held now, and the bytes they hold (beside lm_chat_cache_bytes). */
int lm_chat_marks(const LmChat *c);
size_t lm_chat_mark_bytes(const LmChat *c);

/* Replace the conversation with msgs[0..n): an optional system message first, then user and
 * assistant messages (content only for assistants; reasoning and gen_ids are ignored). This
 * is how a client that sends the whole conversation each time (lm-serve) talks to the chat.
 * Leading messages that equal the chat's own history (same role, content fingerprint,
 * including image SHAs, tool calls and tool name, in order) keep what the chat knows about them, so an answer it
 * generated is rendered from its exact ids and the next turn's prompt still extends the fed
 * state. Multimodal user messages must carry parts with owned-copyable image rows; content
 * is the fingerprint string (text plus image SHAs). Returns 0, or -1 with a message in err
 * (a system message that is not first, a message without text, allocation failure); on -1
 * the conversation is unchanged. The backend is not touched: the next turn decides what to
 * feed. */
int lm_chat_sync(LmChat *c, const LmMsg *msgs, int n, char *err, size_t errlen);

/* Tools for the turns that follow: the request's OpenAI "tools" array, or NULL for none.
 * Borrowed: it must stay valid until the next call, and the caller clears it (NULL) when the
 * request is done. With tools set the prompt carries them (the model's own template), a
 * generated tool call is parsed (LmTurnOpts.on_call, LmTurn.calls) and a reply may hold
 * several. Without tools the first tool call ends the turn with LM_STOP_TOOLCALL. */
void lm_chat_set_tools(LmChat *c, const jval *tools);

/* New sampling settings from the next turn on, with the random stream re-seeded. Checked
 * when sampling (an invalid combination makes the turn fail), like LmChatConfig's. */
void lm_chat_set_sampling(LmChat *c, double temperature, double top_p, int top_k, uint64_t seed);

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
    /* May be NULL. A complete, well-formed tool call, as the model finishes writing it
     * (tools set). A malformed one is not reported here: its text goes to the sink as answer. */
    void (*on_call)(void *user, int index, const LmToolCall *call);
} LmTurnOpts;

typedef struct {
    int stop;               /* LM_STOP_* */
    int n_prompt;           /* ids in this turn's rendered prompt */
    int n_prefill;          /* ids actually fed for it */
    int n_cached;           /* ids restored from the prefix cache instead of fed */
    int n_rewound;          /* ids kept by rewinding to a mark instead of fed again */
    int reprefill;          /* 1 if the backend was reset and the state re-fed from scratch */
    int n_gen;              /* generated ids, stop id excluded */
    int tool_calls;         /* tool calls attempted, well-formed or not */
    int n_calls;            /* the well-formed ones: calls[0..n_calls) */
    const LmToolCall *calls; /* owned by the chat, valid until the next turn */
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

/* Answer the conversation as it stands, which must end with a user or a tool message (the
 * results of the calls the last assistant message made): lm_chat_sync it, then this. Same
 * returns as lm_chat_turn; a failure leaves the history as it was. */
int lm_chat_continue(LmChat *c, const LmTurnOpts *opts, LmTurn *res, char *err, size_t errlen);

/* Like lm_chat_turn, but the user message is multimodal parts (text and/or images).
 * parts must remain valid through the call; image rows are copied into history. */
int lm_chat_turn_parts(LmChat *c, const LmMsgPart *parts, int n_parts, const LmTurnOpts *opts,
                       LmTurn *res, char *err, size_t errlen);

#endif
