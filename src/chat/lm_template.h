/* lm_template.h -- chat formats behind one small interface, so the generation loop
 * (lm_chat.c) never learns a model's special tokens. Phase 1 ships Qwen3.5
 * (lm_template_qwen35) and Gemma 4 (lm_template_gemma4), two constructors returning the
 * same LmTemplate, with no change to lm_chat.
 *
 * A template does two things:
 *   render  a conversation -> the exact token ids to feed, ending in the generation prompt;
 *   parser  a streaming parser for the generated ids: reasoning vs answer, stop ids, and
 *           unsupported constructs surfaced (never silently dropped).
 *
 * Tools: LmRenderOpts.tools (the request's OpenAI "tools" array) is rendered where the
 * model's own template puts it (Qwen: a system block; Gemma: <|tool> declarations in the
 * system turn), an assistant message's calls and the tool messages after it are rendered as
 * the template does, and parse_call turns a generated call back into a name and arguments.
 * Images are optional: lm_template_set_vision installs start/pad/end ids so a user message's image
 * parts become start + N pads + end in the rendered prompt (system messages may not
 * contain images). */
#ifndef LM_TEMPLATE_H
#define LM_TEMPLATE_H

#include <stddef.h>

#include "json.h"
#include "tokenizer/lm_tok.h"

/* ---- growable id vector ------------------------------------------------------- */
typedef struct { int *ids; int n, cap; } LmIds;
int  lm_ids_push(LmIds *v, int id);                     /* 0, or -1 on allocation failure */
int  lm_ids_append(LmIds *v, const int *ids, int n);    /* 0, or -1 */
void lm_ids_free(LmIds *v);                             /* leaves an empty vector */

/* ---- messages ---------------------------------------------------------------- */
typedef enum { LM_ROLE_SYSTEM, LM_ROLE_USER, LM_ROLE_ASSISTANT, LM_ROLE_TOOL } LmRole;

/* One function call of an assistant message. arguments is a JSON object text. */
typedef struct {
    const char *name;
    const char *arguments;
} LmToolCall;

/* Free the strings of a call made by parse_call (or by the chat). */
void lm_toolcall_free(LmToolCall *c);

enum { LM_PART_TEXT = 0, LM_PART_IMAGE = 1 };

/* One content part of a user message. Image parts carry already-encoded vision rows
 * (lm_vision_encode) and a SHA-256 hex of the source image bytes for conversation reuse. */
typedef struct {
    int kind;                 /* LM_PART_TEXT or LM_PART_IMAGE */
    const char *text;         /* LM_PART_TEXT */
    const float *rows;        /* LM_PART_IMAGE: n_rows * width floats */
    int n_rows;
    int width;
    int grid_h, grid_w;
    char sha_hex[65];         /* lowercase hex + NUL; required for image parts */
} LmMsgPart;

typedef struct {
    LmRole role;
    const char *content;   /* the message text; for an assistant message, the answer.
                            * For a multimodal user message this is also the fingerprint
                            * used by lm_chat_sync (text plus image SHAs). */
    const char *reasoning; /* assistant only; NULL = none (the template's reasoning_content) */
    /* Assistant messages produced by this program carry the exact ids the model generated
     * (stop id excluded) and whether the generation prompt had thinking on. The
     * "verbatim" history mode renders from them so the prompt is byte-for-byte what was
     * fed; without them the message is rendered from its text like any other. */
    const int *gen_ids;
    int n_gen_ids;
    int gen_thinking;
    /* Optional multimodal content (user messages). NULL / 0 = text-only via content. */
    const LmMsgPart *parts;
    int n_parts;
    /* Assistant: the function calls it made (content is the text around them). Tool: the
     * result of one call, as content, and the name of the function that was called. */
    const LmToolCall *calls;
    int n_calls;
    const char *tool_name;
} LmMsg;

/* Where an image block sits in the rendered id stream: ids[id_start .. id_start+n_ids)
 * are start + n_rows pads + end; img_index is the order of image parts across the
 * conversation (0-based). */
typedef struct {
    int id_start;
    int n_ids;
    int img_index;
} LmImgSpan;

typedef struct {
    int enable_thinking;   /* generation prompt opens a think block (1) or closes an empty one (0) */
    /* 0 = exactly what the model's own chat template does: an earlier assistant turn is
     *     rendered as its answer only, reasoning dropped. The prompt then differs from
     *     what was fed in the previous turn, so the KV/recurrent state cannot be reused.
     * 1 = history assistant turns that carry gen_ids are rendered as they were generated
     *     (generation prompt + generated ids + end of turn), so each turn's prompt extends
     *     the previous one and nothing is re-fed. Deviates from the model's template for
     *     those turns (it would have dropped the think block). */
    int verbatim_history;
    /* Optional: render appends one LmImgSpan per image part (realloc). Caller frees *spans. */
    LmImgSpan **spans;
    int *n_spans;
    /* The request's tools: a JSON array of {"type":"function","function":{name,description,
     * parameters}}, or NULL. Borrowed for the call. */
    const jval *tools;
} LmRenderOpts;

/* ---- streaming parser ---------------------------------------------------------- */
typedef enum { LM_OUT_REASONING, LM_OUT_ANSWER, LM_OUT_TOOLCALL } LmOutKind;

/* Receives complete UTF-8 text, never a split character. Leading and trailing whitespace
 * of each reasoning / answer stream is withheld (the template trims both), so what a
 * sink sees concatenates to the trimmed text. For LM_OUT_TOOLCALL the text is the notice
 * "tool calls are not supported", sent once per call, and the call's body is not forwarded;
 * with LmParser.collect_calls set it is instead the body of each complete call (or of one cut
 * short at the end of the stream), for LmTemplate.parse_call. In a Gemma body the <|"|>
 * delimiter is the byte '\x01'. */
typedef void (*LmSink)(void *user, LmOutKind kind, const char *s, size_t n);

enum {
    LM_PARSE_CONT       = 0, /* keep generating */
    LM_PARSE_STOP       = 1, /* a stop id: the id is not part of the output, do not feed it */
    LM_PARSE_STOP_AFTER = 2  /* the output is complete with this id (end of a tool call); it
                                is part of the generated sequence but nothing follows */
};

typedef struct LmParser LmParser;
struct LmParser {
    int  (*feed)(LmParser *p, int id, LmSink sink, void *user);
    void (*finish)(LmParser *p, LmSink sink, void *user); /* end of stream: flush a held partial character */
    void (*destroy)(LmParser *p);
    /* Valid at any time. */
    int thinking_open; /* inside a think block that has not been closed */
    int tool_calls;    /* number of tool calls attempted */
    /* Set by the caller before the first feed. 0: the first complete tool call ends the
     * output (LM_PARSE_STOP_AFTER) and only a notice is reported. 1: every call is reported
     * whole and generation goes on until a stop id, so a reply can hold several calls. */
    int collect_calls;
};

/* ---- the template ------------------------------------------------------------ */
typedef struct LmTemplate LmTemplate;
struct LmTemplate {
    const char *name;
    /* Render msgs[0..n) plus the generation prompt into out (appended after clearing it).
     * Returns 0, or -1 with a message in err: no messages, a system message that is not
     * first, no user message, a system message with images, vision unset for an image
     * part, or allocation failure. */
    int (*render)(const LmTemplate *t, const LmMsg *msgs, int n, const LmRenderOpts *opts,
                  LmIds *out, char *err, size_t errlen);
    /* thinking: the generation prompt opened a think block, so the first generated tokens
     * are reasoning. */
    LmParser *(*parser_new)(const LmTemplate *t, int thinking);
    int  (*is_stop)(const LmTemplate *t, int id);
    /* The body of a generated tool call (see LmSink) -> name + arguments (a compact JSON
     * object), both malloc'd (lm_toolcall_free). 0, or -1 with a message in err when the
     * body is not a well-formed call. tools gives the declared parameter types (may be NULL). */
    int  (*parse_call)(const LmTemplate *t, const char *raw, const jval *tools, LmToolCall *out,
                       char *err, size_t errlen);
    void (*destroy)(LmTemplate *t);
    void *impl;
};

/* Install the vision special ids used when rendering image parts. start/pad/end of -1
 * clear vision (image parts then fail at render). */
void lm_template_set_vision(LmTemplate *t, int start, int pad, int end);

/* Qwen3.5 (the template of the model's chat_template.jinja, text-only). eos_ids are extra
 * stop ids from config.json (eos_token_id may be one id or several); <|im_end|> is always
 * a stop id. NULL with a message in err if the tokenizer lacks <|im_start|>, <|im_end|>,
 * <think>, </think>, <tool_call> or </tool_call>. tok must outlive the template. */
LmTemplate *lm_template_qwen35(const LmTok *tok, const int *eos_ids, int n_eos, char *err, size_t errlen);
/* Plain ChatML, the template inside the GGUF of Qwen2.5-Coder fine-tunes: <|im_start|>role\n
 * content<|im_end|>\n per message (content verbatim), then <|im_start|>assistant\n. No default
 * system prompt, no reasoning block, no tools, no images. <|im_end|> is always a stop id; eos_ids
 * are extra ones. NULL with a message in err if the tokenizer lacks <|im_start|> or <|im_end|>. */
LmTemplate *lm_template_chatml(const LmTok *tok, const int *eos_ids, int n_eos, char *err, size_t errlen);
/* Gemma 4 (the model's chat_template.jinja, text-only): <bos>, an optional system turn that
 * carries the thinking flag, <|turn>user|model\n ... <turn|>\n turns, generation prompt
 * <|turn>model\n. <turn|> is always a stop id; eos_ids are the others (generation_config.json
 * eos_token_id: <eos>, <turn|>, <|tool_response>). NULL with a message in err if the
 * tokenizer lacks <bos>, <|turn>, <turn|>, <|think|>, <|channel>, <channel|>, <|tool_call>
 * or <tool_call|>. tok must outlive the template. The parser reports "<|channel>thought\n
 * ... <channel|>" as reasoning. */
LmTemplate *lm_template_gemma4(const LmTok *tok, const int *eos_ids, int n_eos, char *err, size_t errlen);
void        lm_template_free(LmTemplate *t);

#endif
