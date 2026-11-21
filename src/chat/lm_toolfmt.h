/* lm_toolfmt.h -- the text formats of tool calling that the two chat templates share the
 * machinery for: JSON the way Python's json.dumps writes it (what the models' jinja
 * templates print for `tool | tojson`), the Gemma 4 declaration / argument syntax, and the
 * parsers that turn a generated tool call back into a name and a JSON object. Internal to
 * src/chat; lm_template.c is the only caller. Original to this repository. */
#ifndef LM_TOOLFMT_H
#define LM_TOOLFMT_H

#include <stddef.h>

#include "chat/lm_template.h"
#include "json.h"

/* Gemma's <|"|> string delimiter is a control token. Inside text that later becomes ids it is
 * written as this byte, and the builder splits on it; user text never carries it (stripped). */
#define LM_QS '\x01'

typedef struct { char *s; size_t n, cap; int oom; } LmSb;

void lm_sb_add(LmSb *b, const char *s, size_t n);
void lm_sb_str(LmSb *b, const char *s);
void lm_sb_free(LmSb *b);
/* user text for the Gemma builder: LM_QS bytes are dropped so it cannot forge <|"|> */
void lm_sb_user(LmSb *b, const char *s);

/* v as JSON. py = 1: Python's default separators (", " and ": "), keys in input order, no
 * ASCII escaping; py = 0: compact (no spaces). */
void lm_json_dump(LmSb *b, const jval *v, int py);
/* a JSON string literal for s[0..n) */
void lm_json_string(LmSb *b, const char *s, size_t n);

/* Qwen3.5: Python's str() of a scalar argument value, JSON text for an object or array. */
void lm_qwen_arg_text(LmSb *b, const jval *v);

/* Gemma 4: "declaration:NAME{...}" for one {"type":"function","function":{...}} tool, with
 * LM_QS standing for <|"|>; and format_argument(v, escape_keys) of the jinja. */
void lm_gemma_declaration(LmSb *b, const jval *tool);
void lm_gemma_argument(LmSb *b, const jval *v, int escape_keys);

/* Parse the text generated between a family's tool-call markers (Gemma's <|"|> as LM_QS).
 * Returns 0 and a call whose name and arguments (compact JSON object) are malloc'd, or -1
 * with a message in err. tools (may be NULL) gives the Qwen parameter types. */
int lm_qwen_parse_call(const char *raw, const jval *tools, LmToolCall *out, char *err, size_t errlen);
int lm_gemma_parse_call(const char *raw, LmToolCall *out, char *err, size_t errlen);

#endif
