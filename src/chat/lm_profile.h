/* lm_profile.h -- what lm-chat needs to know about a model directory: the family
 * (config.json model_type), its stop ids and its sampling defaults, and the chat template
 * that goes with the family. The one place that maps a model family to its chat format.
 *
 *   qwen3_5  stop ids from config.json eos_token_id (text_config first); sampling defaults
 *            are lm-chat's own (generation_config.json is not read).
 *   gemma4   stop ids from generation_config.json eos_token_id ([1, 106, 50]: <eos>,
 *            <turn|>, <|tool_response>) when present, else config.json; sampling defaults
 *            (temperature, top_k, top_p) from generation_config.json when present.
 *
 * Any other model_type is an error, never a guess. */
#ifndef LM_PROFILE_H
#define LM_PROFILE_H

#include <stddef.h>

#include "chat/lm_template.h"
#include "tokenizer/lm_tok.h"

#define LM_PROFILE_MAX_EOS 8

typedef struct {
    char model_type[64];
    int eos[LM_PROFILE_MAX_EOS];
    int n_eos;
    /* sampling defaults found in the model directory; has_* = 0 means "use the program's" */
    int has_temperature, has_top_p, has_top_k;
    double temperature, top_p;
    int top_k;
} LmProfile;

/* 0, or -1 with a message in err: config.json missing / malformed / without model_type, or
 * (gemma4) a generation_config.json that exists but is malformed or holds an invalid
 * temperature, top_p or top_k. A missing generation_config.json is not an error. */
int lm_profile_read(const char *dir, LmProfile *p, char *err, size_t errlen);

/* The template for p->model_type, or NULL with a message in err (unsupported model_type, a
 * tokenizer without the family's special tokens). tok must outlive the template. */
LmTemplate *lm_profile_template(const LmProfile *p, const LmTok *tok, char *err, size_t errlen);

#endif
