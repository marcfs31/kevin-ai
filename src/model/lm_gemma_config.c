/* lm_gemma_config.c - see lm_gemma_config.h. Original code. */
#include "io/lm_portable_io.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "core/lm_bf16.h"
#include "io/lm_io.h"
#include "json.h"
#include "model/lm_config.h"
#include "model/lm_gemma_config.h"

#define DIM_MAX (1 << 24)

static int req_int(jval *o, const char *key, int lo, int *out, char *err, size_t n)
{
    jval *v = json_get(o, key);
    if (!v) return lm_fail(err, n, "config.json is missing \"%s\"", key);
    if (v->t != J_NUM || v->num != floor(v->num) || v->num < lo || v->num > DIM_MAX)
        return lm_fail(err, n, "config.json: \"%s\" must be an integer in [%d, %d]", key, lo, DIM_MAX);
    *out = (int)v->num;
    return 0;
}

/* Absent is -1 (image prefill then refuses). Present but not an integer is an error. */
static int opt_int(jval *o, const char *key, int *out, char *err, size_t n)
{
    if (!json_get(o, key)) { *out = -1; return 0; }
    return req_int(o, key, 0, out, err, n);
}

static int req_num(jval *o, const char *key, double *out, char *err, size_t n)
{
    jval *v = json_get(o, key);
    if (!v) return lm_fail(err, n, "config.json is missing \"%s\"", key);
    if (v->t != J_NUM || !isfinite(v->num)) return lm_fail(err, n, "config.json: \"%s\" must be a number", key);
    *out = v->num;
    return 0;
}

static int req_str(jval *o, const char *key, const char **out, char *err, size_t n)
{
    jval *v = json_get(o, key);
    if (!v) return lm_fail(err, n, "config.json is missing \"%s\"", key);
    if (v->t != J_STR) return lm_fail(err, n, "config.json: \"%s\" must be a string", key);
    *out = v->str;
    return 0;
}

/* a boolean the engine only supports as false */
static int req_false(jval *o, const char *key, char *err, size_t n)
{
    jval *v = json_get(o, key);
    if (!v) return lm_fail(err, n, "config.json is missing \"%s\"", key);
    if (v->t != J_BOOL) return lm_fail(err, n, "config.json: \"%s\" must be a boolean", key);
    if (v->boolean) return lm_fail(err, n, "%s=true is not supported", key);
    return 0;
}

/* a key that must be present and null */
static int req_null(jval *o, const char *key, char *err, size_t n)
{
    jval *v = json_get(o, key);
    if (!v) return lm_fail(err, n, "config.json is missing \"%s\"", key);
    if (v->t != J_NULL) return lm_fail(err, n, "%s is not supported (it must be null)", key);
    return 0;
}

static float bf16_round(double v) { return lm_bf16_to_f32(lm_f32_to_bf16((float)v)); }

static int read_rope(jval *rp, const char *layer_type, int slot, LmGemmaCfg *c, char *err, size_t n)
{
    jval *e = json_get(rp, layer_type);
    if (!e || e->t != J_OBJ) return lm_fail(err, n, "config.json: rope_parameters is missing \"%s\"", layer_type);
    const char *rtype = NULL;
    double theta = 0, partial = 1.0, factor = 1.0;
    if (req_str(e, "rope_type", &rtype, err, n) || req_num(e, "rope_theta", &theta, err, n)) return -1;
    if (!(theta > 0)) return lm_fail(err, n, "rope_parameters.%s: rope_theta must be positive", layer_type);
    jval *f = json_get(e, "factor");
    if (f && (f->t != J_NUM || !(f->num > 0))) return lm_fail(err, n, "rope_parameters.%s: factor must be a positive number", layer_type);
    if (!strcmp(rtype, "default")) {
        jval *p = json_get(e, "partial_rotary_factor");
        if (p && (p->t != J_NUM || p->num != 1.0))
            return lm_fail(err, n, "rope_parameters.%s: a default rope with partial_rotary_factor != 1 is not supported", layer_type);
    } else if (!strcmp(rtype, "proportional")) {
        if (req_num(e, "partial_rotary_factor", &partial, err, n)) return -1;
        if (!(partial > 0 && partial <= 1.0))
            return lm_fail(err, n, "rope_parameters.%s: partial_rotary_factor must be in (0, 1]", layer_type);
        if (f) factor = f->num;
    } else {
        return lm_fail(err, n, "rope_type \"%s\" is not supported (only default and proportional)", rtype);
    }
    c->theta[slot] = theta;
    c->partial[slot] = partial;
    c->factor[slot] = factor;
    return 0;
}

int lm_gemma_config_load(const char *path, LmGemmaCfg *c, char *err, size_t errlen)
{
    memset(c, 0, sizeof *c);
    jval *root = lm_config_read_json(path, err, errlen);
    if (!root) return -1;

    int rc = -1;
    const char *mt = NULL, *tmt = NULL, *act = NULL;
    double eps = 0, cap = 0;
    jval *t = json_get(root, "text_config");
    if (!t) { lm_fail(err, errlen, "config.json is missing \"text_config\""); goto out; }
    if (t->t != J_OBJ) { lm_fail(err, errlen, "config.json: \"text_config\" must be an object"); goto out; }
    if (req_str(root, "model_type", &mt, err, errlen)) goto out;
    if (strcmp(mt, "gemma4") != 0) { lm_fail(err, errlen, "model_type \"%s\" is not supported (only gemma4)", mt); goto out; }
    if (req_str(t, "model_type", &tmt, err, errlen)) goto out;
    if (strcmp(tmt, "gemma4_text") != 0) {
        lm_fail(err, errlen, "text_config.model_type \"%s\" is not supported (only gemma4_text)", tmt);
        goto out;
    }
    if (req_int(t, "hidden_size", 1, &c->hidden, err, errlen) || req_int(t, "intermediate_size", 1, &c->inter, err, errlen) ||
        req_int(t, "num_hidden_layers", 1, &c->layers, err, errlen) || req_int(t, "vocab_size", 1, &c->vocab, err, errlen) ||
        req_num(t, "rms_norm_eps", &eps, err, errlen) ||
        req_int(t, "num_attention_heads", 1, &c->heads, err, errlen) || req_int(t, "num_key_value_heads", 1, &c->kv_heads, err, errlen) ||
        req_int(t, "head_dim", 1, &c->head_dim, err, errlen) || req_int(t, "global_head_dim", 1, &c->global_head_dim, err, errlen) ||
        req_int(t, "sliding_window", 1, &c->window, err, errlen) ||
        req_int(t, "max_position_embeddings", 1, &c->max_pos, err, errlen) ||
        req_int(t, "num_kv_shared_layers", 0, &c->kv_shared, err, errlen) ||
        req_int(t, "hidden_size_per_layer_input", 0, &c->ple_dim, err, errlen) ||
        req_str(t, "hidden_activation", &act, err, errlen)) goto out;
    int vocab_ple = 0;
    if (req_int(t, "vocab_size_per_layer_input", 1, &vocab_ple, err, errlen)) goto out;

    jval *fc = json_get(t, "final_logit_softcapping");
    if (!fc) { lm_fail(err, errlen, "config.json is missing \"final_logit_softcapping\""); goto out; }
    if (fc->t != J_NUM || !isfinite(fc->num) || !(fc->num > 0)) {
        lm_fail(err, errlen, "final_logit_softcapping must be a positive number (the logit softcap is part of the model)");
        goto out;
    }
    cap = fc->num;

    if (req_false(t, "enable_moe_block", err, errlen) || req_false(t, "attention_k_eq_v", err, errlen) ||
        req_false(t, "use_double_wide_mlp", err, errlen) || req_false(t, "attention_bias", err, errlen) ||
        req_null(t, "use_bidirectional_attention", err, errlen) || req_null(t, "num_global_key_value_heads", err, errlen)) goto out;
    jval *plc = json_get(t, "per_layer_config");
    if (plc && plc->t != J_NULL) { lm_fail(err, errlen, "an explicit per_layer_config is not supported"); goto out; }

    if (strcmp(act, "gelu_pytorch_tanh") != 0) {
        lm_fail(err, errlen, "hidden_activation \"%s\" is not supported (only gelu_pytorch_tanh)", act);
        goto out;
    }
    jval *tie = json_get(t, "tie_word_embeddings");
    if (!tie || tie->t != J_BOOL) { lm_fail(err, errlen, "config.json is missing boolean \"tie_word_embeddings\""); goto out; }
    jval *tie_top = json_get(root, "tie_word_embeddings");
    if (!tie->boolean || (tie_top && (tie_top->t != J_BOOL || !tie_top->boolean))) {
        lm_fail(err, errlen, "tie_word_embeddings=false is not supported (the unembedding is the tied embedding)");
        goto out;
    }
    if (c->ple_dim == 0) { lm_fail(err, errlen, "hidden_size_per_layer_input=0 is not supported"); goto out; }
    if (vocab_ple != c->vocab) {
        lm_fail(err, errlen, "vocab_size_per_layer_input (%d) differs from vocab_size (%d): not supported", vocab_ple, c->vocab);
        goto out;
    }
    if (!(eps > 0)) { lm_fail(err, errlen, "rms_norm_eps must be positive"); goto out; }
    c->eps = (float)eps;
    c->cap = (float)cap;

    jval *lt = json_get(t, "layer_types");
    if (!lt) { lm_fail(err, errlen, "config.json is missing \"layer_types\""); goto out; }
    if (lt->t != J_ARR || lt->len != c->layers) {
        lm_fail(err, errlen, "layer_types must be an array of num_hidden_layers (%d) entries", c->layers);
        goto out;
    }
    c->full = (int *)malloc((size_t)c->layers * sizeof(int));
    c->producer = (int *)malloc((size_t)c->layers * sizeof(int));
    if (!c->full || !c->producer) { lm_fail(err, errlen, "out of memory"); goto out; }
    int any_full = 0, any_sliding = 0;
    for (int i = 0; i < c->layers; i++) {
        const jval *e = lt->kids[i];
        if (e->t == J_STR && !strcmp(e->str, "sliding_attention")) { c->full[i] = 0; any_sliding = 1; }
        else if (e->t == J_STR && !strcmp(e->str, "full_attention")) { c->full[i] = 1; any_full = 1; }
        else { lm_fail(err, errlen, "layer %d: unsupported layer type (only sliding_attention and full_attention)", i); goto out; }
    }
    if (!c->full[c->layers - 1]) { lm_fail(err, errlen, "the last layer must be full_attention"); goto out; }
    if (c->heads % c->kv_heads) { lm_fail(err, errlen, "head counts are not divisible (heads %% kv_heads != 0)"); goto out; }
    if ((c->head_dim & 1) || (c->global_head_dim & 1)) { lm_fail(err, errlen, "head_dim and global_head_dim must be even"); goto out; }
    if (c->kv_shared >= c->layers) { lm_fail(err, errlen, "num_kv_shared_layers (%d) must be below num_hidden_layers (%d)", c->kv_shared, c->layers); goto out; }
    if ((long long)c->layers * c->ple_dim > DIM_MAX) { lm_fail(err, errlen, "num_hidden_layers * hidden_size_per_layer_input is implausibly large"); goto out; }

    c->first_shared = c->layers - c->kv_shared;
    /* configuration_gemma4.kv_sharing_modes: a reuse layer reads the cache of the LAST layer of
     * its type before first_shared; that layer must exist */
    int last_of[2] = { -1, -1 };
    for (int i = 0; i < c->first_shared; i++) last_of[c->full[i]] = i;
    for (int i = 0; i < c->layers; i++) {
        if (i < c->first_shared) { c->producer[i] = i; continue; }
        if (last_of[c->full[i]] < 0) {
            lm_fail(err, errlen, "layer %d (%s) shares KV but no earlier layer of its type produces it", i,
                    c->full[i] ? "full_attention" : "sliding_attention");
            goto out;
        }
        c->producer[i] = last_of[c->full[i]];
    }

    jval *rp = json_get(t, "rope_parameters");
    if (!rp || rp->t != J_OBJ) { lm_fail(err, errlen, "config.json is missing \"rope_parameters\""); goto out; }
    if (any_sliding && read_rope(rp, "sliding_attention", 0, c, err, errlen)) goto out;
    if (any_full && read_rope(rp, "full_attention", 1, c, err, errlen)) goto out;

    c->embed_scale = bf16_round(sqrt((double)c->hidden));
    c->ple_scale = bf16_round(sqrt((double)c->ple_dim));
    /* Image-block ids. A text-only open does not need them; lm_prefill_image refuses -1.
     * They may sit outside vocab_size (the tiny text fixture keeps the real ids). */
    if (opt_int(t, "pad_token_id", &c->pad_id, err, errlen) ||
        opt_int(root, "boi_token_id", &c->boi, err, errlen) ||
        opt_int(root, "eoi_token_id", &c->eoi, err, errlen)) goto out;
    rc = 0;
out:
    json_free_tree(root);
    if (rc != 0) lm_gemma_config_free(c);
    return rc;
}

void lm_gemma_config_free(LmGemmaCfg *c)
{
    free(c->full);
    free(c->producer);
    memset(c, 0, sizeof *c);
}
