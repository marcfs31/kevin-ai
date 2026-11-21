/* lm_config.c - see lm_config.h. Original code. */
#include "io/lm_portable_io.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "io/lm_io.h"
#include "json.h"
#include "model/lm_config.h"

#define DIM_MAX (1 << 24)

static int req_int(jval *o, const char *key, int *out, char *err, size_t n)
{
    jval *v = json_get(o, key);
    if (!v) return lm_fail(err, n, "config.json is missing \"%s\"", key);
    if (v->t != J_NUM || v->num != floor(v->num) || v->num < 1 || v->num > DIM_MAX)
        return lm_fail(err, n, "config.json: \"%s\" must be an integer in [1, %d]", key, DIM_MAX);
    *out = (int)v->num;
    return 0;
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

struct jval *lm_config_read_json(const char *path, char *err, size_t errlen)
{
    char file[4096];
    struct stat st;
    if (stat(path, &st) != 0) { lm_fail(err, errlen, "cannot open %s", path); return NULL; }
    if (S_ISDIR(st.st_mode)) snprintf(file, sizeof file, "%s/config.json", path);
    else snprintf(file, sizeof file, "%s", path);
    FILE *f = fopen(file, "rb");
    if (!f) { lm_fail(err, errlen, "cannot open %s", file); return NULL; }
    fseek(f, 0, SEEK_END);
    const long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0 || sz > (16 << 20)) { fclose(f); lm_fail(err, errlen, "%s has an implausible size", file); return NULL; }
    char *text = (char *)malloc((size_t)sz + 1);
    if (!text) { fclose(f); lm_fail(err, errlen, "out of memory"); return NULL; }
    const size_t got = fread(text, 1, (size_t)sz, f);
    fclose(f);
    text[got] = '\0';
    jval *root = json_parse(text, NULL);
    free(text);
    if (!root || root->t != J_OBJ) { json_free_tree(root); lm_fail(err, errlen, "%s is not a valid JSON object", file); return NULL; }
    return root;
}

int lm_config_model_type(const char *path, char *out, size_t n)
{
    jval *root = lm_config_read_json(path, NULL, 0);
    if (!root) return -1;
    jval *v = json_get(root, "model_type");
    int rc = -1;
    if (v && v->t == J_STR && strlen(v->str) < n) { snprintf(out, n, "%s", v->str); rc = 0; }
    json_free_tree(root);
    return rc;
}

int lm_config_load(const char *path, LmCfg *c, char *err, size_t errlen)
{
    memset(c, 0, sizeof *c);
    jval *root = lm_config_read_json(path, err, errlen);
    if (!root) return -1;

    int rc = -1;
    const char *mt = NULL, *act = NULL, *rtype = NULL;
    double theta = 0, partial = 0, eps = 0;
    int gate_ok = 0;
    jval *t = json_get(root, "text_config");
    if (!t) { lm_fail(err, errlen, "config.json is missing \"text_config\""); goto out; }
    if (t->t != J_OBJ) { lm_fail(err, errlen, "config.json: \"text_config\" must be an object"); goto out; }
    if (req_str(root, "model_type", &mt, err, errlen) != 0) goto out;
    if (strcmp(mt, "qwen3_5") != 0) {
        lm_fail(err, errlen, "model_type \"%s\" is not supported (only qwen3_5)", mt);
        goto out;
    }
    if (req_int(t, "hidden_size", &c->hidden, err, errlen) || req_int(t, "intermediate_size", &c->inter, err, errlen) ||
        req_int(t, "num_hidden_layers", &c->layers, err, errlen) || req_int(t, "vocab_size", &c->vocab, err, errlen) ||
        req_num(t, "rms_norm_eps", &eps, err, errlen) ||
        req_int(t, "num_attention_heads", &c->heads, err, errlen) || req_int(t, "num_key_value_heads", &c->kv_heads, err, errlen) ||
        req_int(t, "head_dim", &c->head_dim, err, errlen) ||
        req_int(t, "linear_num_key_heads", &c->lk_heads, err, errlen) || req_int(t, "linear_num_value_heads", &c->lv_heads, err, errlen) ||
        req_int(t, "linear_key_head_dim", &c->lk_dim, err, errlen) || req_int(t, "linear_value_head_dim", &c->lv_dim, err, errlen) ||
        req_int(t, "linear_conv_kernel_dim", &c->conv_k, err, errlen) ||
        req_int(t, "max_position_embeddings", &c->max_pos, err, errlen) ||
        req_str(t, "hidden_act", &act, err, errlen)) goto out;

    jval *g = json_get(t, "attn_output_gate");
    if (!g) { lm_fail(err, errlen, "config.json is missing \"attn_output_gate\""); goto out; }
    gate_ok = g->t == J_BOOL && g->boolean;
    if (!gate_ok) { lm_fail(err, errlen, "attn_output_gate must be true (the only supported attention form)"); goto out; }

    jval *tie = json_get(t, "tie_word_embeddings");
    if (!tie) tie = json_get(root, "tie_word_embeddings");
    if (!tie || tie->t != J_BOOL) { lm_fail(err, errlen, "config.json is missing boolean \"tie_word_embeddings\""); goto out; }
    c->tied = tie->boolean;

    jval *rp = json_get(t, "rope_parameters");
    if (!rp || rp->t != J_OBJ) { lm_fail(err, errlen, "config.json is missing \"rope_parameters\""); goto out; }
    if (req_num(rp, "rope_theta", &theta, err, errlen) || req_num(rp, "partial_rotary_factor", &partial, err, errlen) ||
        req_str(rp, "rope_type", &rtype, err, errlen)) goto out;
    if (strcmp(rtype, "default") != 0) { lm_fail(err, errlen, "rope_type \"%s\" is not supported (only default)", rtype); goto out; }
    if (strcmp(act, "silu") != 0) { lm_fail(err, errlen, "hidden_act \"%s\" is not supported (only silu)", act); goto out; }
    if (!(theta > 0) || !(eps > 0) || !(partial > 0 && partial <= 1.0)) {
        lm_fail(err, errlen, "rope_theta, rms_norm_eps and partial_rotary_factor must be positive (factor <= 1)");
        goto out;
    }
    c->theta = (float)theta;
    c->eps = (float)eps;
    c->rot = (int)((double)c->head_dim * partial);


    jval *lt = json_get(t, "layer_types");
    if (!lt) { lm_fail(err, errlen, "config.json is missing \"layer_types\""); goto out; }
    if (lt->t != J_ARR || lt->len != c->layers) {
        lm_fail(err, errlen, "layer_types must be an array of num_hidden_layers (%d) entries", c->layers);
        goto out;
    }
    c->layer_type = (int *)malloc((size_t)c->layers * sizeof(int));
    if (!c->layer_type) { lm_fail(err, errlen, "out of memory"); goto out; }
    for (int i = 0; i < c->layers; i++) {
        const jval *e = lt->kids[i];
        if (e->t == J_STR && !strcmp(e->str, "linear_attention")) c->layer_type[i] = LM_LAYER_LINEAR;
        else if (e->t == J_STR && !strcmp(e->str, "full_attention")) c->layer_type[i] = LM_LAYER_FULL;
        else { lm_fail(err, errlen, "layer %d: unsupported layer type (only linear_attention and full_attention)", i); goto out; }
    }
    if (c->heads % c->kv_heads || c->lv_heads % c->lk_heads) {
        lm_fail(err, errlen, "head counts are not divisible (heads %% kv_heads or linear value %% key heads != 0)");
        goto out;
    }
    if (c->rot < 2 || (c->rot & 1) || c->rot > c->head_dim) {
        lm_fail(err, errlen, "rotary dimension %d (head_dim * partial_rotary_factor) must be even and in [2, head_dim]", c->rot);
        goto out;
    }
    jval *ml = json_get(rp, "mrope_interleaved");
    if (!ml || ml->t != J_BOOL || !ml->boolean) {
        lm_fail(err, errlen, "rope_parameters.mrope_interleaved must be true (the only supported M-RoPE layout)");
        goto out;
    }
    jval *ms = json_get(rp, "mrope_section");
    if (!ms || ms->t != J_ARR || ms->len != 3 || ms->kids[0]->t != J_NUM || ms->kids[1]->t != J_NUM ||
        ms->kids[2]->t != J_NUM) {
        lm_fail(err, errlen, "config.json is missing rope_parameters.mrope_section (3 numbers)");
        goto out;
    }
    {
        const double st = ms->kids[0]->num, sh = ms->kids[1]->num, sw = ms->kids[2]->num;
        if (st < 0 || sh < 0 || sw < 0 || st != (int)st || sh != (int)sh || sw != (int)sw ||
            (int)st + (int)sh + (int)sw != c->rot / 2) {
            lm_fail(err, errlen, "rope_parameters.mrope_section must be 3 non-negative integers summing to rot/2 (%d)", c->rot / 2);
            goto out;
        }
        c->sec_h = (int)sh;
        c->sec_w = (int)sw;
    }
    c->key_dim = c->lk_heads * c->lk_dim;
    c->value_dim = c->lv_heads * c->lv_dim;
    c->conv_dim = 2 * c->key_dim + c->value_dim;
    rc = 0;
out:
    json_free_tree(root);
    if (rc != 0) lm_config_free(c);
    return rc;
}

void lm_config_free(LmCfg *c)
{
    free(c->layer_type);
    memset(c, 0, sizeof *c);
}
