/* lm_profile.c -- see lm_profile.h. Original to this repository; the config.json reading is
 * the code that used to live in src/cli/lm-chat.c. */
#include "chat/lm_profile.h"
#include "io/lm_gguf.h"

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* json.h is a header-only library of file-scope functions; the suppression is scoped to
 * this include so first-party dead code still warns. */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#include "json.h"
#pragma GCC diagnostic pop

/* Reads and parses dir/leaf. Returns the tree, or NULL with err set; *missing is set to 1
 * (and err left empty) when the file does not exist. */
static jval *read_json(const char *dir, const char *leaf, int *missing, char *err, size_t errlen)
{
    char path[4096];
    snprintf(path, sizeof path, "%s/%s", dir, leaf);
    if (missing) *missing = 0;
    FILE *f = fopen(path, "rb");
    if (!f) {
        if (errno == ENOENT && missing) { *missing = 1; return NULL; }
        snprintf(err, errlen, "cannot open %s: %s", path, strerror(errno));
        return NULL;
    }
    char *b = NULL;
    long n = -1;
    if (fseek(f, 0, SEEK_END) == 0) n = ftell(f);
    if (n >= 0 && n < (1L << 26) && fseek(f, 0, SEEK_SET) == 0) b = (char *)malloc((size_t)n + 1);
    if (!b || fread(b, 1, (size_t)n, f) != (size_t)n) {
        snprintf(err, errlen, "cannot read %s", path);
        free(b);
        fclose(f);
        return NULL;
    }
    fclose(f);
    b[n] = 0;
    jval *root = json_parse(b, NULL);
    free(b);
    if (!root || root->t != J_OBJ) {
        snprintf(err, errlen, "%s is not valid JSON (an object is required)", path);
        json_free_tree(root);
        return NULL;
    }
    return root;
}

static void add_eos(LmProfile *p, const jval *v)
{
    if (v && v->t == J_NUM && v->num >= 0 && v->num < 1e9 && p->n_eos < LM_PROFILE_MAX_EOS) p->eos[p->n_eos++] = (int)v->num;
}

static void read_eos(LmProfile *p, const jval *e)
{
    p->n_eos = 0;
    if (e && e->t == J_ARR) for (int i = 0; i < e->len; i++) add_eos(p, e->kids[i]);
    else add_eos(p, e);
}

int lm_profile_read(const char *dir, LmProfile *p, char *err, size_t errlen)
{
    memset(p, 0, sizeof *p);
    char gp[4096];
    if (lm_gguf_locate(dir, gp, sizeof gp)) {   /* a GGUF model: architecture and stop id from its metadata */
        LmGguf g;
        if (lm_gguf_open(&g, gp, err, errlen) != 0) return -1;
        LmGgufStr arch;
        uint64_t eos;
        if (lm_gguf_get_str(&g, "general.architecture", &arch) != 0 || arch.n != 5 || memcmp(arch.s, "qwen2", 5) != 0) {
            snprintf(err, errlen, "%s: only the qwen2 GGUF architecture is supported", gp);
            lm_gguf_close(&g);
            return -1;
        }
        snprintf(p->model_type, sizeof p->model_type, "qwen2");
        if (lm_gguf_get_u64(&g, "tokenizer.ggml.eos_token_id", &eos) == 0 && eos < 1000000000u) p->eos[p->n_eos++] = (int)eos;
        lm_gguf_close(&g);
        return 0;
    }
    jval *root = read_json(dir, "config.json", NULL, err, errlen);
    if (!root) return -1;
    jval *mt = json_get(root, "model_type");
    if (!mt || mt->t != J_STR || !mt->str) {
        snprintf(err, errlen, "%s/config.json has no model_type", dir);
        json_free_tree(root);
        return -1;
    }
    snprintf(p->model_type, sizeof p->model_type, "%s", mt->str);
    jval *tc = json_get(root, "text_config");
    jval *e = json_get(tc && tc->t == J_OBJ ? tc : root, "eos_token_id");
    if (!e) e = json_get(root, "eos_token_id");
    read_eos(p, e);
    json_free_tree(root);

    if (strcmp(p->model_type, "gemma4") != 0) return 0;

    int missing = 0;
    jval *g = read_json(dir, "generation_config.json", &missing, err, errlen);
    if (!g) return missing ? 0 : -1;
    jval *ge = json_get(g, "eos_token_id");
    if (ge) read_eos(p, ge);
    jval *v = json_get(g, "temperature");
    if (v) {
        if (v->t != J_NUM || !isfinite(v->num) || v->num < 0) {
            snprintf(err, errlen, "%s/generation_config.json: temperature must be a number >= 0", dir);
            json_free_tree(g);
            return -1;
        }
        p->has_temperature = 1;
        p->temperature = v->num;
    }
    v = json_get(g, "top_p");
    if (v) {
        if (v->t != J_NUM || !(v->num > 0 && v->num <= 1)) {
            snprintf(err, errlen, "%s/generation_config.json: top_p must be a number in (0, 1]", dir);
            json_free_tree(g);
            return -1;
        }
        p->has_top_p = 1;
        p->top_p = v->num;
    }
    v = json_get(g, "top_k");
    if (v) {
        if (v->t != J_NUM || v->num < 0 || v->num > (double)(1 << 20) || v->num != floor(v->num)) {
            snprintf(err, errlen, "%s/generation_config.json: top_k must be an integer >= 0", dir);
            json_free_tree(g);
            return -1;
        }
        p->has_top_k = 1;
        p->top_k = (int)v->num;
    }
    json_free_tree(g);
    return 0;
}

LmTemplate *lm_profile_template(const LmProfile *p, const LmTok *tok, char *err, size_t errlen)
{
    if (!strcmp(p->model_type, "qwen3_5")) return lm_template_qwen35(tok, p->eos, p->n_eos, err, errlen);
    if (!strcmp(p->model_type, "qwen2")) return lm_template_chatml(tok, p->eos, p->n_eos, err, errlen);
    if (!strcmp(p->model_type, "gemma4")) return lm_template_gemma4(tok, p->eos, p->n_eos, err, errlen);
    snprintf(err, errlen, "no chat template for model_type \"%s\" (supported: qwen3_5, gemma4, qwen2)", p->model_type);
    return NULL;
}
