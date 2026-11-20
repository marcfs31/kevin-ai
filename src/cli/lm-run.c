/* lm-run.c - bin/lm-run --model DIR [--weights FILE] --tokens 1,2,3 [--logits-out F]
 *                       [--greedy N] [--ctx N] [--threads N] [--fast] [--no-prefill] [--profile]
 *                       [--mtp [--mtp-out F]]
 *        bin/lm-run --model DIR [--weights FILE] --image F.ppm --vision-rows-out F
 *
 * Feeds the token ids (through lm_prefill when no logits are written, token by token with
 * --no-prefill or --logits-out: the results are bit-identical), writes one little-endian f32
 * row of `vocab` logits per fed token to F, then generates N tokens by argmax (lowest id wins a tie) and prints one id per
 * line on stdout. Nothing else goes to stdout; errors go to stderr with a non-zero exit.
 * --profile prints, on stderr, where the time went for the prompt and for the generation
 * (lm_times_get: multiplies, linear attention, attention, the rest); the ids are unchanged.
 * --mtp loads Qwen3.5's MTP head and generates by speculative decoding (lm_draft, lm_verify,
 * lm_accept): the ids are exactly those without it; stderr gets how many drafts were accepted.
 * --mtp-out F (with --mtp) writes the head's draft logits for each token position but the last
 * (position p sees the token at p+1), one row each, fed token by token.
 * --image runs only the vision encoder (src/model/lm_vision.h) on a binary PPM already at its
 * grid size and writes one little-endian f32 row per image token to the --vision-rows-out file.
 * Original code; contract in docs/PLAN.md. */
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "io/lm_image.h"
#include "lm/lm.h"
#include "model/lm_vision.h"

static int usage(void)
{
    fprintf(stderr, "usage: lm-run --model DIR [--weights FILE] --tokens 1,2,3 [--logits-out F]\n"
                    "              [--greedy N] [--ctx N] [--threads N] [--fast] [--no-prefill] [--profile]\n"
                    "              [--mtp [--mtp-out F]]\n"
                    "       lm-run --model DIR [--weights FILE] --image F.ppm --vision-rows-out F\n");
    return 2;
}

static int parse_int(const char *s, int lo, int *out)
{
    char *end;
    errno = 0;
    const long v = strtol(s, &end, 10);
    if (end == s || *end || errno || v < lo || v > 2147483647L) return -1;
    *out = (int)v;
    return 0;
}

static const char *code_name(int rc)
{
    switch (rc) {
    case LM_ERR_CTX:   return "context window full";
    case LM_ERR_TOKEN: return "token id outside the vocabulary";
    case LM_ERR_IO:    return "weight read failed";
    case LM_ERR_STATE: return "call out of order";
    default:           return "unknown error";
    }
}

static int argmax(const float *row, int n)
{
    int best = 0;
    for (int i = 1; i < n; i++) if (row[i] > row[best]) best = i;
    return best;
}

/* One stderr line: the time between two lm_times_get snapshots, split by stage. */
static void print_profile(const char *what, int ntok, const LmTimes *a, const LmTimes *b)
{
    const double total = b->total - a->total, mm = b->matmul - a->matmul;
    const double lin = b->linear - a->linear, att = b->attention - a->attention;
    double rest = total - mm - lin - att;
    if (rest < 0) rest = 0;
    const double pc = total > 0 ? 100.0 / total : 0;
    fprintf(stderr, "profile %s: %d tokens in %.3f s (%.2f tok/s): matmul %.3f s (%.0f%%), "
                    "linear attention %.3f s (%.0f%%), attention %.3f s (%.0f%%), rest %.3f s (%.0f%%)\n",
            what, ntok, total, total > 0 ? ntok / total : 0.0, mm, mm * pc, lin, lin * pc, att, att * pc,
            rest, rest * pc);
}

static int run_image(const char *model, const char *weights, const char *image, const char *outp)
{
    char err[1024] = "";
    uint8_t *rgb = NULL;
    float *rows = NULL;
    int w, h, rc = 1;
    LmVision *v = lm_vision_open(model, weights, err, sizeof err);
    if (!v) { fprintf(stderr, "lm-run: %s\n", err); return 1; }
    if (lm_ppm_read(image, &rgb, &w, &h, err, sizeof err) != 0) { fprintf(stderr, "lm-run: %s\n", err); goto done; }
    const int nt = lm_vision_tokens(v, h, w), dim = lm_vision_info(v)->out_dim;
    if (nt < 1) {
        fprintf(stderr, "lm-run: image %dx%d: sides must be multiples of %d, at most %d patches\n", w, h,
                lm_vision_info(v)->unit, LM_VISION_MAX_PATCHES);
        goto done;
    }
    rows = (float *)malloc((size_t)nt * dim * sizeof(float));
    if (!rows) { fprintf(stderr, "lm-run: out of memory\n"); goto done; }
    if (lm_vision_encode(v, rgb, h, w, rows, err, sizeof err) != 0) { fprintf(stderr, "lm-run: %s\n", err); goto done; }
    FILE *f = fopen(outp, "wb");
    if (!f) { fprintf(stderr, "lm-run: cannot create %s: %s\n", outp, strerror(errno)); goto done; }
    const int ok = fwrite(rows, sizeof(float), (size_t)nt * dim, f) == (size_t)nt * dim;
    if (fclose(f) != 0 || !ok) { fprintf(stderr, "lm-run: write to %s failed\n", outp); goto done; }
    rc = 0;
done:
    free(rows);
    free(rgb);
    lm_vision_close(v);
    return rc;
}

int main(int argc, char **argv)
{
    const char *model = NULL, *weights = NULL, *tokstr = NULL, *outp = NULL, *mtp_out = NULL;
    const char *image = NULL, *vrows = NULL;
    int greedy = 0, ctx = 0, threads = 0, fast = 0, no_prefill = 0, profile = 0, mtp = 0;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        const int has = i + 1 < argc;
        if (!strcmp(a, "--model") && has) model = argv[++i];
        else if (!strcmp(a, "--weights") && has) weights = argv[++i];
        else if (!strcmp(a, "--tokens") && has) tokstr = argv[++i];
        else if (!strcmp(a, "--logits-out") && has) outp = argv[++i];
        else if (!strcmp(a, "--greedy") && has) { if (parse_int(argv[++i], 0, &greedy)) { fprintf(stderr, "lm-run: bad --greedy\n"); return 2; } }
        else if (!strcmp(a, "--ctx") && has) { if (parse_int(argv[++i], 1, &ctx)) { fprintf(stderr, "lm-run: bad --ctx\n"); return 2; } }
        else if (!strcmp(a, "--fast")) fast = 1;
        else if (!strcmp(a, "--no-prefill")) no_prefill = 1;
        else if (!strcmp(a, "--profile")) profile = 1;
        else if (!strcmp(a, "--mtp")) mtp = 1;
        else if (!strcmp(a, "--mtp-out") && has) mtp_out = argv[++i];
        else if (!strcmp(a, "--image") && has) image = argv[++i];
        else if (!strcmp(a, "--vision-rows-out") && has) vrows = argv[++i];
        else if (!strcmp(a, "--threads") && has) { if (parse_int(argv[++i], 1, &threads)) { fprintf(stderr, "lm-run: bad --threads\n"); return 2; } }
        else return usage();
    }
    if (image || vrows) {
        if (!model || !image || !vrows || tokstr || outp || mtp || mtp_out || greedy || ctx || threads || fast ||
            no_prefill || profile) return usage();
        return run_image(model, weights, image, vrows);
    }
    if (!model || !tokstr || (mtp_out && !mtp)) return usage();

    int ntok = 0;
    int *tok = (int *)malloc((strlen(tokstr) / 2 + 2) * sizeof(int));
    if (!tok) { fprintf(stderr, "lm-run: out of memory\n"); return 1; }
    {
        char *copy = strdup(tokstr), *save = NULL;
        if (!copy) { fprintf(stderr, "lm-run: out of memory\n"); return 1; }
        for (char *p = strtok_r(copy, ",", &save); p; p = strtok_r(NULL, ",", &save)) {
            if (parse_int(p, 0, &tok[ntok])) { fprintf(stderr, "lm-run: bad token id \"%s\"\n", p); free(copy); free(tok); return 2; }
            ntok++;
        }
        free(copy);
    }
    if (ntok == 0) { fprintf(stderr, "lm-run: --tokens is empty\n"); free(tok); return 2; }

    char err[1024] = "";
    LmOpts o = { weights, ctx, threads, fast, mtp };
    LmModel *m = lm_open(model, &o, err, sizeof err);
    if (!m) { fprintf(stderr, "lm-run: %s\n", err); free(tok); return 1; }
    const int vocab = lm_n_vocab(m);
    float *logits = (float *)malloc((size_t)vocab * sizeof(float));
    float *rows = (float *)malloc((size_t)(LM_VERIFY_MAX + 1) * vocab * sizeof(float)); /* verify rows + a draft row */
    FILE *fo = NULL;
    int rc = 1;
    if (!logits || !rows) { fprintf(stderr, "lm-run: out of memory\n"); goto done; }
    if (mtp_out) {   /* the head's rows, then a fresh state for the run below */
        FILE *fm = fopen(mtp_out, "wb");
        if (!fm) { fprintf(stderr, "lm-run: cannot create %s: %s\n", mtp_out, strerror(errno)); goto done; }
        int r = lm_feed(m, tok[0], NULL);
        for (int p = 0; r == LM_OK && p + 1 < ntok; p++) {
            r = lm_draft(m, tok[p + 1], rows);
            if (r == LM_OK && fwrite(rows, sizeof(float), (size_t)vocab, fm) != (size_t)vocab) r = LM_ERR_IO;
            if (r == LM_OK) r = lm_feed(m, tok[p + 1], NULL);
        }
        if (fclose(fm) != 0 && r == LM_OK) r = LM_ERR_IO;
        if (r != LM_OK) { fprintf(stderr, "lm-run: --mtp-out: %s\n", code_name(r)); goto done; }
        lm_reset(m);
    }
    if (outp) {
        fo = fopen(outp, "wb");
        if (!fo) { fprintf(stderr, "lm-run: cannot create %s: %s\n", outp, strerror(errno)); goto done; }
    }

    LmTimes t_start, t_prompt, t_end;
    lm_times_enable(profile);
    lm_times_get(&t_start);
    if (!fo && !no_prefill) {   /* no per-token rows wanted: the prompt goes in as chunks */
        const int r = lm_prefill(m, tok, ntok, greedy > 0 ? logits : NULL);
        if (r != LM_OK) { fprintf(stderr, "lm-run: prompt of %d tokens: %s\n", ntok, code_name(r)); goto done; }
    } else {
        for (int i = 0; i < ntok; i++) {
            const int want = fo != NULL || (greedy > 0 && i == ntok - 1);
            const int r = lm_feed(m, tok[i], want ? logits : NULL);
            if (r != LM_OK) { fprintf(stderr, "lm-run: token %d (id %d): %s\n", i, tok[i], code_name(r)); goto done; }
            if (fo && fwrite(logits, sizeof(float), (size_t)vocab, fo) != (size_t)vocab) {
                fprintf(stderr, "lm-run: write to %s failed\n", outp);
                goto done;
            }
        }
    }
    lm_times_get(&t_prompt);
    int drafts = 0, accepted = 0;
    for (int k = 0; k < greedy;) {
        const int best = argmax(logits, vocab);
        printf("%d\n", best);
        if (++k == greedy) break;
        int r;
        if (mtp && k + 1 < greedy && lm_pos(m) + 2 <= lm_n_ctx(m)) {
            /* draft the id after best, check both in one pass, keep what greedy agrees with */
            float *dl = rows + (size_t)LM_VERIFY_MAX * vocab;
            int kept = 0;
            r = lm_draft(m, best, dl);
            const int d = argmax(dl, vocab);
            if (r == LM_OK) r = lm_greedy_check(m, best, d, rows, logits, &kept);
            if (r == LM_OK) drafts++;
            if (kept == 2) {
                accepted++;
                printf("%d\n", d);
                k++;
            }
        } else {
            r = lm_feed(m, best, logits);
        }
        if (r != LM_OK) { fprintf(stderr, "lm-run: generating token %d: %s\n", k, code_name(r)); goto done; }
    }
    if (mtp && greedy > 0)
        fprintf(stderr, "lm-run: mtp drafts %d, accepted %d (%.0f%%)\n", drafts, accepted, drafts ? 100.0 * accepted / drafts : 0.0);
    if (fflush(stdout) != 0) { fprintf(stderr, "lm-run: write to stdout failed\n"); goto done; }
    lm_times_get(&t_end);
    if (profile) {
        print_profile("prompt", ntok, &t_start, &t_prompt);
        if (greedy > 1) print_profile("generation", greedy - 1, &t_prompt, &t_end);
    }
    rc = 0;
done:
    if (fo && fclose(fo) != 0 && rc == 0) { fprintf(stderr, "lm-run: close of %s failed\n", outp); rc = 1; }
    free(logits);
    free(rows);
    lm_close(m);
    free(tok);
    return rc;
}
