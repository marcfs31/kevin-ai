/* lm-run.c - bin/lm-run --model DIR [--weights FILE] --tokens 1,2,3 [--logits-out F]
 *                       [--greedy N] [--ctx N] [--threads N] [--fast] [--no-prefill] [--profile]
 *
 * Feeds the token ids (through lm_prefill when no logits are written, token by token with
 * --no-prefill or --logits-out: the results are bit-identical), writes one little-endian f32
 * row of `vocab` logits per fed token to F, then generates N tokens by argmax (lowest id wins a tie) and prints one id per
 * line on stdout. Nothing else goes to stdout; errors go to stderr with a non-zero exit.
 * --profile prints, on stderr, where the time went for the prompt and for the generation
 * (lm_times_get: multiplies, linear attention, attention, the rest); the ids are unchanged.
 * Original code; contract in docs/PLAN.md. */
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "lm/lm.h"

static int usage(void)
{
    fprintf(stderr, "usage: lm-run --model DIR [--weights FILE] --tokens 1,2,3 [--logits-out F]\n"
                    "              [--greedy N] [--ctx N] [--threads N] [--fast] [--no-prefill] [--profile]\n");
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
    default:           return "unknown error";
    }
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

int main(int argc, char **argv)
{
    const char *model = NULL, *weights = NULL, *tokstr = NULL, *outp = NULL;
    int greedy = 0, ctx = 0, threads = 0, fast = 0, no_prefill = 0, profile = 0;
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
        else if (!strcmp(a, "--threads") && has) { if (parse_int(argv[++i], 1, &threads)) { fprintf(stderr, "lm-run: bad --threads\n"); return 2; } }
        else return usage();
    }
    if (!model || !tokstr) return usage();

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
    LmOpts o = { weights, ctx, threads, fast };
    LmModel *m = lm_open(model, &o, err, sizeof err);
    if (!m) { fprintf(stderr, "lm-run: %s\n", err); free(tok); return 1; }
    const int vocab = lm_n_vocab(m);
    float *logits = (float *)malloc((size_t)vocab * sizeof(float));
    FILE *fo = NULL;
    int rc = 1;
    if (!logits) { fprintf(stderr, "lm-run: out of memory\n"); goto done; }
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
    for (int k = 0; k < greedy; k++) {
        int best = 0;
        for (int i = 1; i < vocab; i++) if (logits[i] > logits[best]) best = i;
        printf("%d\n", best);
        if (k + 1 < greedy) {
            const int r = lm_feed(m, best, logits);
            if (r != LM_OK) { fprintf(stderr, "lm-run: generating token %d: %s\n", k + 1, code_name(r)); goto done; }
        }
    }
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
    lm_close(m);
    free(tok);
    return rc;
}
