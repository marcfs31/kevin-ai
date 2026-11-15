/* test_fast.c - the opt-in int8 fast path (LmOpts.q8_int8_acts) on the tiny Qwen3.5 and
 * Gemma 4 fixtures' q8 files: it must actually change the arithmetic (a flag that did
 * nothing would pass an error bound), stay close to the exact path, be deterministic and
 * independent of the thread count, and refuse the configurations it cannot honour.
 * Run from the repository root; a missing fixture is a FAILURE. */
#include "lm_test.h"

#include "lm/lm.h"

#define NTOK 10
static const int TOK[NTOK] = { 5, 17, 3, 42, 9, 31, 7, 2, 11, 24 };

/* A gross-breakage limit, not the accuracy claim. Measured on these fixtures: every matrix
 * multiply is off by 0.7 % (rms, max 1.9 %), but the 64-wide random 8-layer fixtures
 * compound that to 12 % (Gemma) and 27 % (Qwen) of the largest logit; the real models are
 * far wider, and `make gate-fast` and the PR notes report the real numbers. A wrong sign,
 * a wrong scale or a dropped group moves logits by about 100 %. */
#define MAX_REL_ERR 0.50

static float *run(const char *dir, const char *weights, int fast, int threads, int *vocab_out)
{
    char err[512] = "";
    LmOpts o = { weights, 0, threads, fast };
    LmModel *m = lm_open(dir, &o, err, sizeof err);
    CHECK_MSG(m != NULL, "lm_open(%s, fast=%d): %s", dir, fast, err);
    if (!m) return NULL;
    const int V = lm_n_vocab(m);
    float *all = (float *)malloc((size_t)NTOK * V * sizeof *all);
    for (int t = 0; t < NTOK; t++) CHECK(lm_feed(m, TOK[t], all + (size_t)t * V) == LM_OK);
    lm_close(m);
    *vocab_out = V;
    return all;
}

static void check_family(const char *dir, const char *label)
{
    char q8[600];
    snprintf(q8, sizeof q8, "%s/model.q8.safetensors", dir);
    int V = 0, V2 = 0;
    float *exact = run(dir, q8, 0, 0, &V);
    float *fast = run(dir, q8, 1, 0, &V2);
    float *fast1 = run(dir, q8, 1, 1, &V2);
    float *fast4 = run(dir, q8, 1, 4, &V2);
    if (!exact || !fast || !fast1 || !fast4) { free(exact); free(fast); free(fast1); free(fast4); return; }
    const size_t n = (size_t)NTOK * V;

    double maxref = 0, maxerr = 0;
    size_t differ = 0;
    for (size_t i = 0; i < n; i++) {
        CHECK(isfinite(fast[i]));
        maxref = fmax(maxref, fabs((double)exact[i]));
        maxerr = fmax(maxerr, fabs((double)fast[i] - (double)exact[i]));
        if (fast[i] != exact[i]) differ++;
    }
    printf("  %s: fast vs exact: max |diff| = %.3g (%.2f%% of max |logit| %.3g), %zu of %zu logits differ\n",
           label, maxerr, 100.0 * maxerr / maxref, maxref, differ, n);
    CHECK_MSG(differ > 0, "%s: the fast path produced bit-identical logits; the flag did nothing", label);
    CHECK_MSG(maxerr <= MAX_REL_ERR * maxref, "%s: fast path error %.3g above %.0f%% of max logit %.3g", label,
              maxerr, 100 * MAX_REL_ERR, maxref);
    CHECK_MSG(!memcmp(fast, fast1, n * sizeof *fast), "%s: fast path differs between runs/thread counts (1)", label);
    CHECK_MSG(!memcmp(fast, fast4, n * sizeof *fast), "%s: fast path differs with 4 threads", label);
    free(exact); free(fast); free(fast1); free(fast4);
}

static void expect_open_fail(const char *dir, const LmOpts *o, const char *needle, const char *label)
{
    char err[512] = "";
    LmModel *m = lm_open(dir, o, err, sizeof err);
    CHECK_MSG(m == NULL && strstr(err, needle), "%s: expected an error containing \"%s\", got %s \"%s\"", label,
              needle, m ? "a model" : "an error", err);
    if (m) lm_close(m);
}

int main(void)
{
    static const char *const FAM[2][2] = { { "tests/fixtures/qwen35-tiny", "qwen35" },
                                           { "tests/fixtures/gemma4-tiny", "gemma4" } };
    for (int f = 0; f < 2; f++) {
        char probe[600];
        snprintf(probe, sizeof probe, "%s/model.q8.safetensors", FAM[f][0]);
        if (access(probe, R_OK) != 0) {
            fprintf(stderr, "  FAIL fixture %s not found (run from the repository root)\n", probe);
            return 1;
        }
        check_family(FAM[f][0], FAM[f][1]);
        /* NEGATIVE: the fast path has nothing to round on a bf16 checkpoint, and 2 is not a flag */
        LmOpts nofile = { NULL, 0, 0, 1 };
        expect_open_fail(FAM[f][0], &nofile, "needs a q8 weights file", "fast without q8 weights");
        LmOpts two = { probe, 0, 0, 2 };
        expect_open_fail(FAM[f][0], &two, "q8_int8_acts must be 0 or 1", "flag value 2");
    }
    return lt_finish("test_fast");
}
