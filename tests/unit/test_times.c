/* test_times.c - lm_times_enable / lm_times_get on the tiny Qwen3.5 and Gemma 4 fixtures (run
 * from the repository root; a missing fixture is a FAILURE): off by default and when stopped
 * nothing is counted; on, every stage the family has is counted, the stages never add up to
 * more than the total, Gemma (no linear attention) counts none, enabling again starts from
 * zero, and profiling leaves the logits bit-identical. */
#include "lm_test.h"

#include "lm/lm.h"

#define N 40   /* more than one prefill chunk */

static int tok[N];

/* N tokens through lm_prefill, then 3 through lm_feed; returns the last logits (malloc'd) */
static float *run(const char *dir, const char *weights, int *vocab)
{
    char err[512] = "";
    LmOpts o = { weights, 0, 0, 0 };
    LmModel *m = lm_open(dir, &o, err, sizeof err);
    CHECK_MSG(m != NULL, "lm_open(%s): %s", dir, err);
    if (!m) return NULL;
    const int V = lm_n_vocab(m);
    float *l = (float *)malloc((size_t)V * sizeof *l);
    for (int i = 0; i < N; i++) tok[i] = (i * 37 + 5) % V;
    CHECK(lm_prefill(m, tok, N, NULL) == LM_OK);
    for (int k = 0; k < 3; k++) CHECK(lm_feed(m, tok[k], l) == LM_OK);
    lm_close(m);
    *vocab = V;
    return l;
}

static int all_zero(const LmTimes *t) { return t->total == 0 && t->matmul == 0 && t->linear == 0 && t->attention == 0; }

static void check_family(const char *dir, const char *weights, int has_linear, const char *label)
{
    LmTimes t;
    int V = 0;

    lm_times_enable(0);
    lm_times_get(&t);
    const LmTimes before = t;
    float *off = run(dir, weights, &V);
    lm_times_get(&t);
    CHECK_MSG(!memcmp(&t, &before, sizeof t), "%s: counters moved while profiling was off", label);

    lm_times_enable(1);
    lm_times_get(&t);
    CHECK_MSG(all_zero(&t), "%s: enabling did not zero the counters", label);
    float *on = run(dir, weights, &V);
    lm_times_get(&t);
    printf("  %s: total %.6f s, matmul %.6f, linear %.6f, attention %.6f\n", label, t.total, t.matmul, t.linear,
           t.attention);
    CHECK_MSG(t.total > 0 && t.matmul > 0 && t.attention > 0, "%s: a stage that ran was not counted", label);
    if (has_linear) CHECK_MSG(t.linear > 0, "%s: linear attention ran but was not counted", label);
    /* NEGATIVE: Gemma has no linear-attention layers, so nothing may be booked there */
    else CHECK_MSG(t.linear == 0, "%s: %.9f s booked to linear attention, which Gemma does not have", label, t.linear);
    /* a mixer's own time is a difference of clock readings: allow rounding, not real overlap */
    CHECK(t.linear >= -1e-9 && t.attention >= -1e-9);
    CHECK_MSG(t.matmul + t.linear + t.attention <= t.total * (1 + 1e-9) + 1e-9,
              "%s: stages %.9f s exceed the total %.9f s (a second counted twice)", label,
              t.matmul + t.linear + t.attention, t.total);

    lm_times_enable(0);
    const LmTimes stopped = t;
    free(run(dir, weights, &V));
    lm_times_get(&t);
    CHECK_MSG(!memcmp(&t, &stopped, sizeof t), "%s: counters moved after profiling was stopped", label);

    if (off && on) CHECK_MSG(!memcmp(off, on, (size_t)V * sizeof *on), "%s: profiling changed the logits", label);
    free(off);
    free(on);
}

int main(void)
{
    static const struct { const char *dir, *weights, *label; int linear; } C[] = {
        { "tests/fixtures/qwen35-tiny", NULL, "qwen bf16", 1 },
        { "tests/fixtures/qwen35-tiny", "tests/fixtures/qwen35-tiny/model.q8.safetensors", "qwen q8", 1 },
        { "tests/fixtures/gemma4-tiny", "tests/fixtures/gemma4-tiny/model.q8.safetensors", "gemma q8", 0 },
    };
    for (size_t i = 0; i < sizeof C / sizeof *C; i++) {
        char probe[600];
        snprintf(probe, sizeof probe, "%s/config.json", C[i].dir);
        if (access(probe, R_OK) != 0) {
            fprintf(stderr, "  FAIL fixture %s not found (run from the repository root)\n", probe);
            return 1;
        }
        check_family(C[i].dir, C[i].weights, C[i].linear, C[i].label);
    }
    return lt_finish("test_times");
}
