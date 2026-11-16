/* lm_ctx.c - the context window from free memory; see lm_ctx.h. */
#include "model/lm_ctx.h"

#include "io/lm_io.h"

#define GIB(x) ((double)(x) / (1024.0 * 1024.0 * 1024.0))

int lm_ctx_choose(int requested, int max_pos, uint64_t weights, LmStateBytesFn fn, const void *arg,
                  LmMemStatus st, uint64_t avail, int *ctx, char *err, size_t errlen)
{
    if (!ctx || !fn || max_pos <= 0 || requested < 0) {
        lm_fail(err, errlen, "lm_ctx_choose: bad arguments");
        return -1;
    }
    const int fallback = max_pos < LM_CTX_DEFAULT ? max_pos : LM_CTX_DEFAULT;
    if (st != LM_MEM_OK) {   /* unknown is not "none": refuse nothing, keep the old default */
        *ctx = requested ? requested : fallback;
        return 0;
    }
    const uint64_t room = avail > LM_CTX_MARGIN ? avail - LM_CTX_MARGIN : 0;
    const int resident = weights <= room;
    int best = 0;   /* the largest window that keeps the weights and the state in memory */
    if (resident && weights + (uint64_t)fn(arg, 1) <= room) {
        int lo = 1, hi = max_pos;
        while (lo < hi) {
            const int mid = lo + (hi - lo + 1) / 2;
            if (weights + (uint64_t)fn(arg, mid) <= room) lo = mid;
            else hi = mid - 1;
        }
        best = lo;
    }
    if (requested > 0) {
        const uint64_t s = fn(arg, requested);
        if (s > room) {
            lm_fail(err, errlen, "a %d-token context needs %.2f GiB of state and %.2f GiB is available "
                    "(%.1f GiB kept free); lower ctx", requested, GIB(s), GIB(avail), GIB(LM_CTX_MARGIN));
            return -1;
        }
        if (resident && weights + s > room) {
            lm_fail(err, errlen, "a %d-token context needs %.2f GiB of state, which with the %.2f GiB of "
                    "weights does not fit in the %.2f GiB available (%.1f GiB kept free); the largest that "
                    "fits is %d, or pass ctx 0", requested, GIB(s), GIB(weights), GIB(avail), GIB(LM_CTX_MARGIN), best);
            return -1;
        }
        *ctx = requested;
        return 0;
    }
    if (best >= fallback) {
        int c = best == max_pos ? best : best - best % LM_CTX_ALIGN;
        *ctx = c < fallback ? fallback : c;
        return 0;
    }
    /* the weights leave no room for a default window: they stream from disk, the state must fit */
    const uint64_t s = fn(arg, fallback);
    if (s > room) {
        lm_fail(err, errlen, "even a %d-token context needs %.2f GiB of state and %.2f GiB is available "
                "(%.1f GiB kept free)", fallback, GIB(s), GIB(avail), GIB(LM_CTX_MARGIN));
        return -1;
    }
    *ctx = fallback;
    return 0;
}

int lm_ctx_choose_here(int requested, int max_pos, uint64_t weights, LmStateBytesFn fn, const void *arg,
                       int *ctx, char *err, size_t errlen)
{
    uint64_t avail = 0;
    const LmMemStatus st = lm_mem_available(&avail);
    return lm_ctx_choose(requested, max_pos, weights, fn, arg, st, avail, ctx, err, errlen);
}
