/* test_ctx.c - the context window chosen from free memory (lm_ctx_choose), with synthetic
 * sizes: a state of base + per * ctx bytes. */
#include "lm_test.h"

#include <string.h>

#include "model/lm_ctx.h"

#define MIB ((uint64_t)1 << 20)
#define GIB ((uint64_t)1 << 30)

typedef struct { uint64_t base, per; } Lin;

static size_t lin_bytes(const void *arg, int ctx)
{
    const Lin *l = (const Lin *)arg;
    return (size_t)(l->base + l->per * (uint64_t)ctx);
}

static int choose(int req, int max_pos, uint64_t weights, const Lin *l, LmMemStatus st, uint64_t avail, int *ctx,
                  char *err)
{
    return lm_ctx_choose(req, max_pos, weights, lin_bytes, l, st, avail, ctx, err, 256);
}

int main(void)
{
    char err[256];
    int ctx;
    const Lin qwen = { 100 * MIB, 64 * 1024 };   /* about Qwen3.5-4B: 64 KiB of KV per position */

    /* memory unknown: nothing refused, 0 means the default (or max_pos when smaller) */
    CHECK(choose(0, 262144, 5 * GIB, &qwen, LM_MEM_UNAVAILABLE, 0, &ctx, err) == 0 && ctx == LM_CTX_DEFAULT);
    CHECK(choose(0, 2048, 5 * GIB, &qwen, LM_MEM_FAILED, 0, &ctx, err) == 0 && ctx == 2048);
    CHECK(choose(1000000, 262144, 5 * GIB, &qwen, LM_MEM_FAILED, 0, &ctx, err) == 0 && ctx == 1000000);

    /* automatic: the largest aligned window that keeps weights and state in memory */
    {
        const uint64_t avail = 11 * GIB, weights = 4608 * MIB, room = avail - LM_CTX_MARGIN;
        CHECK(choose(0, 262144, weights, &qwen, LM_MEM_OK, avail, &ctx, err) == 0);
        CHECK(ctx % LM_CTX_ALIGN == 0);
        CHECK(weights + lin_bytes(&qwen, ctx) <= room);
        CHECK(weights + lin_bytes(&qwen, ctx + LM_CTX_ALIGN) > room);
        CHECK(ctx == 96512);   /* (11 GiB - 0.5 - 4.5 - 100 MiB) / 64 KiB = 96704, aligned down */
    }
    /* capped at max_pos, which need not be aligned */
    CHECK(choose(0, 40000, 4 * GIB, &qwen, LM_MEM_OK, 64 * GIB, &ctx, err) == 0 && ctx == 40000);
    /* exactly at the boundary: a window whose state fills the room to the byte is taken */
    {
        const Lin flat = { 0, 1024 };
        const uint64_t avail = LM_CTX_MARGIN + GIB + 8192 * 1024;
        CHECK(choose(0, 1 << 20, GIB, &flat, LM_MEM_OK, avail, &ctx, err) == 0 && ctx == 8192);
        CHECK(choose(0, 1 << 20, GIB, &flat, LM_MEM_OK, avail - 1, &ctx, err) == 0 && ctx == 7936);
    }
    /* weights that leave no room for the default window (bf16 streamed from disk): the
     * default, as long as its state alone fits */
    CHECK(choose(0, 262144, 16 * GIB, &qwen, LM_MEM_OK, 11 * GIB, &ctx, err) == 0 && ctx == LM_CTX_DEFAULT);
    CHECK(choose(0, 262144, 10 * GIB + 300 * MIB, &qwen, LM_MEM_OK, 11 * GIB, &ctx, err) == 0 && ctx == LM_CTX_DEFAULT);
    /* an explicit window is judged by the same rule: no refusal that ctx 0 would then ignore */
    CHECK(choose(4096, 262144, 10 * GIB + 300 * MIB, &qwen, LM_MEM_OK, 11 * GIB, &ctx, err) == 0 && ctx == 4096);
    CHECK(choose(8192, 262144, 10 * GIB + 300 * MIB, &qwen, LM_MEM_OK, 11 * GIB, &ctx, err) == 0 && ctx == 8192);
    /* ...and refused when even that state does not fit */
    ctx = -7;
    err[0] = 0;
    CHECK(choose(0, 262144, 16 * GIB, &qwen, LM_MEM_OK, 600 * MIB, &ctx, err) == -1 && ctx == -7);
    CHECK(strstr(err, "4096-token") != NULL);
    CHECK(choose(0, 262144, 0, &qwen, LM_MEM_OK, 100 * MIB, &ctx, err) == -1);   /* below the margin */

    /* explicit: taken unchanged when it fits, never aligned */
    CHECK(choose(5000, 262144, 4608 * MIB, &qwen, LM_MEM_OK, 11 * GIB, &ctx, err) == 0 && ctx == 5000);
    CHECK(choose(96704, 262144, 4608 * MIB, &qwen, LM_MEM_OK, 11 * GIB, &ctx, err) == 0 && ctx == 96704);
    /* refused when its state would push resident weights out, naming the largest that fits */
    ctx = -7;
    err[0] = 0;
    CHECK(choose(96705, 262144, 4608 * MIB, &qwen, LM_MEM_OK, 11 * GIB, &ctx, err) == -1 && ctx == -7);
    CHECK(strstr(err, "96704") != NULL && strstr(err, "ctx 0") != NULL);
    /* refused when its state does not fit at all */
    err[0] = 0;
    CHECK(choose(262144, 262144, 16 * GIB, &qwen, LM_MEM_OK, 11 * GIB, &ctx, err) == -1);
    CHECK(strstr(err, "262144-token") != NULL && strstr(err, "lower ctx") != NULL);
    /* streamed weights: an explicit window is taken when its state alone fits */
    CHECK(choose(32768, 262144, 16 * GIB, &qwen, LM_MEM_OK, 11 * GIB, &ctx, err) == 0 && ctx == 32768);

    /* bad arguments */
    CHECK(lm_ctx_choose(0, 4096, 0, NULL, &qwen, LM_MEM_OK, GIB, &ctx, err, sizeof err) == -1);
    CHECK(choose(0, 0, 0, &qwen, LM_MEM_OK, GIB, &ctx, err) == -1);
    CHECK(choose(-1, 4096, 0, &qwen, LM_MEM_OK, GIB, &ctx, err) == -1);
    CHECK(lm_ctx_choose(0, 4096, 0, lin_bytes, &qwen, LM_MEM_OK, GIB, NULL, err, sizeof err) == -1);

    /* the live probe gives some answer on this machine */
    CHECK(lm_ctx_choose_here(0, 4096, 0, lin_bytes, &(Lin){ 0, 1 }, &ctx, err, sizeof err) == 0 && ctx == 4096);

    return lt_finish("test_ctx");
}
