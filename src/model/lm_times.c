/* lm_times.c - see lm_times.h. Original code. */
#include "io/lm_portable_io.h"

#include <string.h>
#include <time.h>

#include "model/lm_times.h"

int lm_times_on = 0;
LmTimes lm_times_acc;

double lm_times_now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

void lm_times_enable(int on)
{
    if (on) memset(&lm_times_acc, 0, sizeof lm_times_acc);
    lm_times_on = on != 0;
}

void lm_times_get(LmTimes *out) { *out = lm_times_acc; }
