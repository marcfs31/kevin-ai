/* lm_times.h - the counters behind lm_times_enable / lm_times_get (include/lm/lm.h): one
 * process-wide set, owned by lm_times.c. Each timed region reads the clock only while
 * profiling is on, so the default path pays one branch. A mixer's own time is its elapsed
 * time minus the multiplies it ran, so no second is counted twice. Original code. */
#ifndef LM_TIMES_H
#define LM_TIMES_H

#include "lm/lm.h"

extern int lm_times_on;
extern LmTimes lm_times_acc;
double lm_times_now(void);

#define LM_TIMES_T0(t0) const double t0 = lm_times_on ? lm_times_now() : 0.0
#define LM_TIMES_ADD(field, t0) \
    do { if (lm_times_on) lm_times_acc.field += lm_times_now() - (t0); } while (0)
#define LM_TIMES_ADD_OWN(field, t0, mm0) \
    do { if (lm_times_on) lm_times_acc.field += lm_times_now() - (t0) - (lm_times_acc.matmul - (mm0)); } while (0)

#endif /* LM_TIMES_H */
