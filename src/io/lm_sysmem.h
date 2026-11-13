/* lm_sysmem.h - how much memory this machine can actually hand out.
 *
 * Ported from kimi-k3-in-c src/cli/k3_sysmem.h (Apache-2.0) via an earlier unreviewed
 * salvage (oss_sysmem.[ch], reviewed here). Changes: the Windows branch is dropped; the
 * result is a tri-state instead of a double where 0.0 meant "unknown" (which a caller
 * comparing "available > 0" turned into a silently disabled guard); the /proc/meminfo
 * field parser is exposed as a pure function so it is testable on every platform; the
 * output is integer bytes.
 *
 *   Linux   MemAvailable, the kernel's estimate of what can be handed out without
 *           swapping (it counts reclaimable page cache; MemFree does not).
 *   Darwin  (free + inactive + purgeable) pages from host_statistics64.
 *   other   LM_MEM_UNAVAILABLE.
 *
 * A value of 0 bytes with LM_MEM_OK is a real answer ("none"); it is not "unknown". */
#ifndef LM_SYSMEM_H
#define LM_SYSMEM_H

#include <stdint.h>

typedef enum {
    LM_MEM_OK          = 0, /* *bytes holds the answer */
    LM_MEM_UNAVAILABLE = 1, /* this platform has no probe; *bytes is untouched */
    LM_MEM_FAILED      = 2  /* a probe exists but failed or returned unparseable data;
                               *bytes is untouched */
} LmMemStatus;

LmMemStatus lm_mem_available(uint64_t *bytes);
LmMemStatus lm_mem_total(uint64_t *bytes);

/* Pure parser for /proc/meminfo-style text: finds a line "Field:   <number> kB" (Field
 * given without the colon, matched at the start of a line) and returns kB*1024 in
 * *bytes. LM_MEM_FAILED when the field is absent, has no number, has a unit other than
 * kB, or the product would overflow. Never guesses. */
LmMemStatus lm_meminfo_parse(const char *text, const char *field, uint64_t *bytes);

#endif /* LM_SYSMEM_H */
