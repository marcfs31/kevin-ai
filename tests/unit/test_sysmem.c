/* test_sysmem.c - the available-memory probe: the pure /proc/meminfo parser on every
 * platform, and the live probe on the platform running the test. */
#include "lm_test.h"

#include "io/lm_sysmem.h"

int main(void)
{
    uint64_t v = 12345;
    const char *good =
        "MemTotal:        8123456 kB\n"
        "MemFree:          100000 kB\n"
        "MemAvailable:    4000000 kB\n"
        "Buffers:               0 kB\n";
    CHECK(lm_meminfo_parse(good, "MemAvailable", &v) == LM_MEM_OK && v == 4000000ull * 1024);
    CHECK(lm_meminfo_parse(good, "MemTotal", &v) == LM_MEM_OK && v == 8123456ull * 1024);
    CHECK(lm_meminfo_parse(good, "Buffers", &v) == LM_MEM_OK && v == 0);    /* zero is an answer, not "unknown" */

    /* each failure leaves *bytes untouched and says FAILED: never a guessed number */
    v = 777;
    CHECK(lm_meminfo_parse(good, "MemShared", &v) == LM_MEM_FAILED && v == 777);               /* absent */
    CHECK(lm_meminfo_parse("MemAvailable: 12 MB\n", "MemAvailable", &v) == LM_MEM_FAILED && v == 777);  /* unit */
    CHECK(lm_meminfo_parse("MemAvailable: lots kB\n", "MemAvailable", &v) == LM_MEM_FAILED && v == 777); /* no number */
    CHECK(lm_meminfo_parse("MemAvailable: 12\n", "MemAvailable", &v) == LM_MEM_FAILED && v == 777);      /* no unit */
    CHECK(lm_meminfo_parse("MemAvailable: 99999999999999999999 kB\n", "MemAvailable", &v) == LM_MEM_FAILED && v == 777);
    CHECK(lm_meminfo_parse("MemAvailable: 18014398509481984 kB\n", "MemAvailable", &v) == LM_MEM_FAILED && v == 777); /* *1024 overflows */
    CHECK(lm_meminfo_parse("", "MemAvailable", &v) == LM_MEM_FAILED && v == 777);
    CHECK(lm_meminfo_parse(NULL, "MemAvailable", &v) == LM_MEM_FAILED);
    CHECK(lm_meminfo_parse(good, "", &v) == LM_MEM_FAILED);
    /* a field name is matched whole and at line start: "MemAvailable" must not match a longer name */
    CHECK(lm_meminfo_parse("XMemAvailable: 5 kB\n", "MemAvailable", &v) == LM_MEM_FAILED && v == 777);
    CHECK(lm_meminfo_parse("MemAvailableX: 5 kB\n", "MemAvailable", &v) == LM_MEM_FAILED && v == 777);

    /* the live probe: OK with a plausible value, or an explicit non-OK, never a made-up number */
    uint64_t avail = 0, total = 0;
    const LmMemStatus sa = lm_mem_available(&avail), st = lm_mem_total(&total);
#if defined(__linux__) || defined(__APPLE__)
    CHECK(sa == LM_MEM_OK && st == LM_MEM_OK);
    CHECK(total >= ((uint64_t)256 << 20));
    CHECK(avail > 0 && avail <= total);
#else
    CHECK(sa == LM_MEM_UNAVAILABLE && st == LM_MEM_UNAVAILABLE);
#endif
    if (sa == LM_MEM_OK) printf("  probe: %llu MB available of %llu MB\n",
                                (unsigned long long)(avail >> 20), (unsigned long long)(total >> 20));
    return lt_finish("test_sysmem");
}
