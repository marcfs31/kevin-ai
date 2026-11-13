/* lm_sysmem.c - see lm_sysmem.h. Ported from kimi-k3-in-c src/cli/k3_sysmem.h
 * (Apache-2.0) via the oss_sysmem.c salvage; changes listed in the header. */
#include "io/lm_portable_io.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__APPLE__)
#include <mach/mach.h>
#include <sys/sysctl.h>
#endif

#include "io/lm_sysmem.h"

LmMemStatus lm_meminfo_parse(const char *text, const char *field, uint64_t *bytes)
{
    if (!text || !field || !*field) return LM_MEM_FAILED;
    const size_t fl = strlen(field);
    const char *p = text;
    while (*p) {
        if (strncmp(p, field, fl) == 0 && p[fl] == ':') {
            const char *q = p + fl + 1;
            while (*q == ' ' || *q == '\t') q++;
            if (*q < '0' || *q > '9') return LM_MEM_FAILED;
            uint64_t kb = 0;
            while (*q >= '0' && *q <= '9') {
                const unsigned d = (unsigned)(*q++ - '0');
                if (kb > (UINT64_MAX - d) / 10) return LM_MEM_FAILED;
                kb = kb * 10 + d;
            }
            while (*q == ' ' || *q == '\t') q++;
            if (strncmp(q, "kB", 2) != 0) return LM_MEM_FAILED;
            if (kb > UINT64_MAX / 1024) return LM_MEM_FAILED;
            *bytes = kb * 1024;
            return LM_MEM_OK;
        }
        const char *nl = strchr(p, '\n');
        if (!nl) break;
        p = nl + 1;
    }
    return LM_MEM_FAILED;
}

#if defined(__linux__)
static LmMemStatus meminfo_field(const char *field, uint64_t *bytes)
{
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f) return LM_MEM_FAILED;
    char buf[8192];
    const size_t n = fread(buf, 1, sizeof buf - 1, f);
    fclose(f);
    buf[n] = '\0';
    return lm_meminfo_parse(buf, field, bytes);
}
#endif

LmMemStatus lm_mem_available(uint64_t *bytes)
{
#if defined(__APPLE__)
    vm_statistics64_data_t vm;
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    vm_size_t page = 0;
    const mach_port_t host = mach_host_self();
    if (host_statistics64(host, HOST_VM_INFO64, (host_info64_t)&vm, &count) != KERN_SUCCESS)
        return LM_MEM_FAILED;
    if (host_page_size(host, &page) != KERN_SUCCESS || page == 0) return LM_MEM_FAILED;
    *bytes = ((uint64_t)vm.free_count + (uint64_t)vm.inactive_count +
              (uint64_t)vm.purgeable_count) * (uint64_t)page;
    return LM_MEM_OK;
#elif defined(__linux__)
    return meminfo_field("MemAvailable", bytes);
#else
    (void)bytes;
    return LM_MEM_UNAVAILABLE;
#endif
}

LmMemStatus lm_mem_total(uint64_t *bytes)
{
#if defined(__APPLE__)
    uint64_t b = 0;
    size_t len = sizeof b;
    if (sysctlbyname("hw.memsize", &b, &len, NULL, 0) != 0 || b == 0) return LM_MEM_FAILED;
    *bytes = b;
    return LM_MEM_OK;
#elif defined(__linux__)
    return meminfo_field("MemTotal", bytes);
#else
    (void)bytes;
    return LM_MEM_UNAVAILABLE;
#endif
}
