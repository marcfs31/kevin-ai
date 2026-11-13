/* lm_io.c - see lm_io.h. */
#include "io/lm_portable_io.h"

#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "io/lm_io.h"

int lm_fail(char *err, size_t errlen, const char *fmt, ...)
{
    if (err && errlen) {
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(err, errlen, fmt, ap);
        va_end(ap);
    }
    return -1;
}

int lm_pread_full(int fd, void *buf, size_t n, int64_t off, char *err, size_t errlen)
{
    if (off < 0) return lm_fail(err, errlen, "negative read offset %lld", (long long)off);
    size_t got = 0;
    while (got < n) {
        size_t req = n - got;
        if (req > (size_t)LM_PREAD_MAX) req = (size_t)LM_PREAD_MAX;
        const ssize_t r = pread(fd, (char *)buf + got, req, (off_t)(off + (int64_t)got));
        if (r < 0) {
            if (errno == EINTR) continue;
            return lm_fail(err, errlen, "read of %zu bytes at offset %lld failed after %zu: %s",
                           n, (long long)off, got, strerror(errno));
        }
        if (r == 0)
            return lm_fail(err, errlen, "short read: EOF after %zu of %zu bytes at offset %lld",
                           got, n, (long long)off);
        got += (size_t)r;
    }
    return 0;
}
