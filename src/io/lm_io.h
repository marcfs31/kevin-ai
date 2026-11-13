/* lm_io.h - error reporting and positional reads shared by the safetensors reader and
 * writer. Original code; the full-read loop follows kimi-k3-in-c src/io/k3_st.c
 * k3_st_read (Apache-2.0), except that every failure is an error return with a message,
 * never a short count the caller has to remember to compare. */
#ifndef LM_IO_H
#define LM_IO_H

#include <stddef.h>
#include <stdint.h>

/* snprintf into err when err is non-NULL; always returns -1 so callers can write
 * `return lm_fail(...)`. */
int lm_fail(char *err, size_t errlen, const char *fmt, ...)
#if defined(__GNUC__)
    __attribute__((format(printf, 3, 4)))
#endif
    ;

/* Read exactly n bytes at offset off into buf. Loops over short reads and EINTR, chunked
 * to LM_PREAD_MAX per syscall. Returns 0, or -1 with a message in err: a read error, or
 * EOF before n bytes (a short read is an error, never a partial success). */
int lm_pread_full(int fd, void *buf, size_t n, int64_t off, char *err, size_t errlen);

#endif /* LM_IO_H */
