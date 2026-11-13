/* lm_portable_io.h - feature macros and one constant shared by the I/O sources.
 *
 * Ported from kimi-k3-in-c src/io/k3_portable_io.h (Apache-2.0) via an earlier unreviewed
 * salvage (oss_portable_io.h). Changes against the original: renamed K3_ to LM_; the
 * Windows branch, the O_DIRECT / F_NOCACHE shims and posix_fadvise are dropped (this
 * engine maps or preads files and never bypasses the page cache); _GNU_SOURCE,
 * _FILE_OFFSET_BITS and _DARWIN_C_SOURCE are set here.
 *
 *   LM_PREAD_MAX  Darwin's pread() returns EINVAL for one request of 2^31 bytes or more;
 *                 every read loop chunks against this cap. Harmless on Linux.
 *
 * Include this header BEFORE any libc header. */
#ifndef LM_PORTABLE_IO_H
#define LM_PORTABLE_IO_H

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#ifndef _FILE_OFFSET_BITS
#define _FILE_OFFSET_BITS 64
#endif
#if defined(__APPLE__) && !defined(_DARWIN_C_SOURCE)
#define _DARWIN_C_SOURCE
#endif

#include <fcntl.h>
#include <stdint.h>

#define LM_PREAD_MAX ((int64_t)1 << 30)   /* 1 GiB per syscall */

#endif /* LM_PORTABLE_IO_H */
