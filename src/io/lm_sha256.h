/* lm_sha256.h - SHA-256 (FIPS 180-4), streaming. Original code; used by lm-quant for the
 * "source" metadata hash so no third-party dependency is needed. */
#ifndef LM_SHA256_H
#define LM_SHA256_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    uint32_t h[8];
    uint64_t nbytes;
    unsigned char buf[64];
    size_t   fill;
} LmSha256;

void lm_sha256_init(LmSha256 *c);
void lm_sha256_update(LmSha256 *c, const void *data, size_t n);
void lm_sha256_final(LmSha256 *c, unsigned char out[32]);

/* 64 lowercase hex characters plus NUL. */
void lm_sha256_hex(const unsigned char digest[32], char out[65]);

/* Hash a whole file in 1 MiB reads. Returns 0, or -1 with a message in err. */
int  lm_sha256_file(const char *path, char hex[65], char *err, size_t errlen);

#endif /* LM_SHA256_H */
