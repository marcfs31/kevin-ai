/* lm_test.h - tiny shared helpers for tests/unit/test_*.c: checks, a mkdtemp scratch
 * directory with cleanup, and a hand-rolled safetensors file writer that is deliberately
 * independent of src/io/lm_stw.c so the reader is not tested against its own writer.
 * Not a test itself (the Makefile only builds tests/unit/test_*.c). */
#ifndef LM_TEST_H
#define LM_TEST_H

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#if defined(__APPLE__) && !defined(_DARWIN_C_SOURCE)
#define _DARWIN_C_SOURCE
#endif
#include <ftw.h>
#include <math.h>
#include <stdarg.h>
#include <sys/stat.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define LT_UNUSED __attribute__((unused))
static int lt_fails = 0, lt_checks = 0;

#define CHECK(cond) do { lt_checks++; if (!(cond)) { lt_fails++; \
    fprintf(stderr, "  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); } } while (0)

#define CHECK_MSG(cond, ...) do { lt_checks++; if (!(cond)) { lt_fails++; \
    fprintf(stderr, "  FAIL %s:%d: %s: ", __FILE__, __LINE__, #cond); \
    fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); } } while (0)

#define CHECK_NEAR(a, b, tol) do { const double a_ = (double)(a), b_ = (double)(b); lt_checks++; \
    if (!(fabs(a_ - b_) <= (tol))) { lt_fails++; \
    fprintf(stderr, "  FAIL %s:%d: |%s - %s| = |%.9g - %.9g| > %g\n", __FILE__, __LINE__, \
            #a, #b, a_, b_, (double)(tol)); } } while (0)

/* An error message must contain `needle`: proves the right check fired, not just any. */
#define CHECK_ERR(rc, err, needle) do { lt_checks++; \
    if (!((rc) != 0 && strstr((err), (needle)))) { lt_fails++; \
    fprintf(stderr, "  FAIL %s:%d: expected failure containing \"%s\", rc=%d err=\"%s\"\n", \
            __FILE__, __LINE__, (needle), (int)(rc), (err)); } } while (0)

LT_UNUSED static int lt_finish(const char *name)
{
    if (lt_fails) { fprintf(stderr, "%s: %d of %d checks FAILED\n", name, lt_fails, lt_checks); return 1; }
    printf("%s: %d checks passed\n", name, lt_checks);
    return 0;
}

/* ---- scratch directory ---- */

LT_UNUSED static int lt_rm_cb(const char *p, const struct stat *sb, int flag, struct FTW *ftw)
{
    (void)sb; (void)flag; (void)ftw;
    return remove(p);
}

LT_UNUSED static void lt_rmtree(const char *dir) { nftw(dir, lt_rm_cb, 16, FTW_DEPTH | FTW_PHYS); }

/* dir must hold at least 512 bytes. Aborts the test on failure. */
LT_UNUSED static void lt_mkdtemp(char *dir)
{
    const char *base = getenv("TMPDIR");
    if (!base || !*base) base = "/tmp";
    snprintf(dir, 512, "%s/lm_test_XXXXXX", base);
    if (!mkdtemp(dir)) { perror("mkdtemp"); exit(1); }
}

LT_UNUSED static void lt_path(char *out, const char *dir, const char *leaf)
{
    snprintf(out, 1024, "%s/%s", dir, leaf);
}

/* ---- raw file helpers ---- */

LT_UNUSED static void lt_write_file(const char *path, const void *data, size_t n)
{
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); exit(1); }
    if (n && fwrite(data, 1, n, f) != n) { perror("fwrite"); exit(1); }
    fclose(f);
}

LT_UNUSED static void lt_write_text(const char *path, const char *text) { lt_write_file(path, text, strlen(text)); }

LT_UNUSED static void lt_le64(unsigned char out[8], uint64_t v)
{
    for (int i = 0; i < 8; i++) { out[i] = (unsigned char)(v & 0xff); v >>= 8; }
}

/* A safetensors file from a literal header string (written as is, no padding) and data. */
LT_UNUSED static void lt_write_st_raw(const char *path, const char *header, const void *data, size_t ndata)
{
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); exit(1); }
    unsigned char len[8];
    lt_le64(len, strlen(header));
    fwrite(len, 1, 8, f);
    fwrite(header, 1, strlen(header), f);
    if (ndata) fwrite(data, 1, ndata, f);
    fclose(f);
}

/* Same, with the header padded with spaces to a multiple of 8 so tensor data is 8-aligned
 * (what real writers do, and what lm_st_data's alignment rule relies on). */
LT_UNUSED static void lt_write_st_raw8(const char *path, const char *header, const void *data, size_t ndata)
{
    const size_t n = strlen(header), pn = (n + 7) / 8 * 8;
    char *h = (char *)malloc(pn + 1);
    memcpy(h, header, n);
    memset(h + n, ' ', pn - n);
    h[pn] = '\0';
    lt_write_st_raw(path, h, data, ndata);
    free(h);
}

/* Same as lt_write_st_raw, but the length prefix is given explicitly (to lie about it). */
LT_UNUSED static void lt_write_st_len(const char *path, uint64_t hlen, const char *header, const void *data, size_t ndata)
{
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); exit(1); }
    unsigned char len[8];
    lt_le64(len, hlen);
    fwrite(len, 1, 8, f);
    fwrite(header, 1, strlen(header), f);
    if (ndata) fwrite(data, 1, ndata, f);
    fclose(f);
}

LT_UNUSED static uint16_t lt_bf16(float f)
{
    uint32_t u;
    memcpy(&u, &f, 4);
    u += 0x7fffu + ((u >> 16) & 1u);   /* round to nearest even; callers pass finite values */
    return (uint16_t)(u >> 16);
}

/* Deterministic pseudo-random floats in [-1, 1), independent of libc rand(). */
static uint64_t lt_rng_state = 88172645463325252ull;
LT_UNUSED static float lt_rand(void)
{
    lt_rng_state ^= lt_rng_state << 13;
    lt_rng_state ^= lt_rng_state >> 7;
    lt_rng_state ^= lt_rng_state << 17;
    return (float)((double)(lt_rng_state >> 11) / (double)(1ull << 53) * 2.0 - 1.0);
}

#endif /* LM_TEST_H */
