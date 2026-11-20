/* lm_image.c - see lm_image.h. Original code. */
#include "io/lm_portable_io.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>

#include "io/lm_image.h"
#include "io/lm_io.h"

/* One whitespace-delimited header field as a non-negative integer, or a magic word. */
static int field(FILE *f, char *buf, size_t cap)
{
    int c;
    do { c = fgetc(f); } while (c != EOF && isspace(c));
    size_t n = 0;
    while (c != EOF && !isspace(c)) {
        if (n + 1 >= cap) return -1;
        buf[n++] = (char)c;
        c = fgetc(f);
    }
    buf[n] = '\0';
    return n ? 0 : -1;       /* the single whitespace after maxval is consumed here */
}

static int number(const char *s, long *out)
{
    char *end;
    const long v = strtol(s, &end, 10);
    if (end == s || *end || v < 0) return -1;
    *out = v;
    return 0;
}

int lm_ppm_read(const char *path, uint8_t **rgb, int *w, int *h, char *err, size_t errlen)
{
    *rgb = NULL;
    FILE *f = fopen(path, "rb");
    if (!f) return lm_fail(err, errlen, "cannot open %s", path);
    char a[16], b[16], c[16], d[16];
    long W, H, M;
    if (field(f, a, sizeof a) || field(f, b, sizeof b) || field(f, c, sizeof c) || field(f, d, sizeof d) ||
        a[0] != 'P' || a[1] != '6' || a[2] || number(b, &W) || number(c, &H) || number(d, &M)) {
        fclose(f);
        return lm_fail(err, errlen, "%s: only binary P6 PPM (no comments) is supported", path);
    }
    if (M != 255) { fclose(f); return lm_fail(err, errlen, "%s: maxval %ld, only 255 is supported", path, M); }
    if (W < 1 || H < 1 || W > LM_IMAGE_MAX_SIDE || H > LM_IMAGE_MAX_SIDE) {
        fclose(f);
        return lm_fail(err, errlen, "%s: size %ldx%ld outside 1..%d", path, W, H, LM_IMAGE_MAX_SIDE);
    }
    const size_t n = (size_t)W * (size_t)H * 3;
    uint8_t *px = (uint8_t *)malloc(n);
    if (!px) { fclose(f); return lm_fail(err, errlen, "out of memory reading %s", path); }
    const size_t got = fread(px, 1, n, f);
    fclose(f);
    if (got != n) { free(px); return lm_fail(err, errlen, "%s: truncated (%zu of %zu pixel bytes)", path, got, n); }
    *rgb = px;
    *w = (int)W;
    *h = (int)H;
    return 0;
}
