/* lm_nfc.c -- see lm_nfc.h. Original to this repository. */
#include "tokenizer/lm_nfc.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "tok_unicode_nfc.h"

/* A raw (invalid UTF-8) byte b is carried through the algorithm as RAW_BASE + b: above
 * every code point, combining class 0, never in a table, so it is a plain starter. */
#define RAW_BASE 0x110000u

typedef struct { uint32_t cp; unsigned char ccc; } Ch;

static int ccc_of(uint32_t cp)
{
    if (cp < 0x300) return 0; /* the lowest code point with a nonzero class is U+0300 */
    int lo = 0, hi = nfc_ccc_n - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        if (cp < nfc_ccc[mid][0]) hi = mid - 1;
        else if (cp > nfc_ccc[mid][1]) lo = mid + 1;
        else return (int)nfc_ccc[mid][2];
    }
    return 0;
}

static int dec_find(uint32_t cp)
{
    int lo = 0, hi = nfc_dec_n - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        if (cp < nfc_dec[mid][0]) hi = mid - 1;
        else if (cp > nfc_dec[mid][0]) lo = mid + 1;
        else return mid;
    }
    return -1;
}

/* Primary composite of (a, b), or 0. Hangul L+V and LV+T are algorithmic. */
static uint32_t compose_pair(uint32_t a, uint32_t b)
{
    if (a >= 0x1100 && a < 0x1100 + 19 && b >= 0x1161 && b < 0x1161 + 21)
        return 0xAC00 + ((a - 0x1100) * 21 + (b - 0x1161)) * 28;
    if (a >= 0xAC00 && a < 0xAC00 + 11172 && (a - 0xAC00) % 28 == 0 && b > 0x11A7 && b < 0x11A7 + 28)
        return a + (b - 0x11A7);
    int lo = 0, hi = nfc_comp_n - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        uint32_t fa = nfc_comp[mid][0], fb = nfc_comp[mid][1];
        if (a < fa || (a == fa && b < fb)) hi = mid - 1;
        else if (a > fa || b > fb) lo = mid + 1;
        else return nfc_comp[mid][2];
    }
    return 0;
}

static int utf8_put(unsigned char *o, uint32_t cp)
{
    if (cp >= RAW_BASE) { o[0] = (unsigned char)(cp - RAW_BASE); return 1; }
    if (cp < 0x80) { o[0] = (unsigned char)cp; return 1; }
    if (cp < 0x800) { o[0] = (unsigned char)(0xC0 | (cp >> 6)); o[1] = (unsigned char)(0x80 | (cp & 0x3F)); return 2; }
    if (cp < 0x10000) {
        o[0] = (unsigned char)(0xE0 | (cp >> 12)); o[1] = (unsigned char)(0x80 | ((cp >> 6) & 0x3F));
        o[2] = (unsigned char)(0x80 | (cp & 0x3F)); return 3;
    }
    o[0] = (unsigned char)(0xF0 | (cp >> 18)); o[1] = (unsigned char)(0x80 | ((cp >> 12) & 0x3F));
    o[2] = (unsigned char)(0x80 | ((cp >> 6) & 0x3F)); o[3] = (unsigned char)(0x80 | (cp & 0x3F));
    return 4;
}

/* Strict UTF-8 decode of one sequence at s[i]; returns its length, or 0 if invalid
 * (overlong, surrogate, above U+10FFFF, truncated). */
static int utf8_get(const unsigned char *s, size_t n, size_t i, uint32_t *cp)
{
    unsigned char c = s[i];
    if (c < 0x80) { *cp = c; return 1; }
    int len = c >= 0xC2 && c <= 0xDF ? 2 : c >= 0xE0 && c <= 0xEF ? 3 : c >= 0xF0 && c <= 0xF4 ? 4 : 0;
    if (!len || i + (size_t)len > n) return 0;
    uint32_t v = c & (0x3Fu >> (len - 1));
    for (int k = 1; k < len; k++) {
        if ((s[i + (size_t)k] & 0xC0) != 0x80) return 0;
        v = (v << 6) | (s[i + (size_t)k] & 0x3F);
    }
    if ((len == 3 && v < 0x800) || (len == 4 && (v < 0x10000 || v > 0x10FFFF)) || (v >= 0xD800 && v < 0xE000)) return 0;
    *cp = v;
    return len;
}

char *lm_nfc(const char *str, size_t n, size_t *out_n)
{
    const unsigned char *s = (const unsigned char *)str;
    size_t i;
    for (i = 0; i < n && s[i] < 0x80; i++) {}
    if (i == n) { /* ASCII is already NFC */
        char *o = (char *)malloc(n + 1);
        if (!o) return NULL;
        memcpy(o, str, n);
        o[n] = 0;
        if (out_n) *out_n = n;
        return o;
    }

    /* Decompose. A code point decomposes to at most 18 (U+FDFA), Hangul to 3. */
    Ch *a = (Ch *)malloc((n * 6 + 3) * sizeof *a);
    if (!a) return NULL;
    size_t na = 0;
    for (i = 0; i < n;) {
        uint32_t cp;
        int k = utf8_get(s, n, i, &cp);
        if (!k) { a[na].cp = RAW_BASE + s[i]; a[na].ccc = 0; na++; i++; continue; }
        i += (size_t)k;
        if (cp >= 0xAC00 && cp < 0xAC00 + 11172) {
            uint32_t sidx = cp - 0xAC00;
            a[na].cp = 0x1100 + sidx / 588; a[na++].ccc = 0;
            a[na].cp = 0x1161 + (sidx % 588) / 28; a[na++].ccc = 0;
            if (sidx % 28) { a[na].cp = 0x11A7 + sidx % 28; a[na++].ccc = 0; }
            continue;
        }
        int d = cp < 0xC0 ? -1 : dec_find(cp);
        if (d < 0) { a[na].cp = cp; a[na].ccc = (unsigned char)ccc_of(cp); na++; continue; }
        for (uint32_t j = 0; j < nfc_dec[d][2]; j++) {
            uint32_t c = nfc_pool[nfc_dec[d][1] + j];
            a[na].cp = c; a[na].ccc = (unsigned char)ccc_of(c); na++;
        }
    }

    /* Canonical ordering: stable insertion sort inside each run of nonzero classes. */
    for (i = 1; i < na; i++) {
        if (!a[i].ccc) continue;
        Ch x = a[i];
        size_t j = i;
        while (j > 0 && a[j - 1].ccc > x.ccc) { a[j] = a[j - 1]; j--; }
        a[j] = x;
    }

    /* Canonical composition, in place. */
    size_t no = 0;
    long starter = -1;
    for (i = 0; i < na; i++) {
        Ch ch = a[i];
        if (starter >= 0) {
            int adjacent = (long)no - 1 == starter;
            int last_cc = adjacent ? 0 : a[no - 1].ccc;
            if (adjacent || last_cc < ch.ccc) {
                uint32_t c = compose_pair(a[starter].cp, ch.cp);
                if (c) { a[starter].cp = c; continue; }
            }
        }
        if (!ch.ccc) starter = (long)no;
        a[no++] = ch;
    }

    char *o = (char *)malloc(no * 4 + 1);
    if (!o) { free(a); return NULL; }
    size_t w = 0;
    for (i = 0; i < no; i++) w += (size_t)utf8_put((unsigned char *)o + w, a[i].cp);
    o[w] = 0;
    free(a);
    if (out_n) *out_n = w;
    return o;
}
