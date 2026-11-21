/* lm_image.c - see lm_image.h. Original code; PNG/JPEG via stb_image. */
#include "io/lm_portable_io.h"

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "io/lm_image.h"
#include "io/lm_io.h"

#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#define STBI_NO_STDIO
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

/* ---- PPM (fixtures) ------------------------------------------------------------- */

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
    return n ? 0 : -1;
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

/* ---- PNG / JPEG via stb --------------------------------------------------------- */

static int is_png(const unsigned char *p, size_t n)
{
    static const unsigned char sig[8] = { 137, 80, 78, 71, 13, 10, 26, 10 };
    return n >= 8 && !memcmp(p, sig, 8);
}

static int is_jpeg(const unsigned char *p, size_t n)
{
    return n >= 3 && p[0] == 0xff && p[1] == 0xd8 && p[2] == 0xff;
}

int lm_image_decode(const void *data, size_t n, uint8_t **rgb, int *w, int *h,
                    char *err, size_t errlen)
{
    *rgb = NULL;
    if (!data || n == 0) return lm_fail(err, errlen, "empty image data");
    const unsigned char *p = (const unsigned char *)data;
    if (!is_png(p, n) && !is_jpeg(p, n))
        return lm_fail(err, errlen, "only PNG and JPEG images are supported");
    int sw = 0, sh = 0, comp = 0;
    if (!stbi_info_from_memory(p, (int)n, &sw, &sh, &comp))
        return lm_fail(err, errlen, "cannot read image header: %s", stbi_failure_reason());
    if (sw < 1 || sh < 1 || sw > LM_IMAGE_MAX_DECODE || sh > LM_IMAGE_MAX_DECODE)
        return lm_fail(err, errlen, "decoded image %dx%d exceeds %d on a side", sw, sh, LM_IMAGE_MAX_DECODE);
    int dw = 0, dh = 0;
    unsigned char *px = stbi_load_from_memory(p, (int)n, &dw, &dh, &comp, 3);
    if (!px) return lm_fail(err, errlen, "cannot decode image: %s", stbi_failure_reason());
    if (dw != sw || dh != sh || dw < 1 || dh < 1 || dw > LM_IMAGE_MAX_DECODE || dh > LM_IMAGE_MAX_DECODE) {
        stbi_image_free(px);
        return lm_fail(err, errlen, "decoded image size is invalid");
    }
    const size_t bytes = (size_t)dw * (size_t)dh * 3;
    uint8_t *out = (uint8_t *)malloc(bytes);
    if (!out) { stbi_image_free(px); return lm_fail(err, errlen, "out of memory decoding image"); }
    memcpy(out, px, bytes);
    stbi_image_free(px);
    *rgb = out;
    *w = dw;
    *h = dh;
    return 0;
}

int lm_image_decode_file(const char *path, uint8_t **rgb, int *w, int *h,
                         char *err, size_t errlen)
{
    *rgb = NULL;
    FILE *f = fopen(path, "rb");
    if (!f) return lm_fail(err, errlen, "cannot open %s", path);
    if (fseek(f, 0, SEEK_END)) { fclose(f); return lm_fail(err, errlen, "cannot seek %s", path); }
    const long sz = ftell(f);
    if (sz < 0 || fseek(f, 0, SEEK_SET)) { fclose(f); return lm_fail(err, errlen, "cannot seek %s", path); }
    /* A side of 4096 RGB is ~48 MiB; refuse absurd files early. */
    if (sz > 64L * 1024 * 1024) { fclose(f); return lm_fail(err, errlen, "%s: file too large", path); }
    unsigned char *buf = (unsigned char *)malloc((size_t)sz ? (size_t)sz : 1);
    if (!buf) { fclose(f); return lm_fail(err, errlen, "out of memory reading %s", path); }
    const size_t got = fread(buf, 1, (size_t)sz, f);
    fclose(f);
    if (got != (size_t)sz) { free(buf); return lm_fail(err, errlen, "%s: truncated read", path); }
    const int rc = lm_image_decode(buf, got, rgb, w, h, err, errlen);
    free(buf);
    return rc;
}

/* ---- data URL + base64 ---------------------------------------------------------- */

static int b64_val(int c)
{
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return c - 'a' + 26;
    if (c >= '0' && c <= '9') return c - '0' + 52;
    if (c == '+') return 62;
    if (c == '/') return 63;
    return -1;
}

static int b64_decode(const char *s, unsigned char **out, size_t *nout, char *err, size_t errlen)
{
    size_t n = strlen(s), pad = 0, i = 0, o = 0;
    while (n && (s[n - 1] == '\n' || s[n - 1] == '\r' || s[n - 1] == ' ')) n--;
    if (n >= 1 && s[n - 1] == '=') { pad++; if (n >= 2 && s[n - 2] == '=') pad++; }
    if (n % 4) return lm_fail(err, errlen, "invalid base64 length");
    const size_t out_n = (n / 4) * 3 - pad;
    unsigned char *buf = (unsigned char *)malloc(out_n ? out_n : 1);
    if (!buf) return lm_fail(err, errlen, "out of memory decoding base64");
    while (i < n) {
        int v[4];
        for (int k = 0; k < 4; k++) {
            const char c = s[i++];
            if (c == '=') v[k] = 0;
            else {
                v[k] = b64_val((unsigned char)c);
                if (v[k] < 0) { free(buf); return lm_fail(err, errlen, "invalid base64 character"); }
            }
        }
        if (o < out_n) buf[o++] = (unsigned char)((v[0] << 2) | (v[1] >> 4));
        if (o < out_n) buf[o++] = (unsigned char)(((v[1] & 15) << 4) | (v[2] >> 2));
        if (o < out_n) buf[o++] = (unsigned char)(((v[2] & 3) << 6) | v[3]);
    }
    *out = buf;
    *nout = out_n;
    return 0;
}

int lm_image_decode_data_url(const char *url, uint8_t **rgb, int *w, int *h,
                             char *err, size_t errlen)
{
    *rgb = NULL;
    if (!url) return lm_fail(err, errlen, "missing data URL");
    static const char png[] = "data:image/png;base64,";
    static const char jpeg[] = "data:image/jpeg;base64,";
    static const char jpg[] = "data:image/jpg;base64,";
    const char *payload = NULL;
    if (!strncmp(url, png, sizeof png - 1)) payload = url + sizeof png - 1;
    else if (!strncmp(url, jpeg, sizeof jpeg - 1)) payload = url + sizeof jpeg - 1;
    else if (!strncmp(url, jpg, sizeof jpg - 1)) payload = url + sizeof jpg - 1;
    else return lm_fail(err, errlen, "only data:image/png|jpeg|jpg;base64,... URLs are supported");
    unsigned char *raw = NULL;
    size_t nraw = 0;
    if (b64_decode(payload, &raw, &nraw, err, errlen)) return -1;
    const int rc = lm_image_decode(raw, nraw, rgb, w, h, err, errlen);
    free(raw);
    return rc;
}

/* ---- bicubic resize (Keys a = -0.5) ---------------------------------------------- */

static float keys_weight(float x, float a)
{
    x = fabsf(x);
    if (x < 1.f) return ((a + 2.f) * x - (a + 3.f)) * x * x + 1.f;
    if (x < 2.f) return ((a * x - 5.f * a) * x + 8.f * a) * x - 4.f * a;
    return 0.f;
}

static int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

static uint8_t sat_u8(float v)
{
    if (v <= 0.f) return 0;
    if (v >= 255.f) return 255;
    return (uint8_t)(v + 0.5f);
}

int lm_image_resize(const uint8_t *src, int sw, int sh, int ow, int oh, uint8_t **out,
                    char *err, size_t errlen)
{
    *out = NULL;
    if (!src || sw < 1 || sh < 1 || ow < 1 || oh < 1)
        return lm_fail(err, errlen, "invalid resize size");
    uint8_t *dst = (uint8_t *)malloc((size_t)ow * (size_t)oh * 3);
    if (!dst) return lm_fail(err, errlen, "out of memory resizing image");
    if (ow == sw && oh == sh) {
        memcpy(dst, src, (size_t)ow * (size_t)oh * 3);
        *out = dst;
        return 0;
    }
    /* Separable: horizontal into a temp of (ow x sh), then vertical into (ow x oh). */
    float *tmp = (float *)malloc((size_t)ow * (size_t)sh * 3 * sizeof(float));
    if (!tmp) { free(dst); return lm_fail(err, errlen, "out of memory resizing image"); }
    const float a = -0.5f;
    const float scale_x = (float)sw / (float)ow;
    const float scale_y = (float)sh / (float)oh;
    const float support_x = scale_x > 1.f ? 2.f * scale_x : 2.f;
    const float support_y = scale_y > 1.f ? 2.f * scale_y : 2.f;

    for (int y = 0; y < sh; y++) {
        for (int x = 0; x < ow; x++) {
            const float center = ((float)x + 0.5f) * scale_x - 0.5f;
            float acc[3] = { 0, 0, 0 }, wsum = 0.f;
            const int i0 = (int)floorf(center - support_x);
            const int i1 = (int)ceilf(center + support_x);
            for (int i = i0; i <= i1; i++) {
                const int sx = clampi(i, 0, sw - 1);
                /* Shrink: widen the Keys support by scale and divide the weight (PIL). */
                float wx = scale_x > 1.f
                    ? keys_weight((center - (float)i) / scale_x, a) / scale_x
                    : keys_weight(center - (float)i, a);
                if (wx == 0.f) continue;
                const uint8_t *p = src + ((size_t)y * (size_t)sw + (size_t)sx) * 3;
                acc[0] += wx * p[0]; acc[1] += wx * p[1]; acc[2] += wx * p[2];
                wsum += wx;
            }
            float *q = tmp + ((size_t)y * (size_t)ow + (size_t)x) * 3;
            if (wsum > 0.f) { q[0] = acc[0] / wsum; q[1] = acc[1] / wsum; q[2] = acc[2] / wsum; }
            else { q[0] = q[1] = q[2] = 0.f; }
        }
    }

    for (int y = 0; y < oh; y++) {
        const float center = ((float)y + 0.5f) * scale_y - 0.5f;
        const int j0 = (int)floorf(center - support_y);
        const int j1 = (int)ceilf(center + support_y);
        for (int x = 0; x < ow; x++) {
            float acc[3] = { 0, 0, 0 }, wsum = 0.f;
            for (int j = j0; j <= j1; j++) {
                const int sy = clampi(j, 0, sh - 1);
                float wy;
                if (scale_y > 1.f) wy = keys_weight((center - (float)j) / scale_y, a) / scale_y;
                else wy = keys_weight(center - (float)j, a);
                if (wy == 0.f) continue;
                const float *p = tmp + ((size_t)sy * (size_t)ow + (size_t)x) * 3;
                acc[0] += wy * p[0]; acc[1] += wy * p[1]; acc[2] += wy * p[2];
                wsum += wy;
            }
            uint8_t *q = dst + ((size_t)y * (size_t)ow + (size_t)x) * 3;
            if (wsum > 0.f) {
                q[0] = sat_u8(acc[0] / wsum);
                q[1] = sat_u8(acc[1] / wsum);
                q[2] = sat_u8(acc[2] / wsum);
            } else {
                q[0] = q[1] = q[2] = 0;
            }
        }
    }
    free(tmp);
    *out = dst;
    return 0;
}

/* ---- Qwen smart_resize / Gemma aspect size -------------------------------------- */

void lm_image_smart_resize(int h, int w, int factor, int min_pixels, int max_pixels,
                           int *oh, int *ow)
{
    if (factor < 1) factor = 1;
    if (h < 1) h = 1;
    if (w < 1) w = 1;
    int h_bar = (int)llround((double)h / (double)factor) * factor;
    int w_bar = (int)llround((double)w / (double)factor) * factor;
    if ((long long)h_bar * w_bar > max_pixels) {
        const double beta = sqrt(((double)h * (double)w) / (double)max_pixels);
        h_bar = (int)(floor(h / beta / factor) * factor);
        w_bar = (int)(floor(w / beta / factor) * factor);
        if (h_bar < factor) h_bar = factor;
        if (w_bar < factor) w_bar = factor;
    } else if ((long long)h_bar * w_bar < min_pixels) {
        const double beta = sqrt((double)min_pixels / ((double)h * (double)w));
        h_bar = (int)(ceil(h * beta / factor) * factor);
        w_bar = (int)(ceil(w * beta / factor) * factor);
    }
    *oh = h_bar;
    *ow = w_bar;
}

int lm_image_gemma_size(int h, int w, int unit, int max_patches, int *oh, int *ow,
                        char *err, size_t errlen)
{
    if (h < 1 || w < 1 || unit < 1 || max_patches < 1)
        return lm_fail(err, errlen, "invalid gemma resize arguments");
    const long long total_px = (long long)h * w;
    const long long target_px = (long long)max_patches * 256; /* patch 16^2, as in the reference */
    const double factor = sqrt((double)target_px / (double)total_px);
    int th = (int)(floor(factor * h / (double)unit) * unit);
    int tw = (int)(floor(factor * w / (double)unit) * unit);
    if (th == 0 && tw == 0) return lm_fail(err, errlen, "image too small to resize");
    const int max_side = (max_patches / 9) * unit; /* pool 3 -> /9 soft tokens */
    if (th == 0) {
        th = unit;
        tw = (int)floor((double)w / (double)h) * unit;
        if (tw > max_side) tw = max_side;
        if (tw < unit) tw = unit;
    } else if (tw == 0) {
        tw = unit;
        th = (int)floor((double)h / (double)w) * unit;
        if (th > max_side) th = max_side;
        if (th < unit) th = unit;
    }
    if ((long long)th * tw > target_px) return lm_fail(err, errlen, "resized image exceeds max_patches");
    *oh = th;
    *ow = tw;
    return 0;
}
