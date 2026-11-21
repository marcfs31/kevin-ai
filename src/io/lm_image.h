/* lm_image.h - image files and buffers to RGB8 pixels, plus bicubic resize.
 * Original code; PNG/JPEG decode via vendored stb_image (public domain / MIT).
 *
 * Binary PPM (P6, maxval 255, no comments) remains for the vision fixtures.
 * PNG and JPEG (memory or file) are for lm-serve / lm-chat. Decoded sides above
 * LM_IMAGE_MAX_DECODE are refused before allocation. Resampling is a separable
 * bicubic (Keys a = -0.5), matching docs/PLAN.md "Resampling". */
#ifndef LM_IMAGE_H
#define LM_IMAGE_H

#include <stddef.h>
#include <stdint.h>

/* Largest side accepted for PPM fixtures and for decoded PNG/JPEG before resize. */
#define LM_IMAGE_MAX_SIDE 16384
#define LM_IMAGE_MAX_DECODE 4096

/* *rgb: malloc'd [h][w][3] row-major, freed by the caller. 0, or -1 with a message in err
 * (unreadable file, not P6, maxval other than 255, a side of 0 or above LM_IMAGE_MAX_SIDE,
 * truncated pixel data). */
int lm_ppm_read(const char *path, uint8_t **rgb, int *w, int *h, char *err, size_t errlen);

/* Decode PNG or JPEG bytes to RGB8. Other formats are refused. Sides must be in
 * 1..LM_IMAGE_MAX_DECODE. 0, or -1 with a message in err. */
int lm_image_decode(const void *data, size_t n, uint8_t **rgb, int *w, int *h,
                    char *err, size_t errlen);

/* Read a PNG or JPEG file and decode it. */
int lm_image_decode_file(const char *path, uint8_t **rgb, int *w, int *h,
                         char *err, size_t errlen);

/* OpenAI-style data URL: data:image/png;base64,... or data:image/jpeg;base64,...
 * (image/jpg accepted). Any other scheme or media type is refused; nothing is fetched.
 * 0, or -1 with a message in err. */
int lm_image_decode_data_url(const char *url, uint8_t **rgb, int *w, int *h,
                             char *err, size_t errlen);

/* Separable bicubic resize (Keys a = -0.5). Identity when src and dst sizes match
 * (bytes are copied). A constant image stays constant. oh/ow must be >= 1.
 * *out is malloc'd [oh][ow][3]. 0, or -1 (out of memory / bad size). */
int lm_image_resize(const uint8_t *src, int sw, int sh, int ow, int oh, uint8_t **out,
                    char *err, size_t errlen);

/* Qwen smart_resize: round each side to a multiple of factor (32); scale so area is in
 * [min_pixels, max_pixels]. See docs/PLAN.md and tools/ref_vision_qwen35.py. */
void lm_image_smart_resize(int h, int w, int factor, int min_pixels, int max_pixels,
                           int *oh, int *ow);

/* Gemma aspect-preserving size: floor each side to a multiple of unit (48);
 * at most max_patches patches (2520 for E4B). Returns 0, or -1 when the reference
 * would error (zero side / over budget after flooring). */
int lm_image_gemma_size(int h, int w, int unit, int max_patches, int *oh, int *ow,
                        char *err, size_t errlen);

#endif /* LM_IMAGE_H */
