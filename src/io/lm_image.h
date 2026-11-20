/* lm_image.h - image files to RGB8 pixels. Original code.
 *
 * Binary PPM only (P6, maxval 255, no comments), the format of the test fixtures and of
 * tools/ref_vision_*.py's read_ppm. */
#ifndef LM_IMAGE_H
#define LM_IMAGE_H

#include <stddef.h>
#include <stdint.h>

/* Largest side accepted, so width * height * 3 cannot overflow and a corrupt header cannot
 * ask for gigabytes. */
#define LM_IMAGE_MAX_SIDE 16384

/* *rgb: malloc'd [h][w][3] row-major, freed by the caller. 0, or -1 with a message in err
 * (unreadable file, not P6, maxval other than 255, a side of 0 or above LM_IMAGE_MAX_SIDE,
 * truncated pixel data). */
int lm_ppm_read(const char *path, uint8_t **rgb, int *w, int *h, char *err, size_t errlen);

#endif /* LM_IMAGE_H */
