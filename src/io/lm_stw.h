/* lm_stw.h - streaming safetensors writer (used by bin/lm-quant). Original code.
 *
 * The header precedes the data and records every tensor's size, so the caller DECLARES
 * every tensor (name, dtype, shape) first, then streams the bytes tensor by tensor and
 * in chunks. Nothing but the header is held in memory.
 *
 * CANONICAL LAYOUT (the C and Python writers must agree byte for byte):
 *   - tensors are laid out in ascending byte order of their names (strcmp order), data
 *     packed back to back with NO padding between tensors
 *   - header JSON is compact (no spaces): {"__metadata__":{...},"<name>":{"dtype":"..",
 *     "shape":[..],"data_offsets":[s,e]},...} with metadata keys sorted, the metadata
 *     object first (omitted when empty), then tensors in the layout order; each tensor
 *     entry's keys in the order dtype, shape, data_offsets
 *   - the header is padded with spaces (0x20) to a multiple of 8 bytes, and the 8 byte
 *     little-endian length N counts the padding
 *   - names and metadata strings must be printable ASCII; '"' and '\' are escaped
 * The file is written to "<path>.tmp" and renamed over <path> by lm_stw_finish, so a
 * failed run never leaves a plausible-looking truncated file under the final name. */
#ifndef LM_STW_H
#define LM_STW_H

#include <stddef.h>
#include <stdint.h>

#include "io/lm_st.h"

typedef struct LmStWriter LmStWriter;

/* NULL with a message in err on failure. */
LmStWriter *lm_stw_create(const char *path, char *err, size_t errlen);

/* Phase 1: describe the file. Call before lm_stw_begin. Duplicate names are refused. */
int lm_stw_meta(LmStWriter *w, const char *key, const char *value, char *err, size_t errlen);
int lm_stw_declare(LmStWriter *w, const char *name, LmDtype dtype, int ndim, const int64_t *shape,
                   char *err, size_t errlen);

/* Phase 2: sort, write the header. */
int lm_stw_begin(LmStWriter *w, char *err, size_t errlen);

/* Phase 3: the data, in the layout order (ascending name). lm_stw_tensor_begin refuses any
 * name but the next one; lm_stw_write may be called repeatedly and refuses to exceed the
 * declared size; lm_stw_tensor_end refuses a short tensor. */
int lm_stw_tensor_begin(LmStWriter *w, const char *name, char *err, size_t errlen);
int lm_stw_write(LmStWriter *w, const void *buf, int64_t nbytes, char *err, size_t errlen);
int lm_stw_tensor_end(LmStWriter *w, char *err, size_t errlen);

/* Name of the next tensor to write in layout order, or NULL when all are done. */
const char *lm_stw_next_name(const LmStWriter *w);

/* Check every tensor was written, flush, fsync, rename into place and free. Returns 0 or
 * -1; on -1 the temporary file is removed and the writer freed. */
int lm_stw_finish(LmStWriter *w, char *err, size_t errlen);

/* Drop a partially written file and free the writer (safe on NULL). */
void lm_stw_abort(LmStWriter *w);

#endif /* LM_STW_H */
