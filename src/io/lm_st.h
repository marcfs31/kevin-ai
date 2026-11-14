/* lm_st.h - safetensors reader: one file, or a sharded checkpoint described by
 * model.safetensors.index.json.
 *
 * Ported from kimi-k3-in-c src/io/k3_st.[ch] (Apache-2.0) via an earlier unreviewed
 * salvage (oss_st.[ch], reviewed and extended here). Kept from the original: the header
 * scanner (str_/i64_/skip_value), the saturating integer parse, the span sweep and the
 * FNV-1a open-addressed hash. Changes: dtypes BF16, F16, F32, I8, U8 (the K3 per-row int8
 * trunk dtype is dropped); sharded sets via the index file with a name -> shard
 * cross-check in both directions; read-only mmap as the default access path with a pread
 * fallback; every failure returns -1 with a message in err instead of writing to stderr;
 * zero-length tensors are exempt from the overlap check; the header is size-capped; text
 * after the header object, duplicate keys and missing keys are errors; __metadata__ is
 * parsed (string -> string) and exposed.
 *
 * FORMAT:  [8 bytes little-endian N] [N bytes of JSON header] [tensor data]
 *   Each header entry: {"dtype": ..., "shape": [...], "data_offsets": [start, end]}.
 *   data_offsets are relative to the END of the header: absolute offset = 8 + N + start.
 *   Every LmTensor handed out has an ABSOLUTE file offset.
 *
 * WHAT IS REFUSED AT OPEN (each with its own message, never a default):
 *   - a file shorter than 8 bytes, a zero header length, a header longer than the file
 *     (truncated) or longer than LM_ST_MAX_HEADER (oversized / corrupt)
 *   - a header that is not one JSON object, malformed entries, missing dtype / shape /
 *     data_offsets, duplicate names (within a file or across shards), rank above 4,
 *     negative dims, an unsupported dtype, a non-string __metadata__ value
 *   - data_offsets that are negative, inverted, or end past the end of the file (this is
 *     how a truncated data region shows up)
 *   - two tensors whose byte spans overlap
 *   - a byte length that is not numel * element size (shape/dtype/length mismatch)
 *   - an index that is malformed, names a shard file that does not exist or is not a bare
 *     file name, lists a tensor its shard lacks, or omits a tensor its shard holds
 * Gaps between tensors and trailing bytes after the last tensor are tolerated (they are
 * not a hazard to a reader; an overlap is).
 *
 * ACCESS: lm_st_data() hands out a pointer into a read-only MAP_PRIVATE mapping (the
 * default). The mapping is only as stable as the file: truncating a mapped file while it
 * is in use raises SIGBUS, which is the standard mmap contract. use_mmap = 0 opens the
 * pread path instead (lm_st_data returns NULL; use lm_st_read*). */
#ifndef LM_ST_H
#define LM_ST_H

#include <stddef.h>
#include <stdint.h>

#include "core/lm_bf16.h"

typedef enum { LM_DT_UNKNOWN = 0, LM_DT_BF16, LM_DT_F16, LM_DT_F32, LM_DT_I8, LM_DT_U8 } LmDtype;

#define LM_ST_MAX_HEADER ((int64_t)100 * 1024 * 1024)   /* the safetensors spec's limit */
#define LM_ST_MAX_INDEX  ((int64_t)256 * 1024 * 1024)

typedef struct {
    char     *name;
    LmDtype   dtype;
    int       ndim;           /* 0..4; 0 is a scalar */
    int       rank_overflow;  /* true rank > 4: shape holds the first 4 dims only */
    int64_t   numel;          /* product of ALL dims, saturating; set by the parser */
    int64_t   shape[4];
    int64_t   off;            /* ABSOLUTE byte offset within its file */
    int64_t   nbytes;         /* == numel * element size, verified at open */
    int       file;           /* index into LmSt.files */
} LmTensor;

typedef struct {
    char          *path;
    int            fd;
    int64_t        size;
    int64_t        data_base; /* 8 + header length */
    const unsigned char *map; /* whole-file read-only mapping, or NULL on the pread path */
    char         **meta_key;  /* __metadata__ entries */
    char         **meta_val;
    int            nmeta;
} LmStFile;

typedef struct {
    LmStFile  *files;
    int        nfiles;
    LmTensor  *t;             /* every tensor of every file, shard by shard */
    int        nt;
    int32_t   *bucket;        /* open-addressed hash index, -1 empty */
    int        nbucket;
} LmSt;

/* path: a .safetensors file; a model.safetensors.index.json; or a directory holding
 * model.safetensors.index.json (preferred) or model.safetensors. Returns 0, or -1 with a
 * message in err (the set is then closed and safe to pass to lm_st_close). A zero-filled
 * LmSt is also safe to close. */
int  lm_st_open(LmSt *s, const char *path, int use_mmap, char *err, size_t errlen);
void lm_st_close(LmSt *s);

/* O(1) lookup; NULL when absent. A missing weight must be treated as fatal by the caller. */
const LmTensor *lm_st_find(const LmSt *s, const char *name);

/* __metadata__ value for key from the first file that carries it, or NULL. */
const char *lm_st_meta(const LmSt *s, const char *key);

int64_t     lm_st_numel(const LmTensor *t);
/* Product of the dims, saturating at INT64_MAX; a zero dim gives 0. Negative dims give INT64_MAX. */
int64_t     lm_shape_numel(int ndim, const int64_t *shape);
int         lm_dtype_size(LmDtype d);        /* 0 for LM_DT_UNKNOWN */
const char *lm_dtype_name(LmDtype d);        /* "BF16", "F16", "F32", "I8", "U8" */

/* Pointer to the tensor's bytes inside the mapping, or NULL when the set was opened with
 * use_mmap = 0 or the offset is not aligned for the element type (the caller then falls
 * back to lm_st_read). */
const void *lm_st_data(const LmSt *s, const LmTensor *t);

/* Raw bytes exactly as stored. buf must hold t->nbytes. Returns 0 or -1 (message in err). */
int lm_st_read(const LmSt *s, const LmTensor *t, void *buf, char *err, size_t errlen);

/* nbytes at byte offset byte_off INSIDE the tensor (a row of the embedding table). Refuses
 * a range outside the tensor. */
int lm_st_read_slice(const LmSt *s, const LmTensor *t, int64_t byte_off, int64_t nbytes,
                     void *buf, char *err, size_t errlen);

/* count elements starting at element first, widened to float32 (F32 copy, BF16 shift,
 * F16 convert, I8 and U8 the numeric value). Reads in bounded chunks. Returns 0 or -1. */
int lm_st_read_f32_range(const LmSt *s, const LmTensor *t, int64_t first, int64_t count,
                         float *out, char *err, size_t errlen);

/* The whole tensor widened to float32; out must hold lm_st_numel(t) floats. */
int lm_st_read_f32(const LmSt *s, const LmTensor *t, float *out, char *err, size_t errlen);

#endif /* LM_ST_H */
