/* lm_gguf.h - GGUF v3 reader: header, metadata, tensor table, read-only mmap. Original code;
 * the container format is the one llama.cpp documents (docs/gguf.md).
 *
 * FORMAT: "GGUF", u32 version (3), u64 tensor count, u64 metadata count; metadata key/value
 * pairs (string key, u32 type, value; arrays carry an element type and a count); tensor
 * infos (name, n_dims, dims innermost first, ggml type, offset into the data region); then
 * the data region at the next multiple of general.alignment (default 32). Little endian.
 *
 * WHAT IS REFUSED AT OPEN, each with its own message: a short file, bad magic, a version
 * other than 2 or 3, counts past the limits below, a string or array that runs past the end
 * of the file, nested arrays, an unknown value type, a zero-sized or non-power-of-two
 * general.alignment, more than 4 dims, a dim of 0, an element count that overflows, a
 * duplicate tensor name, and a tensor of a SUPPORTED type whose row length is not a whole
 * number of blocks or whose bytes run past the end of the file. A tensor of a type this
 * reader has no block size for opens (its nbytes is 0): the binder that wants it refuses it
 * by name.
 *
 * Nothing is copied: strings and arrays point into the mapping, which lives until
 * lm_gguf_close. A file truncated while mapped raises SIGBUS (the standard mmap contract). */
#ifndef LM_GGUF_H
#define LM_GGUF_H

#include <stddef.h>
#include <stdint.h>

/* ggml tensor types the engine reads (the numbers are ggml's own). */
enum {
    LM_GG_F32  = 0,
    LM_GG_F16  = 1,
    LM_GG_Q8_0 = 8,
    LM_GG_Q4_K = 12,
    LM_GG_Q5_K = 13,
    LM_GG_Q6_K = 14
};

/* Elements per block and bytes per block of a supported type; -1 for any other. */
int lm_gguf_type_block(int type, int *block_elems, size_t *block_bytes);
const char *lm_gguf_type_name(int type);   /* "Q4_K"; "type N" for an unknown number (static buffer) */

/* Metadata value types (GGUF spec numbers). */
enum {
    LM_GGUF_U8 = 0, LM_GGUF_I8, LM_GGUF_U16, LM_GGUF_I16, LM_GGUF_U32, LM_GGUF_I32, LM_GGUF_F32,
    LM_GGUF_BOOL, LM_GGUF_STR, LM_GGUF_ARR, LM_GGUF_U64, LM_GGUF_I64, LM_GGUF_F64
};

typedef struct {
    const char *s;      /* points into the mapping, NOT NUL-terminated */
    size_t      n;
} LmGgufStr;

typedef struct {
    LmGgufStr key;
    int       vtype;    /* LM_GGUF_*; LM_GGUF_ARR for an array */
    int       etype;    /* array element type (never LM_GGUF_ARR) */
    uint64_t  n;        /* array element count */
    union { uint64_t u; int64_t i; double f; LmGgufStr s; const unsigned char *arr; } v;
} LmGgufKv;

typedef struct {
    char     *name;
    int       type;      /* ggml type number */
    int       ndim;      /* 1..4 */
    int64_t   ne[4];     /* innermost (contiguous) dim first, as the file lists them */
    uint64_t  off;       /* ABSOLUTE byte offset in the file */
    uint64_t  nbytes;    /* 0 for a type without a known block size */
    int64_t   numel;
} LmGgufTensor;

#define LM_GGUF_MAX_KV      (1 << 16)
#define LM_GGUF_MAX_TENSORS (1 << 16)
#define LM_GGUF_MAX_ARRAY   (1 << 24)

typedef struct {
    char                *path;
    int                  fd;
    uint64_t             size;
    const unsigned char *map;
    uint32_t             version;
    uint64_t             alignment;
    uint64_t             data_start;
    LmGgufKv            *kv;
    int                  nkv;
    LmGgufTensor        *t;
    int                  nt;
} LmGguf;

/* 0, or -1 with a message in err (g is then closed and safe to pass to lm_gguf_close). */
int  lm_gguf_open(LmGguf *g, const char *path, char *err, size_t errlen);
void lm_gguf_close(LmGguf *g);

/* NULL when absent. */
const LmGgufKv *lm_gguf_kv(const LmGguf *g, const char *key);

/* Scalars: 0 on success; -1 when the key is absent or of another kind (nothing written).
 * lm_gguf_get_u64 accepts any integer type holding a non-negative value and bool;
 * lm_gguf_get_f64 any float or integer type. */
int lm_gguf_get_u64(const LmGguf *g, const char *key, uint64_t *out);
int lm_gguf_get_f64(const LmGguf *g, const char *key, double *out);
int lm_gguf_get_str(const LmGguf *g, const char *key, LmGgufStr *out);

/* Arrays. A string array becomes n {pointer, length} entries (malloc'd; free it); an integer
 * array n int64 values. NULL with a message in err when the key is absent, not an array of
 * that element kind, or on allocation failure. */
LmGgufStr *lm_gguf_str_array(const LmGguf *g, const char *key, int *n, char *err, size_t errlen);
int64_t   *lm_gguf_int_array(const LmGguf *g, const char *key, int *n, char *err, size_t errlen);

const LmGgufTensor *lm_gguf_find(const LmGguf *g, const char *name);
/* The tensor's bytes inside the mapping. */
const void *lm_gguf_data(const LmGguf *g, const LmGgufTensor *t);

/* 1 when path is a .gguf file, or a directory with no config.json that holds exactly one
 * file ending in .gguf (its path goes into out); 0 otherwise. */
int lm_gguf_locate(const char *path, char *out, size_t outlen);

#endif /* LM_GGUF_H */
