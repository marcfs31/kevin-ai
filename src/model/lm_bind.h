/* lm_bind.h - weight binding shared by the model families: open tensors of a safetensors
 * set (bf16 checkpoint or q8 file) as LmMat matrices and owned f32 vectors, with every
 * shape and dtype checked against the caller's expectation. Original code, moved out of
 * lm_model.c unchanged so Qwen3.5 and Gemma 4 bind the same way.
 *
 * bf16 matrices are used in place when the mapping is aligned (else one owned copy). q8
 * matrices use the int8 bytes in place and COPY the scale table into an owned float
 * buffer (the file only guarantees 2-byte alignment for it). Every owned buffer is
 * recorded and freed by lm_bind_free. */
#ifndef LM_BIND_H
#define LM_BIND_H

#include <stddef.h>
#include <stdint.h>

#include "io/lm_st.h"
#include "model/lm_layers.h"

typedef struct {
    LmSt   st;
    int    q8;                 /* weights come from an lm-q8 file */
    int    fast;               /* set before binding: q8 matrices multiply with int8 activations */
    void **owned;
    int    nowned, capowned;
} LmBind;

/* Records p; frees it and returns NULL if the list cannot grow. NULL in, NULL out. */
void *lm_bind_own(LmBind *b, void *p);
const LmTensor *lm_bind_need(LmBind *b, const char *name, int ndim, int64_t d0, int64_t d1, char *err, size_t n);
const float *lm_bind_vec(LmBind *b, const char *name, int64_t numel, int ndim, int64_t d0, int64_t d1, char *err, size_t n);
int lm_bind_mat(LmBind *b, const char *name, int rows, int cols, LmMat *out, char *err, size_t n);
/* Opens path (file, index or directory); with q8 set, checks the lm-q8 group-32 metadata. */
int lm_bind_open(LmBind *b, const char *path, int q8, char *err, size_t n);
void lm_bind_free(LmBind *b);

#endif /* LM_BIND_H */
