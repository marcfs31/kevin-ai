/* lm_quantfile.h - checkpoint -> q8 file, the library behind bin/lm-quant (kept out of
 * src/cli so the unit tests can call it). Original code; the format is owned by
 * tools/quant_ref.py and this must produce the same bytes.
 *
 * WHAT IS WRITTEN (byte-level layout rules in src/io/lm_stw.h):
 *   - every 2-D tensor whose name ends in ".weight" is quantised: "<name>" as I8
 *     [out, in] and "<name>.q8scale" as F32 [out, in/32] (see src/core/lm_quant.h)
 *   - every other tensor is copied byte for byte with its name, dtype and shape
 *   - only tensors named "model.language_model.*" are kept; everything else (visual, mtp)
 *     is DROPPED, as tools/quant_ref.py does
 *   - __metadata__ {"format":"lm-q8","group":"32","source":"<sha256>"} where source is
 *     the SHA-256 of model.safetensors.index.json when the checkpoint directory has one,
 *     else of the single safetensors file
 * Memory: tensors are read and written in blocks of about 16 MB, so peak RSS is a few
 * tens of MB plus one tensor's scale table (79 MB for the embedding), however large the
 * checkpoint. The source is opened with pread, not mmap, so mapped pages do not count
 * against the process. */
#ifndef LM_QUANTFILE_H
#define LM_QUANTFILE_H

#include <stddef.h>

typedef struct {
    int n_quantised;   /* 2-D .weight tensors written as I8 + .q8scale */
    int n_copied;      /* everything else, byte for byte */
    int n_ignored;     /* visual / mtp tensors dropped */
} LmQuantStats;

/* model: a checkpoint directory (index.json or model.safetensors) or a safetensors file.
 * out: the file to write (via out + ".tmp", renamed on success; nothing is left behind on
 * failure). Returns 0, or -1 with a message in err: an unreadable or malformed source, a
 * 2-D .weight whose in is not a positive multiple of 32 or whose dtype is not BF16/F16/F32,
 * a NaN or Inf weight, a source tensor named "*.q8scale", no 2-D weights at all, or any
 * write failure. stats may be NULL. */
int lm_quantize_checkpoint(const char *model, const char *out, LmQuantStats *stats,
                           char *err, size_t errlen);

#endif /* LM_QUANTFILE_H */
