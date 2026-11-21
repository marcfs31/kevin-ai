/* lm_vision.h - the image encoders: RGB8 pixels at the encoder's grid size -> one row of the
 * text model's hidden width per image token. Original code; the maths is docs/PLAN.md
 * "Phase 3" and the oracles tools/ref_vision_qwen35.py and tools/ref_vision_gemma4.py.
 *
 * Resampling an arbitrary image to the grid size is not done here (docs/PLAN.md,
 * "Resampling"): lm_vision_encode refuses a size that is not a multiple of info.unit.
 * Weights come from the same place as the text model's: the bf16 checkpoint, or the q8 file,
 * where the quantiser keeps the vision tensors (src/io/lm_quantfile.h). */
#ifndef LM_VISION_H
#define LM_VISION_H

#include <stddef.h>
#include <stdint.h>

enum { LM_VISION_QWEN = 1, LM_VISION_GEMMA = 2 };

typedef struct {
    int family;              /* LM_VISION_* */
    int out_dim;             /* width of each output row: the text model's hidden size */
    int unit;                /* image height and width must be positive multiples of this */
    int image_token;         /* the placeholder id each output row replaces */
    int start_token, end_token;   /* the ids around the placeholders */
    int slot_token;          /* the id the text model sees at a placeholder position: Gemma's
                                text_config.pad_token_id (its per-layer table lookup), Qwen's
                                image_token */
} LmVisionInfo;

typedef struct LmVision LmVision;

/* model_dir: the checkpoint directory (config.json); weights: the q8 file, or NULL to read the
 * bf16 checkpoint in model_dir. NULL with a message in err when the config has no supported
 * vision encoder or a tensor is missing or misshapen. */
LmVision *lm_vision_open(const char *model_dir, const char *weights, char *err, size_t errlen);
void      lm_vision_close(LmVision *v);
const LmVisionInfo *lm_vision_info(const LmVision *v);

/* Rows an h x w image encodes to, or -1 when h or w is not a positive multiple of unit or
 * the image is larger than the encoder accepts (LM_VISION_MAX_PATCHES patches). */
#define LM_VISION_MAX_PATCHES 16384
int lm_vision_tokens(const LmVision *v, int h, int w);

/* rgb: [h][w][3] row-major; rows: lm_vision_tokens(v, h, w) * info.out_dim floats, in the
 * order the placeholders are fed. 0, or -1 with a message in err (bad size, out of memory). */
int lm_vision_encode(LmVision *v, const uint8_t *rgb, int h, int w, float *rows, char *err, size_t errlen);

#endif /* LM_VISION_H */
