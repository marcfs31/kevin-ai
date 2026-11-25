/* lm_catalog.h — the ten GGUF models the chat can start, and how a GGUF architecture
 * that this engine does not run should be served instead.
 *
 * The native GGUF loader is qwen2 only (lm_qwen2.c). Gemma 4 E2B and Nemotron 3.5 Bolt
 * are gemma4, the same family as the safetensors Gemma path, and run on llama-server.
 * Granite 4.2 is the granite architecture. Spark-X2.6 is spark2_5 and needs llama.cpp
 * b10828 or later. LFM3-VL is lfm3 and needs llama.cpp b11938 or later. The Qwen3.5-2B
 * GGUF is the Qwen path (qwen3.5), the 2B file, not the installed 4B safetensors.
 * A row has a vision projector only when that file is the matching mmproj. */
#ifndef LM_CATALOG_H
#define LM_CATALOG_H

#include <stddef.h>

typedef struct LmCatalog {
    const char *id;       /* start-remote-lm-serve.sh --id */
    const char *dir;      /* ~/models/<dir>, also the llama-server --alias */
    const char *arch;     /* GGUF general.architecture */
    const char *weights;  /* file name inside dir */
    const char *mmproj;   /* matching projector, or "" when the model is text only */
    const char *llama;    /* llama.cpp build: v1.0.0, b10828 or b11938 */
} LmCatalog;

/* The ten rows, in picker order: five stock, then five with fewer safety filters. */
const LmCatalog *lm_catalog_rows(int *n);

/* Static note for a GGUF architecture this engine does not load, or NULL when arch
 * is not one of the ten (the caller keeps the qwen2-only error). */
const char *lm_catalog_arch_hint(const char *arch, size_t n);

#endif
