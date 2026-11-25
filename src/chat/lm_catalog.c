/* lm_catalog.c — see lm_catalog.h. Original code. The file names are the contract with
 * scripts/start-remote-lm-serve.sh; tests/unit/test_catalog.c checks both. */
#include "chat/lm_catalog.h"

#include <string.h>

static const LmCatalog ROWS[] = {
    { "gemma-e2b", "gemma-4-e2b", "gemma4",
      "gemma-4-E2B-it-Q6_K.gguf", "mmproj-F16.gguf", "v1.0.0" },
    { "nemotron", "nemotron-3.5-bolt-e4b", "gemma4",
      "NVIDIA-Nemotron-3.5-Bolt-E4B-UD-Q4_K_XL.gguf", "", "v1.0.0" },
    { "granite", "granite-4.2-3b", "granite",
      "granite-4.2-3b-Q8_0.gguf", "", "v1.0.0" },
    { "spark", "spark-x2.6-4b", "spark2_5",
      "Spark-X2.6-4B-Q4_K_M.gguf", "", "b10828" },
    { "lfm3", "lfm3-vl-2b", "lfm3",
      "LFM3-VL-2B-Q8_0.gguf", "mmproj-LFM3-VL-2B-F16.gguf", "b11938" },
    { "gemma-e2b-heretic", "gemma-4-e2b-heretic", "gemma4",
      "gemma-4-E2B-it-ultra-uncensored-heretic-Q6_K.gguf", "gemma-4-E2B-it-mmproj-BF16.gguf", "v1.0.0" },
    { "spark-heretic", "spark-x2.6-4b-heretic", "spark2_5",
      "Spark-X2.6-4B-heretic.Q6_K.gguf", "", "b10828" },
    { "granite-heretic", "granite-4.2-3b-heretic", "granite",
      "granite-4.2-3b-Heretic.Q8_0.gguf", "", "v1.0.0" },
    { "qwen2b", "qwen3.5-2b-abliterated", "qwen35",
      "Huihui-Qwen3.5-2B-abliterated.Q8_0.gguf", "Huihui-Qwen3.5-2B-abliterated.mmproj-f16.gguf", "v1.0.0" },
    { "lfm3-abliterated", "lfm3-vl-2b-abliterated", "lfm3",
      "LFM3-VL-2B-Surgical-Abliterated.Q8_0.gguf", "LFM3-VL-2B-Surgical-Abliterated.mmproj-f16.gguf", "b11938" },
};

const LmCatalog *lm_catalog_rows(int *n)
{
    if (n) *n = (int)(sizeof ROWS / sizeof ROWS[0]);
    return ROWS;
}

static int same(const char *s, size_t n, const char *lit)
{
    size_t m = strlen(lit);
    return n == m && memcmp(s, lit, n) == 0;
}

const char *lm_catalog_arch_hint(const char *arch, size_t n)
{
    if (!arch) return NULL;
    if (same(arch, n, "gemma4"))
        return "gemma4 is the Gemma 4 path (Gemma 4 E2B and Nemotron 3.5 Bolt on llama-server)";
    if (same(arch, n, "qwen35") || same(arch, n, "qwen3_5") || same(arch, n, "qwen3.5"))
        return "qwen3.5 GGUF is the Qwen path for the 2B, not the installed 4B safetensors (llama-server)";
    if (same(arch, n, "granite"))
        return "granite is the Granite 4.2 architecture (llama-server)";
    if (same(arch, n, "spark2_5"))
        return "spark2_5 needs llama.cpp b10828 or later";
    if (same(arch, n, "lfm3"))
        return "lfm3 needs llama.cpp b11938 or later";
    return NULL;
}
