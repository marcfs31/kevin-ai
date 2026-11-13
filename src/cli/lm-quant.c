/* lm-quant.c - bin/lm-quant --model DIR --out FILE
 *
 * Reads a bf16 Hugging Face checkpoint (single file or sharded with
 * model.safetensors.index.json) and writes the q8 file of docs/PLAN.md. The work is in
 * src/io/lm_quantfile.c (see its header for exactly what is written); this is the argument
 * parsing and the one-line summary. Original code. */
#include <stdio.h>
#include <string.h>
#include <sys/resource.h>

#include "io/lm_quantfile.h"

int main(int argc, char **argv)
{
    const char *model = NULL, *outp = NULL;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--model") && i + 1 < argc) model = argv[++i];
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) outp = argv[++i];
        else { fprintf(stderr, "usage: lm-quant --model DIR --out FILE\n"); return 2; }
    }
    if (!model || !outp) { fprintf(stderr, "usage: lm-quant --model DIR --out FILE\n"); return 2; }

    char err[1024] = "";
    LmQuantStats st;
    if (lm_quantize_checkpoint(model, outp, &st, err, sizeof err) != 0) {
        fprintf(stderr, "lm-quant: %s\n", err);
        return 1;
    }
    struct rusage ru;
    double mb = -1.0;
    if (getrusage(RUSAGE_SELF, &ru) == 0) {
#if defined(__APPLE__)
        mb = (double)ru.ru_maxrss / (1024.0 * 1024.0);   /* bytes on Darwin */
#else
        mb = (double)ru.ru_maxrss / 1024.0;              /* KiB elsewhere */
#endif
    }
    fprintf(stderr, "lm-quant: %d quantised, %d copied, %d ignored; peak RSS %.1f MB; wrote %s\n",
            st.n_quantised, st.n_copied, st.n_ignored, mb, outp);
    return 0;
}
