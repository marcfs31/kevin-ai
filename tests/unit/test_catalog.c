/* The ten-model catalog: architectures, GGUF names, vision only with a matching mmproj,
 * and the same file names in scripts/start-remote-lm-serve.sh. */
#include "chat/lm_catalog.h"
#include "lm_test.h"

#include <stdio.h>
#include <string.h>

static int contains(const char *hay, const char *needle)
{
    return needle[0] && strstr(hay, needle) != NULL;
}

static void test_rows(void)
{
    int n = 0;
    const LmCatalog *rows = lm_catalog_rows(&n);
    CHECK(n == 10);
    CHECK(!strcmp(rows[0].arch, "gemma4") && !strcmp(rows[1].arch, "gemma4"));
    CHECK(!strcmp(rows[2].arch, "granite") && !strcmp(rows[7].arch, "granite"));
    CHECK(!strcmp(rows[3].arch, "spark2_5") && !strcmp(rows[3].llama, "b10828"));
    CHECK(!strcmp(rows[6].arch, "spark2_5") && !strcmp(rows[6].llama, "b10828"));
    CHECK(!strcmp(rows[4].arch, "lfm3") && !strcmp(rows[4].llama, "b11938"));
    CHECK(!strcmp(rows[9].arch, "lfm3") && !strcmp(rows[9].llama, "b11938"));
    CHECK(!strcmp(rows[8].arch, "qwen35") && !strcmp(rows[8].dir, "qwen3.5-2b-abliterated"));
    CHECK(!strstr(rows[8].dir, "4b") && !strstr(rows[8].weights, "4B"));
    /* Vision only where that row names a projector. Nemotron's repo has none. */
    CHECK(rows[0].mmproj[0] && rows[4].mmproj[0] && rows[5].mmproj[0] && rows[8].mmproj[0] && rows[9].mmproj[0]);
    CHECK(!rows[1].mmproj[0] && !rows[2].mmproj[0] && !rows[3].mmproj[0] && !rows[6].mmproj[0] && !rows[7].mmproj[0]);
}

static void test_hints(void)
{
    CHECK(lm_catalog_arch_hint("qwen2", 5) == NULL);
    CHECK(lm_catalog_arch_hint("llama", 5) == NULL);
    const char *g = lm_catalog_arch_hint("gemma4", 6);
    const char *q = lm_catalog_arch_hint("qwen3_5", 7);
    const char *s = lm_catalog_arch_hint("spark2_5", 8);
    const char *l = lm_catalog_arch_hint("lfm3", 4);
    const char *gr = lm_catalog_arch_hint("granite", 7);
    CHECK(g && strstr(g, "Gemma 4") && strstr(g, "Nemotron"));
    CHECK(q && strstr(q, "2B") && strstr(q, "4B"));
    CHECK(s && strstr(s, "b10828"));
    CHECK(l && strstr(l, "b11938"));
    CHECK(gr && strstr(gr, "Granite 4.2"));
}

static void test_script_names_the_same_files(void)
{
    FILE *f = fopen("scripts/start-remote-lm-serve.sh", "r");
    CHECK(f != NULL);
    if (!f) return;
    char buf[256 * 1024];
    size_t n = fread(buf, 1, sizeof buf - 1, f);
    fclose(f);
    buf[n] = 0;
    CHECK(n > 0 && n < sizeof buf - 1);
    int rows_n = 0;
    const LmCatalog *rows = lm_catalog_rows(&rows_n);
    for (int i = 0; i < rows_n; i++) {
        CHECK_MSG(contains(buf, rows[i].id), "script missing --id %s", rows[i].id);
        CHECK_MSG(contains(buf, rows[i].dir), "script missing dir %s", rows[i].dir);
        CHECK_MSG(contains(buf, rows[i].weights), "script missing %s", rows[i].weights);
        if (rows[i].mmproj[0]) CHECK_MSG(contains(buf, rows[i].mmproj), "script missing %s", rows[i].mmproj);
        if (!strcmp(rows[i].llama, "b10828")) CHECK(contains(buf, "llama.cpp-b10828"));
        if (!strcmp(rows[i].llama, "b11938")) CHECK(contains(buf, "llama.cpp-b11938"));
    }
}

int main(void)
{
    test_rows();
    test_hints();
    test_script_names_the_same_files();
    if (lt_fails) {
        fprintf(stderr, "test_catalog: %d of %d checks FAILED\n", lt_fails, lt_checks);
        return 1;
    }
    printf("test_catalog: %d checks passed\n", lt_checks);
    return 0;
}
