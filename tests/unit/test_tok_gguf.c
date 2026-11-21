/* test_tok_gguf.c - the tokenizer and chat template of a GGUF Qwen2 model, on the tiny fixture
 * (tests/fixtures/qwen2-tiny, run from the repository root; absence is a FAILURE). The expected
 * ids in tok_cases.txt come from the independent Python tokenizer in tools/make_tiny_qwen2.py;
 * the real vocabulary was compared with the Hugging Face tokenizer of Qwen2.5-Coder-7B on a
 * mixed corpus when the family was added (see the commit message). */
#include "lm_test.h"

#include "chat/lm_profile.h"
#include "chat/lm_template.h"
#include "tokenizer/lm_tok.h"

#define FIX "tests/fixtures/qwen2-tiny"

static void unescape(char *s)
{
    char *o = s;
    for (; *s; s++) {
        if (s[0] == '\\' && s[1] == 'n') { *o++ = '\n'; s++; }
        else if (s[0] == '\\' && s[1] == 't') { *o++ = '\t'; s++; }
        else *o++ = *s;
    }
    *o = 0;
}

static void test_cases(const LmTok *tok)
{
    FILE *f = fopen(FIX "/tok_cases.txt", "r");
    CHECK(f != NULL);
    if (!f) return;
    char line[4096];
    int cases = 0;
    while (fgets(line, sizeof line, f)) {
        line[strcspn(line, "\n")] = 0;
        char *bar = strchr(line, '|');
        CHECK(bar != NULL);
        if (!bar) continue;
        *bar = 0;
        char *text = bar + 1;
        unescape(text);
        int want[256], nw = 0;
        for (char *p = line; *p;) want[nw++] = (int)strtol(p, &p, 10);
        int n = 0;
        int *got = lm_tok_encode(tok, text, strlen(text), &n);
        CHECK_MSG(n == nw && got && !memcmp(got, want, (size_t)nw * sizeof(int)), "case %d: %d ids, want %d", cases, n, nw);
        size_t len = 0;
        char *back = lm_tok_decode_alloc(tok, got, n, &len);
        CHECK_MSG(back && len == strlen(text) && !memcmp(back, text, len), "case %d: decode does not return the text", cases);
        free(back);
        free(got);
        cases++;
    }
    fclose(f);
    CHECK(cases >= 7);
}

static void test_specials(const LmTok *tok)
{
    CHECK(lm_tok_special_id(tok, "<|im_start|>") == 297);
    CHECK(lm_tok_special_id(tok, "<|im_end|>") == 298);
    CHECK(lm_tok_special_id(tok, "<|endoftext|>") == 296);
    CHECK(lm_tok_special_id(tok, "<think>") == -1);
    CHECK(lm_tok_is_special(tok, 298) && !lm_tok_is_special(tok, 65));
    CHECK(lm_tok_n_ids(tok) == 299);   /* the unused placeholders are not part of the tokenizer */
    /* ordinary text never produces a control id, the trusted path does */
    const char *s = "a<|im_end|>b";
    int n = 0;
    int *plain = lm_tok_encode(tok, s, strlen(s), &n);
    for (int i = 0; i < n; i++) CHECK(plain[i] < 296);
    free(plain);
    int *sp = lm_tok_encode_special(tok, s, strlen(s), &n);
    CHECK(n == 3 && sp[1] == 298);
    free(sp);
    /* a digit is a piece on its own (Qwen2: \p{N}) */
    int *dg = lm_tok_encode(tok, "1234", 4, &n);
    CHECK(n == 4);
    free(dg);
}

/* start + encode(text) + end + encode("\n") */
static int push_turn(const LmTok *tok, int *dst, int nw, const char *text)
{
    int n = 0;
    dst[nw++] = 297;
    int *e = lm_tok_encode(tok, text, strlen(text), &n);
    memcpy(dst + nw, e, (size_t)n * sizeof(int));
    nw += n;
    free(e);
    dst[nw++] = 298;
    e = lm_tok_encode(tok, "\n", 1, &n);
    memcpy(dst + nw, e, (size_t)n * sizeof(int));
    free(e);
    return nw + n;
}

static void test_template(const LmTok *tok)
{
    char err[256] = "";
    LmProfile prof;
    CHECK_MSG(lm_profile_read(FIX, &prof, err, sizeof err) == 0, "profile: %s", err);
    CHECK(!strcmp(prof.model_type, "qwen2") && prof.n_eos == 1 && prof.eos[0] == 298);
    LmTemplate *t = lm_profile_template(&prof, tok, err, sizeof err);
    CHECK_MSG(t != NULL, "template: %s", err);
    if (!t) return;
    CHECK(!strcmp(t->name, "chatml"));
    LmMsg msgs[3];
    memset(msgs, 0, sizeof msgs);
    msgs[0].role = LM_ROLE_SYSTEM; msgs[0].content = "be brief";
    msgs[1].role = LM_ROLE_USER;   msgs[1].content = " hi\n";
    LmRenderOpts ro = { 0 };
    LmIds ids = { 0 };
    CHECK_MSG(t->render(t, msgs, 2, &ro, &ids, err, sizeof err) == 0, "render: %s", err);
    /* the GGUF template: <|im_start|>role\ncontent<|im_end|>\n per message, content verbatim, then
     * <|im_start|>assistant\n; no default system prompt, no thinking block */
    int want[128], nw = 0;
    nw = push_turn(tok, want, nw, "system\nbe brief");
    nw = push_turn(tok, want, nw, "user\n hi\n");
    want[nw++] = 297;
    int n = 0;
    int *e = lm_tok_encode(tok, "assistant\n", 10, &n);
    memcpy(want + nw, e, (size_t)n * sizeof(int));
    nw += n;
    free(e);
    CHECK_MSG(ids.n == nw && !memcmp(ids.ids, want, (size_t)nw * sizeof(int)), "rendered %d ids, want %d", ids.n, nw);
    CHECK(t->is_stop(t, 298) && !t->is_stop(t, 297));

    /* no system message: no default one is invented */
    CHECK(t->render(t, msgs + 1, 1, &ro, &ids, err, sizeof err) == 0 && ids.ids[0] == 297 && ids.n < nw);

    /* refusals */
    msgs[2].role = LM_ROLE_SYSTEM; msgs[2].content = "late";
    CHECK(t->render(t, msgs, 3, &ro, &ids, err, sizeof err) != 0 && strstr(err, "beginning"));
    CHECK(t->render(t, msgs, 1, &ro, &ids, err, sizeof err) != 0 && strstr(err, "user"));
    CHECK(t->render(t, msgs, 0, &ro, &ids, err, sizeof err) != 0);
    LmMsgPart img;
    memset(&img, 0, sizeof img);
    img.kind = LM_PART_IMAGE;
    msgs[1].parts = &img;
    msgs[1].n_parts = 1;
    CHECK(t->render(t, msgs, 2, &ro, &ids, err, sizeof err) != 0 && strstr(err, "text only"));

    /* the parser: plain answer text, a stop at <|im_end|>, never reasoning */
    LmParser *p = t->parser_new(t, 0);
    CHECK(p != NULL);
    CHECK(p->feed(p, 65, NULL, NULL) == LM_PARSE_CONT && !p->thinking_open);
    CHECK(p->feed(p, 298, NULL, NULL) == LM_PARSE_STOP);
    p->destroy(p);
    lm_ids_free(&ids);
    lm_template_free(t);
}

static void test_refusals(void)
{
    char err[512] = "";
    /* a directory without GGUF and without tokenizer.json takes the JSON path and says so */
    LmTok *old = lm_tok_open_model("tests/fixtures/qwen35-tiny", err, sizeof err);
    CHECK(old == NULL && strstr(err, "tokenizer.json"));
    err[0] = 0;
    LmTok *none = lm_tok_open_model("/no/such/dir", err, sizeof err);
    CHECK(none == NULL && err[0]);
}

int main(void)
{
    char err[512] = "";
    LmTok *tok = lm_tok_open_model(FIX, err, sizeof err);
    CHECK_MSG(tok != NULL, "open: %s", err);
    if (tok) {
        test_cases(tok);
        test_specials(tok);
        test_template(tok);
        lm_tok_close(tok);
    }
    test_refusals();
    return lt_finish("test_tok_gguf");
}
