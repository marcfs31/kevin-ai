/* test_st.c - safetensors reader, writer, sha256. Style follows kimi-k3-in-c
 * tests/unit/test_st.c and test_st_faults.c (Apache-2.0). Everything builds its own files
 * in a mkdtemp directory; the reader tests use a hand-rolled writer (lm_test.h) so they do
 * not depend on lm_stw. */
#include "lm_test.h"

#include "io/lm_sha256.h"
#include "io/lm_st.h"
#include "io/lm_stw.h"

static char g_dir[512];

static char *P(const char *leaf)
{
    static char b[8][1024];
    static int i;
    i = (i + 1) & 7;
    lt_path(b[i], g_dir, leaf);
    return b[i];
}

/* A header-only open helper returning the rc and message. */
static int try_open(const char *path, int mm, char *err, size_t n)
{
    LmSt s;
    err[0] = '\0';
    const int rc = lm_st_open(&s, path, mm, err, n);
    if (rc == 0) lm_st_close(&s);
    return rc;
}

/* ------------------------------------------------------------------ happy path */

static void test_single_file(int use_mmap)
{
    /* data region: bf16[2,3] (12 B) | f32[2] (8) | f16[2] (4) | i8[2,2] (4) | u8[3] (3) |
     * scalar f32 (4) | f32[1,1,1,2] (8) */
    unsigned char d[51];
    memset(d, 0, sizeof d);
    const uint16_t bf[6] = { 0x3fc0 /*1.5*/, 0xc000 /*-2*/, 0x0000, 0x8000 /*-0*/, 0x3f80, 0x7f80 /*inf*/ };
    const float f2[2] = { 3.25f, -0.5f };
    const uint16_t h[2] = { 0x3c00 /*1*/, 0xc000 /*-2*/ };
    const int8_t i8[4] = { -128, -1, 0, 127 };
    const uint8_t u8[3] = { 0, 7, 255 };
    const float sc = 42.0f;
    const float f4[2] = { 1.0f, 2.0f };
    memcpy(d, bf, 12); memcpy(d + 12, f2, 8); memcpy(d + 20, h, 4); memcpy(d + 24, i8, 4);
    memcpy(d + 28, u8, 3); memcpy(d + 31, &sc, 4); memcpy(d + 35, f4, 8);
    lt_write_st_raw8(P("one.safetensors"),
        "{\"__metadata__\":{\"format\":\"pt\",\"note\":\"hi\"},"
        "\"a.bf\":{\"dtype\":\"BF16\",\"shape\":[2,3],\"data_offsets\":[0,12]},"
        "\"b.f32\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[12,20]},"
        "\"c.f16\":{\"dtype\":\"F16\",\"shape\":[2],\"data_offsets\":[20,24]},"
        "\"d.i8\":{\"dtype\":\"I8\",\"shape\":[2,2],\"data_offsets\":[24,28]},"
        "\"e.u8\":{\"dtype\":\"U8\",\"shape\":[3],\"data_offsets\":[28,31]},"
        "\"f.scalar\":{\"dtype\":\"F32\",\"shape\":[],\"data_offsets\":[31,35]},"
        "\"g.rank4\":{\"dtype\":\"F32\",\"shape\":[1,1,1,2],\"data_offsets\":[35,43]},"
        "\"h.empty\":{\"dtype\":\"F32\",\"shape\":[0,5],\"data_offsets\":[43,43]},"
        "\"i.rank5\":{\"dtype\":\"F32\",\"shape\":[1,1,1,1,2],\"data_offsets\":[43,51]}}", d, sizeof d);

    char err[256];
    LmSt s;
    if (use_mmap) {
        /* The same bytes behind an UNPADDED header of odd length: the data is not
         * 2-aligned, so lm_st_data must return NULL (never a misaligned pointer) while
         * lm_st_read still works. */
        const char *hdr = "{\"a\":{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[0,2]}}";
        CHECK(strlen(hdr) % 2 == 1);
        lt_write_st_raw(P("odd.safetensors"), hdr, bf, 2);
        CHECK_MSG(lm_st_open(&s, P("odd.safetensors"), 1, err, sizeof err) == 0, "%s", err);
        const LmTensor *ot = lm_st_find(&s, "a");
        float of = 0;
        CHECK(ot && (ot->off % 2) == 1 && lm_st_data(&s, ot) == NULL);
        CHECK(ot && lm_st_read_f32(&s, ot, &of, err, sizeof err) == 0 && of == 1.5f);
        lm_st_close(&s);
    }
    CHECK_MSG(lm_st_open(&s, P("one.safetensors"), use_mmap, err, sizeof err) == 0, "%s", err);
    CHECK(s.nt == 9 && s.nfiles == 1);   /* 8 entries plus i.rank5 */
    CHECK(s.files[0].map != NULL ? use_mmap == 1 : use_mmap == 0);
    CHECK(lm_st_meta(&s, "format") && !strcmp(lm_st_meta(&s, "format"), "pt"));
    CHECK(lm_st_meta(&s, "note") && !strcmp(lm_st_meta(&s, "note"), "hi"));
    CHECK(lm_st_meta(&s, "absent") == NULL);

    const LmTensor *t = lm_st_find(&s, "a.bf");
    CHECK(t && t->dtype == LM_DT_BF16 && t->ndim == 2 && t->shape[0] == 2 && t->shape[1] == 3 && t->nbytes == 12);
    float f[6];
    CHECK(t && lm_st_read_f32(&s, t, f, err, sizeof err) == 0);
    CHECK(f[0] == 1.5f && f[1] == -2.0f && f[2] == 0.0f && signbit(f[3]) && f[3] == 0.0f && f[4] == 1.0f && isinf(f[5]));
    const uint16_t *raw = (const uint16_t *)lm_st_data(&s, t);
    CHECK(use_mmap ? (raw != NULL && raw[0] == 0x3fc0) : raw == NULL);

    t = lm_st_find(&s, "b.f32");
    CHECK(t && t->dtype == LM_DT_F32 && lm_st_read_f32(&s, t, f, err, sizeof err) == 0 && f[0] == 3.25f && f[1] == -0.5f);
    t = lm_st_find(&s, "c.f16");
    CHECK(t && lm_st_read_f32(&s, t, f, err, sizeof err) == 0 && f[0] == 1.0f && f[1] == -2.0f);
    t = lm_st_find(&s, "d.i8");
    CHECK(t && t->dtype == LM_DT_I8 && lm_st_read_f32(&s, t, f, err, sizeof err) == 0 &&
          f[0] == -128.0f && f[1] == -1.0f && f[2] == 0.0f && f[3] == 127.0f);
    t = lm_st_find(&s, "e.u8");
    CHECK(t && lm_st_read_f32(&s, t, f, err, sizeof err) == 0 && f[0] == 0.0f && f[1] == 7.0f && f[2] == 255.0f);
    t = lm_st_find(&s, "f.scalar");
    CHECK(t && t->ndim == 0 && lm_st_numel(t) == 1 && lm_st_read_f32(&s, t, f, err, sizeof err) == 0 && f[0] == 42.0f);
    t = lm_st_find(&s, "g.rank4");
    CHECK(t && t->ndim == 4 && lm_st_numel(t) == 2);
    t = lm_st_find(&s, "h.empty");
    CHECK(t && lm_st_numel(t) == 0 && t->nbytes == 0 && lm_st_read(&s, t, f, err, sizeof err) == 0);
    /* rank above 4 opens (a tensor the engine never uses must not block the file); the
     * element count covers every dim and the overflow flag is set for users to check */
    t = lm_st_find(&s, "i.rank5");
    CHECK(t && t->rank_overflow == 1 && lm_st_numel(t) == 2 && t->nbytes == 8);
    t = lm_st_find(&s, "g.rank4");
    CHECK(t && t->rank_overflow == 0);
    CHECK(lm_st_find(&s, "nope") == NULL);

    /* slice reads and their bounds */
    t = lm_st_find(&s, "a.bf");
    uint16_t two[2];
    CHECK(lm_st_read_slice(&s, t, 4, 4, two, err, sizeof err) == 0 && two[0] == 0x0000 && two[1] == 0x8000);
    int rc = lm_st_read_slice(&s, t, 10, 4, two, err, sizeof err);
    CHECK_ERR(rc, err, "outside tensor");
    rc = lm_st_read_slice(&s, t, -1, 2, two, err, sizeof err);
    CHECK_ERR(rc, err, "outside tensor");
    CHECK(lm_st_read_f32_range(&s, t, 2, 2, f, err, sizeof err) == 0 && f[0] == 0.0f && f[1] == 0.0f);
    rc = lm_st_read_f32_range(&s, t, 5, 2, f, err, sizeof err);
    CHECK_ERR(rc, err, "outside tensor");
    lm_st_close(&s);
}

static void test_f16_widening(void)
{
    /* 0x0001 = smallest subnormal 2^-24, 0x03ff largest subnormal, 0x7bff max finite, 0x7c00 inf, 0x7e00 nan */
    CHECK(lm_f16_to_f32(0x0001) == 5.9604644775390625e-08f);
    CHECK(lm_f16_to_f32(0x03ff) == 6.097555160522461e-05f);
    CHECK(lm_f16_to_f32(0x7bff) == 65504.0f);
    CHECK(isinf(lm_f16_to_f32(0x7c00)) && lm_f16_to_f32(0x7c00) > 0);
    CHECK(isinf(lm_f16_to_f32(0xfc00)) && lm_f16_to_f32(0xfc00) < 0);
    CHECK(isnan(lm_f16_to_f32(0x7e00)));
    CHECK(lm_f16_to_f32(0x0000) == 0.0f && !signbit(lm_f16_to_f32(0x0000)));
    CHECK(signbit(lm_f16_to_f32(0x8000)));
    CHECK(lm_f16_to_f32(0x3555) == 0.333251953125f);
    /* bf16 round trip: widening is exact, narrowing is nearest-even */
    CHECK(lm_bf16_to_f32(lm_f32_to_bf16(1.5f)) == 1.5f);
    CHECK(lm_f32_to_bf16(1.00390625f) == 0x3f80);      /* tie between 1.0 and 1.0078125: even wins */
    CHECK(lm_f32_to_bf16(1.01171875f) == 0x3f82);      /* tie, rounds up to even */
    CHECK(isnan(lm_bf16_to_f32(lm_f32_to_bf16(NAN))));
}

/* ------------------------------------------------------------------ rejections */

static void test_rejections(void)
{
    char err[512];
    const float z[8] = { 0 };
    int rc;

    /* path problems */
    rc = try_open(P("missing.safetensors"), 1, err, sizeof err);
    CHECK_ERR(rc, err, "cannot open");
    mkdir(P("emptydir"), 0755);
    rc = try_open(P("emptydir"), 1, err, sizeof err);
    CHECK_ERR(rc, err, "neither model.safetensors.index.json nor model.safetensors");

    /* shorter than the length prefix */
    lt_write_file(P("tiny.st"), "abc", 3);
    rc = try_open(P("tiny.st"), 1, err, sizeof err);
    CHECK_ERR(rc, err, "too short");
    lt_write_file(P("empty.st"), "", 0);
    rc = try_open(P("empty.st"), 1, err, sizeof err);
    CHECK_ERR(rc, err, "too short");

    /* header length: zero, past EOF, absurd (2^64-1 must not wrap), over the cap */
    lt_write_st_len(P("hz.st"), 0, "{}", NULL, 0);
    rc = try_open(P("hz.st"), 1, err, sizeof err);
    CHECK_ERR(rc, err, "header length is zero");
    lt_write_st_len(P("hp.st"), 50, "{}", NULL, 0);
    rc = try_open(P("hp.st"), 1, err, sizeof err);
    CHECK_ERR(rc, err, "truncated");
    lt_write_st_len(P("hmax.st"), UINT64_MAX, "{}", NULL, 0);
    rc = try_open(P("hmax.st"), 1, err, sizeof err);
    CHECK_ERR(rc, err, "exceeds");
    lt_write_st_len(P("hbig.st"), (uint64_t)LM_ST_MAX_HEADER + 1, "{}", NULL, 0);
    rc = try_open(P("hbig.st"), 1, err, sizeof err);
    CHECK_ERR(rc, err, "exceeds");

    /* header content */
    lt_write_st_raw(P("notobj.st"), "[1,2]", NULL, 0);
    rc = try_open(P("notobj.st"), 1, err, sizeof err);
    CHECK_ERR(rc, err, "not a JSON object");
    lt_write_st_raw(P("trail.st"), "{} junk", NULL, 0);
    rc = try_open(P("trail.st"), 1, err, sizeof err);
    CHECK_ERR(rc, err, "unexpected data after the header");
    lt_write_st_raw(P("badent.st"), "{\"a\":5}", NULL, 0);
    rc = try_open(P("badent.st"), 1, err, sizeof err);
    CHECK_ERR(rc, err, "not an object");
    lt_write_st_raw(P("trunchdr.st"), "{\"a\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]", z, 4);
    rc = try_open(P("trunchdr.st"), 1, err, sizeof err);
    CHECK(rc != 0);
    lt_write_st_raw(P("nodt.st"), "{\"a\":{\"shape\":[1],\"data_offsets\":[0,4]}}", z, 4);
    rc = try_open(P("nodt.st"), 1, err, sizeof err);
    CHECK_ERR(rc, err, "missing dtype");
    lt_write_st_raw(P("nosh.st"), "{\"a\":{\"dtype\":\"F32\",\"data_offsets\":[0,4]}}", z, 4);
    rc = try_open(P("nosh.st"), 1, err, sizeof err);
    CHECK_ERR(rc, err, "missing shape");
    lt_write_st_raw(P("noof.st"), "{\"a\":{\"dtype\":\"F32\",\"shape\":[1]}}", z, 4);
    rc = try_open(P("noof.st"), 1, err, sizeof err);
    CHECK_ERR(rc, err, "missing data_offsets");
    lt_write_st_raw(P("i64.st"), "{\"a\":{\"dtype\":\"I64\",\"shape\":[1],\"data_offsets\":[0,8]}}", z, 8);
    rc = try_open(P("i64.st"), 1, err, sizeof err);
    CHECK_ERR(rc, err, "unsupported dtype 'I64'");
    /* rank above 4 is accepted at open (see i.rank5 above): only a tensor that is used fails */
    lt_write_st_raw(P("rank5.st"), "{\"a\":{\"dtype\":\"F32\",\"shape\":[1,1,1,1,1],\"data_offsets\":[0,4]}}", z, 4);
    rc = try_open(P("rank5.st"), 1, err, sizeof err);
    CHECK(rc == 0);
    lt_write_st_raw(P("negdim.st"), "{\"a\":{\"dtype\":\"F32\",\"shape\":[-1],\"data_offsets\":[0,4]}}", z, 4);
    rc = try_open(P("negdim.st"), 1, err, sizeof err);
    CHECK_ERR(rc, err, "negative dimension");
    lt_write_st_raw(P("badmeta.st"), "{\"__metadata__\":{\"k\":5}}", NULL, 0);
    rc = try_open(P("badmeta.st"), 1, err, sizeof err);
    CHECK_ERR(rc, err, "non-string");
    lt_write_st_raw(P("dupname.st"),
        "{\"a\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]},"
        "\"a\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[4,8]}}", z, 8);
    rc = try_open(P("dupname.st"), 1, err, sizeof err);
    CHECK(rc != 0);   /* rejected as a duplicate or as overlapping, both are refusals */

    /* ranges */
    lt_write_st_raw(P("inv.st"), "{\"a\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[4,0]}}", z, 4);
    rc = try_open(P("inv.st"), 1, err, sizeof err);
    CHECK_ERR(rc, err, "impossible data_offsets");
    lt_write_st_raw(P("negoff.st"), "{\"a\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[-4,0]}}", z, 4);
    rc = try_open(P("negoff.st"), 1, err, sizeof err);
    CHECK_ERR(rc, err, "impossible data_offsets");
    lt_write_st_raw(P("past.st"), "{\"a\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]}}", z, 4);
    rc = try_open(P("past.st"), 1, err, sizeof err);
    CHECK_ERR(rc, err, "truncated file");
    lt_write_st_raw(P("huge.st"),
        "{\"a\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,9223372036854775807]}}", z, 4);
    rc = try_open(P("huge.st"), 1, err, sizeof err);
    CHECK_ERR(rc, err, "truncated file");
    lt_write_st_raw(P("overlap.st"),
        "{\"a\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]},"
        "\"b\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[4,12]}}", z, 12);
    rc = try_open(P("overlap.st"), 1, err, sizeof err);
    CHECK_ERR(rc, err, "overlap");
    /* the same two spans listed in the other order are still caught (sweep sorts by offset) */
    lt_write_st_raw(P("overlap2.st"),
        "{\"b\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[4,12]},"
        "\"a\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]}}", z, 12);
    rc = try_open(P("overlap2.st"), 1, err, sizeof err);
    CHECK_ERR(rc, err, "overlap");
    /* a zero-length tensor inside another's span is not an overlap */
    lt_write_st_raw(P("zerolen.st"),
        "{\"a\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]},"
        "\"z\":{\"dtype\":\"F32\",\"shape\":[0],\"data_offsets\":[4,4]}}", z, 8);
    rc = try_open(P("zerolen.st"), 1, err, sizeof err);
    CHECK_MSG(rc == 0, "%s", err);

    /* dtype/shape/length mismatch, both directions, and a shape whose product wraps */
    lt_write_st_raw(P("mis1.st"), "{\"a\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,4]}}", z, 4);
    rc = try_open(P("mis1.st"), 1, err, sizeof err);
    CHECK_ERR(rc, err, "its F32 shape implies 8");
    lt_write_st_raw(P("mis2.st"), "{\"a\":{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[0,4]}}", z, 4);
    rc = try_open(P("mis2.st"), 1, err, sizeof err);
    CHECK_ERR(rc, err, "its BF16 shape implies 2");
    lt_write_st_raw(P("wrap.st"),
        "{\"a\":{\"dtype\":\"U8\",\"shape\":[4294967296,4294967296],\"data_offsets\":[0,0]}}", z, 4);
    rc = try_open(P("wrap.st"), 1, err, sizeof err);
    CHECK_ERR(rc, err, "implies");

    /* a truncated file: a good file with its last byte cut off */
    {
        unsigned char d[16] = { 0 };
        lt_write_st_raw(P("full.st"), "{\"a\":{\"dtype\":\"F32\",\"shape\":[4],\"data_offsets\":[0,16]}}", d, 16);
        rc = try_open(P("full.st"), 1, err, sizeof err);
        CHECK(rc == 0);
        lt_write_st_raw(P("cut.st"), "{\"a\":{\"dtype\":\"F32\",\"shape\":[4],\"data_offsets\":[0,16]}}", d, 15);
        rc = try_open(P("cut.st"), 1, err, sizeof err);
        CHECK_ERR(rc, err, "truncated file");
        rc = try_open(P("cut.st"), 0, err, sizeof err);
        CHECK_ERR(rc, err, "truncated file");
    }
}

/* ------------------------------------------------------------------ shards */

static void test_shards(void)
{
    char err[512];
    int rc;
    const float a[2] = { 1.0f, 2.0f }, b[3] = { 3.0f, 4.0f, 5.0f };
    mkdir(P("sh"), 0755);
    lt_write_st_raw(P("sh/model-00001-of-00002.safetensors"),
        "{\"__metadata__\":{\"format\":\"pt\"},\"x.weight\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]}}", a, 8);
    lt_write_st_raw(P("sh/model-00002-of-00002.safetensors"),
        "{\"y.weight\":{\"dtype\":\"F32\",\"shape\":[3],\"data_offsets\":[0,12]}}", b, 12);
    lt_write_text(P("sh/model.safetensors.index.json"),
        "{\"metadata\":{\"total_size\":20},\"weight_map\":{\"x.weight\":\"model-00001-of-00002.safetensors\","
        "\"y.weight\":\"model-00002-of-00002.safetensors\"}}");

    for (int mm = 0; mm <= 1; mm++) {
        LmSt s;
        CHECK_MSG(lm_st_open(&s, P("sh"), mm, err, sizeof err) == 0, "%s", err);   /* directory form */
        CHECK(s.nfiles == 2 && s.nt == 2);
        const LmTensor *x = lm_st_find(&s, "x.weight"), *y = lm_st_find(&s, "y.weight");
        CHECK(x && y && x->file != y->file);
        float f[3];
        CHECK(x && lm_st_read_f32(&s, x, f, err, sizeof err) == 0 && f[0] == 1.0f && f[1] == 2.0f);
        CHECK(y && lm_st_read_f32(&s, y, f, err, sizeof err) == 0 && f[0] == 3.0f && f[2] == 5.0f);
        CHECK(lm_st_meta(&s, "format") && !strcmp(lm_st_meta(&s, "format"), "pt"));
        lm_st_close(&s);
    }
    {
        LmSt s;
        CHECK(lm_st_open(&s, P("sh/model.safetensors.index.json"), 1, err, sizeof err) == 0 && s.nt == 2);  /* file form */
        lm_st_close(&s);
    }

    /* index names a shard that is not there */
    mkdir(P("miss"), 0755);
    lt_write_st_raw(P("miss/s1.safetensors"), "{\"x\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]}}", a, 8);
    lt_write_text(P("miss/model.safetensors.index.json"),
        "{\"weight_map\":{\"x\":\"s1.safetensors\",\"y\":\"s2.safetensors\"}}");
    rc = try_open(P("miss"), 1, err, sizeof err);
    CHECK_ERR(rc, err, "names shard s2.safetensors");
    CHECK(strstr(err, "does not exist") != NULL);

    /* index lists a tensor its shard does not hold */
    mkdir(P("lack"), 0755);
    lt_write_st_raw(P("lack/s1.safetensors"), "{\"x\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]}}", a, 8);
    lt_write_text(P("lack/model.safetensors.index.json"),
        "{\"weight_map\":{\"x\":\"s1.safetensors\",\"ghost\":\"s1.safetensors\"}}");
    rc = try_open(P("lack"), 1, err, sizeof err);
    CHECK_ERR(rc, err, "which has no such tensor");

    /* a shard holds a tensor the index omits */
    mkdir(P("omit"), 0755);
    {
        const float ab[4] = { 1, 2, 3, 4 };
        lt_write_st_raw(P("omit/s1.safetensors"),
            "{\"x\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]},"
            "\"extra\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[8,16]}}", ab, 16);
    }
    lt_write_text(P("omit/model.safetensors.index.json"), "{\"weight_map\":{\"x\":\"s1.safetensors\"}}");
    rc = try_open(P("omit"), 1, err, sizeof err);
    CHECK_ERR(rc, err, "does not list");

    /* index says shard 1, tensor is in shard 2 */
    mkdir(P("wrong"), 0755);
    lt_write_st_raw(P("wrong/s1.safetensors"), "{\"x\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]}}", a, 8);
    lt_write_st_raw(P("wrong/s2.safetensors"), "{\"y\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]}}", a, 8);
    lt_write_text(P("wrong/model.safetensors.index.json"),
        "{\"weight_map\":{\"x\":\"s2.safetensors\",\"y\":\"s1.safetensors\"}}");
    rc = try_open(P("wrong"), 1, err, sizeof err);
    CHECK_ERR(rc, err, "but it is in");

    /* the same tensor name in two shards */
    mkdir(P("dup"), 0755);
    lt_write_st_raw(P("dup/s1.safetensors"), "{\"x\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]}}", a, 8);
    lt_write_st_raw(P("dup/s2.safetensors"), "{\"x\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]}}", a, 8);
    lt_write_text(P("dup/model.safetensors.index.json"),
        "{\"weight_map\":{\"x\":\"s1.safetensors\",\"q\":\"s2.safetensors\"}}");
    rc = try_open(P("dup"), 1, err, sizeof err);
    CHECK_ERR(rc, err, "appears in both");

    /* path traversal and non-string entries */
    lt_write_text(P("dup/model.safetensors.index.json"), "{\"weight_map\":{\"x\":\"../sh/model-00001-of-00002.safetensors\"}}");
    rc = try_open(P("dup"), 1, err, sizeof err);
    CHECK_ERR(rc, err, "not a bare shard file name");
    lt_write_text(P("dup/model.safetensors.index.json"), "{\"weight_map\":{\"x\":7}}");
    rc = try_open(P("dup"), 1, err, sizeof err);
    CHECK_ERR(rc, err, "not a bare shard file name");

    /* malformed / empty indexes */
    lt_write_text(P("dup/model.safetensors.index.json"), "{\"weight_map\":{\"x\":\"s1.safetensors\"");
    rc = try_open(P("dup"), 1, err, sizeof err);
    CHECK_ERR(rc, err, "not a valid JSON object");
    lt_write_text(P("dup/model.safetensors.index.json"), "{\"weight_map\":{}}");
    rc = try_open(P("dup"), 1, err, sizeof err);
    CHECK_ERR(rc, err, "weight_map");
    lt_write_text(P("dup/model.safetensors.index.json"), "{\"other\":1}");
    rc = try_open(P("dup"), 1, err, sizeof err);
    CHECK_ERR(rc, err, "weight_map");

    /* a truncated shard inside an otherwise valid set */
    mkdir(P("tr"), 0755);
    lt_write_st_raw(P("tr/s1.safetensors"), "{\"x\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]}}", a, 6);
    lt_write_text(P("tr/model.safetensors.index.json"), "{\"weight_map\":{\"x\":\"s1.safetensors\"}}");
    rc = try_open(P("tr"), 1, err, sizeof err);
    CHECK_ERR(rc, err, "truncated file");

    /* a lone model.safetensors in a directory opens without an index */
    mkdir(P("lone"), 0755);
    lt_write_st_raw(P("lone/model.safetensors"), "{\"x\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]}}", a, 8);
    rc = try_open(P("lone"), 1, err, sizeof err);
    CHECK_MSG(rc == 0, "%s", err);
}

/* ------------------------------------------------------------------ writer */

static void test_writer(void)
{
    char err[512];
    const int64_t sh2[2] = { 2, 2 }, sh1[1] = { 3 };
    LmStWriter *w = lm_stw_create(P("out.st"), err, sizeof err);
    CHECK(w != NULL);
    if (!w) return;
    CHECK(lm_stw_meta(w, "source", "abc", err, sizeof err) == 0);
    CHECK(lm_stw_meta(w, "format", "lm-q8", err, sizeof err) == 0);
    CHECK(lm_stw_meta(w, "format", "x", err, sizeof err) != 0);                       /* duplicate key */
    /* declared out of order on purpose: the layout is sorted by name */
    CHECK(lm_stw_declare(w, "z.bf", LM_DT_BF16, 1, sh1, err, sizeof err) == 0);
    CHECK(lm_stw_declare(w, "a.w", LM_DT_I8, 2, sh2, err, sizeof err) == 0);
    CHECK(lm_stw_declare(w, "a.w.q8scale", LM_DT_F32, 1, sh1, err, sizeof err) == 0);
    CHECK(lm_stw_declare(w, "a.w", LM_DT_I8, 2, sh2, err, sizeof err) != 0);          /* duplicate */
    CHECK(lm_stw_declare(w, "bad\x01name", LM_DT_F32, 1, sh1, err, sizeof err) != 0); /* non-printable */
    CHECK(lm_stw_declare(w, "neg", LM_DT_F32, 1, (const int64_t[]){ -1 }, err, sizeof err) != 0);
    CHECK(lm_stw_tensor_begin(w, "a.w", err, sizeof err) != 0);                       /* before begin */
    CHECK(lm_stw_begin(w, err, sizeof err) == 0);
    CHECK(lm_stw_declare(w, "late", LM_DT_F32, 1, sh1, err, sizeof err) != 0);        /* after begin */

    int rc = lm_stw_tensor_begin(w, "z.bf", err, sizeof err);
    CHECK_ERR(rc, err, "out of order");
    CHECK(lm_stw_next_name(w) && !strcmp(lm_stw_next_name(w), "a.w"));
    CHECK(lm_stw_tensor_begin(w, "a.w", err, sizeof err) == 0);
    const int8_t q[4] = { 1, -2, 3, -4 };
    CHECK(lm_stw_write(w, q, 3, err, sizeof err) == 0);
    rc = lm_stw_write(w, q, 2, err, sizeof err);
    CHECK_ERR(rc, err, "exceed");
    rc = lm_stw_tensor_end(w, err, sizeof err);
    CHECK_ERR(rc, err, "short");
    CHECK(lm_stw_write(w, q + 3, 1, err, sizeof err) == 0);
    CHECK(lm_stw_tensor_end(w, err, sizeof err) == 0);
    CHECK(lm_stw_tensor_begin(w, "a.w.q8scale", err, sizeof err) == 0);
    const float sc[3] = { 0.5f, 1.0f, 2.0f };
    CHECK(lm_stw_write(w, sc, 12, err, sizeof err) == 0 && lm_stw_tensor_end(w, err, sizeof err) == 0);
    rc = lm_stw_finish(w, err, sizeof err);          /* z.bf never written */
    CHECK_ERR(rc, err, "never written");
    CHECK(access(P("out.st"), F_OK) != 0 && access(P("out.st.tmp"), F_OK) != 0);   /* nothing left behind */

    /* the full write, byte-exact against the documented layout */
    w = lm_stw_create(P("out.st"), err, sizeof err);
    CHECK(w != NULL);
    lm_stw_meta(w, "source", "abc", err, sizeof err);
    lm_stw_meta(w, "format", "lm-q8", err, sizeof err);
    lm_stw_declare(w, "z.bf", LM_DT_BF16, 1, sh1, err, sizeof err);
    lm_stw_declare(w, "a.w", LM_DT_I8, 2, sh2, err, sizeof err);
    lm_stw_declare(w, "a.w.q8scale", LM_DT_F32, 1, sh1, err, sizeof err);
    CHECK(lm_stw_begin(w, err, sizeof err) == 0);
    const uint16_t bf[3] = { 0x3f80, 0x4000, 0x4040 };
    lm_stw_tensor_begin(w, "a.w", err, sizeof err);
    lm_stw_write(w, q, 4, err, sizeof err);
    lm_stw_tensor_end(w, err, sizeof err);
    lm_stw_tensor_begin(w, "a.w.q8scale", err, sizeof err);
    lm_stw_write(w, sc, 12, err, sizeof err);
    lm_stw_tensor_end(w, err, sizeof err);
    lm_stw_tensor_begin(w, "z.bf", err, sizeof err);
    lm_stw_write(w, bf, 4, err, sizeof err);              /* chunked */
    lm_stw_write(w, (const char *)bf + 4, 2, err, sizeof err);
    lm_stw_tensor_end(w, err, sizeof err);
    CHECK(lm_stw_next_name(w) == NULL);
    CHECK_MSG(lm_stw_finish(w, err, sizeof err) == 0, "%s", err);
    CHECK(access(P("out.st.tmp"), F_OK) != 0);

    const char *want =
        "{\"__metadata__\":{\"format\":\"lm-q8\",\"source\":\"abc\"},"
        "\"a.w\":{\"dtype\":\"I8\",\"shape\":[2,2],\"data_offsets\":[0,4]},"
        "\"a.w.q8scale\":{\"dtype\":\"F32\",\"shape\":[3],\"data_offsets\":[4,16]},"
        "\"z.bf\":{\"dtype\":\"BF16\",\"shape\":[3],\"data_offsets\":[16,22]}}";
    FILE *f = fopen(P("out.st"), "rb");
    CHECK(f != NULL);
    if (f) {
        unsigned char buf[1024];
        const size_t n = fread(buf, 1, sizeof buf, f);
        fclose(f);
        size_t hl = strlen(want);
        const size_t padded = (hl + 7) / 8 * 8;
        uint64_t N = 0;
        for (int i = 7; i >= 0; i--) N = (N << 8) | buf[i];
        CHECK(N == padded);                                  /* length counts the padding */
        CHECK(padded % 8 == 0 && padded > hl);
        CHECK(memcmp(buf + 8, want, hl) == 0);
        for (size_t i = hl; i < padded; i++) CHECK(buf[8 + i] == ' ');
        CHECK(n == 8 + padded + 22);
        CHECK(memcmp(buf + 8 + padded, q, 4) == 0);
        CHECK(memcmp(buf + 8 + padded + 4, sc, 12) == 0);
        CHECK(memcmp(buf + 8 + padded + 16, bf, 6) == 0);
    }

    /* and the reader reads it back */
    LmSt s;
    CHECK_MSG(lm_st_open(&s, P("out.st"), 1, err, sizeof err) == 0, "%s", err);
    CHECK(s.nt == 3);
    const LmTensor *t = lm_st_find(&s, "a.w");
    CHECK(t && t->dtype == LM_DT_I8 && t->shape[0] == 2 && t->shape[1] == 2);
    int8_t qr[4];
    CHECK(t && lm_st_read(&s, t, qr, err, sizeof err) == 0 && !memcmp(qr, q, 4));
    t = lm_st_find(&s, "a.w.q8scale");
    float sr[3];
    CHECK(t && lm_st_read(&s, t, sr, err, sizeof err) == 0 && !memcmp(sr, sc, 12));
    t = lm_st_find(&s, "z.bf");
    float zf[3];
    CHECK(t && lm_st_read_f32(&s, t, zf, err, sizeof err) == 0 && zf[0] == 1.0f && zf[1] == 2.0f && zf[2] == 3.0f);
    CHECK(!strcmp(lm_st_meta(&s, "source"), "abc") && !strcmp(lm_st_meta(&s, "format"), "lm-q8"));
    /* a bf16 tensor after an odd-sized I8 tensor can sit at an odd offset: lm_st_data must refuse, not mis-align */
    CHECK(lm_st_data(&s, t) == NULL || ((uintptr_t)lm_st_data(&s, t) & 1) == 0);
    lm_st_close(&s);

    /* an empty metadata set omits the object; empty file with no tensors is valid */
    w = lm_stw_create(P("empty.out"), err, sizeof err);
    CHECK(w && lm_stw_begin(w, err, sizeof err) == 0 && lm_stw_finish(w, err, sizeof err) == 0);
    rc = try_open(P("empty.out"), 1, err, sizeof err);
    CHECK_MSG(rc == 0, "%s", err);

    /* creating in a directory that does not exist fails cleanly */
    CHECK(lm_stw_create(P("nodir/x.st"), err, sizeof err) == NULL && strstr(err, "cannot create"));
    lm_stw_abort(NULL);   /* safe on NULL */
}

/* ------------------------------------------------------------------ sha256 */

static void test_sha256(void)
{
    char hex[65], err[256];
    unsigned char d[32];
    LmSha256 c;
    lm_sha256_init(&c);
    lm_sha256_final(&c, d);
    lm_sha256_hex(d, hex);
    CHECK(!strcmp(hex, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"));
    lm_sha256_init(&c);
    lm_sha256_update(&c, "abc", 3);
    lm_sha256_final(&c, d);
    lm_sha256_hex(d, hex);
    CHECK(!strcmp(hex, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"));
    /* two-block message, fed in awkward pieces to exercise the buffering */
    const char *m = "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
    lm_sha256_init(&c);
    lm_sha256_update(&c, m, 1);
    lm_sha256_update(&c, m + 1, 30);
    lm_sha256_update(&c, m + 31, strlen(m) - 31);
    lm_sha256_final(&c, d);
    lm_sha256_hex(d, hex);
    CHECK(!strcmp(hex, "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"));
    /* one million 'a', through the file helper (also covers a read longer than one block) */
    char *big = (char *)malloc(1000000);
    memset(big, 'a', 1000000);
    lt_write_file(P("million"), big, 1000000);
    free(big);
    CHECK(lm_sha256_file(P("million"), hex, err, sizeof err) == 0);
    CHECK(!strcmp(hex, "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"));
    CHECK(lm_sha256_file(P("no-such-file"), hex, err, sizeof err) != 0);
}

int main(void)
{
    lt_mkdtemp(g_dir);
    test_f16_widening();
    test_single_file(1);
    test_single_file(0);
    test_rejections();
    test_shards();
    test_writer();
    test_sha256();
    lt_rmtree(g_dir);
    return lt_finish("test_st");
}
