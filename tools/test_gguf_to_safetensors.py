#!/usr/bin/env python3
"""Tests for gguf_to_safetensors: a tiny synthetic GGUF plus a tiny fake base checkpoint.

    cd tools && python3 -m unittest test_gguf_to_safetensors -v
"""

import contextlib
import io
import json
import math
import os
import struct
import sys
import tempfile
import unittest

TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS)

import gguf_to_safetensors as g  # noqa: E402
import safetensors as st  # noqa: E402

H, VOCAB, N_K, N_V, KD, VD = 4, 6, 2, 4, 2, 2   # hidden, vocab, k heads, v heads, head dims
RATIO = N_V // N_K
TOKENS = ["a", "b", "c", "<|x|>", "Ġd", "[PAD5]"]
P = "model.language_model."
L0, L1 = P + "layers.0.", P + "layers.1."          # linear attention, full attention


def vals(n, seed):
    return [((i * 7 + seed * 3) % 23 - 11) / 8 for i in range(n)]


def rows(flat, ncols):
    return [flat[i:i + ncols] for i in range(0, len(flat), ncols)]


def flat(rs):
    return [x for r in rs for x in r]


def tile(blocks):
    """llama.cpp order: value head r*N_K + k holds the HF head k*RATIO + r."""
    return [blocks[k * RATIO + r] for r in range(RATIO) for k in range(N_K)]


def base_tensors():
    """HF name -> (dtype, shape, floats)."""
    t = {
        P + "embed_tokens.weight": ("BF16", (VOCAB, H), vals(VOCAB * H, 1)),
        P + "norm.weight": ("BF16", (H,), [0.25, -0.5, 0.75, 1.0]),
        L0 + "input_layernorm.weight": ("BF16", (H,), [0.5, 0.25, -0.25, 0.125]),
        L0 + "post_attention_layernorm.weight": ("BF16", (H,), [0.125, 0.5, 0.75, -0.5]),
        L0 + "linear_attn.A_log": ("F32", (N_V,), [0.5, 1.0, 1.5, 2.0]),
        L0 + "linear_attn.dt_bias": ("BF16", (N_V,), [1.0, 2.0, 3.0, 4.0]),
        L0 + "linear_attn.conv1d.weight": ("BF16", (2 * N_K * KD + N_V * VD, 1, 4),
                                           vals(16 * 4, 2)),
        L0 + "linear_attn.in_proj_qkv.weight": ("BF16", (16, H), vals(16 * H, 3)),
        L0 + "linear_attn.in_proj_z.weight": ("BF16", (N_V * VD, H), vals(N_V * VD * H, 4)),
        L0 + "linear_attn.in_proj_a.weight": ("BF16", (N_V, H), vals(N_V * H, 5)),
        L0 + "linear_attn.in_proj_b.weight": ("BF16", (N_V, H), vals(N_V * H, 6)),
        L0 + "linear_attn.norm.weight": ("F32", (VD,), [0.75, 0.875]),
        L0 + "linear_attn.out_proj.weight": ("BF16", (H, N_V * VD), vals(H * N_V * VD, 7)),
        L0 + "mlp.gate_proj.weight": ("BF16", (6, H), vals(24, 8)),
        L0 + "mlp.up_proj.weight": ("BF16", (6, H), vals(24, 9)),
        L0 + "mlp.down_proj.weight": ("BF16", (H, 6), vals(24, 10)),
        L1 + "input_layernorm.weight": ("BF16", (H,), [0.5, -0.125, 0.25, 0.375]),
        L1 + "post_attention_layernorm.weight": ("BF16", (H,), [0.0, 0.5, 0.25, 0.125]),
        L1 + "self_attn.q_proj.weight": ("BF16", (8, H), vals(8 * H, 11)),
        L1 + "self_attn.k_proj.weight": ("BF16", (2, H), vals(2 * H, 12)),
        L1 + "self_attn.v_proj.weight": ("BF16", (2, H), vals(2 * H, 13)),
        L1 + "self_attn.o_proj.weight": ("BF16", (H, 8), vals(H * 8, 14)),
        L1 + "self_attn.q_norm.weight": ("BF16", (KD,), [0.25, 0.5]),
        L1 + "self_attn.k_norm.weight": ("BF16", (KD,), [-0.25, 0.125]),
        L1 + "mlp.gate_proj.weight": ("BF16", (6, H), vals(24, 15)),
        L1 + "mlp.up_proj.weight": ("BF16", (6, H), vals(24, 16)),
        L1 + "mlp.down_proj.weight": ("BF16", (H, 6), vals(24, 17)),
        "model.visual.patch_embed.proj.bias": ("BF16", (3,), [0.5, 1.5, -2.5]),
        "model.visual.merger.norm.weight": ("F32", (2,), [1.0, 2.0]),
        "mtp.fc.weight": ("BF16", (H, H), vals(16, 18)),
        "mtp.norm.weight": ("BF16", (H,), [0.5, 0.5, 0.25, 0.25]),
    }
    return t


def pack(dtype, values):
    return st.floats_to_bf16(values) if dtype == "BF16" else st.floats_to_f32(values)


def gguf_tensors(base):
    """The llama.cpp direction: HF floats -> (gguf name, gguf dims, type, floats)."""
    out = {}

    def put(name, shape, vs, f32=False):
        out[name] = (tuple(reversed(shape)), g.GGUF_F32 if f32 else g.GGUF_BF16, vs)

    def get(name):
        return base[name][1], base[name][2]

    for hf, gn in ((P + "embed_tokens.weight", "token_embd.weight"),
                   (P + "layers.0.mlp.gate_proj.weight", "blk.0.ffn_gate.weight"),
                   (P + "layers.0.mlp.up_proj.weight", "blk.0.ffn_up.weight"),
                   (P + "layers.0.mlp.down_proj.weight", "blk.0.ffn_down.weight"),
                   (P + "layers.1.mlp.gate_proj.weight", "blk.1.ffn_gate.weight"),
                   (P + "layers.1.mlp.up_proj.weight", "blk.1.ffn_up.weight"),
                   (P + "layers.1.mlp.down_proj.weight", "blk.1.ffn_down.weight"),
                   (L1 + "self_attn.q_proj.weight", "blk.1.attn_q.weight"),
                   (L1 + "self_attn.k_proj.weight", "blk.1.attn_k.weight"),
                   (L1 + "self_attn.v_proj.weight", "blk.1.attn_v.weight"),
                   (L1 + "self_attn.o_proj.weight", "blk.1.attn_output.weight"),
                   (L0 + "linear_attn.norm.weight", "blk.0.ssm_norm.weight")):
        shape, vs = get(hf)
        put(gn, shape, vs, f32=hf.endswith("linear_attn.norm.weight"))
    for hf, gn in ((P + "norm.weight", "output_norm.weight"),
                   (L0 + "input_layernorm.weight", "blk.0.attn_norm.weight"),
                   (L0 + "post_attention_layernorm.weight", "blk.0.post_attention_norm.weight"),
                   (L1 + "input_layernorm.weight", "blk.1.attn_norm.weight"),
                   (L1 + "post_attention_layernorm.weight", "blk.1.post_attention_norm.weight"),
                   (L1 + "self_attn.q_norm.weight", "blk.1.attn_q_norm.weight"),
                   (L1 + "self_attn.k_norm.weight", "blk.1.attn_k_norm.weight")):
        shape, vs = get(hf)
        put(gn, shape, [x + 1.0 for x in vs], f32=True)          # norms stored as w + 1
    shape, vs = get(L0 + "linear_attn.A_log")
    put("blk.0.ssm_a", shape, [-math.exp(x) for x in tile(vs)], f32=True)
    shape, vs = get(L0 + "linear_attn.dt_bias")
    put("blk.0.ssm_dt.bias", shape, tile(vs), f32=True)
    shape, vs = get(L0 + "linear_attn.conv1d.weight")       # squeezed, v channels tiled
    r = rows(vs, 4)
    put("blk.0.ssm_conv1d.weight", (shape[0], shape[2]),
        flat(r[:8] + flat(tile([r[8 + h * VD:8 + (h + 1) * VD] for h in range(N_V)]))), f32=True)
    shape, vs = get(L0 + "linear_attn.in_proj_qkv.weight")
    r = rows(vs, H)
    put("blk.0.attn_qkv.weight", shape,
        flat(r[:8]) + flat(flat(tile([r[8 + h * VD:8 + (h + 1) * VD] for h in range(N_V)]))))
    shape, vs = get(L0 + "linear_attn.in_proj_z.weight")
    r = rows(vs, H)
    put("blk.0.attn_gate.weight", shape,
        flat(flat(tile([r[h * VD:(h + 1) * VD] for h in range(N_V)]))))
    for hf, gn in ((L0 + "linear_attn.in_proj_a.weight", "blk.0.ssm_alpha.weight"),
                   (L0 + "linear_attn.in_proj_b.weight", "blk.0.ssm_beta.weight")):
        shape, vs = get(hf)
        put(gn, shape, flat(tile(rows(vs, H))))
    shape, vs = get(L0 + "linear_attn.out_proj.weight")     # v heads are the columns
    put("blk.0.ssm_out.weight", shape,
        flat(flat(tile([r[h * VD:(h + 1) * VD] for h in range(N_V)])) for r in rows(vs, N_V * VD)))
    return out


def gguf_bytes(meta, tensors, alignment=32):
    def s(x):
        b = x.encode()
        return struct.pack("<Q", len(b)) + b

    def kv(key, vtype, payload):
        return s(key) + struct.pack("<I", vtype) + payload

    kvs = [kv("general.architecture", 8, s("qwen35")),
           kv("tokenizer.ggml.tokens", 9, struct.pack("<IQ", 8, len(meta["tokens"]))
              + b"".join(s(t) for t in meta["tokens"])),
           kv("general.alignment", 4, struct.pack("<I", alignment))]
    infos, blobs, off = [], [], 0
    for name, (dims, ttype, raw) in tensors.items():
        infos.append(s(name) + struct.pack("<I", len(dims)) + struct.pack("<%dQ" % len(dims), *dims)
                     + struct.pack("<IQ", ttype, off))
        pad = b"\0" * ((-len(raw)) % alignment)
        blobs.append(raw + pad)
        off += len(raw) + len(pad)
    head = (b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kvs)) + b"".join(kvs)
            + b"".join(infos))
    return head + b"\0" * ((-len(head)) % alignment) + b"".join(blobs)


class Fixture:
    def __init__(self, root, tokens=None, gguf_edit=None, alignment=32):
        self.base_dir = os.path.join(root, "base")
        self.out_dir = os.path.join(root, "out")
        self.gguf_path = os.path.join(root, "m.gguf")
        os.makedirs(self.base_dir)
        self.base = base_tensors()
        st.write_file(os.path.join(self.base_dir, "model.safetensors"),
                      {n: (d, s, pack(d, v)) for n, (d, s, v) in self.base.items()})
        tj = {"model": {"vocab": {"a": 0, "b": 1, "c": 2, "Ġd": 4}},
              "added_tokens": [{"id": 3, "content": "<|x|>"}]}
        cfg = {"text_config": {"vocab_size": VOCAB, "linear_num_key_heads": N_K,
                               "linear_num_value_heads": N_V, "linear_key_head_dim": KD,
                               "linear_value_head_dim": VD}}
        for name, doc in (("config.json", cfg), ("tokenizer.json", tj),
                          ("tokenizer_config.json", {}), ("preprocessor_config.json", {})):
            with open(os.path.join(self.base_dir, name), "w") as f:
                json.dump(doc, f)
        with open(os.path.join(self.base_dir, "chat_template.jinja"), "w") as f:
            f.write("{{ x }}")
        self.gg = {n: (dims, t, pack("F32" if t == g.GGUF_F32 else "BF16", v))
                   for n, (dims, t, v) in gguf_tensors(self.base).items()}
        if gguf_edit:
            gguf_edit(self.gg)
        with open(self.gguf_path, "wb") as f:
            f.write(gguf_bytes({"tokens": tokens or TOKENS}, self.gg, alignment))

    def convert(self):
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            g.convert(self.base_dir, self.gguf_path, self.out_dir)
        return buf.getvalue()

    def out(self):
        return st.open_checkpoint(self.out_dir)


class Case(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def fixture(self, sub="f", **kw):
        root = os.path.join(self.tmp.name, sub)
        os.makedirs(root)
        return Fixture(root, **kw)


class NameMappingTests(Case):
    def test_every_gguf_tensor_maps_to_a_base_name(self):
        fx = self.fixture()
        base_names = set(fx.base)
        self.assertEqual({g.hf_name(n) for n in fx.gg}, {n for n in base_names
                                                        if not n.startswith(g.COPIED_PREFIXES)})

    def test_specific_names(self):
        self.assertEqual(g.hf_name("token_embd.weight"), P + "embed_tokens.weight")
        self.assertEqual(g.hf_name("output_norm.weight"), P + "norm.weight")
        self.assertEqual(g.hf_name("blk.17.ssm_dt.bias"), P + "layers.17.linear_attn.dt_bias")
        self.assertEqual(g.hf_name("blk.3.attn_output.weight"),
                         P + "layers.3.self_attn.o_proj.weight")

    def test_unknown_gguf_tensor_is_rejected(self):
        for name in ("output.weight", "blk.0.mystery.weight", "rope_freqs.weight"):
            with self.assertRaisesRegex(g.ConvertError, "no HF name"):
                g.hf_name(name)


class ReaderTests(Case):
    def test_dims_are_reversed_to_hf_shape(self):
        fx = self.fixture()
        gg = g.Gguf(fx.gguf_path)
        self.assertEqual(gg.tensors["blk.0.ffn_down.weight"][0], (6, H))
        self.assertEqual(gg.hf_shape("blk.0.ffn_down.weight"), (H, 6))
        self.assertEqual(gg.hf_shape("blk.0.ssm_conv1d.weight"), (16, 4))

    def test_alignment_comes_from_the_metadata(self):
        fx = self.fixture(alignment=64)
        gg = g.Gguf(fx.gguf_path)
        self.assertEqual(gg.data_start % 64, 0)
        fx.convert()
        out = fx.out()
        self.assertEqual(out.floats(P + "embed_tokens.weight"), fx.base[P + "embed_tokens.weight"][2])

    def test_unsupported_dtype_is_rejected(self):
        def edit(gg):
            dims, _, raw = gg["blk.0.ffn_up.weight"]
            gg["blk.0.ffn_up.weight"] = (dims, 1, raw)          # GGML_TYPE_F16
        fx = self.fixture(gguf_edit=edit)
        with self.assertRaisesRegex(g.ConvertError, "unsupported GGUF type 1"):
            g.Gguf(fx.gguf_path)
        with self.assertRaises(g.ConvertError):
            fx.convert()

    def test_not_gguf(self):
        path = os.path.join(self.tmp.name, "x.gguf")
        with open(path, "wb") as f:
            f.write(b"nope" * 8)
        with self.assertRaisesRegex(g.ConvertError, "not a GGUF"):
            g.Gguf(path)


class ConvertTests(Case):
    def setUp(self):
        super().setUp()
        self.fx = self.fixture()
        self.summary = self.fx.convert()
        self.out = self.fx.out()

    def want(self, name):
        dtype, shape, v = self.fx.base[name]
        return dtype, shape, pack(dtype, v)

    def gguf_floats(self, gname):
        dims, t, raw = self.fx.gg[gname]
        return g.to_floats("F32" if t == g.GGUF_F32 else "BF16", raw)

    def assertSame(self, name):
        dtype, shape, raw = self.want(name)
        self.assertEqual((self.out.dtype(name), tuple(self.out.shape(name))), (dtype, shape))
        self.assertEqual(self.out.raw(name), raw, name)

    def test_same_names_shapes_and_dtypes_as_base(self):
        self.assertEqual(self.out.names(), sorted(self.fx.base))
        self.assertFalse([n for n in self.out.names() if "lm_head" in n])
        for name in self.fx.base:
            if not name.endswith("A_log"):          # log(exp(x)) is only close, see below
                self.assertSame(name)

    def test_plain_tensors_pass_through(self):
        for name in (P + "embed_tokens.weight", L0 + "mlp.down_proj.weight",
                     L1 + "self_attn.q_proj.weight", L1 + "self_attn.o_proj.weight",
                     L0 + "linear_attn.norm.weight"):
            self.assertSame(name)

    def test_norm_plus_one_is_undone(self):
        for gn, name in (("blk.0.attn_norm.weight", L0 + "input_layernorm.weight"),
                         ("output_norm.weight", P + "norm.weight"),
                         ("blk.1.attn_q_norm.weight", L1 + "self_attn.q_norm.weight")):
            self.assertNotEqual(self.gguf_floats(gn), self.fx.base[name][2])   # skipping fails
            self.assertSame(name)

    def test_gated_norm_is_not_shifted(self):
        self.assertEqual(self.gguf_floats("blk.0.ssm_norm.weight"),
                         self.fx.base[L0 + "linear_attn.norm.weight"][2])
        self.assertSame(L0 + "linear_attn.norm.weight")

    def test_a_log_is_log_of_negated_ssm_a_and_untiled(self):
        name = L0 + "linear_attn.A_log"
        self.assertTrue(all(x < 0 for x in self.gguf_floats("blk.0.ssm_a")))
        self.assertNotEqual([math.log(-x) for x in self.gguf_floats("blk.0.ssm_a")],
                            self.fx.base[name][2])                              # no tile undo
        got, want = self.out.floats(name), self.fx.base[name][2]
        self.assertEqual(self.out.dtype(name), "F32")
        for a, b in zip(got, want):
            self.assertAlmostEqual(a, b, places=5)

    def test_dt_bias_is_untiled(self):
        name = L0 + "linear_attn.dt_bias"
        self.assertNotEqual(self.gguf_floats("blk.0.ssm_dt.bias"), self.fx.base[name][2])
        self.assertSame(name)

    def test_conv1d_is_unsqueezed_and_untiled(self):
        name = L0 + "linear_attn.conv1d.weight"
        self.assertEqual(self.fx.gg["blk.0.ssm_conv1d.weight"][0], (4, 16))
        self.assertEqual(tuple(self.out.shape(name)), (16, 1, 4))
        self.assertNotEqual(self.gguf_floats("blk.0.ssm_conv1d.weight"), self.fx.base[name][2])
        self.assertSame(name)

    def test_value_head_reorder_of_projections(self):
        for gn, name in (("blk.0.attn_qkv.weight", L0 + "linear_attn.in_proj_qkv.weight"),
                         ("blk.0.attn_gate.weight", L0 + "linear_attn.in_proj_z.weight"),
                         ("blk.0.ssm_alpha.weight", L0 + "linear_attn.in_proj_a.weight"),
                         ("blk.0.ssm_beta.weight", L0 + "linear_attn.in_proj_b.weight"),
                         ("blk.0.ssm_out.weight", L0 + "linear_attn.out_proj.weight")):
            self.assertNotEqual(self.gguf_floats(gn), self.fx.base[name][2], gn)
            self.assertSame(name)

    def test_qkv_keeps_q_and_k_rows_in_place(self):
        name = L0 + "linear_attn.in_proj_qkv.weight"
        self.assertEqual(self.gguf_floats("blk.0.attn_qkv.weight")[:8 * H],
                         self.fx.base[name][2][:8 * H])

    def test_tensors_missing_from_the_gguf_are_copied_from_base(self):
        for name in ("model.visual.patch_embed.proj.bias", "model.visual.merger.norm.weight",
                     "mtp.fc.weight", "mtp.norm.weight"):
            self.assertSame(name)
        self.assertIn("copied 4 from the base (2 model.visual, 2 mtp)", self.summary)
        self.assertIn("converted 27 tensors", self.summary)

    def test_support_files_are_copied(self):
        for name in g.COPIED_FILES:
            with open(os.path.join(self.fx.base_dir, name), "rb") as a, \
                    open(os.path.join(self.fx.out_dir, name), "rb") as b:
                self.assertEqual(a.read(), b.read(), name)

    def test_missing_text_tensor_is_rejected(self):
        def edit(gg):
            del gg["blk.1.ffn_up.weight"]
        fx = self.fixture("m", gguf_edit=edit)
        with self.assertRaisesRegex(g.ConvertError, "lacks 1 text tensors"):
            fx.convert()

    def test_shape_mismatch_is_rejected(self):
        def edit(gg):
            dims, t, raw = gg["blk.1.ffn_up.weight"]
            gg["blk.1.ffn_up.weight"] = (tuple(reversed(dims)), t, raw)
        fx = self.fixture("m", gguf_edit=edit)
        with self.assertRaisesRegex(g.ConvertError, "does not match base"):
            fx.convert()


class TokenizerTests(Case):
    def test_differing_token_is_rejected(self):
        fx = self.fixture(tokens=["a", "b", "X", "<|x|>", "Ġd", "[PAD5]"])
        with self.assertRaisesRegex(g.ConvertError, "tokenizer mismatch.*first id 2"):
            fx.convert()
        self.assertFalse(os.path.exists(os.path.join(fx.out_dir, "model.safetensors")))

    def test_wrong_token_count_is_rejected(self):
        fx = self.fixture(tokens=TOKENS[:5])
        with self.assertRaisesRegex(g.ConvertError, "GGUF has 5 tokens"):
            fx.convert()


class CheckTests(Case):
    def test_converted_model_passes_and_prints_a_table(self):
        fx = self.fixture()
        fx.convert()
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            self.assertTrue(g.check(fx.base_dir, fx.out_dir))
        self.assertIn("checked 27 text tensors, min cosine 1.0000", buf.getvalue())
        self.assertNotIn("FAIL", buf.getvalue())

    def test_a_skipped_transform_fails_the_check(self):
        fx = self.fixture()
        fx.convert()
        name = L0 + "linear_attn.in_proj_z.weight"
        ck = fx.out()
        dtype, shape = ck.dtype(name), ck.shape(name)
        tensors = {n: (ck.dtype(n), ck.shape(n), ck.raw(n)) for n in ck.names()}
        tensors[name] = (dtype, shape, fx.gg["blk.0.attn_gate.weight"][2])   # still tiled
        del ck
        st.write_file(os.path.join(fx.out_dir, "model.safetensors"), tensors)
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            self.assertFalse(g.check(fx.base_dir, fx.out_dir))
        self.assertIn("FAIL", buf.getvalue())

    def test_main_exit_codes(self):
        fx = self.fixture()
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(g.main(["--base", fx.base_dir, "--gguf", fx.gguf_path,
                                     "--out", fx.out_dir]), 0)
            self.assertEqual(g.main(["--base", fx.base_dir, "--out", fx.out_dir, "--check"]), 0)
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(g.main(["--base", fx.base_dir, "--gguf", fx.base_dir + "/config.json",
                                     "--out", fx.out_dir + "2"]), 2)


if __name__ == "__main__":
    unittest.main()
