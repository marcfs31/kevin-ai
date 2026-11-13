#!/usr/bin/env python3
"""Unit tests for the oracle, quantiser reference, fixtures and the golden checker.

    python3 -m unittest discover -s tools -p 'test_*.py' -v
"""

import hashlib
import json
import math
import os
import random
import stat
import struct
import subprocess
import sys
import tempfile
import textwrap
import unittest

TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS)

import check_golden  # noqa: E402
import make_ops  # noqa: E402
import make_tiny  # noqa: E402
import quant_ref  # noqa: E402
import ref_qwen35 as ref  # noqa: E402
import safetensors as st  # noqa: E402

ROOT = os.path.dirname(TOOLS)
TINY = os.path.join(ROOT, "tests", "fixtures", "qwen35-tiny")
OPS = os.path.join(ROOT, "tests", "fixtures", "ops")

REAL_TEXT_KEYS = [
    "attention_bias", "attention_dropout", "attn_output_gate", "dtype", "eos_token_id",
    "full_attention_interval", "head_dim", "hidden_act", "hidden_size", "initializer_range",
    "intermediate_size", "layer_types", "linear_conv_kernel_dim", "linear_key_head_dim",
    "linear_num_key_heads", "linear_num_value_heads", "linear_value_head_dim",
    "max_position_embeddings", "mlp_only_layers", "model_type", "mtp_num_hidden_layers",
    "mtp_use_dedicated_embeddings", "num_attention_heads", "num_hidden_layers",
    "num_key_value_heads", "rms_norm_eps", "tie_word_embeddings", "use_cache", "vocab_size",
    "mamba_ssm_dtype", "rope_parameters"]


def jload(path):
    with open(path) as f:
        return json.load(f)


def rbytes(path):
    with open(path, "rb") as f:
        return f.read()


def raw_file(header, data=b""):
    h = header if isinstance(header, bytes) else json.dumps(header).encode()
    return struct.pack("<Q", len(h)) + h + data


def entry(dtype, shape, lo, hi):
    return {"dtype": dtype, "shape": shape, "data_offsets": [lo, hi]}


class SafetensorsTests(unittest.TestCase):
    def test_round_trip_all_dtypes_and_metadata(self):
        t = {"a": ("BF16", (2, 2), st.floats_to_bf16([1.0, -2.5, 0.0, 3.0])),
             "b": ("F32", (3,), st.floats_to_f32([0.5, -1.25, 1e-3])),
             "c": ("I8", (4,), struct.pack("<4b", -127, 0, 5, 127)),
             "d": ("U8", (2, 1), b"\x00\xff")}
        blob = st.serialize(t, metadata={"format": "x", "k": "v"})
        p = st.parse(blob)
        self.assertEqual(p.names(), ["a", "b", "c", "d"])
        self.assertEqual(p.metadata, {"format": "x", "k": "v"})
        self.assertEqual(p.floats("a"), [1.0, -2.5, 0.0, 3.0])
        self.assertEqual(p.floats("b"), st.f32_to_floats(st.floats_to_f32([0.5, -1.25, 1e-3])))
        self.assertEqual(p.ints("c"), [-127, 0, 5, 127])
        self.assertEqual(p.ints("d"), [0, 255])
        self.assertEqual(p.shape("a"), (2, 2))
        self.assertEqual(st.serialize({n: (p.dtype(n), p.shape(n), p.raw(n)) for n in p.names()},
                                      metadata=p.metadata), blob)

    def test_header_formatting_is_exact(self):
        blob = st.serialize({"b": ("U8", (1,), b"\x07"), "a": ("U8", (2,), b"\x01\x02")},
                            metadata={"format": "x"})
        header = ('{"__metadata__":{"format":"x"},'
                  '"a":{"dtype":"U8","shape":[2],"data_offsets":[0,2]},'
                  '"b":{"dtype":"U8","shape":[1],"data_offsets":[2,3]}}').encode()
        header += b" " * ((-len(header)) % 8)
        self.assertEqual(blob, struct.pack("<Q", len(header)) + header + b"\x01\x02\x07")
        self.assertEqual(len(header) % 8, 0)

    def test_sharded_checkpoint_reads_all_tensors(self):
        c = st.open_checkpoint(TINY)
        wm = jload(os.path.join(TINY, "model.safetensors.index.json"))["weight_map"]
        self.assertEqual(c.names(), sorted(wm))
        self.assertEqual(len(set(wm.values())), 2)
        self.assertEqual(c.shape("model.language_model.embed_tokens.weight"), (384, 64))
        self.assertEqual(c.dtype("model.language_model.layers.0.linear_attn.A_log"), "F32")
        self.assertTrue(any(n.startswith("model.visual.") for n in c.names()))
        self.assertTrue(any(n.startswith("mtp.") for n in c.names()))
        self.assertEqual(c.source_sha256, hashlib.sha256(
            rbytes(os.path.join(TINY, "model.safetensors.index.json"))).hexdigest())

    def test_single_file_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            st.write_file(os.path.join(d, "model.safetensors"), {"x": ("U8", (1,), b"\x09")})
            self.assertEqual(st.open_checkpoint(d).raw("x"), b"\x09")
            self.assertEqual(st.open_checkpoint(os.path.join(d, "model.safetensors")).raw("x"), b"\x09")

    def test_rejects_malformed(self):
        bad = {
            "empty": b"",
            "short": b"\x01\x00",
            "length_past_eof": struct.pack("<Q", 1000) + b"{}",
            "huge_length": struct.pack("<Q", 1 << 40) + b"{}",
            "not_json": raw_file(b"{nope"),
            "not_object": raw_file(b"[1]"),
            "bad_utf8": raw_file(b'{"a":"\xff"}'),
            "duplicate_key": raw_file(b'{"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},'
                                      b'"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}', b"\x00"),
            "unknown_dtype": raw_file({"a": entry("F16", [1], 0, 2)}, b"\x00\x00"),
            "size_mismatch": raw_file({"a": entry("F32", [2], 0, 4)}, b"\x00" * 4),
            "negative_shape": raw_file({"a": entry("U8", [-1], 0, 0)}),
            "bool_shape": raw_file({"a": entry("U8", [True], 0, 1)}, b"\x00"),
            "reversed_offsets": raw_file({"a": entry("U8", [0], 1, 0)}, b"\x00"),
            "missing_key": raw_file({"a": {"dtype": "U8", "shape": [1]}}, b"\x00"),
            "extra_key": raw_file({"a": dict(entry("U8", [1], 0, 1), x=1)}, b"\x00"),
            "overlap": raw_file({"a": entry("U8", [2], 0, 2), "b": entry("U8", [2], 1, 3)}, b"\x00" * 3),
            "hole": raw_file({"a": entry("U8", [1], 0, 1), "b": entry("U8", [1], 2, 3)}, b"\x00" * 3),
            "uncovered_tail": raw_file({"a": entry("U8", [1], 0, 1)}, b"\x00\x00"),
            "data_past_eof": raw_file({"a": entry("U8", [4], 0, 4)}, b"\x00"),
            "bad_metadata": raw_file({"__metadata__": {"k": 1}}),
            "metadata_not_object": raw_file({"__metadata__": []}),
        }
        for name, blob in bad.items():
            with self.subTest(name):
                with self.assertRaises(st.SafetensorsError):
                    st.parse(blob)

    def test_writer_rejects_bad_input(self):
        with self.assertRaises(st.SafetensorsError):
            st.serialize({"a": ("F16", (1,), b"\x00\x00")})
        with self.assertRaises(st.SafetensorsError):
            st.serialize({"a": ("F32", (2,), b"\x00")})

    def test_index_with_wrong_shard_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            st.write_file(os.path.join(d, "s.safetensors"), {"x": ("U8", (1,), b"\x00")})
            for wm in ({"y": "s.safetensors"}, {"x": "../s.safetensors"}, {"x": "missing.safetensors"}):
                with open(os.path.join(d, "model.safetensors.index.json"), "w") as f:
                    json.dump({"weight_map": wm}, f)
                with self.subTest(wm):
                    with self.assertRaises((st.SafetensorsError, OSError)):
                        st.open_checkpoint(d)

    def test_bf16_helpers(self):
        self.assertEqual(st.round_bf16(1.0), 1.0)
        self.assertEqual(st.round_bf16(1.00390625), 1.0)           # exactly halfway -> even
        self.assertEqual(st.round_bf16(1.01171875), 1.015625)      # halfway above -> even (up)
        self.assertEqual(st.float_to_bf16_bits(-0.0), 0x8000)
        self.assertEqual(st.bf16_to_floats(b"\x80\x3f"), [1.0])


class QuantTests(unittest.TestCase):
    def test_ties_round_half_even(self):
        x = [127.0, 0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 126.5, 125.5] + [0.0] * 23
        scale, q = quant_ref.quantize_group(x)
        self.assertEqual(scale, 1.0)
        self.assertEqual(q[:9], [127, 0, 2, 2, 0, -2, -2, 126, 126])
        self.assertEqual(q[9:], [0] * 23)

    def test_zero_and_negative_zero_groups(self):
        for g in ([0.0] * 32, [-0.0] * 32):
            self.assertEqual(quant_ref.quantize_group(g), (0.0, [0] * 32))
        scale, q = quant_ref.quantize_group([-0.0, 1.0, -0.0, -2.0] + [0.5] * 28)
        self.assertEqual(q[0], 0)
        self.assertEqual(q[2], 0)
        self.assertEqual(q[3], -127)
        self.assertGreater(scale, 0.0)

    def test_scale_rounded_once_to_f32_from_double(self):
        scale, q = quant_ref.quantize_group([1.0] + [0.0] * 31)
        self.assertEqual(scale, struct.unpack("<f", struct.pack("<f", 1.0 / 127.0))[0])
        self.assertNotEqual(scale, 1.0 / 127.0)
        self.assertEqual(q[0], 127)

    def test_extremes_hit_clamp_and_never_minus_128(self):
        rng = random.Random(3)
        for _ in range(200):
            g = [st.round_bf16(rng.gauss(0, 10 ** rng.uniform(-3, 3))) for _ in range(32)]
            scale, q = quant_ref.quantize_group(g)
            self.assertTrue(all(-127 <= v <= 127 for v in q))
            if scale:
                i = max(range(32), key=lambda j: abs(g[j]))
                self.assertEqual(abs(q[i]), 127)
                for v, qv in zip(g, q):
                    self.assertLessEqual(abs(v - qv * scale), scale * 0.5 * (1 + 1e-6))

    def test_bf16_subnormal_scale_is_exact(self):
        tiny = st.bf16_to_floats(b"\x01\x00")[0]
        scale, q = quant_ref.quantize_group([tiny] + [0.0] * 31)
        self.assertGreater(scale, 0.0)
        self.assertEqual(q[0], 127)

    def test_non_finite_is_rejected(self):
        for bad in (float("nan"), float("inf")):
            with self.assertRaises(quant_ref.QuantError):
                quant_ref.quantize_group([bad] + [1.0] * 31)

    def test_in_dimension_must_be_multiple_of_32(self):
        with self.assertRaises(quant_ref.QuantError):
            quant_ref.quantize_matrix([0.0] * 48, 1, 48)
        store = st.MemoryCheckpoint({"model.language_model.w.weight":
                                     ("BF16", (1, 48), st.floats_to_bf16([1.0] * 48))})
        with self.assertRaises(quant_ref.QuantError):
            quant_ref.quantize_store(store, "x")

    def test_quantised_set_copies_and_layout(self):
        q8 = quant_ref.load_q8(os.path.join(TINY, "model.q8.safetensors"))
        src = st.open_checkpoint(TINY)
        text = [n for n in src.names() if n.startswith("model.language_model.")]
        self.assertFalse(any(n.startswith(("model.visual.", "mtp.")) for n in q8.names()))
        self.assertEqual(q8.metadata, {"format": "lm-q8", "group": "32", "source": src.source_sha256})
        self.assertEqual(list(q8.metadata), ["format", "group", "source"])
        self.assertEqual(q8.names(), sorted(q8.info))
        offsets = [q8.info[n][2] for n in q8.names()]
        self.assertEqual(offsets, sorted(offsets))
        for n in text:
            shape = src.shape(n)
            if n.endswith(".weight") and len(shape) == 2:
                self.assertEqual(q8.dtype(n), "I8")
                self.assertEqual(q8.shape(n), shape)
                self.assertEqual(q8.dtype(n + ".q8scale"), "F32")
                self.assertEqual(q8.shape(n + ".q8scale"), (shape[0], shape[1] // 32))
            else:
                self.assertEqual((q8.dtype(n), q8.shape(n), q8.raw(n)),
                                 (src.dtype(n), shape, src.raw(n)), n)
        self.assertEqual(len(q8.names()), len(text) + sum(
            1 for n in text if n.endswith(".weight") and len(src.shape(n)) == 2))
        header_len = struct.unpack("<Q", bytes(q8.buf[:8]))[0]
        self.assertEqual(header_len % 8, 0)

    def test_rebuild_is_byte_identical_and_deterministic(self):
        with tempfile.TemporaryDirectory() as d:
            a = quant_ref.quantize_checkpoint(TINY, os.path.join(d, "a"))
            b = quant_ref.quantize_checkpoint(TINY, os.path.join(d, "b"))
        self.assertEqual(a, b)
        self.assertEqual(a, rbytes(os.path.join(TINY, "model.q8.safetensors")))

    def test_dequantise_matches_formula(self):
        src = st.open_checkpoint(TINY)
        name = "model.language_model.layers.3.self_attn.q_proj.weight"
        q8 = quant_ref.load_q8(os.path.join(TINY, "model.q8.safetensors"))
        rows, cols = src.shape(name)
        deq = quant_ref.dequantize_matrix(q8.ints(name), q8.floats(name + ".q8scale"), rows, cols)
        orig = src.floats(name)
        scales = q8.floats(name + ".q8scale")
        for i, (a, b) in enumerate(zip(orig, deq)):
            self.assertLessEqual(abs(a - b), scales[i // 32] * 0.5 * (1 + 1e-6))

    def test_no_text_tensor_is_an_error(self):
        store = st.MemoryCheckpoint({"mtp.fc.weight": ("BF16", (1, 32), st.floats_to_bf16([1.0] * 32))})
        with self.assertRaises(quant_ref.QuantError):
            quant_ref.quantize_store(store, "x")


class OpTests(unittest.TestCase):
    def test_rmsnorm_zero_centred_hand_values(self):
        y = ref.rmsnorm_zero_centred([3.0, 4.0], [0.0, 1.0], 0.0)
        self.assertAlmostEqual(y[0], 0.8485281374238570, places=12)   # 3 / sqrt(12.5)
        self.assertAlmostEqual(y[1], 2.2627416997969520, places=12)   # 4 / sqrt(12.5) * 2

    def test_rmsnorm_gated_uses_plain_weight_and_silu_gate(self):
        y = ref.rmsnorm_gated([3.0, 4.0], [0.0, 100.0], [2.0, 3.0], 0.0)
        self.assertEqual(y[0], 0.0)
        self.assertAlmostEqual(y[1], 339.4112549695428, delta=1e-9)
        # plain w: w=1 and z large leaves silu(z)*normalised x (NOT 2x as 1+w would give)
        y = ref.rmsnorm_gated([3.0, 4.0], [50.0, 50.0], [1.0, 1.0], 0.0)
        self.assertAlmostEqual(y[0], 50.0 * 0.8485281374238570, places=9)

    def test_l2norm_hand_values(self):
        y = ref.l2norm([3.0, 4.0], 0.0)
        self.assertAlmostEqual(y[0], 0.6, places=15)
        self.assertAlmostEqual(y[1], 0.8, places=15)
        self.assertEqual(ref.l2norm([0.0] * 4), [0.0] * 4)
        self.assertAlmostEqual(ref.l2norm([3.0, 4.0])[0], 3.0 / math.sqrt(25.000001), places=15)

    def test_rope_hand_values(self):
        y = ref.rope_apply([1.0, 0.0, 0.0, 0.0], 1, 4, 10000.0)
        self.assertAlmostEqual(y[0], math.cos(1.0), places=15)
        self.assertAlmostEqual(y[2], math.sin(1.0), places=15)
        y = ref.rope_apply([0.0, 1.0, 0.0, 0.0], 1, 4, 10000.0)
        self.assertAlmostEqual(y[1], math.cos(0.01), places=15)
        self.assertAlmostEqual(y[3], math.sin(0.01), places=15)
        x = [0.3, -1.0, 0.7, 2.0]
        self.assertEqual(ref.rope_apply(x, 0, 4, 10000.0), x)

    def test_partial_rope_leaves_tail_and_preserves_norm(self):
        rng = random.Random(1)
        x = [rng.gauss(0, 1) for _ in range(32)]
        y = ref.rope_apply(x, 37, 8, 1e7)
        self.assertEqual(y[8:], x[8:])
        self.assertNotEqual(y[:8], x[:8])
        self.assertAlmostEqual(sum(v * v for v in y[:8]), sum(v * v for v in x[:8]), places=12)

    def test_inv_freq_matches_formula(self):
        f = ref.rope_inv_freq(64, 1e7)
        self.assertEqual(len(f), 32)
        self.assertEqual(f[0], 1.0)
        self.assertAlmostEqual(f[1], 1e7 ** (-2.0 / 64), places=15)

    def test_conv1d_hand_values(self):
        weight = [[1.0, 2.0, 3.0]]
        state = [[0.0], [0.0]]
        ys = [ref.conv1d_silu_step(state, [x], weight)[0] for x in (1.0, 0.0, 0.0, 0.0)]
        expect = [ref.silu(3.0), ref.silu(2.0), ref.silu(1.0), 0.0]
        for a, b in zip(ys, expect):
            self.assertAlmostEqual(a, b, places=15)

    def test_stable_sigmoid_softplus(self):
        self.assertEqual(ref.sigmoid(-800.0), 0.0)
        self.assertEqual(ref.sigmoid(800.0), 1.0)
        self.assertAlmostEqual(ref.softplus(800.0), 800.0, places=9)
        self.assertAlmostEqual(ref.softplus(0.0), math.log(2.0), places=15)
        self.assertAlmostEqual(ref.softplus(-40.0), math.exp(-40.0), places=30)

    def test_delta_rule_matches_matrix_form(self):
        # S_t = (I - beta k k^T) (exp(g) S_{t-1}) + beta k v^T ; o_t = S_t^T q  (independent formulation)
        rng = random.Random(5)
        dk, dv, T = 3, 4, 6
        steps = [([rng.gauss(0, 1) for _ in range(dk)], [rng.gauss(0, 1) for _ in range(dk)],
                  [rng.gauss(0, 1) for _ in range(dv)], [-rng.uniform(0.05, 1.5)], [rng.uniform(0.1, 0.9)])
                 for _ in range(T)]
        outs, _ = ref.gated_delta_rule(steps, 1, 1, dk, dv)
        S = [[0.0] * dv for _ in range(dk)]
        for (q, k, v, gs, bs), o in zip(steps, outs):
            g, beta = gs[0], bs[0]
            qn = [c / math.sqrt(dk) for c in ref.l2norm(q)]
            kn = ref.l2norm(k)
            a = math.exp(g)
            decayed = [[a * c for c in row] for row in S]
            proj = [[sum((1.0 if i == j else 0.0) * decayed[j][m] - beta * kn[i] * kn[j] * decayed[j][m]
                         for j in range(dk)) for m in range(dv)] for i in range(dk)]
            S = [[proj[i][m] + beta * kn[i] * v[m] for m in range(dv)] for i in range(dk)]
            want = [sum(S[i][m] * qn[i] for i in range(dk)) for m in range(dv)]
            for a_, b_ in zip(o, want):
                self.assertAlmostEqual(a_, b_, places=12)

    def test_delta_rule_gqa_mapping_is_consecutive(self):
        rng = random.Random(6)
        nk, nv, dk, dv, T = 2, 4, 5, 3, 4
        seq = [([rng.gauss(0, 1) for _ in range(nk * dk)], [rng.gauss(0, 1) for _ in range(nk * dk)],
                [rng.gauss(0, 1) for _ in range(nv * dv)], [-rng.uniform(0.1, 1) for _ in range(nv)],
                [rng.uniform(0.2, 0.8) for _ in range(nv)]) for _ in range(T)]
        full, _ = ref.gated_delta_rule(seq, nk, nv, dk, dv)
        for i in range(nv):
            kh = i // (nv // nk)  # value heads 0,1 -> key head 0; 2,3 -> key head 1
            single = [(q[kh * dk:(kh + 1) * dk], k[kh * dk:(kh + 1) * dk], v[i * dv:(i + 1) * dv],
                       [g[i]], [b[i]]) for q, k, v, g, b in seq]
            one, _ = ref.gated_delta_rule(single, 1, 1, dk, dv)
            for t in range(T):
                self.assertEqual(full[t][i * dv:(i + 1) * dv], one[t])

    def test_delta_rule_state_carries_information(self):
        dk = dv = 4

        def seq(first_v, beta=0.6):
            out = []
            for t in range(3):
                r = random.Random(100 + t)
                out.append(([r.gauss(0, 1) for _ in range(dk)], [r.gauss(0, 1) for _ in range(dk)],
                            first_v if t == 0 else [0.5] * dv, [-0.1], [beta]))
            return out
        a, _ = ref.gated_delta_rule(seq([1.0, 2.0, 3.0, 4.0]), 1, 1, dk, dv)
        b, _ = ref.gated_delta_rule(seq([-1.0, 0.0, 9.0, 4.0]), 1, 1, dk, dv)
        self.assertNotEqual(a[2], b[2])   # step 2 depends on step 0's value only through the state
        z, state = ref.gated_delta_rule(seq([1.0, 2.0, 3.0, 4.0], beta=0.0), 1, 1, dk, dv)
        self.assertTrue(all(v == 0.0 for row in z for v in row))      # beta = 0 never writes the state
        self.assertTrue(all(v == 0.0 for row in state[0] for v in row))

    def test_softmax_and_argmax_tie_break(self):
        self.assertAlmostEqual(sum(ref.softmax([1.0, 2.0, 3.0])), 1.0, places=15)
        self.assertEqual(ref.argmax([1.0, 5.0, 5.0, 2.0]), 1)
        self.assertEqual(ref.argmax([5.0, 5.0]), 0)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = ref.Model.load(TINY)
        with open(os.path.join(TINY, "golden_bf16.json")) as f:
            cls.gold = json.load(f)
        cls.tokens = cls.gold["tokens"]

    def test_causality_prefix_rows_equal_full_run(self):
        full = self.model.forward(self.tokens)
        for t in (0, 3, 9, 17):
            self.assertEqual(self.model.forward(self.tokens[:t + 1])[t], full[t])

    def test_later_tokens_do_not_change_earlier_rows(self):
        a = self.model.forward(self.tokens[:6] + [5])
        b = self.model.forward(self.tokens[:6] + [250, 7])
        self.assertEqual(a[:6], b[:6])
        self.assertNotEqual(a[6], b[6][:])

    def test_generate_equals_recompute_of_growing_prefix(self):
        prompt = self.tokens[:5]
        _, ids, _ = self.model.generate(prompt, 4)
        seq = list(prompt)
        for want in ids:
            nxt = ref.argmax(self.model.forward(seq)[-1])
            self.assertEqual(nxt, want)
            seq.append(nxt)

    def test_delta_state_matters_for_the_model(self):
        base = self.model.forward(self.tokens[:10])
        orig = ref.gated_delta_rule
        try:
            ref.gated_delta_rule = lambda seq, *a, **k: orig(
                [(q, kk, v, g, [0.0] * len(b)) for q, kk, v, g, b in seq], *a, **k)
            zeroed = ref.Model.load(TINY).forward(self.tokens[:10])
        finally:
            ref.gated_delta_rule = orig
        diff = max(abs(x - y) for r1, r2 in zip(base, zeroed) for x, y in zip(r1, r2))
        self.assertGreater(diff, 0.5)

    def test_decay_matters_for_the_model(self):
        base = self.model.forward(self.tokens[:10])
        m = ref.Model.load(TINY)
        for i, lt in enumerate(m.cfg.layer_types):
            if lt == "linear_attention":
                m.w._cache["%slayers.%d.linear_attn.A_log" % (ref.PREFIX, i)] = [8.0] * m.cfg.lv_heads
        other = m.forward(self.tokens[:10])
        diff = max(abs(x - y) for r1, r2 in zip(base, other) for x, y in zip(r1, r2))
        self.assertGreater(diff, 0.5)

    def test_goldens_reproduce(self):
        for name, weights in (("golden_bf16.json", None), ("golden_q8.json", "model.q8.safetensors")):
            with self.subTest(name):
                gold = jload(os.path.join(TINY, name))
                m = ref.Model.load(TINY, os.path.join(TINY, weights) if weights else None)
                rows, ids, steps = m.generate(gold["greedy"]["prompt"], len(gold["greedy"]["out"]))
                full = m.forward(gold["tokens"])
                self.assertEqual(ids, gold["greedy"]["out"])
                for r, g in zip(full, gold["logits"]):
                    for a, b in zip(r, g):
                        self.assertAlmostEqual(a, b, delta=1e-7 * max(1.0, abs(a)))
                for r in full + steps:
                    self.assertGreater(ref.top_gap(r), 1e-2)
                self.assertEqual(len(gold["tokens"]), 24)
                self.assertEqual((len(gold["greedy"]["prompt"]), len(gold["greedy"]["out"])), (8, 16))

    def test_q8_golden_differs_from_bf16(self):
        a = jload(os.path.join(TINY, "golden_bf16.json"))["logits"]
        b = jload(os.path.join(TINY, "golden_q8.json"))["logits"]
        top = max(abs(v) for r in a for v in r)
        diff = max(abs(x - y) for r1, r2 in zip(a, b) for x, y in zip(r1, r2))
        self.assertGreater(diff, 3 * 2e-4 * top)

    def test_activations_are_not_degenerate(self):
        rows = self.gold["logits"]
        self.assertGreaterEqual(len({ref.argmax(r) for r in rows}), 8)
        for r in rows:
            mean = sum(r) / len(r)
            std = math.sqrt(sum((v - mean) ** 2 for v in r) / len(r))
            self.assertGreater(std, 1.0)
        self.assertNotEqual(rows[0], rows[1])

    def test_config_has_real_schema(self):
        cfg = jload(os.path.join(TINY, "config.json"))
        self.assertEqual(cfg["model_type"], "qwen3_5")
        self.assertEqual(cfg["text_config"]["model_type"], "qwen3_5_text")
        self.assertEqual(list(cfg["text_config"]), REAL_TEXT_KEYS)
        t = cfg["text_config"]
        self.assertEqual(t["layer_types"], (["linear_attention"] * 3 + ["full_attention"]) * 2)
        self.assertEqual(t["full_attention_interval"], 4)
        self.assertEqual(t["rope_parameters"]["rope_theta"], 10000000)
        self.assertEqual(int(t["head_dim"] * t["rope_parameters"]["partial_rotary_factor"]), 8)

    def test_unsupported_config_is_rejected(self):
        raw = jload(os.path.join(TINY, "config.json"))
        raw["text_config"]["attn_output_gate"] = False
        with self.assertRaises(ValueError):
            ref.Config(raw)
        raw["text_config"]["attn_output_gate"] = True
        raw["text_config"]["rope_parameters"]["rope_type"] = "yarn"
        with self.assertRaises(ValueError):
            ref.Config(raw)

    def test_cli_matches_module(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "l.f32")
            p = subprocess.run([sys.executable, os.path.join(TOOLS, "ref_qwen35.py"), "--model", TINY,
                                "--tokens", ",".join(map(str, self.tokens[:4])), "--greedy", "3",
                                "--logits-out", out, "--threads", "2"], capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stderr)
            raw = rbytes(out)
            self.assertEqual(len(raw), 4 * 384 * 4)
            rows, ids, _ = self.model.generate(self.tokens[:4], 3)
            self.assertEqual([int(x) for x in p.stdout.split()], ids)
            first = struct.unpack("<384f", raw[:384 * 4])
            for a, b in zip(first, rows[0]):
                self.assertAlmostEqual(a, b, delta=1e-6 * max(1, abs(b)))


class FixtureTests(unittest.TestCase):
    def test_tiny_fixture_is_reproducible_and_small(self):
        files, info = make_tiny.build(make_tiny.SEED)
        total = 0
        self.assertEqual(sorted(files), sorted(os.listdir(TINY)))
        for name, data in files.items():
            self.assertEqual(rbytes(os.path.join(TINY, name)), data, name)
            total += len(data)
        self.assertLess(total, 2 * 1024 * 1024)
        self.assertGreater(info["min_gap_bf16"], 1e-2)
        self.assertGreater(info["min_gap_q8"], 1e-2)

    def test_ops_fixtures_are_reproducible_and_small(self):
        files = make_ops.build()
        self.assertEqual(sorted(files), sorted(os.listdir(OPS)))
        for name, data in files.items():
            self.assertEqual(rbytes(os.path.join(OPS, name)), data, name)
            self.assertLess(len(data), 1 << 20, name)
            json.loads(data)

    def test_ops_q8_cases_cover_the_required_edges(self):
        d = jload(os.path.join(OPS, "q8_quantize.json"))
        names = {g["name"] for g in d["groups"]}
        for want in ("ties_at_half_scale_1", "zero_group", "negative_zero_group", "max_negative_at_clamp"):
            self.assertIn(want, names)
        ties = next(g for g in d["groups"] if g["name"] == "ties_at_half_scale_1")
        self.assertEqual(ties["scale"], 1.0)
        self.assertEqual(ties["q"][:6], [0, 2, 2, 0, -2, -2])   # 0.5 1.5 2.5 -0.5 -1.5 -2.5, scale 1
        zero = next(g for g in d["groups"] if g["name"] == "zero_group")
        self.assertEqual((zero["scale"], zero["q"]), (0.0, [0] * 32))
        m = d["matrix"]
        self.assertEqual(m["q"][64:96], [0] * 32)
        self.assertEqual(m["scale"][2], 0.0)

    def test_ops_layer_fixture_recomputes(self):
        d = jload(os.path.join(OPS, "mlp.json"))
        cfg = make_ops.mini_cfg(32, 48, 2, 1, 8, 0.5, 1e7, 1, 1, 1, 1, 4, "full_attention")
        m = make_ops.layer_model(cfg, d["weights"])
        for x, y in zip(d["x"], d["y"]):
            self.assertEqual(m._mlp("", x), y)


FAKE_RUN = textwrap.dedent('''\
    #!{py}
    import os, struct, sys
    sys.path.insert(0, {tools!r})
    import ref_qwen35 as ref
    a = sys.argv[1:]
    def opt(name):
        return a[a.index(name) + 1] if name in a else None
    mode = os.environ.get("FAKE_MODE", "ok")
    model = ref.Model.load(opt("--model"), opt("--weights"))
    tokens = [int(t) for t in opt("--tokens").split(",")]
    greedy = int(opt("--greedy") or 0)
    rows, ids, _ = model.generate(tokens, greedy)
    if mode == "exit":
        sys.exit(3)
    if mode == "perturb":
        rows[5][3] += 0.05
    if mode == "nan":
        rows[2][7] = float("nan")
    if mode == "missingrow":
        rows = rows[:-1]
    if mode == "wrongid" and ids:
        ids[0] = (ids[0] + 1) % model.cfg.vocab
    if mode == "extraid" and ids:
        ids.append(0)
    if mode == "noise":
        print("hello")
    if opt("--logits-out"):
        with open(opt("--logits-out"), "wb") as f:
            for r in rows:
                f.write(struct.pack("<%df" % len(r), *r))
    for i in ids:
        print(i)
''')

FAKE_QUANT = textwrap.dedent('''\
    #!{py}
    import os, sys
    sys.path.insert(0, {tools!r})
    import quant_ref
    a = sys.argv[1:]
    data = bytearray(quant_ref.quantize_checkpoint(a[a.index("--model") + 1], a[a.index("--out") + 1]))
    if os.environ.get("FAKE_QUANT") == "corrupt":
        data[-1] ^= 1
    elif os.environ.get("FAKE_QUANT") == "header":
        data[20] ^= 1
    elif os.environ.get("FAKE_QUANT") == "short":
        del data[-4:]
    open(a[a.index("--out") + 1], "wb").write(data)
''')


class CheckGoldenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.run_bin = os.path.join(cls.tmp.name, "lm-run")
        cls.quant_bin = os.path.join(cls.tmp.name, "lm-quant")
        for path, body in ((cls.run_bin, FAKE_RUN), (cls.quant_bin, FAKE_QUANT)):
            with open(path, "w") as f:
                f.write(body.format(py=sys.executable, tools=TOOLS))
            os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def go(self, fake_mode="ok", fake_quant="ok", run_bin=None, quant_bin=None, fixtures=TINY):
        env = dict(os.environ, FAKE_MODE=fake_mode, FAKE_QUANT=fake_quant)
        return subprocess.run(
            [sys.executable, os.path.join(TOOLS, "check_golden.py"), "--run", run_bin or self.run_bin,
             "--quant", quant_bin or self.quant_bin, "--fixtures", fixtures, "--timeout", "120"],
            capture_output=True, text=True, env=env)

    def test_correct_engine_passes(self):
        p = self.go()
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(len(p.stdout.strip().splitlines()), 1)
        self.assertIn("check_golden OK", p.stdout)
        self.assertIn("quantiser byte-identical", p.stdout)

    def assert_fails(self, p, needle):
        self.assertEqual(p.returncode, 1, p.stdout + p.stderr)
        self.assertIn("check_golden FAILED", p.stderr)
        self.assertIn(needle, p.stderr)
        self.assertEqual(p.stdout, "")

    def test_perturbed_logits_fail(self):
        self.assert_fails(self.go(fake_mode="perturb"), "bf16 row 5")

    def test_wrong_greedy_id_fails(self):
        self.assert_fails(self.go(fake_mode="wrongid"), "greedy id 0")

    def test_extra_greedy_id_fails(self):
        self.assert_fails(self.go(fake_mode="extraid"), "greedy printed 17 ids")

    def test_missing_row_fails(self):
        self.assert_fails(self.go(fake_mode="missingrow"), "logits file has")

    def test_nan_fails(self):
        self.assert_fails(self.go(fake_mode="nan"), "NaN")

    def test_non_zero_exit_fails(self):
        self.assert_fails(self.go(fake_mode="exit"), "exited with status 3")

    def test_stdout_noise_fails(self):
        self.assert_fails(self.go(fake_mode="noise"), "stdout")

    def test_quantiser_byte_mismatch_fails(self):
        for mode in ("corrupt", "header", "short"):
            with self.subTest(mode):
                self.assert_fails(self.go(fake_quant=mode), "differs from model.q8.safetensors")

    def test_missing_pieces_fail_closed(self):
        self.assert_fails(self.go(run_bin="/nonexistent/lm-run"), "lm-run not found")
        self.assert_fails(self.go(quant_bin="/nonexistent/lm-quant"), "lm-quant not found")
        self.assert_fails(self.go(fixtures="/nonexistent"), "fixtures directory not found")
        with tempfile.TemporaryDirectory() as d:
            for n in ("config.json", "model.q8.safetensors"):
                with open(os.path.join(d, n), "w") as f:
                    f.write('{"vocab_size": 4}')
            self.assert_fails(self.go(fixtures=d), "golden file not found")


if __name__ == "__main__":
    unittest.main()
