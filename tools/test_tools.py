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
import make_tiny_gemma4  # noqa: E402
import quant_ref  # noqa: E402
import ref_gemma4 as gem  # noqa: E402
import ref_qwen35 as ref  # noqa: E402
import safetensors as st  # noqa: E402

ROOT = os.path.dirname(TOOLS)
TINY = os.path.join(ROOT, "tests", "fixtures", "qwen35-tiny")
GTINY = os.path.join(ROOT, "tests", "fixtures", "gemma4-tiny")
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


REAL_GEMMA_TEXT_KEYS = [
    "attention_bias", "attention_dropout", "attention_k_eq_v", "bos_token_id", "dtype", "enable_moe_block",
    "eos_token_id", "expert_intermediate_size", "final_logit_softcapping", "global_head_dim", "head_dim",
    "hidden_activation", "hidden_size", "hidden_size_per_layer_input", "initializer_range",
    "intermediate_size", "layer_types", "max_position_embeddings", "model_type", "num_attention_heads",
    "num_experts", "num_global_key_value_heads", "num_hidden_layers", "num_key_value_heads",
    "num_kv_shared_layers", "pad_token_id", "rms_norm_eps", "rope_parameters", "sliding_window",
    "tie_word_embeddings", "top_k_experts", "use_bidirectional_attention", "use_cache",
    "use_double_wide_mlp", "vocab_size", "vocab_size_per_layer_input"]
REAL_GEMMA_TOP_KEYS = [
    "architectures", "audio_config", "audio_token_id", "boa_token_id", "boi_token_id", "dtype",
    "eoa_token_id", "eoa_token_index", "eoi_token_id", "eos_token_id", "image_token_id",
    "initializer_range", "model_type", "text_config", "tie_word_embeddings", "transformers_version",
    "video_token_id", "vision_config", "vision_soft_tokens_per_image"]


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
        text = [n for n in src.names() if n.startswith(("model.language_model.", "mtp."))]
        self.assertTrue(any(n.startswith("mtp.") for n in text))     # the MTP head is kept
        self.assertFalse(any(n.startswith("model.visual.") for n in q8.names()))
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
    import {mod} as ref
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
        cls.gemma_run_bin = os.path.join(cls.tmp.name, "lm-run-gemma")
        cls.quant_bin = os.path.join(cls.tmp.name, "lm-quant")
        for path, body, mod in ((cls.run_bin, FAKE_RUN, "ref_qwen35"), (cls.gemma_run_bin, FAKE_RUN, "ref_gemma4"),
                                (cls.quant_bin, FAKE_QUANT, "")):
            with open(path, "w") as f:
                f.write(body.format(py=sys.executable, tools=TOOLS, mod=mod))
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

    def test_gemma_fixture_passes_and_is_labelled(self):
        p = self.go(fixtures=GTINY, run_bin=self.gemma_run_bin)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("[gemma4]", p.stdout)
        self.assertIn("40 rows", p.stdout)

    def test_qwen_summary_is_unchanged_by_family_detection(self):
        self.assertNotIn("[", self.go().stdout)

    def test_gemma_fixture_fails_closed(self):
        for mode, msg in (("perturb", "tol"), ("wrongid", "greedy id"), ("nan", "NaN")):
            with self.subTest(mode):
                self.assert_fails(self.go(fake_mode=mode, fixtures=GTINY, run_bin=self.gemma_run_bin), msg)
        self.assert_fails(self.go(fake_quant="corrupt", fixtures=GTINY, run_bin=self.gemma_run_bin),
                          "differs from model.q8.safetensors")
        with tempfile.TemporaryDirectory() as d:
            for n in os.listdir(GTINY):
                if n != "model.safetensors":
                    with open(os.path.join(d, n), "wb") as f:
                        f.write(rbytes(os.path.join(GTINY, n)))
            self.assert_fails(self.go(fixtures=d, run_bin=self.gemma_run_bin), "model.safetensors not found")


def gemma_raw(**text):
    """The tiny fixture's config with text_config overrides."""
    raw = jload(os.path.join(GTINY, "config.json"))
    raw["text_config"].update(text)
    return raw


def toy_gemma_cfg(hidden, ple, layers=1):
    return gem.Config({"model_type": "gemma4", "text_config": {
        "model_type": "gemma4_text", "hidden_size": hidden, "intermediate_size": 4, "num_hidden_layers": layers,
        "layer_types": [gem.FULL] * layers, "vocab_size": 3, "vocab_size_per_layer_input": 3,
        "rms_norm_eps": 1e-6, "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 4,
        "global_head_dim": 4, "hidden_activation": "gelu_pytorch_tanh", "sliding_window": 4,
        "num_kv_shared_layers": 0, "hidden_size_per_layer_input": ple, "final_logit_softcapping": 5.0,
        "tie_word_embeddings": True, "attention_bias": False, "enable_moe_block": False,
        "attention_k_eq_v": False, "use_double_wide_mlp": False, "use_bidirectional_attention": None,
        "num_global_key_value_heads": None,
        "rope_parameters": {"full_attention": {"rope_type": "proportional", "rope_theta": 1e6,
                                               "partial_rotary_factor": 0.25}}}})


class DictWeights:
    def __init__(self, d):
        self.d = d

    def get(self, name):
        return self.d[name]


class GemmaOpTests(unittest.TestCase):
    def test_rmsnorm_uses_plain_weight_not_one_plus_w(self):
        x, w = [3.0, 4.0], [2.0, 0.5]
        r = 1.0 / math.sqrt(12.5 + 1e-6)
        got = gem.rmsnorm(x, w, 1e-6)
        for g, want in zip(got, [3.0 * r * 2.0, 4.0 * r * 0.5]):
            self.assertAlmostEqual(g, want, places=14)
        zc = ref.rmsnorm_zero_centred(x, w, 1e-6)
        self.assertNotAlmostEqual(got[0], zc[0], places=3)
        for g, want in zip(gem.rmsnorm(x, None, 1e-6), [3.0 * r, 4.0 * r]):
            self.assertAlmostEqual(g, want, places=14)
        self.assertEqual(gem.rmsnorm(x, [1.0, 1.0], 1e-6), gem.rmsnorm(x, None, 1e-6))
        self.assertEqual(gem.rmsnorm([0.0] * 4, [1.0] * 4, 1e-6), [0.0] * 4)

    def test_gelu_tanh_known_values(self):
        self.assertEqual(gem.gelu_tanh(0.0), 0.0)
        self.assertAlmostEqual(gem.gelu_tanh(1.0), 0.8411919906082768, places=12)
        self.assertAlmostEqual(gem.gelu_tanh(-1.0), -0.15880800939172324, places=12)
        self.assertAlmostEqual(gem.gelu_tanh(30.0), 30.0, places=9)
        self.assertAlmostEqual(gem.gelu_tanh(-30.0), 0.0, places=9)
        self.assertNotAlmostEqual(gem.gelu_tanh(1.0), ref.silu(1.0), places=2)  # not SiLU

    def test_softcap_bounds_and_shape(self):
        xs = [0.0, 1e-9, -3.0, 15.0, 30.0, 300.0, -300.0]
        ys = gem.softcap(xs, 30.0)
        self.assertTrue(all(abs(y) < 30.0 for y in ys), ys)
        self.assertAlmostEqual(ys[1], 1e-9, places=15)
        self.assertAlmostEqual(ys[3], 30.0 * math.tanh(0.5), places=12)
        self.assertEqual(ys[2], -gem.softcap([3.0], 30.0)[0])
        self.assertLessEqual(abs(gem.softcap([1e9], 30.0)[0]), 30.0)

    def test_rope_default_hand_values(self):
        f = gem.rope_freqs(4, 100.0)
        self.assertEqual(f, [1.0, 100.0 ** -0.5])
        y = gem.rope_apply([1.0, 2.0, 3.0, 4.0], 2, f)   # pairs (0,2) and (1,3)
        a0, a1 = 2.0, 2.0 * 0.1
        want = [1.0 * math.cos(a0) - 3.0 * math.sin(a0), 2.0 * math.cos(a1) - 4.0 * math.sin(a1),
                3.0 * math.cos(a0) + 1.0 * math.sin(a0), 4.0 * math.cos(a1) + 2.0 * math.sin(a1)]
        for u, v in zip(y, want):
            self.assertAlmostEqual(u, v, places=14)
        self.assertEqual(gem.rope_apply([1.0, 2.0, 3.0, 4.0], 0, f), [1.0, 2.0, 3.0, 4.0])

    def test_proportional_rope_leaves_dims_beyond_r_untouched(self):
        d, partial, theta = 32, 0.25, 1e6
        f = gem.rope_freqs(d, theta, "proportional", partial)
        r = int(partial * d / 2)
        self.assertEqual(r, 4)
        self.assertEqual(f[:r], [theta ** (-2.0 * i / d) for i in range(r)])   # exponent denominator is d
        self.assertEqual(f[r:], [0.0] * (d // 2 - r))
        rng = random.Random(3)
        x = [rng.gauss(0, 1) for _ in range(d)]
        for pos in (1, 77, 100000):
            y = gem.rope_apply(x, pos, f)
            for i in range(d // 2):
                if i >= r:
                    self.assertEqual((y[i], y[i + d // 2]), (x[i], x[i + d // 2]), (pos, i))
                else:
                    self.assertNotEqual(y[i], x[i])
                self.assertAlmostEqual(math.hypot(y[i], y[i + d // 2]),
                                       math.hypot(x[i], x[i + d // 2]), places=12)

    def test_proportional_rope_real_shape_and_factor(self):
        f = gem.rope_freqs(512, 1e6, "proportional", 0.25)
        self.assertEqual(sum(1 for v in f if v), 64)
        self.assertEqual(f[1], 1e6 ** (-2.0 / 512))            # not 1e6 ** (-2/128)
        self.assertEqual(gem.rope_freqs(512, 1e6, "proportional", 0.25, factor=4.0)[1], f[1] / 4.0)
        self.assertEqual(gem.rope_freqs(512, 1e6, "proportional", 0.25, factor=4.0)[100], 0.0)
        with self.assertRaises(ValueError):
            gem.rope_freqs(8, 10.0, "yarn")

    def test_sliding_mask_excludes_position_i_minus_window(self):
        w = 8
        for i in (0, 3, 8, 9, 40):
            self.assertFalse(gem.mask_allows(i, i - w, w), i)
            self.assertFalse(gem.mask_allows(i, i + 1, w))
            self.assertTrue(gem.mask_allows(i, i, w))
            if i - w + 1 >= 0:
                self.assertTrue(gem.mask_allows(i, i - w + 1, w))
        self.assertEqual(sum(gem.mask_allows(30, j, w) for j in range(100)), w)   # w keys incl. current
        self.assertTrue(gem.mask_allows(30, 0, None))

    def test_attention_window_boundary_through_the_op(self):
        rng = random.Random(5)
        heads, kvh, d, window, pos = 4, 2, 8, 5, 12
        q = [[rng.gauss(0, 0.5) for _ in range(d)] for _ in range(heads)]
        ks = [[[rng.gauss(0, 0.5) for _ in range(d)] for _ in range(kvh)] for _ in range(pos + 1)]
        vs = [[[rng.gauss(0, 1) for _ in range(d)] for _ in range(kvh)] for _ in range(pos + 1)]
        base = gem.attention_context(q, ks, vs, pos, window)

        def with_v(j, win):
            v2 = [[list(h) for h in row] for row in vs]
            v2[j][0][0] += 10.0
            return gem.attention_context(q, ks, v2, pos, win)

        self.assertEqual(with_v(pos - window, window), base)    # excluded key: no effect at all
        self.assertEqual(with_v(0, window), base)
        self.assertNotEqual(with_v(pos - window + 1, window), base)   # first visible key matters
        self.assertNotEqual(with_v(pos, window), base)
        full = gem.attention_context(q, ks, vs, pos, None)
        self.assertNotEqual(full, base)
        self.assertNotEqual(with_v(0, None), full)

    def test_attention_scaling_is_one_and_gqa_mapping(self):
        # one head, d=2, two keys: scores must be q.k (scaling 1.0), not q.k / sqrt(2)
        out = gem.attention_context([[1.0, 0.0]], [[[2.0, 0.0]], [[0.0, 0.0]]], [[[1.0, 0.0]], [[0.0, 1.0]]], 1, None)
        p0 = math.exp(2.0) / (math.exp(2.0) + 1.0)
        self.assertAlmostEqual(out[0], p0, places=14)
        self.assertAlmostEqual(out[1], 1.0 - p0, places=14)
        # 4 query heads, 2 KV heads: heads 0,1 -> KV head 0; heads 2,3 -> KV head 1
        ks = [[[0.0], [0.0]]]
        vs = [[[10.0], [20.0]]]
        self.assertEqual(gem.attention_context([[1.0]] * 4, ks, vs, 0, None), [10.0, 10.0, 20.0, 20.0])
        with self.assertRaises(ValueError):
            gem.attention_context([[1.0]], ks, vs, 3, None)

    def test_ple_for_a_one_layer_toy_by_hand(self):
        c = toy_gemma_cfg(hidden=4, ple=2)
        self.assertEqual(c.ple_scale, 1.4140625)            # bf16(sqrt(2)); sqrt(2) itself is 1.41421356
        self.assertEqual(c.embed_scale, 2.0)                # bf16(sqrt(4))
        m = object.__new__(gem.Model)
        m.cfg = c
        m.w = DictWeights({
            gem.PREFIX + "embed_tokens_per_layer.weight": [[0.0, 0.0], [1.0, -2.0], [0.0, 0.0]],
            gem.PREFIX + "per_layer_model_projection.weight": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            gem.PREFIX + "per_layer_projection_norm.weight": [1.0, 2.0]})
        got = m.per_layer_inputs(1, [1.0, 2.0, 3.0, 4.0])
        # ctx = [1, 2] * 4^-1/2 = [0.5, 1.0]; mean(ctx^2)+eps = 0.625 + 1e-6; row = ctx * rsqrt * [1, 2]
        # tok = [1, -2] * 1.4140625; ple = (row + tok) / sqrt(2); numbers computed separately by hand
        self.assertEqual(len(got), 1)
        self.assertAlmostEqual(got[0][0], 1.4471064205011135, places=12)
        self.assertAlmostEqual(got[0][1], -0.21093341462516135, places=12)
        with self.assertRaises(ValueError):
            m.per_layer_inputs(3, [1.0, 2.0, 3.0, 4.0])


class GemmaConfigTests(unittest.TestCase):
    def test_tiny_config_parses_with_plan_values(self):
        c = gem.Config(gemma_raw())
        self.assertEqual((c.layers, c.kv_shared, c.first_shared, c.window), (12, 4, 8, 8))
        self.assertEqual(c.producer, {gem.SLIDING: 7, gem.FULL: 5})
        self.assertEqual([c.reuses(i) for i in range(12)], [False] * 8 + [True] * 4)
        self.assertEqual((c.hd(0), c.hd(5)), (16, 32))
        self.assertEqual(c.rope[gem.FULL], ("proportional", 1e6, 0.25, 1.0))
        self.assertEqual(c.rope[gem.SLIDING][:2], ("default", 1e4))

    def test_real_e4b_sharing_layout(self):
        types = ([gem.SLIDING] * 5 + [gem.FULL]) * 7
        c = gem.Config(gemma_raw(num_hidden_layers=42, layer_types=types, num_kv_shared_layers=18))
        self.assertEqual(c.producer, {gem.SLIDING: 22, gem.FULL: 23})
        self.assertEqual(c.first_shared, 24)
        self.assertEqual(types.index(gem.FULL), 5)

    def test_no_sharing_means_no_reuse_layers(self):
        c = gem.Config(gemma_raw(num_kv_shared_layers=0))
        self.assertEqual(c.producer, {})
        self.assertFalse(any(c.reuses(i) for i in range(c.layers)))

    def test_each_unsupported_flag_is_rejected(self):
        for key, val, text in (
                ("enable_moe_block", True, "MoE"), ("attention_k_eq_v", True, "attention_k_eq_v"),
                ("use_double_wide_mlp", True, "use_double_wide_mlp"), ("attention_bias", True, "attention_bias"),
                ("use_bidirectional_attention", "all", "use_bidirectional_attention"),
                ("use_bidirectional_attention", "vision", "use_bidirectional_attention"),
                ("num_global_key_value_heads", 2, "num_global_key_value_heads"),
                ("per_layer_config", {"0": {"head_dim": 8}}, "per_layer_config"),
                ("hidden_size_per_layer_input", 0, "hidden_size_per_layer_input"),
                ("hidden_activation", "silu", "hidden_activation"),
                ("tie_word_embeddings", False, "tie_word_embeddings"),
                ("final_logit_softcapping", None, "final_logit_softcapping"),
                ("final_logit_softcapping", 0.0, "final_logit_softcapping"),
                ("sliding_window", 0, "sliding_window"),
                ("num_kv_shared_layers", 12, "num_kv_shared_layers")):
            with self.subTest(key=key, val=val):
                with self.assertRaisesRegex(ValueError, text):
                    gem.Config(gemma_raw(**{key: val}))

    def test_other_unsupported_shapes_are_rejected(self):
        types = list(gemma_raw()["text_config"]["layer_types"])
        sliding_last = types[:-1] + [gem.SLIDING]
        with self.assertRaisesRegex(ValueError, "last layer"):
            gem.Config(gemma_raw(layer_types=sliding_last))
        with self.assertRaisesRegex(ValueError, "layer_types"):
            gem.Config(gemma_raw(layer_types=types[:-1] + [gem.FULL, gem.FULL]))
        with self.assertRaisesRegex(ValueError, "layer type"):
            gem.Config(gemma_raw(layer_types=["linear_attention"] + types[1:]))
        for rope in ({"rope_type": "yarn", "rope_theta": 1e4}, {"rope_type": "linear", "rope_theta": 1e4}):
            raw = gemma_raw()
            raw["text_config"]["rope_parameters"]["sliding_attention"] = rope
            with self.assertRaisesRegex(ValueError, "rope_type"):
                gem.Config(raw)
        raw = gemma_raw()
        raw["text_config"]["rope_parameters"]["sliding_attention"]["partial_rotary_factor"] = 0.5
        with self.assertRaisesRegex(ValueError, "partial_rotary_factor"):
            gem.Config(raw)
        raw = gemma_raw()
        raw["model_type"] = "qwen3_5"
        with self.assertRaisesRegex(ValueError, "model_type"):
            gem.Config(raw)
        raw = gemma_raw()
        raw["text_config"]["model_type"] = "gemma4_vision"
        with self.assertRaisesRegex(ValueError, "model_type"):
            gem.Config(raw)
        with self.assertRaisesRegex(ValueError, "no producing layer"):
            gem.Config(gemma_raw(num_kv_shared_layers=11))   # only layer 0 (sliding) left unshared

    def test_every_missing_field_is_an_error(self):
        required = ["attention_bias", "attention_k_eq_v", "enable_moe_block", "final_logit_softcapping",
                    "global_head_dim", "head_dim", "hidden_activation", "hidden_size",
                    "hidden_size_per_layer_input", "intermediate_size", "layer_types", "model_type",
                    "num_attention_heads", "num_global_key_value_heads", "num_hidden_layers",
                    "num_key_value_heads", "num_kv_shared_layers", "rms_norm_eps", "rope_parameters",
                    "sliding_window", "tie_word_embeddings", "use_bidirectional_attention",
                    "use_double_wide_mlp", "vocab_size", "vocab_size_per_layer_input"]
        for key in required:
            raw = gemma_raw()
            del raw["text_config"][key]
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, "missing"):
                    gem.Config(raw)
        for lt, key in ((gem.FULL, "partial_rotary_factor"), (gem.FULL, "rope_theta"), (gem.FULL, "rope_type"),
                        (gem.SLIDING, "rope_theta"), (gem.SLIDING, "rope_type")):
            raw = gemma_raw()
            del raw["text_config"]["rope_parameters"][lt][key]
            with self.subTest(rope=lt, key=key):
                with self.assertRaisesRegex(ValueError, "missing"):
                    gem.Config(raw)
        for top in ("model_type", "text_config"):
            raw = gemma_raw()
            del raw[top]
            with self.assertRaisesRegex(ValueError, "missing"):
                gem.Config(raw)
        raw = gemma_raw()
        del raw["text_config"]["rope_parameters"][gem.FULL]
        with self.assertRaisesRegex(ValueError, "missing"):
            gem.Config(raw)

    def test_missing_tensor_and_wrong_shape_are_errors(self):
        store = st.open_checkpoint(GTINY)
        cfg = jload(os.path.join(GTINY, "config.json"))

        def tensors():
            return {n: (store.dtype(n), store.shape(n), store.raw(n)) for n in store.names()}

        names = tensors()
        del names[gem.PREFIX + "layers.3.layer_scalar"]
        with self.assertRaisesRegex(ValueError, "layer_scalar is missing"):
            gem.Model(gem.Config(cfg), gem.Weights(st.MemoryCheckpoint(names)))
        names = tensors()
        names[gem.PREFIX + "layers.5.self_attn.q_proj.weight"] = ("BF16", (64, 64), b"\x00\x00" * (64 * 64))
        with self.assertRaisesRegex(ValueError, "shape"):       # layer 5 is full: 4 heads x 32, not x 16
            gem.Model(gem.Config(cfg), gem.Weights(st.MemoryCheckpoint(names)))
        names = tensors()
        del names[gem.PREFIX + "layers.9.self_attn.k_proj.weight"]   # a reuse layer's K projection: not required
        gem.Model(gem.Config(cfg), gem.Weights(st.MemoryCheckpoint(names)))


class GemmaModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = gem.Model.load(GTINY)
        cls.gold = jload(os.path.join(GTINY, "golden_bf16.json"))
        cls.tokens = cls.gold["tokens"]

    def test_causality_and_generate_consistency(self):
        full = self.model.forward(self.tokens)
        for t in (0, 7, 8, 20, 39):
            self.assertEqual(self.model.forward(self.tokens[:t + 1])[t], full[t])
        a = self.model.forward(self.tokens[:10] + [5])
        b = self.model.forward(self.tokens[:10] + [250, 7])
        self.assertEqual(a[:10], b[:10])
        self.assertNotEqual(a[10], b[10])
        _, ids, _ = self.model.generate(self.tokens[:12], 3)
        seq = list(self.tokens[:12])
        for want in ids:
            nxt = gem.argmax(self.model.forward(seq)[-1])
            self.assertEqual(nxt, want)
            seq.append(nxt)

    def test_goldens_reproduce(self):
        for name, weights in (("golden_bf16.json", None), ("golden_q8.json", "model.q8.safetensors")):
            with self.subTest(name):
                gold = jload(os.path.join(GTINY, name))
                m = gem.Model.load(GTINY, os.path.join(GTINY, weights) if weights else None)
                rows, ids, steps = m.generate(gold["greedy"]["prompt"], len(gold["greedy"]["out"]))
                full = m.forward(gold["tokens"])
                self.assertEqual(ids, gold["greedy"]["out"])
                for r, g in zip(full, gold["logits"]):
                    for a, b in zip(r, g):
                        self.assertAlmostEqual(a, b, delta=1e-7 * max(1.0, abs(a)))
                for r in full + steps:
                    self.assertGreater(gem.top_gap(r), make_tiny_gemma4.MIN_GAP)
                self.assertGreater(len(gold["tokens"]), 2 * m.cfg.window)
                self.assertGreater(len(gold["greedy"]["prompt"]), 2 * m.cfg.window)
                self.assertEqual((len(gold["tokens"]), len(gold["greedy"]["prompt"]),
                                  len(gold["greedy"]["out"])), (40, 20, 16))

    def test_q8_golden_differs_from_bf16(self):
        a = jload(os.path.join(GTINY, "golden_bf16.json"))["logits"]
        b = jload(os.path.join(GTINY, "golden_q8.json"))["logits"]
        top = max(abs(v) for r in a for v in r)
        diff = max(abs(x - y) for r1, r2 in zip(a, b) for x, y in zip(r1, r2))
        self.assertGreater(diff, 3 * 2e-4 * top)

    def test_softcap_bites_and_bounds_the_logits(self):
        stats = {}
        rows = self.model.forward(self.tokens, stats)
        cap = self.model.cfg.cap
        self.assertGreaterEqual(stats["max_precap"], cap / 2)
        self.assertGreater(stats["max_precap"], max(abs(v) for r in rows for v in r))   # capping shrinks it
        for r in rows:
            self.assertTrue(all(abs(v) < cap for v in r))
        self.assertLess(max(abs(v) for r in self.gold["logits"] for v in r), cap)

    def test_activations_are_not_degenerate(self):
        rows = self.gold["logits"]
        self.assertGreaterEqual(len({gem.argmax(r) for r in rows}), 8)
        self.assertGreaterEqual(len(set(self.gold["greedy"]["out"])), 4)
        for r in rows:
            mean = sum(r) / len(r)
            self.assertGreater(math.sqrt(sum((v - mean) ** 2 for v in r) / len(r)), 1.0)

    def test_reuse_layers_never_read_their_own_kv_tensors(self):
        c = self.model.cfg
        dead = {"%slayers.%d.self_attn.%s" % (gem.PREFIX, i, n)
                for i in range(c.first_shared, c.layers) for n in ("k_proj.weight", "v_proj.weight", "k_norm.weight")}
        self.assertEqual(len(dead), 12)
        base = self.model.forward(self.tokens[:14])
        m = gem.Model.load(GTINY)
        seen, orig = [], m.w.get
        m.w.get = lambda name: (seen.append(name), orig(name))[1]
        self.assertEqual(m.forward(self.tokens[:14]), base)
        self.assertGreater(len(seen), 100)
        self.assertEqual(dead & set(seen), set())
        # corrupt them in memory: nothing changes; the same corruption on a PRODUCER layer does change
        m2 = gem.Model.load(GTINY)
        for name in dead:
            shape = m2.w.shape(name)
            m2.w._cache[name] = [float("nan")] * shape[0] if len(shape) == 1 else \
                [[float("nan")] * shape[1] for _ in range(shape[0])]
        self.assertEqual(m2.forward(self.tokens[:14]), base)
        m3 = gem.Model.load(GTINY)
        n = "%slayers.7.self_attn.k_norm.weight" % gem.PREFIX
        m3.w._cache[n] = [v * 3.0 for v in m3.w.get(n)]
        self.assertNotEqual(m3.forward(self.tokens[:14]), base)

    def test_shared_layers_read_the_producers_cache_under_their_own_mask(self):
        c = self.model.cfg
        state = self.model.new_state()
        self.assertEqual(sorted(state["kv"]), list(range(c.first_shared)))   # only unshared layers keep K/V
        for t in self.tokens[:20]:
            self.model.step(state, t)
        h = [0.3 * ((i % 5) - 2) for i in range(c.hidden)]
        pos = state["pos"] - 1                     # the cache holds positions 0..19; query as position 19
        outs = {i: self.model._attn(i, h, state, pos) for i in (8, 11)}   # 8: sliding (producer 7); 11: full (5)
        self.assertEqual(len(state["kv"][7]["k"]), 20)                     # reuse layers did not append
        kv7, kv5 = state["kv"][7]["v"], state["kv"][5]["v"]
        bump = lambda vv: [[x + 5.0 for x in hv] for hv in vv]
        kv7[0] = bump(kv7[0])
        self.assertEqual(self.model._attn(8, h, state, pos), outs[8])       # key 0 is outside the window (8)
        self.assertEqual(self.model._attn(11, h, state, pos), outs[11])     # full layer ignores producer 7
        kv7[-1] = bump(kv7[-1])
        self.assertNotEqual(self.model._attn(8, h, state, pos), outs[8])    # the current key is visible
        kv5[0] = bump(kv5[0])
        self.assertNotEqual(self.model._attn(11, h, state, pos), outs[11])  # full layer sees key 0 of producer 5

    def test_sliding_window_is_observable_in_the_model(self):
        base = self.model.forward(self.tokens)
        m = gem.Model.load(GTINY)
        m.cfg.window = 10 ** 6
        wide = m.forward(self.tokens)
        self.assertEqual(wide[:8], base[:8])             # inside the window nothing can differ
        self.assertNotEqual(wide[10], base[10])
        diff = max(abs(a - b) for r1, r2 in zip(base, wide) for a, b in zip(r1, r2))
        self.assertGreater(diff, 1.0)

    def test_layer_scalar_and_embed_scales_matter(self):
        base = self.model.forward(self.tokens[:6])
        m = gem.Model.load(GTINY)
        n = "%slayers.0.layer_scalar" % gem.PREFIX
        m.w._cache[n] = [m.w.get(n)[0] * 2.0]
        self.assertNotEqual(m.forward(self.tokens[:6]), base)
        m = gem.Model.load(GTINY)
        self.assertEqual((m.cfg.embed_scale, m.cfg.ple_scale), (8.0, 5.65625))   # bf16(sqrt(64)), bf16(sqrt(32))
        m.cfg.embed_scale = 8.0 * 1.01
        self.assertNotEqual(m.forward(self.tokens[:6]), base)
        m = gem.Model.load(GTINY)
        m.cfg.ple_scale = math.sqrt(32)                  # unrounded scale differs from the bf16 one
        self.assertNotEqual(m.forward(self.tokens[:6]), base)

    def test_cli_matches_module(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "l.f32")
            p = subprocess.run([sys.executable, os.path.join(TOOLS, "ref_gemma4.py"), "--model", GTINY,
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


class GemmaFixtureTests(unittest.TestCase):
    def test_tiny_fixture_is_reproducible(self):
        files, info = make_tiny_gemma4.build(make_tiny_gemma4.SEED)
        self.assertEqual(sorted(files), sorted(os.listdir(GTINY)))
        total = 0
        for name, data in files.items():
            self.assertEqual(rbytes(os.path.join(GTINY, name)), data, name)
            total += len(data)
        self.assertLess(total, 4 * 1024 * 1024)
        self.assertGreater(info["min_gap_bf16"], make_tiny_gemma4.MIN_GAP)
        self.assertGreater(info["min_gap_q8"], make_tiny_gemma4.MIN_GAP)
        self.assertGreater(info["min_gap_bf16"], 10 * info["gate_tolerance"])
        self.assertGreaterEqual(info["max_precap_bf16"], make_tiny_gemma4.CAP / 2)

    def test_config_has_real_schema(self):
        cfg = jload(os.path.join(GTINY, "config.json"))
        self.assertEqual(list(cfg), REAL_GEMMA_TOP_KEYS)
        self.assertEqual(cfg["model_type"], "gemma4")
        t = cfg["text_config"]
        self.assertEqual(list(t), REAL_GEMMA_TEXT_KEYS)
        self.assertEqual(t["model_type"], "gemma4_text")
        self.assertEqual(t["layer_types"], (["sliding_attention"] * 5 + ["full_attention"]) * 2)
        self.assertEqual((t["num_hidden_layers"], t["num_kv_shared_layers"], t["sliding_window"]), (12, 4, 8))
        self.assertEqual((t["head_dim"], t["global_head_dim"], t["num_attention_heads"], t["num_key_value_heads"]),
                         (16, 32, 4, 2))
        self.assertEqual(t["vocab_size"], t["vocab_size_per_layer_input"])
        self.assertEqual(t["rope_parameters"]["full_attention"],
                         {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0, "rope_type": "proportional"})
        self.assertEqual(t["rope_parameters"]["sliding_attention"], {"rope_theta": 10000.0, "rope_type": "default"})
        self.assertEqual(jload(os.path.join(GTINY, "generation_config.json"))["eos_token_id"], [1, 106, 50])

    def test_single_bf16_file_with_all_real_tensor_names(self):
        self.assertFalse(os.path.exists(os.path.join(GTINY, "model.safetensors.index.json")))
        c = st.open_checkpoint(GTINY)
        m = gem.Model.load(GTINY)
        want = set(m.expected_shapes())
        for i in range(m.cfg.first_shared, m.cfg.layers):     # unused by the engine, present in the file
            for n in ("k_proj.weight", "v_proj.weight", "k_norm.weight"):
                want.add("%slayers.%d.self_attn.%s" % (gem.PREFIX, i, n))
        self.assertEqual({n for n in c.names() if n.startswith(gem.PREFIX)}, want)
        others = [n for n in c.names() if not n.startswith(gem.PREFIX)]
        self.assertEqual(len(others), 8)
        for n in others:
            self.assertTrue(n.startswith(("model.vision_tower.", "model.embed_vision.", "model.embed_audio.",
                                          "model.audio_tower.")), n)
        for n in c.names():
            self.assertEqual(c.dtype(n), "BF16", n)
        self.assertEqual(c.shape("model.vision_tower.encoder.layers.0.self_attn.q_proj.input_max"), ())
        for i in range(12):
            name = "%slayers.%d.layer_scalar" % (gem.PREFIX, i)
            (v,) = c.floats(name)
            self.assertTrue(0.5 <= v <= 1.5, v)
            self.assertEqual(c.shape(name), (1,))
        self.assertEqual(c.shape(gem.PREFIX + "layers.8.self_attn.k_proj.weight"), (32, 64))
        self.assertEqual(c.shape(gem.PREFIX + "layers.5.self_attn.q_proj.weight"), (128, 64))
        self.assertEqual(c.shape(gem.PREFIX + "embed_tokens_per_layer.weight"), (384, 12 * 32))

    def test_q8_file_is_text_only_and_quantises_the_ple_matrices(self):
        q = quant_ref.load_q8(os.path.join(GTINY, "model.q8.safetensors"))
        self.assertTrue(all(n.startswith(gem.PREFIX) for n in q.names()))
        for n in ("embed_tokens_per_layer.weight", "per_layer_model_projection.weight",
                  "layers.3.per_layer_projection.weight", "layers.3.per_layer_input_gate.weight"):
            self.assertEqual(q.dtype(gem.PREFIX + n), "I8", n)
            self.assertTrue(q.has(gem.PREFIX + n + ".q8scale"))
        self.assertEqual(q.dtype(gem.PREFIX + "layers.3.layer_scalar"), "BF16")
        self.assertEqual(q.dtype(gem.PREFIX + "layers.3.input_layernorm.weight"), "BF16")
        self.assertEqual(q.metadata["source"],
                         hashlib.sha256(rbytes(os.path.join(GTINY, "model.safetensors"))).hexdigest())

    def test_per_layer_input_dim_keeps_quantiser_alive(self):
        # docs/PLAN.md says P = 8, but per_layer_projection is [hidden, P] and q8 needs in % 32 == 0
        with self.assertRaises(quant_ref.QuantError):
            quant_ref.quantize_matrix([0.0] * (64 * 8), 64, 8)
        self.assertEqual(make_tiny_gemma4.PLE_DIM % 32, 0)

    def test_ops_sliding_attention_recomputes_independently(self):
        d = jload(os.path.join(OPS, "sliding_attention.json"))
        self.assertEqual(d["scaling"], 1.0)
        kinds = set()
        for c in d["cases"]:
            hd, h, kvh, pos, window = c["head_dim"], c["heads"], c["kv_heads"], c["pos"], c["window"]
            first = 0 if window is None else max(0, pos - window + 1)
            self.assertEqual(c["first_visible"], first)
            self.assertEqual(len(c["k"]), pos + 1)
            kinds.add("none" if window is None else "older" if pos + 1 > window else "within")
            rows = range(first, pos + 1)
            for head in range(h):                  # independent: softmax over exactly the visible slice
                kv = head // (h // kvh)
                qv = c["q"][head * hd:(head + 1) * hd]
                scores = [sum(a * b for a, b in zip(qv, c["k"][j][kv * hd:(kv + 1) * hd])) for j in rows]
                ws = [math.exp(s - max(scores)) for s in scores]
                for dim in range(hd):
                    got = sum(wt / sum(ws) * c["v"][j][kv * hd + dim] for wt, j in zip(ws, rows))
                    self.assertAlmostEqual(got, c["y"][head * hd + dim], places=12)
        self.assertEqual(kinds, {"older", "none", "within"})
        self.assertTrue(any(c["window"] and c["pos"] == c["window"] for c in d["cases"]))   # key 0 is at i - window
        self.assertTrue(any(c["window"] and c["pos"] + 1 > 2 * c["window"] for c in d["cases"]))

    def test_ops_gemma_rope_softcap_norm_gelu_fixtures(self):
        d = jload(os.path.join(OPS, "gemma_rope.json"))
        largest, untouched = 0, 0
        for g in d["groups"]:
            hd = g["head_dim"]
            r = hd // 2 if g["rope_type"] == "default" else int(g["partial_rotary_factor"] * hd / 2)
            self.assertEqual(len(g["freqs"]), hd // 2)
            self.assertEqual(sum(1 for f in g["freqs"] if f), r)
            self.assertEqual(g["freqs"][0], 1.0)
            for c in g["cases"]:
                largest = max(largest, c["pos"])
                for i in range(r, hd // 2):
                    untouched += 1
                    self.assertEqual((c["y"][i], c["y"][i + hd // 2]), (c["x"][i], c["x"][i + hd // 2]))
                if c["pos"] == 0:
                    self.assertEqual(c["y"], c["x"])
        self.assertGreaterEqual(largest, 100000)
        self.assertGreater(untouched, 100)
        for c in jload(os.path.join(OPS, "logit_softcap.json"))["cases"]:
            for x, y in zip(c["x"], c["y"]):
                self.assertLessEqual(abs(y), c["cap"])
                if x:
                    self.assertEqual(y > 0, x > 0)
        n = jload(os.path.join(OPS, "gemma_rmsnorm.json"))
        c = n["cases"][1]
        r = (sum(v * v for v in c["x"]) / len(c["x"]) + 1e-6) ** -0.5
        for x, w, y in zip(c["x"], c["w"], c["y"]):
            self.assertAlmostEqual(y, x * r * w, places=14)
        for c in n["cases_noweight"][:-1]:
            r = (sum(v * v for v in c["x"]) / len(c["x"]) + 1e-6) ** -0.5
            for x, y in zip(c["x"], c["y"]):
                self.assertAlmostEqual(y, x * r, places=14)
        for c in jload(os.path.join(OPS, "gelu_tanh_mul.json"))["cases"]:
            for gate, up, y in zip(c["gate"], c["up"], c["y"]):
                want = 0.5 * gate * (1 + math.tanh(math.sqrt(2 / math.pi) * (gate + 0.044715 * gate ** 3))) * up
                self.assertAlmostEqual(y, want, delta=1e-12 * max(1.0, abs(want)))

    def test_gemma_ops_did_not_move_any_qwen_op_file(self):
        qwen = ["rmsnorm", "rmsnorm_gated", "l2norm", "rope", "conv1d_silu", "gated_delta_rule",
                "full_attention_layer", "linear_attention_layer", "mlp", "q8_quantize"]
        files = make_ops.build()
        for name in qwen:
            self.assertEqual(files[name + ".json"], rbytes(os.path.join(OPS, name + ".json")), name)
        self.assertEqual(sorted(n for n in files if n[:-5] not in qwen),
                         ["gelu_tanh.json", "gelu_tanh_mul.json", "gemma_rmsnorm.json", "gemma_rope.json",
                          "logit_softcap.json", "sliding_attention.json"])


if __name__ == "__main__":
    unittest.main()
