#!/usr/bin/env python3
"""Deterministic generator for tests/fixtures/qwen35-tiny/ (see docs/PLAN.md and
tests/fixtures/README.md).

    make_tiny.py                 write the fixture with the committed SEED
    make_tiny.py --check         regenerate in memory, fail unless byte-identical to the files on disk
    make_tiny.py --find-seed [N] search seeds N, N+1, ... until every margin holds; print the first

Everything is derived from `random.Random(seed)` (Mersenne Twister: stable across Python
versions for `random()`, `gauss()` and `randrange()` as used here) plus the float64 oracle in
ref_qwen35.py, so the output is byte-for-byte reproducible. The seed is a constant in this
file (SEED); the golden files carry no seed.

Margins asserted for BOTH goldens (bf16 and q8): top-1 minus top-2 logit > MIN_GAP at each of
the 24 golden positions and at each of the 16 greedy steps; the 24 positions and the greedy
ids are not degenerate (distinct argmaxes); and the q8 logits differ from the bf16 logits by
more than 3x the gate tolerance, so a q8 golden cannot be satisfied by the bf16 path.
"""

import argparse
import hashlib
import json
import math
import os
import random
import sys
import tempfile

import quant_ref
import ref_qwen35 as ref
import safetensors as st

SEED = 8
MIN_GAP = 5e-2  # stricter than the 1e-2 the plan asks for
GATE_REL_TOL = 2e-4  # docs/PLAN.md gate 2: |engine - oracle| <= 2e-4 * max|logit|
N_TOKENS, N_PROMPT, N_GREEDY = 24, 8, 16
SHARD1 = "model.safetensors-00001-of-00002.safetensors"
SHARD2 = "model.safetensors-00002-of-00002.safetensors"
P = ref.PREFIX

HIDDEN, INTER, VOCAB, LAYERS = 64, 128, 384, 8
HEADS, KV_HEADS, HEAD_DIM = 4, 2, 32
LK_HEADS, LV_HEADS, LK_DIM, LV_DIM, CONV_K = 2, 4, 16, 16, 4
KEY_DIM, VALUE_DIM = LK_HEADS * LK_DIM, LV_HEADS * LV_DIM
CONV_DIM = 2 * KEY_DIM + VALUE_DIM


def config():
    layer_types = ["linear_attention"] * 3 + ["full_attention"]
    layer_types = layer_types * (LAYERS // 4)
    text = {
        "attention_bias": False,
        "attention_dropout": 0.0,
        "attn_output_gate": True,
        "dtype": "bfloat16",
        "eos_token_id": VOCAB - 1,
        "full_attention_interval": 4,
        "head_dim": HEAD_DIM,
        "hidden_act": "silu",
        "hidden_size": HIDDEN,
        "initializer_range": 0.02,
        "intermediate_size": INTER,
        "layer_types": layer_types,
        "linear_conv_kernel_dim": CONV_K,
        "linear_key_head_dim": LK_DIM,
        "linear_num_key_heads": LK_HEADS,
        "linear_num_value_heads": LV_HEADS,
        "linear_value_head_dim": LV_DIM,
        "max_position_embeddings": 4096,
        "mlp_only_layers": [],
        "model_type": "qwen3_5_text",
        "mtp_num_hidden_layers": 1,
        "mtp_use_dedicated_embeddings": False,
        "num_attention_heads": HEADS,
        "num_hidden_layers": LAYERS,
        "num_key_value_heads": KV_HEADS,
        "rms_norm_eps": 1e-06,
        "tie_word_embeddings": True,
        "use_cache": True,
        "vocab_size": VOCAB,
        "mamba_ssm_dtype": "float32",
        "rope_parameters": {
            "mrope_interleaved": True,
            "mrope_section": [2, 1, 1],
            "rope_type": "default",
            "rope_theta": 10000000,
            "partial_rotary_factor": 0.25,
        },
    }
    return {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "image_token_id": 248056,
        "model_type": "qwen3_5",
        "text_config": text,
        "tie_word_embeddings": True,
        "transformers_version": "4.57.0.dev0",
        "video_token_id": 248057,
        "vision_config": {
            "deepstack_visual_indexes": [],
            "depth": 1,
            "hidden_act": "gelu_pytorch_tanh",
            "hidden_size": 8,
            "in_channels": 3,
            "initializer_range": 0.02,
            "intermediate_size": 16,
            "model_type": "qwen3_5",
            "num_heads": 2,
            "num_position_embeddings": 16,
            "out_hidden_size": HIDDEN,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
        },
        "vision_end_token_id": 248054,
        "vision_start_token_id": 248053,
    }


class Gen:
    """Tensor factory: bf16-exact values (rounded once), recorded in creation order."""

    def __init__(self, rng):
        self.rng, self.t = rng, {}

    def bf16(self, name, shape, std, mean=0.0):
        n = st.numel(shape)
        vals = [st.round_bf16(self.rng.gauss(mean, std)) for _ in range(n)]
        self.t[name] = ("BF16", tuple(shape), st.floats_to_bf16(vals))

    def bf16_uniform(self, name, shape, lo, hi):
        vals = [st.round_bf16(self.rng.uniform(lo, hi)) for _ in range(st.numel(shape))]
        self.t[name] = ("BF16", tuple(shape), st.floats_to_bf16(vals))

    def f32(self, name, vals):
        self.t[name] = ("F32", (len(vals),), st.floats_to_f32([st.round_f32(v) for v in vals]))


def build_tensors(rng):
    g = Gen(rng)
    g.bf16(P + "embed_tokens.weight", (VOCAB, HIDDEN), 0.6)
    for i, lt in enumerate(config()["text_config"]["layer_types"]):
        p = "%slayers.%d." % (P, i)
        g.bf16(p + "input_layernorm.weight", (HIDDEN,), 0.25)
        g.bf16(p + "post_attention_layernorm.weight", (HIDDEN,), 0.25)
        if lt == "linear_attention":
            a = p + "linear_attn."
            g.bf16(a + "in_proj_qkv.weight", (CONV_DIM, HIDDEN), 1.0 / math.sqrt(HIDDEN))
            g.bf16(a + "in_proj_z.weight", (VALUE_DIM, HIDDEN), 1.0 / math.sqrt(HIDDEN))
            g.bf16(a + "in_proj_b.weight", (LV_HEADS, HIDDEN), 1.5 / math.sqrt(HIDDEN))
            g.bf16(a + "in_proj_a.weight", (LV_HEADS, HIDDEN), 1.5 / math.sqrt(HIDDEN))
            g.bf16(a + "conv1d.weight", (CONV_DIM, 1, CONV_K), 0.5)
            g.bf16(a + "dt_bias", (LV_HEADS,), 0.5, mean=-0.3)
            g.f32(a + "A_log", [math.log(rng.uniform(0.05, 1.0)) for _ in range(LV_HEADS)])
            g.bf16(a + "norm.weight", (LV_DIM,), 0.2, mean=1.0)
            g.bf16(a + "out_proj.weight", (HIDDEN, VALUE_DIM), 0.9 / math.sqrt(VALUE_DIM))
        else:
            a = p + "self_attn."
            g.bf16(a + "q_proj.weight", (HEADS * HEAD_DIM * 2, HIDDEN), 1.2 / math.sqrt(HIDDEN))
            g.bf16(a + "k_proj.weight", (KV_HEADS * HEAD_DIM, HIDDEN), 1.2 / math.sqrt(HIDDEN))
            g.bf16(a + "v_proj.weight", (KV_HEADS * HEAD_DIM, HIDDEN), 1.0 / math.sqrt(HIDDEN))
            g.bf16(a + "o_proj.weight", (HIDDEN, HEADS * HEAD_DIM), 0.9 / math.sqrt(HEADS * HEAD_DIM))
            g.bf16(a + "q_norm.weight", (HEAD_DIM,), 0.3)
            g.bf16(a + "k_norm.weight", (HEAD_DIM,), 0.3)
        m = p + "mlp."
        g.bf16(m + "gate_proj.weight", (INTER, HIDDEN), 1.0 / math.sqrt(HIDDEN))
        g.bf16(m + "up_proj.weight", (INTER, HIDDEN), 1.0 / math.sqrt(HIDDEN))
        g.bf16(m + "down_proj.weight", (HIDDEN, INTER), 0.8 / math.sqrt(INTER))
    g.bf16(P + "norm.weight", (HIDDEN,), 0.25)
    # tensors the engine must ignore (not part of the text model)
    g.bf16("model.visual.patch_embed.proj.weight", (8, 3, 2, 4), 0.1)
    g.bf16("model.visual.patch_embed.proj.bias", (8,), 0.1)
    g.bf16("model.visual.blocks.0.attn.qkv.weight", (24, 8), 0.1)
    g.bf16("model.visual.blocks.0.attn.qkv.bias", (24,), 0.1)
    g.bf16("model.visual.merger.linear_fc1.weight", (8, 8), 0.1)
    g.bf16("mtp.fc.weight", (HIDDEN, 2 * HIDDEN), 0.1)
    g.bf16("mtp.norm.weight", (HIDDEN,), 0.1)
    g.bf16("mtp.layers.0.input_layernorm.weight", (HIDDEN,), 0.1)
    return g.t


def split_shards(tensors):
    """Layers 0-3 and the embedding in shard 1; the rest (incl. visual/mtp) in shard 2."""
    s1 = {}
    for name, v in tensors.items():
        if name == P + "embed_tokens.weight":
            s1[name] = v
        elif name.startswith(P + "layers."):
            if int(name[len(P + "layers."):].split(".")[0]) < LAYERS // 2:
                s1[name] = v
    s2 = {n: v for n, v in tensors.items() if n not in s1}
    return s1, s2


def jdump(obj, **kw):
    return (json.dumps(obj, **kw) + "\n").encode("ascii")


def golden(tokens, rows, ids):
    rounded = [[float("%.9g" % v) for v in r] for r in rows]
    return jdump({"tokens": tokens, "logits": rounded,
                  "greedy": {"prompt": tokens[:N_PROMPT], "out": ids}}, separators=(",", ":"))


class MarginError(Exception):
    pass


def run_model(model, tokens, label):
    rows, ids, steps = model.generate(tokens[:N_PROMPT], N_GREEDY)
    full = model.forward(tokens)
    gaps = [ref.top_gap(r) for r in full] + [ref.top_gap(r) for r in steps]
    if min(gaps) <= MIN_GAP:
        raise MarginError("%s: top-1/top-2 gap %.3g <= %g" % (label, min(gaps), MIN_GAP))
    if len({ref.argmax(r) for r in full}) < 8:
        raise MarginError("%s: fewer than 8 distinct argmax ids in the golden positions" % label)
    if len(set(ids)) < 3:
        raise MarginError("%s: greedy output is degenerate" % label)
    return full, ids, min(gaps)


def build(seed):
    """Returns (files: {relative name: bytes}, info dict). Raises MarginError."""
    rng = random.Random(seed)
    tensors = build_tensors(rng)
    tokens = [rng.randrange(VOCAB) for _ in range(N_TOKENS)]
    s1, s2 = split_shards(tensors)
    files = {"config.json": jdump(config(), indent=4)}
    files[SHARD1] = st.serialize(s1, metadata={"format": "pt"})
    files[SHARD2] = st.serialize(s2, metadata={"format": "pt"})
    wmap = {n: SHARD1 for n in s1}
    wmap.update({n: SHARD2 for n in s2})
    total = sum(len(v[2]) for v in tensors.values())
    files["model.safetensors.index.json"] = jdump(
        {"metadata": {"total_size": total}, "weight_map": wmap}, indent=2, sort_keys=True)
    with tempfile.TemporaryDirectory() as d:
        for name, data in files.items():
            with open(os.path.join(d, name), "wb") as f:
                f.write(data)
        # the q8 file goes through the same entry point as the CLI (sharded dir, sha of the index)
        files["model.q8.safetensors"] = quant_ref.quantize_checkpoint(
            d, os.path.join(d, "model.q8.safetensors"))
        bf = ref.Model.load(d)
        q8 = ref.Model.load(d, os.path.join(d, "model.q8.safetensors"))
        bf_rows, bf_ids, bf_gap = run_model(bf, tokens, "bf16")
        q8_rows, q8_ids, q8_gap = run_model(q8, tokens, "q8")
    top = max(abs(v) for r in bf_rows for v in r)
    diff = max(abs(a - b) for r1, r2 in zip(bf_rows, q8_rows) for a, b in zip(r1, r2))
    if diff <= 3 * GATE_REL_TOL * top:
        raise MarginError("q8 logits are too close to bf16 (max diff %.3g)" % diff)
    files["golden_bf16.json"] = golden(tokens, bf_rows, bf_ids)
    files["golden_q8.json"] = golden(tokens, q8_rows, q8_ids)
    info = {"seed": seed, "min_gap_bf16": bf_gap, "min_gap_q8": q8_gap,
            "max_abs_logit": top, "max_q8_vs_bf16": diff,
            "gate_tolerance": GATE_REL_TOL * top,
            "greedy_bf16": bf_ids, "greedy_q8": q8_ids,
            "sha256_q8": hashlib.sha256(files["model.q8.safetensors"]).hexdigest()}
    return files, info


def default_dir():
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "tests", "fixtures", "qwen35-tiny")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=default_dir())
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--find-seed", nargs="?", const=0, type=int, metavar="START")
    a = ap.parse_args(argv)
    if a.find_seed is not None:
        s = a.find_seed
        while True:
            try:
                _, info = build(s)
            except MarginError as e:
                print("seed %d: %s" % (s, e), file=sys.stderr)
                s += 1
                continue
            print(json.dumps(info))
            return 0
    try:
        files, info = build(SEED)
    except MarginError as e:
        print("make_tiny: seed %d fails its margins: %s" % (SEED, e), file=sys.stderr)
        return 1
    if a.check:
        bad = []
        for name, data in files.items():
            path = os.path.join(a.out, name)
            if not os.path.exists(path):
                bad.append("missing " + name)
            else:
                with open(path, "rb") as f:
                    if f.read() != data:
                        bad.append("differs " + name)
        if os.path.isdir(a.out):
            bad += ["unexpected " + n for n in sorted(os.listdir(a.out)) if n not in files]
        else:
            bad.append("missing directory " + a.out)
        if bad:
            print("make_tiny --check FAILED: " + "; ".join(bad), file=sys.stderr)
            return 1
        print("make_tiny --check OK: %d files byte-identical (seed %d, min gap bf16 %.4g q8 %.4g)"
              % (len(files), SEED, info["min_gap_bf16"], info["min_gap_q8"]))
        return 0
    os.makedirs(a.out, exist_ok=True)
    for name, data in files.items():
        with open(os.path.join(a.out, name), "wb") as f:
            f.write(data)
    print(json.dumps(info))
    return 0


if __name__ == "__main__":
    sys.exit(main())
