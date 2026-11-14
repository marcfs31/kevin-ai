#!/usr/bin/env python3
"""Deterministic generator for tests/fixtures/gemma4-tiny/ (see docs/PLAN.md "Phase 2" and
tests/fixtures/README.md).

    make_tiny_gemma4.py                 write the fixture with the committed SEED
    make_tiny_gemma4.py --check         regenerate in memory, fail unless byte-identical to the files on disk
    make_tiny_gemma4.py --find-seed [N] search seeds N, N+1, ... until every margin holds; print the first

Everything is derived from `random.Random(seed)` plus the float64 oracle in ref_gemma4.py, so the
output is byte-for-byte reproducible. The seed is a constant in this file (SEED).

Margins asserted for BOTH goldens (bf16 and q8): top-1 minus top-2 logit > MIN_GAP at each of the
golden positions and greedy steps; the golden positions and the greedy ids are not degenerate;
the q8 logits differ from the bf16 logits by more than 3x the gate tolerance; the final logit
softcap bites (max |pre-cap logit| >= cap/2, asserted on both models); the golden prompt and the
greedy prompt are longer than 2 x sliding_window; and a model with the sliding window disabled
gives different logits (so the window mask is observable in the goldens).

Deviation from docs/PLAN.md: hidden_size_per_layer_input is 32, not 8. The q8 format needs every
quantised matrix to have an input dimension that is a multiple of 32, and per_layer_projection is
[hidden, P] (input P), so P = 8 would make `bin/lm-quant` (and quant_ref.py) fail on it.
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
import ref_gemma4 as ref
import safetensors as st

SEED = 16  # found with --find-seed
MIN_GAP = 3e-2  # 3x the 1e-2 the plan asks for and >10x the gate tolerance 2e-4 * max|logit|
GATE_REL_TOL = 2e-4  # docs/PLAN.md gate 2
N_TOKENS, N_PROMPT, N_GREEDY = 40, 20, 16
P = ref.PREFIX

HIDDEN, INTER, VOCAB, LAYERS = 64, 128, 384, 12
HEADS, KV_HEADS, HEAD_DIM, GLOBAL_HEAD_DIM = 4, 2, 16, 32
WINDOW, PLE_DIM, KV_SHARED = 8, 32, 4
CAP = 16.0
LAYER_TYPES = ([ref.SLIDING] * 5 + [ref.FULL]) * 2


def config():
    return {
        "architectures": ["Gemma4ForConditionalGeneration"],
        "audio_config": {
            "_name_or_path": "", "architectures": None, "attention_chunk_size": 12,
            "attention_context_left": 13, "attention_context_right": 0,
            "attention_invalid_logits_value": -1000000000.0, "attention_logit_cap": 50.0,
            "chunk_size_feed_forward": 0, "conv_kernel_size": 5, "dtype": "bfloat16",
            "gradient_clipping": 10000000000.0, "hidden_act": "silu", "hidden_size": 8,
            "id2label": {"0": "LABEL_0", "1": "LABEL_1"}, "initializer_range": 0.02,
            "is_encoder_decoder": False, "label2id": {"LABEL_0": 0, "LABEL_1": 1},
            "model_type": "gemma4_audio", "num_attention_heads": 2, "num_hidden_layers": 1,
            "output_attentions": False, "output_hidden_states": False, "output_proj_dims": 8,
            "problem_type": None, "residual_weight": 0.5, "return_dict": True, "rms_norm_eps": 1e-06,
            "subsampling_conv_channels": [4, 2], "use_clipped_linears": True},
        "audio_token_id": 258881,
        "boa_token_id": 256000,
        "boi_token_id": 255999,
        "dtype": "bfloat16",
        "eoa_token_id": 258883,
        "eoa_token_index": 258883,
        "eoi_token_id": 258882,
        "eos_token_id": [1, 106],
        "image_token_id": 258880,
        "initializer_range": 0.02,
        "model_type": "gemma4",
        "text_config": {
            "attention_bias": False,
            "attention_dropout": 0.0,
            "attention_k_eq_v": False,
            "bos_token_id": 2,
            "dtype": "bfloat16",
            "enable_moe_block": False,
            "eos_token_id": 1,
            "expert_intermediate_size": None,
            "final_logit_softcapping": CAP,
            "global_head_dim": GLOBAL_HEAD_DIM,
            "head_dim": HEAD_DIM,
            "hidden_activation": "gelu_pytorch_tanh",
            "hidden_size": HIDDEN,
            "hidden_size_per_layer_input": PLE_DIM,
            "initializer_range": 0.02,
            "intermediate_size": INTER,
            "layer_types": list(LAYER_TYPES),
            "max_position_embeddings": 4096,
            "model_type": "gemma4_text",
            "num_attention_heads": HEADS,
            "num_experts": None,
            "num_global_key_value_heads": None,
            "num_hidden_layers": LAYERS,
            "num_key_value_heads": KV_HEADS,
            "num_kv_shared_layers": KV_SHARED,
            "pad_token_id": 0,
            "rms_norm_eps": 1e-06,
            "rope_parameters": {
                "full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0,
                                   "rope_type": "proportional"},
                "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}},
            "sliding_window": WINDOW,
            "tie_word_embeddings": True,
            "top_k_experts": None,
            "use_bidirectional_attention": None,
            "use_cache": True,
            "use_double_wide_mlp": False,
            "vocab_size": VOCAB,
            "vocab_size_per_layer_input": VOCAB,
        },
        "tie_word_embeddings": True,
        "transformers_version": "5.5.0.dev0",
        "video_token_id": 258884,
        "vision_config": {
            "_name_or_path": "", "architectures": None, "attention_bias": False,
            "attention_dropout": 0.0, "chunk_size_feed_forward": 0, "default_output_length": 4,
            "dtype": "bfloat16", "global_head_dim": 8, "head_dim": 8, "hidden_activation": "gelu_pytorch_tanh",
            "hidden_size": 16, "id2label": {"0": "LABEL_0", "1": "LABEL_1"}, "initializer_range": 0.02,
            "intermediate_size": 32, "is_encoder_decoder": False,
            "label2id": {"LABEL_0": 0, "LABEL_1": 1}, "max_position_embeddings": 4096,
            "model_type": "gemma4_vision", "num_attention_heads": 2, "num_hidden_layers": 1,
            "num_key_value_heads": 2, "output_attentions": False, "output_hidden_states": False,
            "patch_size": 16, "pooling_kernel_size": 3, "position_embedding_size": 4,
            "problem_type": None, "return_dict": True, "rms_norm_eps": 1e-06,
            "rope_parameters": {"rope_theta": 100.0, "rope_type": "default"},
            "standardize": False, "use_clipped_linears": True},
        "vision_soft_tokens_per_image": 4,
    }


class Gen:
    """Tensor factory: bf16-exact values (rounded once), recorded in creation order."""

    def __init__(self, rng):
        self.rng, self.t = rng, {}

    def bf16(self, name, shape, std, mean=0.0):
        vals = [st.round_bf16(self.rng.gauss(mean, std)) for _ in range(st.numel(shape))]
        self.t[name] = ("BF16", tuple(shape), st.floats_to_bf16(vals))

    def bf16_uniform(self, name, shape, lo, hi):
        vals = [st.round_bf16(self.rng.uniform(lo, hi)) for _ in range(st.numel(shape))]
        self.t[name] = ("BF16", tuple(shape), st.floats_to_bf16(vals))


def build_tensors(rng):
    g = Gen(rng)
    inv = lambda n: 1.0 / math.sqrt(n)
    g.bf16(P + "embed_tokens.weight", (VOCAB, HIDDEN), 0.12)
    g.bf16(P + "embed_tokens_per_layer.weight", (VOCAB, LAYERS * PLE_DIM), 0.2)
    g.bf16(P + "per_layer_model_projection.weight", (LAYERS * PLE_DIM, HIDDEN), 1.0 * inv(HIDDEN))
    g.bf16(P + "per_layer_projection_norm.weight", (PLE_DIM,), 0.2, mean=1.0)
    for i, lt in enumerate(LAYER_TYPES):
        p = "%slayers.%d." % (P, i)
        hd = GLOBAL_HEAD_DIM if lt == ref.FULL else HEAD_DIM
        g.bf16(p + "input_layernorm.weight", (HIDDEN,), 0.2, mean=1.0)
        g.bf16_uniform(p + "layer_scalar", (1,), 0.5, 1.5)
        m = p + "mlp."
        g.bf16(m + "down_proj.weight", (HIDDEN, INTER), 0.8 * inv(INTER))
        g.bf16(m + "gate_proj.weight", (INTER, HIDDEN), 1.0 * inv(HIDDEN))
        g.bf16(m + "up_proj.weight", (INTER, HIDDEN), 1.0 * inv(HIDDEN))
        g.bf16(p + "per_layer_input_gate.weight", (PLE_DIM, HIDDEN), 1.0 * inv(HIDDEN))
        g.bf16(p + "per_layer_projection.weight", (HIDDEN, PLE_DIM), 1.0 * inv(PLE_DIM))
        g.bf16(p + "post_attention_layernorm.weight", (HIDDEN,), 0.1, mean=0.6)
        g.bf16(p + "post_feedforward_layernorm.weight", (HIDDEN,), 0.1, mean=0.6)
        g.bf16(p + "post_per_layer_input_norm.weight", (HIDDEN,), 0.1, mean=0.6)
        g.bf16(p + "pre_feedforward_layernorm.weight", (HIDDEN,), 0.2, mean=1.0)
        a = p + "self_attn."
        g.bf16(a + "k_norm.weight", (hd,), 0.1, mean=0.4)
        g.bf16(a + "k_proj.weight", (KV_HEADS * hd, HIDDEN), 1.0 * inv(HIDDEN))
        g.bf16(a + "o_proj.weight", (HIDDEN, HEADS * hd), 0.9 * inv(HEADS * hd))
        g.bf16(a + "q_norm.weight", (hd,), 0.1, mean=0.4)
        g.bf16(a + "q_proj.weight", (HEADS * hd, HIDDEN), 1.0 * inv(HIDDEN))
        g.bf16(a + "v_proj.weight", (KV_HEADS * hd, HIDDEN), 1.0 * inv(HIDDEN))
    g.bf16(P + "norm.weight", (HIDDEN,), 0.4, mean=3.0)
    # tensors the engine must ignore (not part of the text model); names and ranks follow the real file
    g.bf16("model.embed_audio.embedding_projection.weight", (HIDDEN, 8), 0.1)
    g.bf16("model.embed_vision.embedding_projection.weight", (HIDDEN, 16), 0.1)
    g.bf16("model.vision_tower.patch_embedder.input_proj.weight", (16, 16), 0.1)
    g.bf16("model.vision_tower.patch_embedder.position_embedding_table", (2, 4, 16), 0.1)
    g.bf16("model.vision_tower.encoder.layers.0.self_attn.q_proj.linear.weight", (16, 16), 0.1)
    g.bf16("model.vision_tower.encoder.layers.0.self_attn.q_proj.input_max", (), 0.1)
    g.bf16("model.audio_tower.layers.0.feed_forward1.ffw_layer_1.linear.weight", (32, 8), 0.1)
    g.bf16("model.audio_tower.layers.0.norm_out.weight", (8,), 0.1)
    return g.t


def jdump(obj, **kw):
    return (json.dumps(obj, **kw) + "\n").encode("ascii")


def golden(tokens, rows, ids):
    rounded = [[float("%.9g" % v) for v in r] for r in rows]
    return jdump({"tokens": tokens, "logits": rounded,
                  "greedy": {"prompt": tokens[:N_PROMPT], "out": ids}}, separators=(",", ":"))


class MarginError(Exception):
    pass


def run_model(model, tokens, label):
    stats = {}
    rows, ids, steps = model.generate(tokens[:N_PROMPT], N_GREEDY, stats)
    full = model.forward(tokens, stats)
    gaps = [ref.top_gap(r) for r in full] + [ref.top_gap(r) for r in steps]
    if min(gaps) <= MIN_GAP:
        raise MarginError("%s: top-1/top-2 gap %.3g <= %g" % (label, min(gaps), MIN_GAP))
    if len({ref.argmax(r) for r in full}) < 8:
        raise MarginError("%s: fewer than 8 distinct argmax ids in the golden positions" % label)
    if len(set(ids)) < 4:
        raise MarginError("%s: greedy output is degenerate" % label)
    if stats["max_precap"] < CAP / 2:
        raise MarginError("%s: softcap does not bite (max pre-cap |logit| %.3g < %g)"
                          % (label, stats["max_precap"], CAP / 2))
    if max(abs(v) for r in full for v in r) >= CAP:
        raise MarginError("%s: a capped logit reached the cap" % label)
    return full, ids, min(gaps), stats["max_precap"]


def build(seed):
    """Returns (files: {relative name: bytes}, info dict). Raises MarginError."""
    assert N_PROMPT > 2 * WINDOW and N_TOKENS > 2 * WINDOW
    rng = random.Random(seed)
    tensors = build_tensors(rng)
    tokens = [rng.randrange(VOCAB) for _ in range(N_TOKENS)]
    files = {"config.json": jdump(config(), indent=4),
             "generation_config.json": jdump({
                 "bos_token_id": 2, "do_sample": True, "eos_token_id": [1, 106, 50], "pad_token_id": 0,
                 "temperature": 1.0, "top_k": 64, "top_p": 0.95, "transformers_version": "5.5.0.dev0"},
                 indent=2),
             "model.safetensors": st.serialize(tensors, metadata={"format": "pt"}, order=list(tensors))}
    with tempfile.TemporaryDirectory() as d:
        for name, data in files.items():
            with open(os.path.join(d, name), "wb") as f:
                f.write(data)
        files["model.q8.safetensors"] = quant_ref.quantize_checkpoint(
            d, os.path.join(d, "model.q8.safetensors"))
        bf = ref.Model.load(d)
        q8 = ref.Model.load(d, os.path.join(d, "model.q8.safetensors"))
        bf_rows, bf_ids, bf_gap, bf_pre = run_model(bf, tokens, "bf16")
        q8_rows, q8_ids, q8_gap, q8_pre = run_model(q8, tokens, "q8")
        bf.cfg.window = 10 ** 6  # the sliding mask must be observable in the goldens
        nowin = bf.forward(tokens)
    top = max(abs(v) for r in bf_rows for v in r)
    diff = max(abs(a - b) for r1, r2 in zip(bf_rows, q8_rows) for a, b in zip(r1, r2))
    if diff <= 3 * GATE_REL_TOL * top:
        raise MarginError("q8 logits are too close to bf16 (max diff %.3g)" % diff)
    wdiff = max(abs(a - b) for r1, r2 in zip(bf_rows, nowin) for a, b in zip(r1, r2))
    if wdiff <= 3 * GATE_REL_TOL * top:
        raise MarginError("the sliding window does not change the logits (max diff %.3g)" % wdiff)
    files["golden_bf16.json"] = golden(tokens, bf_rows, bf_ids)
    files["golden_q8.json"] = golden(tokens, q8_rows, q8_ids)
    info = {"seed": seed, "min_gap_bf16": bf_gap, "min_gap_q8": q8_gap,
            "max_abs_logit": top, "max_precap_bf16": bf_pre, "max_precap_q8": q8_pre,
            "max_q8_vs_bf16": diff, "max_window_effect": wdiff,
            "gate_tolerance": GATE_REL_TOL * top,
            "greedy_bf16": bf_ids, "greedy_q8": q8_ids,
            "sha256_q8": hashlib.sha256(files["model.q8.safetensors"]).hexdigest()}
    return files, info


def default_dir():
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "tests", "fixtures", "gemma4-tiny")


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
        print("make_tiny_gemma4: seed %d fails its margins: %s" % (SEED, e), file=sys.stderr)
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
            print("make_tiny_gemma4 --check FAILED: " + "; ".join(bad), file=sys.stderr)
            return 1
        print("make_tiny_gemma4 --check OK: %d files byte-identical (seed %d, min gap bf16 %.4g q8 %.4g)"
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
