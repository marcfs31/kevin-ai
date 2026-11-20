#!/usr/bin/env python3
"""Dependency-free float64 oracle for the Gemma 4 text model (standard library only).

Port of the math in Hugging Face transformers `models/gemma4/modeling_gemma4.py`
(Copyright 2026 the HuggingFace Inc. team, Apache License 2.0): Gemma4RMSNorm, Gemma4TextMLP,
Gemma4TextRotaryEmbedding, Gemma4TextAttention, eager_attention_forward, Gemma4TextDecoderLayer,
Gemma4TextScaledWordEmbedding, Gemma4TextModel (get_per_layer_inputs, project_per_layer_inputs),
Gemma4ForCausalLM (final logit softcap), `configuration_gemma4.py` (kv_sharing_modes),
`modeling_rope_utils._compute_proportional_rope_parameters` and
`masking_utils.sliding_window_overlay`. This file is a re-implementation in plain Python
evaluated in float64.

Deliberate differences from the reference, none of which changes the model function:
  * every activation stays float64 (the reference casts to the checkpoint dtype between ops, and
    computes RoPE angles in float32); the embedding scales are still rounded to bf16 first, as the
    reference does (`embed_scale.to(weight.dtype)`);
  * the model runs token by token with a KV cache per producing layer; the attention mask is
    applied exactly (key j visible to query i iff j <= i and, for sliding layers, j > i - window),
    which is what the reference's full-sequence masks encode;
  * shared (reuse) layers read the cache of the last unshared layer of their type, never their
    own k_proj/v_proj/k_norm (those tensors are not even opened);
  * stricter than the reference about configuration: every field is read, a missing one is an
    error, and the flags listed in docs/PLAN.md (and a few more, see Config) are errors instead of
    silently being defaulted or ignored.

Weights come from a bf16 checkpoint directory (single file or sharded) or, with --weights, from an
lm-q8 file written by quant_ref.py, dequantised exactly in double ((double)q * (double)scale).
The config always comes from --model DIR/config.json (top-level model_type gemma4, text_config).

Module: `Model.load(model_dir, weights=None)`, `model.forward(tokens) -> [[vocab] per position]`,
`model.generate(prompt, n) -> (prompt rows, ids, steps)`.
CLI (the `lm-run` contract; engine-only flags are accepted and ignored):
    ref_gemma4.py --model DIR [--weights Q8FILE] --tokens 1,2,3 [--logits-out F] [--greedy N]
Stdout: one generated id per line. `--logits-out` writes one little-endian f32 row of
`vocab` logits per FED prompt token (generated tokens add no rows).
"""

import argparse
import json
import math
import os
import struct
import sys

import quant_ref
import ref_qwen35 as q35
import safetensors as st
from ref_qwen35 import Weights, argmax, dot, matvec, softmax, top_gap  # noqa: F401

PREFIX = q35.PREFIX
SLIDING, FULL = "sliding_attention", "full_attention"


# --------------------------------------------------------------------------- ops (float64)

def rmsnorm(x, w, eps):
    """y = x * (mean(x^2) + eps)^(-1/2) * w   (Gemma4RMSNorm: plain w, NOT 1 + w).

    w=None is the weightless norm (`v_norm`, with_scale=False).
    """
    r = (sum(v * v for v in x) / len(x) + eps) ** -0.5
    if w is None:
        return [v * r for v in x]
    return [v * r * s for v, s in zip(x, w)]


def gelu_tanh(x):
    """gelu_pytorch_tanh: 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))."""
    return 0.5 * x * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x * x * x)))


def softcap(x, cap):
    """cap * tanh(x / cap), applied to every entry."""
    return [cap * math.tanh(v / cap) for v in x]


def bf16_scale(v):
    """The embedding scale, rounded to bf16 like `embed_scale.to(weight.dtype)`."""
    return st.round_bf16(v)


def rope_freqs(head_dim, theta, rope_type="default", partial=1.0, factor=1.0):
    """f[i] for i < head_dim/2 (angle = pos * f[i]); rotate-half pairs (i, i + head_dim/2).

    default:      f[i] = theta^(-2i/d).
    proportional: r = int(partial*d // 2); f[i] = theta^(-2i/d) / factor for i < r, else 0
                  (those dims pass through). The exponent denominator is d, not the rotated width.
    """
    half = head_dim // 2
    if rope_type == "default":
        return [theta ** (-(2.0 * i) / head_dim) for i in range(half)]
    if rope_type == "proportional":
        r = int((partial * head_dim) // 2)
        return [theta ** (-(2.0 * i) / head_dim) / factor if i < r else 0.0 for i in range(half)]
    raise ValueError("rope_type %r is not supported (only default, proportional)" % rope_type)


def rope_apply(x, pos, freqs):
    """Rotate-half RoPE over the whole head vector with per-pair frequencies `freqs`."""
    half = len(freqs)
    out = list(x)
    for i, f in enumerate(freqs):
        c, s = math.cos(pos * f), math.sin(pos * f)
        a, b = x[i], x[i + half]
        out[i] = a * c - b * s
        out[i + half] = b * c + a * s
    return out


def mask_allows(i, j, window):
    """Query position i may see key position j (window None = plain causal)."""
    return j <= i and (window is None or j > i - window)


def attention_context(queries, ks, vs, pos, window, scaling=1.0):
    """One query position of grouped-query attention.

    queries: [heads][d] (already normed and roped). ks, vs: [T][kv_heads][d] for key positions
    0..T-1 with T = pos + 1 (the cache includes the current token). Head h uses KV head
    h // (heads/kv_heads). Returns the concatenated per-head context [heads*d].
    """
    heads, kvh = len(queries), len(ks[0])
    if len(ks) != pos + 1 or len(vs) != pos + 1:
        raise ValueError("KV cache holds %d positions, expected %d" % (len(ks), pos + 1))
    group = heads // kvh
    visible = [j for j in range(pos + 1) if mask_allows(pos, j, window)]
    out = []
    for h in range(heads):
        kv = h // group
        probs = softmax([dot(queries[h], ks[j][kv]) * scaling for j in visible])
        d = len(queries[h])
        out.extend(sum(pr * vs[j][kv][c] for pr, j in zip(probs, visible)) for c in range(d))
    return out


# --------------------------------------------------------------------------- config

UNSUPPORTED_FLAGS = (
    # (key, value that is rejected, text)
    ("enable_moe_block", True, "MoE blocks"),
    ("attention_k_eq_v", True, "attention_k_eq_v"),
    ("use_double_wide_mlp", True, "use_double_wide_mlp"),
    ("attention_bias", True, "attention_bias"),
)


class Config:
    def __init__(self, raw):
        try:
            top_type = raw["model_type"]
            t = raw["text_config"]
        except (KeyError, TypeError) as e:
            raise ValueError("config.json is missing %s" % e)
        if top_type != "gemma4":
            raise ValueError("model_type %r is not supported (only gemma4)" % top_type)
        try:
            if t["model_type"] != "gemma4_text":
                raise ValueError("text_config.model_type %r is not supported (only gemma4_text)"
                                 % t["model_type"])
            self.hidden = t["hidden_size"]
            self.inter = t["intermediate_size"]
            self.layers = t["num_hidden_layers"]
            self.layer_types = list(t["layer_types"])
            self.vocab = t["vocab_size"]
            self.vocab_ple = t["vocab_size_per_layer_input"]
            self.eps = t["rms_norm_eps"]
            self.heads = t["num_attention_heads"]
            self.kv_heads = t["num_key_value_heads"]
            self.head_dim = t["head_dim"]
            self.global_head_dim = t["global_head_dim"]
            self.act = t["hidden_activation"]
            self.window = t["sliding_window"]
            self.kv_shared = t["num_kv_shared_layers"]
            self.ple_dim = t["hidden_size_per_layer_input"]
            self.cap = t["final_logit_softcapping"]
            tied = t["tie_word_embeddings"]
            flags = {k: t[k] for k, _, _ in UNSUPPORTED_FLAGS}
            bidir = t["use_bidirectional_attention"]
            global_kv = t["num_global_key_value_heads"]
            rp = t["rope_parameters"]
            rope = {}
            for lt in sorted(set(self.layer_types)):
                if lt not in (SLIDING, FULL):
                    raise ValueError("layer type %r is not supported" % lt)
                e = rp[lt]
                rtype = e["rope_type"]
                if rtype == "default":
                    if e.get("partial_rotary_factor", 1.0) != 1.0:
                        raise ValueError("default rope with partial_rotary_factor != 1 is not supported")
                    rope[lt] = ("default", float(e["rope_theta"]), 1.0, 1.0)
                elif rtype == "proportional":
                    rope[lt] = ("proportional", float(e["rope_theta"]),
                                float(e["partial_rotary_factor"]), float(e.get("factor", 1.0)))
                else:
                    raise ValueError("rope_type %r is not supported (only default, proportional)" % rtype)
        except KeyError as e:
            raise ValueError("config.json is missing %s" % e)
        for key, bad, text in UNSUPPORTED_FLAGS:
            if flags[key] is True:
                raise ValueError("%s is not supported" % text)
            if flags[key] is not False:
                raise ValueError("%s must be false, got %r" % (key, flags[key]))
        if bidir is not None:
            raise ValueError("use_bidirectional_attention=%r is not supported" % (bidir,))
        if global_kv is not None:
            raise ValueError("num_global_key_value_heads is not supported")
        if t.get("per_layer_config") is not None:
            raise ValueError("an explicit per_layer_config is not supported")
        if self.act != "gelu_pytorch_tanh":
            raise ValueError("hidden_activation %r is not supported (only gelu_pytorch_tanh)" % self.act)
        if tied is not True or raw.get("tie_word_embeddings", True) is not True:
            raise ValueError("tie_word_embeddings=false is not supported")
        if not self.ple_dim:
            raise ValueError("hidden_size_per_layer_input=0 is not supported")
        if not isinstance(self.cap, (int, float)) or isinstance(self.cap, bool) or self.cap <= 0:
            raise ValueError("final_logit_softcapping must be a positive number, got %r" % (self.cap,))
        if len(self.layer_types) != self.layers or self.layers < 1:
            raise ValueError("layer_types does not match num_hidden_layers")
        if self.layer_types[-1] != FULL:
            raise ValueError("the last layer must be full_attention")
        if self.heads % self.kv_heads:
            raise ValueError("head counts are not divisible")
        if self.head_dim % 2 or self.global_head_dim % 2:
            raise ValueError("head dims must be even")
        if self.window < 1:
            raise ValueError("sliding_window must be >= 1")
        if not 0 <= self.kv_shared < self.layers:
            raise ValueError("num_kv_shared_layers out of range")
        self.rope = rope
        self.first_shared = self.layers - self.kv_shared
        # configuration_gemma4.kv_sharing_modes: the producer of a type is the LAST layer of that
        # type before `first_shared`; every layer >= first_shared reuses its type's producer.
        self.producer = {}
        if self.kv_shared:
            before = self.layer_types[:self.first_shared]
            if set(before) != set(self.layer_types):
                raise ValueError("a layer type used by shared layers has no producing layer")
            for lt in set(before):
                self.producer[lt] = self.first_shared - 1 - before[::-1].index(lt)
        self.embed_scale = bf16_scale(math.sqrt(self.hidden))
        self.ple_scale = bf16_scale(math.sqrt(self.ple_dim))

    def reuses(self, layer):
        return self.kv_shared > 0 and layer >= self.first_shared

    def hd(self, layer):
        return self.global_head_dim if self.layer_types[layer] == FULL else self.head_dim

    def freqs(self, layer):
        lt = self.layer_types[layer]
        rtype, theta, partial, factor = self.rope[lt]
        return rope_freqs(self.hd(layer), theta, rtype, partial, factor)


# --------------------------------------------------------------------------- model

class Model:
    def __init__(self, cfg, weights):
        self.cfg, self.w = cfg, weights
        self._check_shapes()
        self._freqs = [cfg.freqs(i) for i in range(cfg.layers)]

    @classmethod
    def load(cls, model_dir, weights=None):
        with open(os.path.join(model_dir, "config.json")) as f:
            cfg = Config(json.load(f))
        if weights is None:
            return cls(cfg, Weights(st.open_checkpoint(model_dir)))
        return cls(cfg, Weights(quant_ref.load_q8(weights), q8=True))

    def expected_shapes(self):
        c = self.cfg
        lp = c.layers * c.ple_dim
        expect = {PREFIX + "embed_tokens.weight": (c.vocab, c.hidden),
                  PREFIX + "embed_tokens_per_layer.weight": (c.vocab_ple, lp),
                  PREFIX + "per_layer_model_projection.weight": (lp, c.hidden),
                  PREFIX + "per_layer_projection_norm.weight": (c.ple_dim,),
                  PREFIX + "norm.weight": (c.hidden,)}
        for i in range(c.layers):
            p = "%slayers.%d." % (PREFIX, i)
            hd = c.hd(i)
            for n in ("input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm",
                      "post_feedforward_layernorm", "post_per_layer_input_norm"):
                expect[p + n + ".weight"] = (c.hidden,)
            expect[p + "layer_scalar"] = (1,)
            expect[p + "mlp.gate_proj.weight"] = (c.inter, c.hidden)
            expect[p + "mlp.up_proj.weight"] = (c.inter, c.hidden)
            expect[p + "mlp.down_proj.weight"] = (c.hidden, c.inter)
            expect[p + "per_layer_input_gate.weight"] = (c.ple_dim, c.hidden)
            expect[p + "per_layer_projection.weight"] = (c.hidden, c.ple_dim)
            a = p + "self_attn."
            expect[a + "q_proj.weight"] = (c.heads * hd, c.hidden)
            expect[a + "o_proj.weight"] = (c.hidden, c.heads * hd)
            expect[a + "q_norm.weight"] = (hd,)
            if not c.reuses(i):  # reuse layers' k_proj/v_proj/k_norm exist in the file and are ignored
                expect[a + "k_proj.weight"] = (c.kv_heads * hd, c.hidden)
                expect[a + "v_proj.weight"] = (c.kv_heads * hd, c.hidden)
                expect[a + "k_norm.weight"] = (hd,)
        return expect

    def _check_shapes(self):
        for name, shape in self.expected_shapes().items():
            if not self.w.store.has(name):
                raise ValueError("tensor %s is missing" % name)
            if tuple(self.w.shape(name)) != shape:
                raise ValueError("tensor %s has shape %s, expected %s"
                                 % (name, tuple(self.w.shape(name)), shape))

    def new_state(self):
        return {"kv": {i: {"k": [], "v": []} for i in range(self.cfg.layers) if not self.cfg.reuses(i)},
                "pos": 0}

    def per_layer_inputs(self, token, e):
        """ple[l] for one token: (rms(proj(e) * hidden^-1/2, norm) + embed_ple[t] * s_P) * 2^-1/2."""
        c, w = self.cfg, self.w
        if not 0 <= token < c.vocab_ple:
            raise ValueError("token id %d out of range for the per-layer table" % token)
        tok = [v * c.ple_scale for v in w.get(PREFIX + "embed_tokens_per_layer.weight")[token]]
        ctx = [v * c.hidden ** -0.5 for v in matvec(w.get(PREFIX + "per_layer_model_projection.weight"), e)]
        pn = w.get(PREFIX + "per_layer_projection_norm.weight")
        P = c.ple_dim
        out = []
        for layer in range(c.layers):
            row = rmsnorm(ctx[layer * P:(layer + 1) * P], pn, c.eps)
            out.append([(a + b) * 2.0 ** -0.5 for a, b in zip(row, tok[layer * P:(layer + 1) * P])])
        return out

    def _mlp(self, p, x):
        g = matvec(self.w.get(p + "gate_proj.weight"), x)
        u = matvec(self.w.get(p + "up_proj.weight"), x)
        return matvec(self.w.get(p + "down_proj.weight"), [gelu_tanh(a) * b for a, b in zip(g, u)])

    def _attn(self, layer, x, state, pos):
        c, w = self.cfg, self.w
        lt, hd = c.layer_types[layer], c.hd(layer)
        p = "%slayers.%d.self_attn." % (PREFIX, layer)
        freqs = self._freqs[layer]
        q = matvec(w.get(p + "q_proj.weight"), x)
        qn = w.get(p + "q_norm.weight")
        queries = [rope_apply(rmsnorm(q[h * hd:(h + 1) * hd], qn, c.eps), pos, freqs)
                   for h in range(c.heads)]
        if c.reuses(layer):
            kv = state["kv"][c.producer[lt]]
        else:
            kv = state["kv"][layer]
            k = matvec(w.get(p + "k_proj.weight"), x)
            v = matvec(w.get(p + "v_proj.weight"), x)
            kn = w.get(p + "k_norm.weight")
            kv["k"].append([rope_apply(rmsnorm(k[h * hd:(h + 1) * hd], kn, c.eps), pos, freqs)
                            for h in range(c.kv_heads)])
            kv["v"].append([rmsnorm(v[h * hd:(h + 1) * hd], None, c.eps) for h in range(c.kv_heads)])
        ctx = attention_context(queries, kv["k"], kv["v"], pos, c.window if lt == SLIDING else None)
        return matvec(w.get(p + "o_proj.weight"), ctx)

    def step_hidden(self, state, h, ple_token, stats=None):
        """One position whose residual is `h` and whose per-layer table row is `ple_token`.

        For a text token, `h` is the scaled embedding and `ple_token` is the token. For a vision
        row, `h` is the row itself (not multiplied by the text embedding scale) and `ple_token`
        is the pad id: Gemma4Model.forward looks the pad id up in the per-layer table and
        projects the embeddings after the vision rows have been scattered in. Returns the vocab
        logits (after the softcap). Mutates `state`. `stats`, if a dict, receives "max_precap".
        """
        c, w = self.cfg, self.w
        if not 0 <= ple_token < c.vocab_ple:
            raise ValueError("token id %d out of range for the per-layer table" % ple_token)
        pos = state["pos"]
        ple = self.per_layer_inputs(ple_token, h)
        embed = w.get(PREFIX + "embed_tokens.weight")
        for i in range(c.layers):
            p = "%slayers.%d." % (PREFIX, i)
            n1 = rmsnorm(h, w.get(p + "input_layernorm.weight"), c.eps)
            a = rmsnorm(self._attn(i, n1, state, pos), w.get(p + "post_attention_layernorm.weight"), c.eps)
            h = [x + y for x, y in zip(h, a)]
            n2 = rmsnorm(h, w.get(p + "pre_feedforward_layernorm.weight"), c.eps)
            m = rmsnorm(self._mlp(p + "mlp.", n2), w.get(p + "post_feedforward_layernorm.weight"), c.eps)
            h = [x + y for x, y in zip(h, m)]
            g = [gelu_tanh(v) * s for v, s in zip(matvec(w.get(p + "per_layer_input_gate.weight"), h), ple[i])]
            d = rmsnorm(matvec(w.get(p + "per_layer_projection.weight"), g),
                        w.get(p + "post_per_layer_input_norm.weight"), c.eps)
            scalar = w.get(p + "layer_scalar")[0]
            h = [(x + y) * scalar for x, y in zip(h, d)]
        state["pos"] = pos + 1
        h = rmsnorm(h, w.get(PREFIX + "norm.weight"), c.eps)
        raw = matvec(embed, h)
        if stats is not None:
            stats["max_precap"] = max(stats.get("max_precap", 0.0), max(abs(v) for v in raw))
        return softcap(raw, c.cap)

    def step(self, state, token, stats=None):
        """Feed one text token; returns the vocab logits row (after the softcap). Mutates `state`."""
        c, w = self.cfg, self.w
        embed = w.get(PREFIX + "embed_tokens.weight")
        if not 0 <= token < c.vocab:
            raise ValueError("token id %d out of range" % token)
        h = [v * c.embed_scale for v in embed[token]]
        return self.step_hidden(state, h, token, stats)

    def image_block(self, state, rows, start, end, pad, stats=None):
        """Start token, one position per vision row, end token. Returns the logits after the end.

        Positions increase by one and attention stays the text model's causal mask. `rows` are
        the vision encoder's output, in order, unscaled.
        """
        self.step(state, start, stats)
        for row in rows:
            self.step_hidden(state, row, pad, stats)
        return self.step(state, end, stats)

    def forward(self, tokens, stats=None):
        state = self.new_state()
        return [self.step(state, t, stats) for t in tokens]

    def generate(self, prompt, n, stats=None):
        """Feed prompt, then n greedy tokens (argmax, lowest id wins ties).

        Returns (rows for the prompt tokens, generated ids, logits row behind each generated id).
        """
        state = self.new_state()
        rows = [self.step(state, t, stats) for t in prompt]
        last, ids, steps = rows[-1], [], []
        for _ in range(n):
            nxt = argmax(last)
            ids.append(nxt)
            steps.append(last)
            last = self.step(state, nxt, stats)
        return rows, ids, steps


# --------------------------------------------------------------------------- CLI

def main(argv=None):
    ap = argparse.ArgumentParser(description="float64 Gemma 4 text oracle (lm-run contract)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--weights")
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--logits-out")
    ap.add_argument("--greedy", type=int, default=0)
    ap.add_argument("--ctx")
    ap.add_argument("--threads")
    a = ap.parse_args(argv)
    try:
        tokens = [int(t) for t in a.tokens.split(",") if t != ""]
    except ValueError:
        ap.error("--tokens must be a comma separated list of ids")
    if not tokens:
        ap.error("--tokens is empty")
    if a.greedy < 0:
        ap.error("--greedy must be >= 0")
    model = Model.load(a.model, a.weights)
    rows, ids, _ = model.generate(tokens, a.greedy)
    if a.logits_out:
        with open(a.logits_out, "wb") as f:
            for r in rows:
                f.write(struct.pack("<%df" % len(r), *r))
    for i in ids:
        sys.stdout.write("%d\n" % i)
    return 0


if __name__ == "__main__":
    sys.exit(main())
