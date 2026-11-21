#!/usr/bin/env python3
"""Dependency-free float64 oracle for the Qwen3.5 text model (standard library only).

Port of the math in Hugging Face transformers `models/qwen3_5/modeling_qwen3_5.py`
(Copyright 2025 The Qwen Team and The HuggingFace Inc. team, Apache License 2.0):
Qwen3_5TextRotaryEmbedding, Qwen3_5RMSNormGated, torch_recurrent_gated_delta_rule,
Qwen3_5GatedDeltaNet, Qwen3_5Attention, Qwen3_5MLP, Qwen3_5RMSNorm, Qwen3_5DecoderLayer,
Qwen3_5TextModel. This file is a re-implementation in plain Python evaluated in float64.

Deliberate differences from the reference, none of which changes the model function:
  * every activation stays float64 (the reference casts to the checkpoint dtype between ops);
  * linear attention runs token by token with the recurrent gated delta rule (the chunked
    algorithm in the reference is the same maths in a parallel form and is not ported);
  * softplus is the exact function (torch switches to identity above 20; difference < 3e-9);
  * for text the three mRoPE position streams are identical, so the interleaved mRoPE
    recomposition is the identity and plain RoPE on the first `rot` dims is applied;
  * `attn_output_gate` is not read by the reference (the gate is always there); this oracle
    requires it to be true and fails otherwise.

Weights come from a bf16 checkpoint directory (sharded or single file) or, with --weights,
from an lm-q8 file written by quant_ref.py, dequantised exactly in double
((double)q * (double)scale). The config always comes from --model DIR/config.json.

Module: `Model.load(model_dir, weights=None)`, `model.forward(tokens) -> [[vocab] per position]`,
`model.generate(prompt, n) -> (prompt rows, ids, steps)`, `model.mtp_rows(tokens)` (the MTP head).

MTP head (`mtp.*`, ported from vLLM `model_executor/models/qwen3_5_mtp.py`, Apache-2.0): at
position p it takes the final normed hidden state h_p of the main model and the token at
p+1, x = fc([norm_e(embed(t_{p+1})), norm_h(h_p)]), runs one full-attention decoder layer
(its own KV cache, RoPE position p), and drafts the token at p+2 from head(norm(x)).
CLI (the `lm-run` contract; engine-only flags are accepted and ignored):
    ref_qwen35.py --model DIR [--weights Q8FILE] --tokens 1,2,3 [--logits-out F] [--greedy N]
Stdout: one generated id per line. `--logits-out` writes one little-endian f32 row of
`vocab` logits per FED prompt token (generated tokens add no rows).
"""

import argparse
import json
import math
import os
import struct
import sys
from operator import mul

import quant_ref
import safetensors as st

PREFIX = "model.language_model."
MTP = "mtp."


# --------------------------------------------------------------------------- ops (float64)

def dot(a, b):
    return sum(map(mul, a, b))


def matvec(rows, x):
    return [sum(map(mul, r, x)) for r in rows]


def sigmoid(v):
    if v >= 0:
        return 1.0 / (1.0 + math.exp(-v))
    e = math.exp(v)
    return e / (1.0 + e)


def silu(v):
    return v * sigmoid(v)


def softplus(v):
    return max(v, 0.0) + math.log1p(math.exp(-abs(v)))


def rmsnorm_zero_centred(x, w, eps):
    """y = x * rsqrt(mean(x^2) + eps) * (1 + w)   (Qwen3_5RMSNorm)."""
    r = 1.0 / math.sqrt(sum(v * v for v in x) / len(x) + eps)
    return [v * r * (1.0 + s) for v, s in zip(x, w)]


def rmsnorm_gated(x, z, w, eps):
    """y = x * rsqrt(mean(x^2) + eps) * w * silu(z)   (Qwen3_5RMSNormGated; plain w)."""
    r = 1.0 / math.sqrt(sum(v * v for v in x) / len(x) + eps)
    return [v * r * s * silu(g) for v, s, g in zip(x, w, z)]


def l2norm(x, eps=1e-6):
    """x * rsqrt(sum(x^2) + eps)."""
    r = 1.0 / math.sqrt(sum(v * v for v in x) + eps)
    return [v * r for v in x]


def rope_inv_freq(rot_dim, theta):
    return [theta ** (-(2.0 * i) / rot_dim) for i in range(rot_dim // 2)]


def section_axis(k, section):
    """Which position (0 = t, 1 = h, 2 = w) rotary frequency k uses:
    Qwen3_5TextRotaryEmbedding.recomposition_frequencies, interleaved."""
    if k % 3 == 1 and k < section[1] * 3:
        return 1
    if k % 3 == 2 and k < section[2] * 3:
        return 2
    return 0


def rope_apply(x, pos, rot_dim, theta, section=None):
    """Rotate-half RoPE on the first rot_dim entries of one head vector; the rest pass through.
    pos is an int, or a (t, h, w) tuple with `section` (mrope_section): frequency k then takes
    pos[section_axis(k, section)]."""
    half = rot_dim // 2
    out = list(x)
    for i, f in enumerate(rope_inv_freq(rot_dim, theta)):
        p = pos[section_axis(i, section)] if isinstance(pos, tuple) else pos
        c, s = math.cos(p * f), math.sin(p * f)
        a, b = x[i], x[i + half]
        out[i] = a * c - b * s
        out[i + half] = b * c + a * s
    return out


def conv1d_silu_step(state, x, weight):
    """Causal depthwise conv + SiLU for one position.

    state: list of K-1 previous input vectors, oldest first (zeros at position 0); it is
    updated in place. weight: [C][K], index 0 multiplies the oldest of the K inputs.
    """
    k = len(weight[0])
    window = state + [x]
    out = [silu(sum(w[j] * window[j][c] for j in range(k))) for c, w in enumerate(weight)]
    state.pop(0)
    state.append(x)
    return out


def delta_rule_step(S, q, k, v, g, beta):
    """One gated delta rule step for one value head; S [dk][dv] updated in place.

    q, k are already l2-normalised (q also scaled by 1/sqrt(dk)); g, beta are scalars.
    S *= exp(g); kv = S^T k; delta = (v - kv) * beta; S += k outer delta; o = S^T q.
    """
    dk, dv = len(k), len(v)
    decay = math.exp(g)
    for d in range(dk):
        row = S[d]
        for j in range(dv):
            row[j] *= decay
    kv = [sum(S[d][j] * k[d] for d in range(dk)) for j in range(dv)]
    delta = [(v[j] - kv[j]) * beta for j in range(dv)]
    for d in range(dk):
        row, kd = S[d], k[d]
        for j in range(dv):
            row[j] += kd * delta[j]
    return [sum(S[d][j] * q[d] for d in range(dk)) for j in range(dv)]


def gated_delta_rule(seq, num_k_heads, num_v_heads, dk, dv, state=None):
    """seq: list of steps (q, k, v, g, beta). q,k: raw [nk*dk] (post-conv, pre-l2norm);
    v [nv*dv]; g, beta [nv]. Returns (outputs [T][nv*dv], state [nv][dk][dv])."""
    rep = num_v_heads // num_k_heads
    if state is None:
        state = [[[0.0] * dv for _ in range(dk)] for _ in range(num_v_heads)]
    outs = []
    for q, k, v, g, beta in seq:
        qn = [[c / math.sqrt(dk) for c in l2norm(q[h * dk:(h + 1) * dk])] for h in range(num_k_heads)]
        kn = [l2norm(k[h * dk:(h + 1) * dk]) for h in range(num_k_heads)]
        o = []
        for i in range(num_v_heads):
            kh = i // rep
            o.extend(delta_rule_step(state[i], qn[kh], kn[kh], v[i * dv:(i + 1) * dv], g[i], beta[i]))
        outs.append(o)
    return outs, state


def softmax(xs):
    m = max(xs)
    es = [math.exp(v - m) for v in xs]
    s = sum(es)
    return [e / s for e in es]


# --------------------------------------------------------------------------- config

class Config:
    def __init__(self, raw):
        t = raw.get("text_config", raw)
        try:
            self.hidden = t["hidden_size"]
            self.inter = t["intermediate_size"]
            self.layers = t["num_hidden_layers"]
            self.layer_types = list(t["layer_types"])
            self.vocab = t["vocab_size"]
            self.eps = t["rms_norm_eps"]
            self.heads = t["num_attention_heads"]
            self.kv_heads = t["num_key_value_heads"]
            self.head_dim = t["head_dim"]
            rp = t["rope_parameters"]
            self.theta = float(rp["rope_theta"])
            self.partial = float(rp.get("partial_rotary_factor", 1.0))
            self.mrope = [int(v) for v in rp["mrope_section"]]
            if rp.get("mrope_interleaved") is not True or len(self.mrope) != 3:
                raise ValueError("mrope_interleaved must be true with a 3-entry mrope_section")
            rope_type = rp.get("rope_type", "default")
            self.lk_heads = t["linear_num_key_heads"]
            self.lv_heads = t["linear_num_value_heads"]
            self.lk_dim = t["linear_key_head_dim"]
            self.lv_dim = t["linear_value_head_dim"]
            self.conv_k = t["linear_conv_kernel_dim"]
            self.act = t["hidden_act"]
            self.tied = bool(raw.get("tie_word_embeddings", t.get("tie_word_embeddings", True)))
            self.attn_gate = t.get("attn_output_gate", True)
        except KeyError as e:
            raise ValueError("config.json is missing %s" % e)
        if rope_type != "default":
            raise ValueError("rope_type %r is not supported (only default)" % rope_type)
        if self.act != "silu":
            raise ValueError("hidden_act %r is not supported (only silu)" % self.act)
        if not self.attn_gate:
            raise ValueError("attn_output_gate=false is not supported")
        if len(self.layer_types) != self.layers or not all(
                x in ("linear_attention", "full_attention") for x in self.layer_types):
            raise ValueError("layer_types does not match num_hidden_layers")
        if self.heads % self.kv_heads or self.lv_heads % self.lk_heads:
            raise ValueError("head counts are not divisible")
        self.rot = int(self.head_dim * self.partial)
        if self.rot % 2:
            raise ValueError("rotary dim must be even")
        if sum(self.mrope) != self.rot // 2:
            raise ValueError("mrope_section must sum to rot/2")
        self.key_dim = self.lk_heads * self.lk_dim
        self.value_dim = self.lv_heads * self.lv_dim
        self.conv_dim = 2 * self.key_dim + self.value_dim


# --------------------------------------------------------------------------- weights

class Weights:
    """Lazy float64 view of a bf16 checkpoint or a q8 file, reshaped to nested lists."""

    def __init__(self, store, q8=False):
        self.store, self.q8, self._cache = store, q8, {}

    def _flat(self, name):
        s = self.store
        if self.q8 and s.has(name + quant_ref.SCALE_SUFFIX):
            rows, cols = s.shape(name)
            return quant_ref.dequantize_matrix(
                s.ints(name), s.floats(name + quant_ref.SCALE_SUFFIX), rows, cols)
        return s.floats(name)

    def get(self, name):
        if name not in self._cache:
            if not self.store.has(name):
                raise ValueError("tensor %s is missing" % name)
            flat, shape = self._flat(name), self.store.shape(name)
            if len(shape) == 1:
                val = flat
            else:
                inner = st.numel(shape[1:])
                # [rows, inner]; the conv1d weight [C, 1, K] becomes [C][K]
                val = [flat[i * inner:(i + 1) * inner] for i in range(shape[0])]
            self._cache[name] = val
        return self._cache[name]

    def shape(self, name):
        return self.store.shape(name)


# --------------------------------------------------------------------------- model

class Model:
    def __init__(self, cfg, weights):
        self.cfg, self.w = cfg, weights
        self._check_shapes()

    @classmethod
    def load(cls, model_dir, weights=None):
        with open(os.path.join(model_dir, "config.json")) as f:
            cfg = Config(json.load(f))
        if weights is None:
            return cls(cfg, Weights(st.open_checkpoint(model_dir)))
        return cls(cfg, Weights(quant_ref.load_q8(weights), q8=True))

    def has_mtp(self):
        return self.w.store.has(MTP + "fc.weight")

    def _attn_shapes(self, a):
        c = self.cfg
        return {a + "q_proj.weight": (c.heads * c.head_dim * 2, c.hidden),
                a + "k_proj.weight": (c.kv_heads * c.head_dim, c.hidden),
                a + "v_proj.weight": (c.kv_heads * c.head_dim, c.hidden),
                a + "o_proj.weight": (c.hidden, c.heads * c.head_dim),
                a + "q_norm.weight": (c.head_dim,), a + "k_norm.weight": (c.head_dim,)}

    def _layer_shapes(self, p):
        c = self.cfg
        return {p + "input_layernorm.weight": (c.hidden,), p + "post_attention_layernorm.weight": (c.hidden,),
                p + "mlp.gate_proj.weight": (c.inter, c.hidden), p + "mlp.up_proj.weight": (c.inter, c.hidden),
                p + "mlp.down_proj.weight": (c.hidden, c.inter)}

    def _check_shapes(self):
        c, w = self.cfg, self.w
        expect = {PREFIX + "embed_tokens.weight": (c.vocab, c.hidden), PREFIX + "norm.weight": (c.hidden,)}
        if self.has_mtp():
            expect[MTP + "fc.weight"] = (c.hidden, 2 * c.hidden)
            for n in ("norm", "pre_fc_norm_embedding", "pre_fc_norm_hidden"):
                expect[MTP + n + ".weight"] = (c.hidden,)
            expect.update(self._layer_shapes(MTP + "layers.0."))
            expect.update(self._attn_shapes(MTP + "layers.0.self_attn."))
        if not c.tied:
            expect["lm_head.weight"] = (c.vocab, c.hidden)
        for i, lt in enumerate(c.layer_types):
            p = "%slayers.%d." % (PREFIX, i)
            expect[p + "input_layernorm.weight"] = (c.hidden,)
            expect[p + "post_attention_layernorm.weight"] = (c.hidden,)
            expect[p + "mlp.gate_proj.weight"] = (c.inter, c.hidden)
            expect[p + "mlp.up_proj.weight"] = (c.inter, c.hidden)
            expect[p + "mlp.down_proj.weight"] = (c.hidden, c.inter)
            if lt == "linear_attention":
                a = p + "linear_attn."
                expect[a + "in_proj_qkv.weight"] = (c.conv_dim, c.hidden)
                expect[a + "in_proj_z.weight"] = (c.value_dim, c.hidden)
                expect[a + "in_proj_b.weight"] = (c.lv_heads, c.hidden)
                expect[a + "in_proj_a.weight"] = (c.lv_heads, c.hidden)
                expect[a + "conv1d.weight"] = (c.conv_dim, 1, c.conv_k)
                expect[a + "dt_bias"] = (c.lv_heads,)
                expect[a + "A_log"] = (c.lv_heads,)
                expect[a + "norm.weight"] = (c.lv_dim,)
                expect[a + "out_proj.weight"] = (c.hidden, c.value_dim)
            else:
                a = p + "self_attn."
                expect[a + "q_proj.weight"] = (c.heads * c.head_dim * 2, c.hidden)
                expect[a + "k_proj.weight"] = (c.kv_heads * c.head_dim, c.hidden)
                expect[a + "v_proj.weight"] = (c.kv_heads * c.head_dim, c.hidden)
                expect[a + "o_proj.weight"] = (c.hidden, c.heads * c.head_dim)
                expect[a + "q_norm.weight"] = (c.head_dim,)
                expect[a + "k_norm.weight"] = (c.head_dim,)
        for name, shape in expect.items():
            if not w.store.has(name):
                raise ValueError("tensor %s is missing" % name)
            if tuple(w.shape(name)) != shape:
                raise ValueError("tensor %s has shape %s, expected %s" % (name, tuple(w.shape(name)), shape))

    # state: one entry per layer
    def new_state(self):
        c = self.cfg
        st_ = []
        for lt in c.layer_types:
            if lt == "linear_attention":
                st_.append({
                    "conv": [[0.0] * c.conv_dim for _ in range(c.conv_k - 1)],
                    "S": [[[0.0] * c.lv_dim for _ in range(c.lk_dim)] for _ in range(c.lv_heads)],
                })
            else:
                st_.append({"k": [], "v": []})
        return {"layers": st_, "pos": 0, "rope": 0}

    def _mlp(self, p, x):
        g = matvec(self.w.get(p + "gate_proj.weight"), x)
        u = matvec(self.w.get(p + "up_proj.weight"), x)
        return matvec(self.w.get(p + "down_proj.weight"), [silu(a) * b for a, b in zip(g, u)])

    def _full_attn(self, p, x, ls, pos):
        c, w = self.cfg, self.w
        hd = c.head_dim
        qg = matvec(w.get(p + "q_proj.weight"), x)
        k = matvec(w.get(p + "k_proj.weight"), x)
        v = matvec(w.get(p + "v_proj.weight"), x)
        qn, kn = w.get(p + "q_norm.weight"), w.get(p + "k_norm.weight")
        queries, gate = [], []
        for h in range(c.heads):
            seg = qg[h * 2 * hd:(h + 1) * 2 * hd]
            queries.append(rope_apply(rmsnorm_zero_centred(seg[:hd], qn, c.eps), pos, c.rot, c.theta, c.mrope))
            gate.extend(seg[hd:])
        ls["k"].append([rope_apply(rmsnorm_zero_centred(k[h * hd:(h + 1) * hd], kn, c.eps),
                                   pos, c.rot, c.theta, c.mrope) for h in range(c.kv_heads)])
        ls["v"].append([v[h * hd:(h + 1) * hd] for h in range(c.kv_heads)])
        group = c.heads // c.kv_heads
        scale = hd ** -0.5
        out = []
        for h in range(c.heads):
            kvh = h // group
            probs = softmax([dot(queries[h], kk[kvh]) * scale for kk in ls["k"]])
            out.extend(sum(pr * vv[kvh][j] for pr, vv in zip(probs, ls["v"])) for j in range(hd))
        out = [o * sigmoid(g) for o, g in zip(out, gate)]
        return matvec(w.get(p + "o_proj.weight"), out)

    def _linear_attn(self, p, x, ls):
        c, w = self.cfg, self.w
        qkv = matvec(w.get(p + "in_proj_qkv.weight"), x)
        z = matvec(w.get(p + "in_proj_z.weight"), x)
        b = matvec(w.get(p + "in_proj_b.weight"), x)
        a = matvec(w.get(p + "in_proj_a.weight"), x)
        qkv = conv1d_silu_step(ls["conv"], qkv, w.get(p + "conv1d.weight"))
        kd = c.key_dim
        a_log, dt_bias = w.get(p + "A_log"), w.get(p + "dt_bias")
        g = [-math.exp(a_log[i]) * softplus(a[i] + dt_bias[i]) for i in range(c.lv_heads)]
        beta = [sigmoid(v) for v in b]
        outs, _ = gated_delta_rule([(qkv[:kd], qkv[kd:2 * kd], qkv[2 * kd:], g, beta)],
                                   c.lk_heads, c.lv_heads, c.lk_dim, c.lv_dim, state=ls["S"])
        o, nw = outs[0], w.get(p + "norm.weight")
        dv = c.lv_dim
        y = []
        for h in range(c.lv_heads):
            y.extend(rmsnorm_gated(o[h * dv:(h + 1) * dv], z[h * dv:(h + 1) * dv], nw, c.eps))
        return matvec(w.get(p + "out_proj.weight"), y)

    def step(self, state, token, emb=None, rp=None):
        """Feed one token; returns the vocab logits row. Mutates `state`.
        emb: an input embedding replacing the lookup (an image row); rp: its (t, h, w) RoPE position
        (default: the running state["rope"], which then advances by one)."""
        return self.step_h(state, token, emb, rp)[0]

    def step_h(self, state, token, emb=None, rp=None):
        """step, also returning the final normed hidden state (the MTP head's input)."""
        c, w = self.cfg, self.w
        embed = w.get(PREFIX + "embed_tokens.weight")
        if not 0 <= token < c.vocab:
            raise ValueError("token id %d out of range" % token)
        h = list(embed[token]) if emb is None else list(emb)
        pos = state["pos"]
        if rp is None:
            rp = (state["rope"],) * 3
            state["rope"] += 1
        for i, lt in enumerate(c.layer_types):
            p = "%slayers.%d." % (PREFIX, i)
            n1 = rmsnorm_zero_centred(h, w.get(p + "input_layernorm.weight"), c.eps)
            if lt == "linear_attention":
                m = self._linear_attn(p + "linear_attn.", n1, state["layers"][i])
            else:
                m = self._full_attn(p + "self_attn.", n1, state["layers"][i], rp)
            h = [a + b for a, b in zip(h, m)]
            n2 = rmsnorm_zero_centred(h, w.get(p + "post_attention_layernorm.weight"), c.eps)
            h = [a + b for a, b in zip(h, self._mlp(p + "mlp.", n2))]
        state["pos"] = pos + 1
        h = rmsnorm_zero_centred(h, w.get(PREFIX + "norm.weight"), c.eps)
        head = embed if c.tied else w.get("lm_head.weight")
        return matvec(head, h), h

    def feed_image(self, state, rows, lh, lw, start, end, mrope=True):
        """The start token, one row per merged-grid cell (row-major over lh x lw), the end token:
        Qwen3_5Model.get_rope_index. The rows sit at (p, p + i, p + j), p the position after the
        start token, and the next token at p + max(lh, lw). mrope=False is an ABLATION (the rows
        take consecutive positions, as text would). Returns the logits after the end token."""
        self.step(state, start)
        p = state["rope"]
        for k, row in enumerate(rows):
            i, j = divmod(k, lw)
            self.step(state, 0, emb=row, rp=(p, p + i, p + j) if mrope else (p + k,) * 3)
        state["rope"] = p + (max(lh, lw) if mrope else len(rows))
        return self.step(state, end)

    def mtp_step(self, ms, h, token, pos):
        """The MTP head at position pos: h = final normed hidden at pos, token = the id at pos+1.
        Appends to the head's KV cache `ms` ({"k": [], "v": []}); returns the draft logits for
        the id at pos+2."""
        c, w = self.cfg, self.w
        embed = w.get(PREFIX + "embed_tokens.weight")
        e = rmsnorm_zero_centred(embed[token], w.get(MTP + "pre_fc_norm_embedding.weight"), c.eps)
        hn = rmsnorm_zero_centred(h, w.get(MTP + "pre_fc_norm_hidden.weight"), c.eps)
        x = matvec(w.get(MTP + "fc.weight"), e + hn)
        p = MTP + "layers.0."
        n1 = rmsnorm_zero_centred(x, w.get(p + "input_layernorm.weight"), c.eps)
        x = [a + b for a, b in zip(x, self._full_attn(p + "self_attn.", n1, ms, pos))]
        n2 = rmsnorm_zero_centred(x, w.get(p + "post_attention_layernorm.weight"), c.eps)
        x = [a + b for a, b in zip(x, self._mlp(p + "mlp.", n2))]
        x = rmsnorm_zero_centred(x, w.get(MTP + "norm.weight"), c.eps)
        return matvec(embed if c.tied else w.get("lm_head.weight"), x)

    def mtp_rows(self, tokens):
        """The MTP draft logits at positions 0..len(tokens)-2 (position p sees tokens[p+1])."""
        state, hs = self.new_state(), []
        for t in tokens:
            hs.append(self.step_h(state, t)[1])
        ms = {"k": [], "v": []}
        return [self.mtp_step(ms, hs[p], tokens[p + 1], p) for p in range(len(tokens) - 1)]

    def forward(self, tokens):
        state = self.new_state()
        return [self.step(state, t) for t in tokens]

    def generate(self, prompt, n):
        """Feed prompt, then n greedy tokens (argmax, lowest id wins ties).

        Returns (rows for the prompt tokens, generated ids, logits row behind each generated id).
        """
        state = self.new_state()
        rows = [self.step(state, t) for t in prompt]
        last, ids, steps = rows[-1], [], []
        for _ in range(n):
            nxt = argmax(last)
            ids.append(nxt)
            steps.append(last)
            last = self.step(state, nxt)
        return rows, ids, steps


def argmax(row):
    best = 0
    for i in range(1, len(row)):
        if row[i] > row[best]:
            best = i
    return best


def top_gap(row):
    """Gap between the largest and second largest logit."""
    a = sorted(row, reverse=True)
    return a[0] - a[1]


# --------------------------------------------------------------------------- CLI

def main(argv=None):
    ap = argparse.ArgumentParser(description="float64 Qwen3.5 text oracle (lm-run contract)")
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
