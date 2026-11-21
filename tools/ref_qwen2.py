#!/usr/bin/env python3
"""Dependency-free float64 oracle for the Qwen2 / Qwen2.5 text model from a GGUF file.

Port of the maths of Hugging Face transformers `models/qwen2/modeling_qwen2.py` (Copyright the
HuggingFace Inc. team, Apache License 2.0): Qwen2RMSNorm (plain weight), Qwen2MLP (SwiGLU),
Qwen2RotaryEmbedding (rotate-half, default rope), Qwen2Attention (bias on q, k, v; grouped
query; scaling head_dim^-0.5; causal) and Qwen2ForCausalLM, re-implemented in plain Python in
float64. Block-quantised matrices are dequantised in double by tools/gguf.py
((double)d * sc * q - dmin * m, no float rounding), so the oracle is the exact function of the
file's bytes that the engine approximates in float32.

Module: `Model.load(gguf_path)`, `model.forward(tokens) -> [[vocab] per position]`,
`model.generate(prompt, n) -> (prompt rows, ids, steps)`.
CLI (the `lm-run` contract; engine-only flags are accepted and ignored):
    ref_qwen2.py --model DIR_OR_FILE --tokens 1,2,3 [--logits-out F] [--greedy N]
"""

import argparse
import glob
import math
import os
import struct
import sys

import gguf as gg


def dot(a, b):
    return math.fsum(x * y for x, y in zip(a, b))


def matvec(rows, x):
    return [math.fsum(map(float.__mul__, r, x)) for r in rows]


def rmsnorm(x, w, eps):
    s = (math.fsum(v * v for v in x) / len(x) + eps) ** -0.5
    return [v * s * wi for v, wi in zip(x, w)]


def softmax(v):
    m = max(v)
    e = [math.exp(a - m) for a in v]
    s = math.fsum(e)
    return [a / s for a in e]


def argmax(row):
    best = 0
    for i in range(1, len(row)):
        if row[i] > row[best]:
            best = i
    return best


def top_gap(row):
    a = sorted(row, reverse=True)
    return a[0] - a[1]


class Config:
    def __init__(self, meta, vocab):
        if meta.get("general.architecture") != "qwen2":
            raise ValueError("architecture %r is not qwen2" % meta.get("general.architecture"))
        for k in ("qwen2.rope.scaling.type",):
            if meta.get(k, "none") != "none":
                raise ValueError("rope scaling is not supported")
        self.layers = meta["qwen2.block_count"]
        self.hidden = meta["qwen2.embedding_length"]
        self.inter = meta["qwen2.feed_forward_length"]
        self.heads = meta["qwen2.attention.head_count"]
        self.kv_heads = meta["qwen2.attention.head_count_kv"]
        self.eps = meta["qwen2.attention.layer_norm_rms_epsilon"]
        self.theta = meta["qwen2.rope.freq_base"]
        self.head_dim = self.hidden // self.heads
        self.vocab = vocab


class Model:
    @classmethod
    def load(cls, path):
        if os.path.isdir(path):
            path = glob.glob(os.path.join(path, "*.gguf"))[0]
        meta, tensors, buf = gg.read_gguf(path)
        self = cls()
        self.meta, self.t, self.buf = meta, tensors, buf
        self.cfg = Config(meta, tensors["token_embd.weight"][0][1])
        c = self.cfg
        self.embed = self.mat("token_embd.weight")
        self.head = self.mat("output.weight") if "output.weight" in tensors else self.embed
        self.final_norm = self.vec("output_norm.weight")
        self.layers = []
        for i in range(c.layers):
            p = "blk.%d." % i
            self.layers.append({
                "attn_norm": self.vec(p + "attn_norm.weight"), "ffn_norm": self.vec(p + "ffn_norm.weight"),
                "bq": self.vec(p + "attn_q.bias"), "bk": self.vec(p + "attn_k.bias"), "bv": self.vec(p + "attn_v.bias"),
                "q": self.mat(p + "attn_q.weight"), "k": self.mat(p + "attn_k.weight"),
                "v": self.mat(p + "attn_v.weight"), "o": self.mat(p + "attn_output.weight"),
                "gate": self.mat(p + "ffn_gate.weight"), "up": self.mat(p + "ffn_up.weight"),
                "down": self.mat(p + "ffn_down.weight")})
        hd = c.head_dim
        self.freqs = [c.theta ** (-2.0 * i / hd) for i in range(hd // 2)]
        return self

    def _flat(self, name):
        dims, ty, off = self.t[name]
        n = 1
        for d in dims:
            n *= d
        return dims, gg.dequantize(ty, gg.tensor_bytes(self.buf, dims, ty, off), n)

    def vec(self, name):
        return self._flat(name)[1]

    def mat(self, name):
        dims, flat = self._flat(name)
        cols, rows = dims[0], dims[1]
        return [flat[r * cols:(r + 1) * cols] for r in range(rows)]

    def rope(self, x, nh, pos):
        hd, half = self.cfg.head_dim, self.cfg.head_dim // 2
        out = list(x)
        for h in range(nh):
            b = h * hd
            for i in range(half):
                a = pos * self.freqs[i]
                cs, sn = math.cos(a), math.sin(a)
                x0, x1 = x[b + i], x[b + i + half]
                out[b + i] = x0 * cs - x1 * sn
                out[b + i + half] = x1 * cs + x0 * sn
        return out

    def step(self, token, cache):
        c = self.cfg
        hd, group = c.head_dim, c.heads // c.kv_heads
        pos = len(cache[0][0])
        h = list(self.embed[token])
        for li, L in enumerate(self.layers):
            n = rmsnorm(h, L["attn_norm"], c.eps)
            q = [a + b for a, b in zip(matvec(L["q"], n), L["bq"])]
            k = [a + b for a, b in zip(matvec(L["k"], n), L["bk"])]
            v = [a + b for a, b in zip(matvec(L["v"], n), L["bv"])]
            q, k = self.rope(q, c.heads, pos), self.rope(k, c.kv_heads, pos)
            cache[li][0].append(k)
            cache[li][1].append(v)
            attn = []
            for hh in range(c.heads):
                g = hh // group
                qh = q[hh * hd:(hh + 1) * hd]
                sc = softmax([dot(qh, kk[g * hd:(g + 1) * hd]) * hd ** -0.5 for kk in cache[li][0]])
                o = [0.0] * hd
                for w, vv in zip(sc, cache[li][1]):
                    for d in range(hd):
                        o[d] += w * vv[g * hd + d]
                attn += o
            h = [a + b for a, b in zip(h, matvec(L["o"], attn))]
            n = rmsnorm(h, L["ffn_norm"], c.eps)
            gt, up = matvec(L["gate"], n), matvec(L["up"], n)
            act = [(a / (1.0 + math.exp(-a))) * b for a, b in zip(gt, up)]
            h = [a + b for a, b in zip(h, matvec(L["down"], act))]
        return matvec(self.head, rmsnorm(h, self.final_norm, c.eps))

    def new_cache(self):
        return [([], []) for _ in range(self.cfg.layers)]

    def forward(self, tokens):
        cache = self.new_cache()
        return [self.step(t, cache) for t in tokens]

    def generate(self, prompt, n):
        cache = self.new_cache()
        rows = [self.step(t, cache) for t in prompt]
        ids, steps = [], []
        last = rows[-1]
        for _ in range(n):
            t = argmax(last)
            ids.append(t)
            last = self.step(t, cache)
            steps.append(last)
        return rows, ids, steps


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--logits-out")
    ap.add_argument("--greedy", type=int, default=0)
    a, _ = ap.parse_known_args(argv)
    m = Model.load(a.model)
    toks = [int(x) for x in a.tokens.split(",")]
    rows, ids, _ = m.generate(toks, a.greedy)
    if a.logits_out:
        with open(a.logits_out, "wb") as f:
            for r in rows:
                f.write(struct.pack("<%df" % len(r), *r))
    for i in ids:
        print(i)
    return 0


if __name__ == "__main__":
    sys.exit(main())
