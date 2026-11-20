#!/usr/bin/env python3
"""Dependency-free float64 oracle for the Gemma 4 vision encoder (standard library only).

Port of the math in Hugging Face transformers (Apache License 2.0), commit 56547cb:
`models/gemma4/modeling_gemma4.py` (Gemma4ClippableLinear, Gemma4RMSNorm,
Gemma4VisionPatchEmbedder, Gemma4VisionRotaryEmbedding, apply_multidimensional_rope,
Gemma4VisionAttention, Gemma4VisionMLP, Gemma4VisionEncoderLayer, Gemma4VisionPooler,
Gemma4VisionModel, Gemma4MultimodalEmbedder) and `models/gemma4/image_processing_gemma4.py`
(get_aspect_ratio_preserving_size, convert_image_to_patches, the position ids).
docs/PLAN.md, "Phase 3", states the same maths in words.

Deliberate differences from the reference, none of which changes the function:
  * every activation stays float64;
  * the image arrives already at its grid size (both sides multiples of patch * pooling
    kernel); resampling is not part of the oracle;
  * no padding to max_patches: padded patches are masked out of attention and pooling in
    the reference, so they change nothing.
Refused loudly: standardize = true, an activation other than gelu_pytorch_tanh.

Module: `VisionModel.load(model_dir, weights=None)`, `vm.encode(rgb, height, width) -> rows`
(one row of the text hidden size per soft token). CLI as ref_vision_qwen35.py.
"""

import argparse
import json
import math
import os
import struct
import sys

import quant_ref
import ref_qwen35 as ref
import ref_vision_qwen35 as rq
import safetensors as st

VT = "model.vision_tower."
EV = "model.embed_vision."


def aspect_ratio_size(height, width, patch=16, max_patches=2520, pool=3):
    """image_processing_gemma4.get_aspect_ratio_preserving_size."""
    total_px = height * width
    target_px = max_patches * (patch ** 2)
    factor = math.sqrt(target_px / total_px)
    side_mult = pool * patch
    th = int(math.floor(factor * height / side_mult)) * side_mult
    tw = int(math.floor(factor * width / side_mult)) * side_mult
    if th == 0 and tw == 0:
        raise ValueError("image too small to resize")
    max_side = (max_patches // pool ** 2) * side_mult
    if th == 0:
        th, tw = side_mult, min(int(math.floor(width / height)) * side_mult, max_side)
    elif tw == 0:
        tw, th = side_mult, min(int(math.floor(height / width)) * side_mult, max_side)
    if th * tw > target_px:
        raise ValueError("resized image exceeds max_patches")
    return th, tw


def rmsnorm(x, w, eps):
    r = (sum(v * v for v in x) / len(x) + eps) ** -0.5
    return [v * r * g for v, g in zip(x, w)] if w is not None else [v * r for v in x]


def clamp(x, lo, hi):
    return [min(max(v, lo), hi) for v in x]


class VisionConfig:
    def __init__(self, raw):
        try:
            v = raw["vision_config"]
            self.hidden = v["hidden_size"]
            self.inter = v["intermediate_size"]
            self.layers = v["num_hidden_layers"]
            self.heads = v["num_attention_heads"]
            self.kv_heads = v["num_key_value_heads"]
            self.head_dim = v["head_dim"]
            self.patch = v["patch_size"]
            self.pool = v["pooling_kernel_size"]
            self.pos_size = v["position_embedding_size"]
            self.eps = v["rms_norm_eps"]
            self.theta = float(v["rope_parameters"]["rope_theta"])
            self.clipped = bool(v["use_clipped_linears"])
            self.standardize = bool(v.get("standardize", False))
            self.act = v["hidden_activation"]
            self.text_hidden = raw["text_config"]["hidden_size"]
            self.image_token = raw["image_token_id"]
            self.start_token = raw["boi_token_id"]
            self.end_token = raw["eoi_token_id"]
        except KeyError as e:
            raise ValueError("config.json is missing %s" % e)
        if self.act != "gelu_pytorch_tanh":
            raise ValueError("vision hidden_activation %r is not supported" % self.act)
        if self.standardize:
            raise ValueError("vision standardize is not supported")
        if self.heads % self.kv_heads or self.head_dim % 4:
            raise ValueError("unsupported vision attention shape")


class VisionModel:
    def __init__(self, cfg, weights):
        self.cfg, self.w = cfg, weights

    @classmethod
    def load(cls, model_dir, weights=None):
        with open(os.path.join(model_dir, "config.json")) as f:
            cfg = VisionConfig(json.load(f))
        if weights is None:
            return cls(cfg, ref.Weights(st.open_checkpoint(model_dir)))
        return cls(cfg, ref.Weights(quant_ref.load_q8(weights), q8=True))

    def _scalar(self, name):
        return self.w.store.floats(name)[0]

    def _clin(self, p, x):
        """Gemma4ClippableLinear at prefix p (no bias)."""
        if self.cfg.clipped:
            x = clamp(x, self._scalar(p + "input_min"), self._scalar(p + "input_max"))
        y = ref.matvec(self.w.get(p + "linear.weight"), x)
        if self.cfg.clipped:
            y = clamp(y, self._scalar(p + "output_min"), self._scalar(p + "output_max"))
        return y

    def _rope(self, vec, pos):
        """apply_multidimensional_rope on one head: channels [0, hd/2) by x, [hd/2, hd) by y."""
        hd = self.cfg.head_dim
        half = hd // 2
        inv = [1.0 / self.cfg.theta ** ((2.0 * i) / half) for i in range(half // 2)]
        out = []
        for part, p in enumerate(pos):
            ang = [p * f for f in inv] * 2
            out.extend(rq.rotate(vec[part * half:(part + 1) * half],
                                 [math.cos(a) for a in ang], [math.sin(a) for a in ang]))
        return out

    def _attention(self, p, xs, pos):
        c = self.cfg
        hd, nh, nkv = c.head_dim, c.heads, c.kv_heads
        qn, kn = self.w.get(p + "q_norm.weight"), self.w.get(p + "k_norm.weight")
        q, k, v = [], [], []
        for x, ps in zip(xs, pos):
            qr, kr, vr = self._clin(p + "q_proj.", x), self._clin(p + "k_proj.", x), self._clin(p + "v_proj.", x)
            q.append([self._rope(rmsnorm(qr[h * hd:(h + 1) * hd], qn, c.eps), ps) for h in range(nh)])
            k.append([self._rope(rmsnorm(kr[h * hd:(h + 1) * hd], kn, c.eps), ps) for h in range(nkv)])
            v.append([rmsnorm(vr[h * hd:(h + 1) * hd], None, c.eps) for h in range(nkv)])
        group = nh // nkv
        out = []
        for i in range(len(xs)):
            o = []
            for h in range(nh):
                kh = h // group
                pr = ref.softmax([ref.dot(q[i][h], k[j][kh]) for j in range(len(xs))])  # scaling 1.0
                o.extend(sum(pr[j] * v[j][kh][d] for j in range(len(xs))) for d in range(hd))
            out.append(self._clin(p + "o_proj.", o))
        return out

    def encode(self, rgb, height, width):
        """-> list of text-hidden rows, one per soft token (pooled grid, row-major)."""
        c, w = self.cfg, self.w
        unit = c.patch * c.pool
        if height % unit or width % unit or height < unit or width < unit:
            raise ValueError("image %dx%d is not a multiple of %d" % (width, height, unit))
        ph, pw = height // c.patch, width // c.patch
        table = w.store.floats(VT + "patch_embedder.position_embedding_table")
        tab = lambda axis, i, d: table[(axis * c.pos_size + i) * c.hidden + d]
        proj = w.get(VT + "patch_embedder.input_proj.weight")
        xs, pos = [], []
        for py in range(ph):
            for px in range(pw):
                vec = [2.0 * (rgb[((py * c.patch + y) * width + px * c.patch + x) * 3 + ch] / 255.0 - 0.5)
                       for y in range(c.patch) for x in range(c.patch) for ch in range(3)]
                h = ref.matvec(proj, vec)
                xs.append([h[d] + tab(0, px, d) + tab(1, py, d) for d in range(c.hidden)])
                pos.append((px, py))
        for i in range(c.layers):
            p = "%sencoder.layers.%d." % (VT, i)
            n = [rmsnorm(x, w.get(p + "input_layernorm.weight"), c.eps) for x in xs]
            a = self._attention(p + "self_attn.", n, pos)
            a = [rmsnorm(x, w.get(p + "post_attention_layernorm.weight"), c.eps) for x in a]
            xs = [[u + t for u, t in zip(x, y)] for x, y in zip(xs, a)]
            n = [rmsnorm(x, w.get(p + "pre_feedforward_layernorm.weight"), c.eps) for x in xs]
            m = []
            for x in n:
                g = self._clin(p + "mlp.gate_proj.", x)
                u = self._clin(p + "mlp.up_proj.", x)
                m.append(self._clin(p + "mlp.down_proj.", [rq.gelu_tanh(a1) * b1 for a1, b1 in zip(g, u)]))
            m = [rmsnorm(x, w.get(p + "post_feedforward_layernorm.weight"), c.eps) for x in m]
            xs = [[u + t for u, t in zip(x, y)] for x, y in zip(xs, m)]
        k = c.pool
        nout = (ph // k) * (pw // k)
        pooled = [[0.0] * c.hidden for _ in range(nout)]
        for x, (px, py) in zip(xs, pos):
            idx = px // k + (pw // k) * (py // k)
            pooled[idx] = [a1 + b1 / (k * k) for a1, b1 in zip(pooled[idx], x)]
        root = c.hidden ** 0.5
        emb = w.get(EV + "embedding_projection.weight")
        return [ref.matvec(emb, rmsnorm([v * root for v in r], None, c.eps)) for r in pooled]


def main(argv=None):
    ap = argparse.ArgumentParser(description="float64 Gemma 4 vision encoder oracle")
    ap.add_argument("--model", required=True)
    ap.add_argument("--weights")
    ap.add_argument("--image", required=True, help="binary PPM at its grid size")
    ap.add_argument("--rows-out")
    a = ap.parse_args(argv)
    vm = VisionModel.load(a.model, a.weights)
    rgb, w, h = rq.read_ppm(a.image)
    rows = vm.encode(rgb, h, w)
    if a.rows_out:
        with open(a.rows_out, "wb") as f:
            for r in rows:
                f.write(struct.pack("<%df" % len(r), *r))
    print("%d rows of %d" % (len(rows), len(rows[0]) if rows else 0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
