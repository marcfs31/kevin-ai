#!/usr/bin/env python3
"""Dependency-free float64 oracle for the Qwen3.5 vision encoder (standard library only).

Port of the math in Hugging Face transformers (Apache License 2.0), commit 56547cb:
`models/qwen3_5/modeling_qwen3_5.py` (Qwen3_5VisionPatchEmbed, Qwen3_5VisionRotaryEmbedding,
Qwen3_5VisionAttention, Qwen3_5VisionMLP, Qwen3_5VisionBlock, Qwen3_5VisionPatchMerger,
Qwen3_5VisionModel, Qwen3_5Model.get_rope_index / get_vision_position_ids),
`vision_utils.py` (get_vision_position_ids, get_vision_interpolation_indices_and_weights) and
`models/qwen2_vl/image_processing_qwen2_vl.py` (smart_resize, patchify). docs/PLAN.md,
"Phase 3", states the same maths in words.

Deliberate differences from the reference, none of which changes the function:
  * every activation stays float64 (the reference casts to the checkpoint dtype between ops);
  * the image arrives already at its grid size (H and W multiples of 32): resampling is not
    part of the oracle (docs/PLAN.md, "Resampling");
  * one image at a time, so the per-image attention segments (cu_seqlens) are implicit.

Weights: a bf16 checkpoint directory or, with `weights=`, an lm-q8 file (dequantised exactly,
as ref_qwen35.Weights does).

Module: `VisionModel.load(model_dir, weights=None)`, `vm.encode(rgb, height, width) -> rows`
(rgb: bytes, H*W*3 RGB8, row-major), `smart_resize(...)`, `mrope_positions(items)`.
CLI: ref_vision_qwen35.py --model DIR [--weights Q8] --image F.ppm [--rows-out F]
writes one little-endian f32 row of out_hidden values per merged token.
"""

import argparse
import json
import math
import os
import struct
import sys

import ref_qwen35 as ref
import quant_ref
import safetensors as st

V = "model.visual."


# --------------------------------------------------------------------------- preprocessing

def smart_resize(height, width, factor=32, min_pixels=65536, max_pixels=1048576):
    """qwen2_vl smart_resize: both sides multiples of factor, area within [min, max]."""
    if max(height, width) / min(height, width) > 200:
        raise ValueError("absolute aspect ratio must be smaller than 200")
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def patchify(rgb, height, width, patch, merge, temporal):
    """Normalised pixels -> (patches [gh*gw][3*temporal*patch*patch], gh, gw), block-major
    over merge x merge blocks; each patch vector is [c][t][py][px] with the frame repeated."""
    if height % (patch * merge) or width % (patch * merge):
        raise ValueError("image %dx%d is not a multiple of %d" % (width, height, patch * merge))
    if len(rgb) != height * width * 3:
        raise ValueError("rgb has %d bytes, expected %d" % (len(rgb), height * width * 3))
    gh, gw = height // patch, width // patch
    px = [(v / 255.0 - 0.5) / 0.5 for v in rgb]  # image_mean = image_std = 0.5
    out = []
    for bh in range(gh // merge):
        for bw in range(gw // merge):
            for mh in range(merge):
                for mw in range(merge):
                    r0, c0 = (bh * merge + mh) * patch, (bw * merge + mw) * patch
                    vec = []
                    for c in range(3):
                        plane = [px[((r0 + y) * width + c0 + x) * 3 + c]
                                 for y in range(patch) for x in range(patch)]
                        vec.extend(plane * temporal)
                    out.append(vec)
    return out, gh, gw


def block_order(gh, gw, merge):
    """(row, col) of each patch in block-major order (vision_utils.get_vision_position_ids)."""
    return [(bh * merge + mh, bw * merge + mw)
            for bh in range(gh // merge) for bw in range(gw // merge)
            for mh in range(merge) for mw in range(merge)]


def interp_taps(index, size, side):
    """Bilinear, align_corners=True, border clamp: [(tap, weight), (tap, weight)]."""
    src = index * (side - 1) / max(size - 1, 1)
    f = math.floor(src)
    taps = []
    for off in (0, 1):
        t = min(max(f + off, 0), side - 1)
        taps.append((t, max(0.0, 1.0 - abs(src - f - off))))
    return taps


# --------------------------------------------------------------------------- ops (float64)

def layernorm(x, w, b, eps):
    n = len(x)
    mean = sum(x) / n
    var = sum((v - mean) ** 2 for v in x) / n
    r = 1.0 / math.sqrt(var + eps)
    return [(v - mean) * r * g + c for v, g, c in zip(x, w, b)]


def gelu_tanh(v):
    return 0.5 * v * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (v + 0.044715 * v * v * v)))


def gelu_erf(v):
    return 0.5 * v * (1.0 + math.erf(v / math.sqrt(2.0)))


def linear(w, b, x):
    y = ref.matvec(w, x)
    return [v + c for v, c in zip(y, b)] if b is not None else y


def rotate(x, cos, sin):
    """x * cos + rotate_half(x) * sin over the whole vector."""
    h = len(x) // 2
    rh = [-v for v in x[h:]] + x[:h]
    return [a * c + r * s for a, r, c, s in zip(x, rh, cos, sin)]


# --------------------------------------------------------------------------- model

class VisionConfig:
    def __init__(self, raw):
        try:
            v = raw["vision_config"]
            self.depth = v["depth"]
            self.hidden = v["hidden_size"]
            self.heads = v["num_heads"]
            self.inter = v["intermediate_size"]
            self.patch = v["patch_size"]
            self.merge = v["spatial_merge_size"]
            self.temporal = v["temporal_patch_size"]
            self.in_ch = v["in_channels"]
            self.out_hidden = v["out_hidden_size"]
            self.n_pos = v["num_position_embeddings"]
            self.act = v["hidden_act"]
            self.theta = float(v.get("rope_parameters", {}).get("rope_theta", 10000.0))
            self.deepstack = v.get("deepstack_visual_indexes", [])
            self.image_token = raw["image_token_id"]
            self.start_token = raw["vision_start_token_id"]
            self.end_token = raw["vision_end_token_id"]
        except KeyError as e:
            raise ValueError("config.json is missing %s" % e)
        if self.act != "gelu_pytorch_tanh":
            raise ValueError("vision hidden_act %r is not supported" % self.act)
        if self.deepstack:
            raise ValueError("deepstack_visual_indexes is not supported")
        if self.in_ch != 3 or self.hidden % self.heads:
            raise ValueError("unsupported vision shape")
        self.head_dim = self.hidden // self.heads
        self.side = int(round(math.sqrt(self.n_pos)))
        if self.side * self.side != self.n_pos:
            raise ValueError("num_position_embeddings is not a square")
        if self.head_dim % 4:
            raise ValueError("vision head_dim must be a multiple of 4")


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

    def _rope_tables(self, order):
        c = self.cfg
        spatial = c.head_dim // 2
        inv = [c.theta ** (-(2.0 * i) / spatial) for i in range(spatial // 2)]
        tables = []
        for row, col in order:
            ang = [row * f for f in inv] + [col * f for f in inv]
            ang = ang + ang
            tables.append(([math.cos(a) for a in ang], [math.sin(a) for a in ang]))
        return tables

    def _pos_embed(self, order, gh, gw):
        c, table = self.cfg, self.w.get(V + "pos_embed.weight")
        out = []
        for row, col in order:
            acc = [0.0] * c.hidden
            for th, wh in interp_taps(row, gh, c.side):
                for tw, ww in interp_taps(col, gw, c.side):
                    wgt = wh * ww
                    if wgt:
                        e = table[th * c.side + tw]
                        acc = [a + wgt * v for a, v in zip(acc, e)]
            out.append(acc)
        return out

    def _attention(self, p, xs, rope):
        c, w = self.cfg, self.w
        hd, nh = c.head_dim, c.heads
        qkv = [linear(w.get(p + "qkv.weight"), w.get(p + "qkv.bias"), x) for x in xs]
        q = [[rotate(r[h * hd:(h + 1) * hd], *rope[i]) for h in range(nh)] for i, r in enumerate(qkv)]
        k = [[rotate(r[c.hidden + h * hd:c.hidden + (h + 1) * hd], *rope[i]) for h in range(nh)]
             for i, r in enumerate(qkv)]
        v = [[r[2 * c.hidden + h * hd:2 * c.hidden + (h + 1) * hd] for h in range(nh)] for r in qkv]
        scale = hd ** -0.5
        out = []
        for i in range(len(xs)):
            o = []
            for h in range(nh):
                pr = ref.softmax([ref.dot(q[i][h], k[j][h]) * scale for j in range(len(xs))])
                o.extend(sum(pr[j] * v[j][h][d] for j in range(len(xs))) for d in range(hd))
            out.append(linear(w.get(p + "proj.weight"), w.get(p + "proj.bias"), o))
        return out

    def encode(self, rgb, height, width):
        """-> list of out_hidden rows, one per merged token (block order, row-major)."""
        c, w = self.cfg, self.w
        patches, gh, gw = patchify(rgb, height, width, c.patch, c.merge, c.temporal)
        order = block_order(gh, gw, c.merge)
        pe_w, pe_b = w.get(V + "patch_embed.proj.weight"), w.get(V + "patch_embed.proj.bias")
        pos = self._pos_embed(order, gh, gw)
        xs = [[a + b for a, b in zip(linear(pe_w, pe_b, pv), pe)] for pv, pe in zip(patches, pos)]
        rope = self._rope_tables(order)
        for i in range(c.depth):
            p = "%sblocks.%d." % (V, i)
            n1 = [layernorm(x, w.get(p + "norm1.weight"), w.get(p + "norm1.bias"), 1e-6) for x in xs]
            a = self._attention(p + "attn.", n1, rope)
            xs = [[u + v for u, v in zip(x, y)] for x, y in zip(xs, a)]
            n2 = [layernorm(x, w.get(p + "norm2.weight"), w.get(p + "norm2.bias"), 1e-6) for x in xs]
            m = []
            for x in n2:
                h = [gelu_tanh(v) for v in linear(w.get(p + "mlp.linear_fc1.weight"),
                                                  w.get(p + "mlp.linear_fc1.bias"), x)]
                m.append(linear(w.get(p + "mlp.linear_fc2.weight"), w.get(p + "mlp.linear_fc2.bias"), h))
            xs = [[u + v for u, v in zip(x, y)] for x, y in zip(xs, m)]
        mp = V + "merger."
        normed = [layernorm(x, w.get(mp + "norm.weight"), w.get(mp + "norm.bias"), 1e-6) for x in xs]
        unit = c.merge * c.merge
        rows = []
        for t in range(len(normed) // unit):
            cat = [v for x in normed[t * unit:(t + 1) * unit] for v in x]
            h = [gelu_erf(v) for v in linear(w.get(mp + "linear_fc1.weight"), w.get(mp + "linear_fc1.bias"), cat)]
            rows.append(linear(w.get(mp + "linear_fc2.weight"), w.get(mp + "linear_fc2.bias"), h))
        return rows


def mrope_positions(items, merge=2):
    """items: list of ("text", n) or ("image", gh, gw) in sequence order (gh, gw = patch grid).
    -> list of (t, h, w) per token, as Qwen3_5Model.get_rope_index for one sequence."""
    out, p = [], 0
    for it in items:
        if it[0] == "text":
            out.extend((p + i, p + i, p + i) for i in range(it[1]))
            p += it[1]
        else:
            lh, lw = it[1] // merge, it[2] // merge
            out.extend((p, p + i, p + j) for i in range(lh) for j in range(lw))
            p += max(lh, lw)
    return out


section_axis = ref.section_axis   # moved to ref_qwen35 (the text model's RoPE owns it)


def read_ppm(path):
    """Binary PPM (P6, maxval 255) -> (rgb bytes, width, height). Comments are not allowed."""
    with open(path, "rb") as f:
        data = f.read()
    parts, i = [], 0
    while len(parts) < 4:
        while data[i:i + 1].isspace():
            i += 1
        j = i
        while not data[j:j + 1].isspace():
            j += 1
        parts.append(data[i:j])
        i = j
    if parts[0] != b"P6" or int(parts[3]) != 255:
        raise ValueError("%s: only binary P6 PPM with maxval 255 is supported" % path)
    w, h = int(parts[1]), int(parts[2])
    rgb = data[i + 1:i + 1 + w * h * 3]
    if len(rgb) != w * h * 3:
        raise ValueError("%s: truncated" % path)
    return rgb, w, h


def main(argv=None):
    ap = argparse.ArgumentParser(description="float64 Qwen3.5 vision encoder oracle")
    ap.add_argument("--model", required=True)
    ap.add_argument("--weights")
    ap.add_argument("--image", required=True, help="binary PPM at its grid size")
    ap.add_argument("--rows-out")
    a = ap.parse_args(argv)
    vm = VisionModel.load(a.model, a.weights)
    rgb, w, h = read_ppm(a.image)
    rows = vm.encode(rgb, h, w)
    if a.rows_out:
        with open(a.rows_out, "wb") as f:
            for r in rows:
                f.write(struct.pack("<%df" % len(r), *r))
    print("%d rows of %d" % (len(rows), len(rows[0]) if rows else 0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
