#!/usr/bin/env python3
"""Deterministic generator for tests/fixtures/{qwen35,gemma4}-vl-tiny/ (tests/fixtures/README.md).

    make_tiny_vl.py [--family F]              write the fixture with the family's committed seed
    make_tiny_vl.py [--family F] --check      regenerate in memory, fail unless identical on disk
    make_tiny_vl.py [--family F] --find-seed [N]  search seeds N, N+1, ... until every margin holds

F is qwen35 (default) or gemma4. The text model is the family's tiny one (make_tiny /
make_tiny_gemma4 with their seeds, their fake vision stubs left out; Qwen's MTP head too); the
vision encoder is real and small and drawn from its own stream. One single-file checkpoint,
its q8 file, three PPM test images at their grid sizes, and the encoder output of the float64
oracle (tools/ref_vision_qwen35.py or tools/ref_vision_gemma4.py) for each image from the
bf16 weights and from the q8 file. Gemma also gets the text logits after an image block
(tools/ref_gemma4.py), bf16 and q8.

Margins asserted: every golden row differs from every other row of its image (no degenerate
encoder); the q8 rows differ from the bf16 rows by more than 3x the gate tolerance, so a q8
golden cannot be met by the bf16 path; for gemma4, turning the clippable-linear clamps off
changes the rows by more than 3x the tolerance, so the clamps are exercised.
"""

import argparse
import json
import math
import os
import random
import sys
import tempfile

import fixture_cmp
import make_tiny
import make_tiny_gemma4
import quant_ref
import ref_gemma4
import ref_vision_gemma4 as rg
import ref_vision_qwen35 as rv
import safetensors as st

VISION_SEED = 1
GEMMA_VISION_SEED = 1
GATE_REL_TOL = 2e-4
DEPTH, VHIDDEN, VHEADS, VINTER, N_POS = 2, 32, 2, 64, 16
PATCH, MERGE, TEMPORAL = 16, 2, 2
IMAGES = (("img_64x64.ppm", 64, 64), ("img_64x96.ppm", 64, 96), ("img_32x32.ppm", 32, 32))
V = rv.V
# Gemma: 2 layers, hidden 32, 2 query heads over 1 KV head (head_dim 16), pooling 3, a 16-entry
# position table per axis (the widest image is 9 patches)
G_LAYERS, G_HIDDEN, G_HEADS, G_KV, G_HD, G_INTER, G_POS, G_POOL = 2, 32, 2, 1, 16, 64, 16, 3
G_IMAGES = (("img_48x48.ppm", 48, 48), ("img_96x144.ppm", 96, 144), ("img_144x96.ppm", 144, 96))
# The tiny vocab is 384. The real ids (boi 255999, image 258880, eoi 258882) do not fit in it,
# so this fixture uses in-range stand-ins. The engine reads the ids from config.json; a real
# checkpoint keeps the real ones. Pad stays 0, which is text_config.pad_token_id.
G_BOI, G_IMAGE, G_EOI, G_PAD = 11, 12, 13, 0
# Prefix text, then the image block. The wide image is 6 tokens (sliding window is 8); the
# square one is a single token.
IMAGE_BLOCKS = (("img_96x144.ppm", (3, 15, 17, 2)), ("img_48x48.ppm", (5,)))


def config():
    c = make_tiny.config()
    c["vision_config"] = {
        "deepstack_visual_indexes": [],
        "depth": DEPTH,
        "hidden_act": "gelu_pytorch_tanh",
        "hidden_size": VHIDDEN,
        "in_channels": 3,
        "initializer_range": 0.02,
        "intermediate_size": VINTER,
        "model_type": "qwen3_5",
        "num_heads": VHEADS,
        "num_position_embeddings": N_POS,
        "out_hidden_size": make_tiny.HIDDEN,
        "patch_size": PATCH,
        "spatial_merge_size": MERGE,
        "temporal_patch_size": TEMPORAL,
    }
    return c


def vision_tensors(g):
    pin = 3 * TEMPORAL * PATCH * PATCH
    g.bf16(V + "patch_embed.proj.weight", (VHIDDEN, 3, TEMPORAL, PATCH, PATCH), 1.0 / math.sqrt(pin))
    g.bf16(V + "patch_embed.proj.bias", (VHIDDEN,), 0.1)
    g.bf16(V + "pos_embed.weight", (N_POS, VHIDDEN), 0.5)
    for i in range(DEPTH):
        p = "%sblocks.%d." % (V, i)
        g.bf16(p + "norm1.weight", (VHIDDEN,), 0.2, mean=1.0)
        g.bf16(p + "norm1.bias", (VHIDDEN,), 0.1)
        g.bf16(p + "norm2.weight", (VHIDDEN,), 0.2, mean=1.0)
        g.bf16(p + "norm2.bias", (VHIDDEN,), 0.1)
        g.bf16(p + "attn.qkv.weight", (3 * VHIDDEN, VHIDDEN), 1.5 / math.sqrt(VHIDDEN))
        g.bf16(p + "attn.qkv.bias", (3 * VHIDDEN,), 0.1)
        g.bf16(p + "attn.proj.weight", (VHIDDEN, VHIDDEN), 1.0 / math.sqrt(VHIDDEN))
        g.bf16(p + "attn.proj.bias", (VHIDDEN,), 0.1)
        g.bf16(p + "mlp.linear_fc1.weight", (VINTER, VHIDDEN), 1.0 / math.sqrt(VHIDDEN))
        g.bf16(p + "mlp.linear_fc1.bias", (VINTER,), 0.1)
        g.bf16(p + "mlp.linear_fc2.weight", (VHIDDEN, VINTER), 1.0 / math.sqrt(VINTER))
        g.bf16(p + "mlp.linear_fc2.bias", (VHIDDEN,), 0.1)
    m, cat = V + "merger.", VHIDDEN * MERGE * MERGE
    g.bf16(m + "norm.weight", (VHIDDEN,), 0.2, mean=1.0)
    g.bf16(m + "norm.bias", (VHIDDEN,), 0.1)
    g.bf16(m + "linear_fc1.weight", (cat, cat), 1.0 / math.sqrt(cat))
    g.bf16(m + "linear_fc1.bias", (cat,), 0.1)
    g.bf16(m + "linear_fc2.weight", (make_tiny.HIDDEN, cat), 1.0 / math.sqrt(cat))
    g.bf16(m + "linear_fc2.bias", (make_tiny.HIDDEN,), 0.1)


def gemma_config():
    c = make_tiny_gemma4.config()
    vc = dict(c["vision_config"])
    vc.update({"global_head_dim": G_HD, "head_dim": G_HD, "hidden_size": G_HIDDEN, "intermediate_size": G_INTER,
               "num_attention_heads": G_HEADS, "num_hidden_layers": G_LAYERS, "num_key_value_heads": G_KV,
               "pooling_kernel_size": G_POOL, "position_embedding_size": G_POS})
    c["vision_config"] = vc
    c["boi_token_id"] = G_BOI
    c["image_token_id"] = G_IMAGE
    c["eoi_token_id"] = G_EOI
    return c


def gemma_vision_tensors(g):
    inv = lambda n: 1.0 / math.sqrt(n)
    pin = 3 * PATCH * PATCH
    g.bf16(rg.VT + "patch_embedder.input_proj.weight", (G_HIDDEN, pin), inv(pin))
    g.bf16(rg.VT + "patch_embedder.position_embedding_table", (2, G_POS, G_HIDDEN), 0.5)

    def clin(p, out, inp, scale):
        g.bf16(p + "linear.weight", (out, inp), scale * inv(inp))
        lo, hi = -g.rng.uniform(1.0, 2.5), g.rng.uniform(1.0, 2.5)
        for k, val in (("input_min", lo), ("input_max", hi), ("output_min", lo), ("output_max", hi)):
            g.t[p + k] = ("BF16", (), st.floats_to_bf16([st.round_bf16(val)]))

    for i in range(G_LAYERS):
        p = "%sencoder.layers.%d." % (rg.VT, i)
        for n in ("input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm",
                  "post_feedforward_layernorm"):
            g.bf16(p + n + ".weight", (G_HIDDEN,), 0.2, mean=1.0)
        a = p + "self_attn."
        clin(a + "q_proj.", G_HEADS * G_HD, G_HIDDEN, 1.5)
        clin(a + "k_proj.", G_KV * G_HD, G_HIDDEN, 1.5)
        clin(a + "v_proj.", G_KV * G_HD, G_HIDDEN, 1.0)
        clin(a + "o_proj.", G_HIDDEN, G_HEADS * G_HD, 1.0)
        g.bf16(a + "q_norm.weight", (G_HD,), 0.2, mean=1.0)
        g.bf16(a + "k_norm.weight", (G_HD,), 0.2, mean=1.0)
        clin(p + "mlp.gate_proj.", G_INTER, G_HIDDEN, 1.5)
        clin(p + "mlp.up_proj.", G_INTER, G_HIDDEN, 1.5)
        clin(p + "mlp.down_proj.", G_HIDDEN, G_INTER, 1.0)
    g.bf16(rg.EV + "embedding_projection.weight", (make_tiny_gemma4.HIDDEN, G_HIDDEN), inv(G_HIDDEN))


def ppm(rng, h, w):
    """A smooth gradient with noise, so neighbouring patches differ but are not white noise."""
    px = bytearray()
    for y in range(h):
        for x in range(w):
            for c in range(3):
                base = 255.0 * ((x / max(w - 1, 1)) if c == 0 else (y / max(h - 1, 1)) if c == 1
                                else 0.5 + 0.5 * math.sin((x + y) / 7.0))
                px.append(max(0, min(255, int(base + rng.gauss(0.0, 24.0)))))
    return b"P6\n%d %d\n255\n" % (w, h) + bytes(px)


class MarginError(Exception):
    pass


def rnd(rows):
    return [[float("%.9g" % v) for v in r] for r in rows]


def gemma_image_logits(text, rows, prefix):
    state = text.new_state()
    for t in prefix:
        text.step(state, t)
    logits = text.image_block(state, rows, G_BOI, G_EOI, G_PAD)
    if state["pos"] != len(prefix) + len(rows) + 2:
        raise MarginError("image block did not advance one position per token")
    return logits


def gemma_text_goldens(model_dir, vis_bf, vis_q8):
    """Logits after each image block. The pad-token per-layer lookup and the vision-row residual
    must both move the logits by more than 3x the gate tolerance, and so must q8 versus bf16."""
    text = {"bf16": ref_gemma4.Model.load(model_dir),
            "q8": ref_gemma4.Model.load(model_dir, os.path.join(model_dir, "model.q8.safetensors"))}
    vis = {"bf16": vis_bf, "q8": vis_q8}
    raw = {}
    files = {}
    for label in ("bf16", "q8"):
        blocks = []
        for name, prefix in IMAGE_BLOCKS:
            rgb, pw, ph = rv.read_ppm(os.path.join(model_dir, name))
            rows = vis[label].encode(rgb, ph, pw)
            logits = gemma_image_logits(text[label], rows, prefix)
            gap = sorted(logits)[-1] - sorted(logits)[-2]
            if gap <= 3 * GATE_REL_TOL * max(abs(v) for v in logits):
                raise MarginError("%s %s: argmax gap %.3g is within 3x the tolerance" % (label, name, gap))
            raw[(label, name)] = logits
            blocks.append({"image": name, "prefix": list(prefix),
                           "logits": [float("%.9g" % v) for v in logits]})
        files["golden_image_%s.json" % label] = make_tiny.jdump({"blocks": blocks}, separators=(",", ":"))
    name, prefix = IMAGE_BLOCKS[0]
    rgb, pw, ph = rv.read_ppm(os.path.join(model_dir, name))
    rows = vis_bf.encode(rgb, ph, pw)
    right = raw[("bf16", name)]
    top = max(abs(v) for v in right)
    tol = GATE_REL_TOL * top
    # The image-token id in the per-layer table, instead of the pad.
    state = text["bf16"].new_state()
    for t in prefix:
        text["bf16"].step(state, t)
    wrong_ple = text["bf16"].image_block(state, rows, G_BOI, G_EOI, G_IMAGE)
    ple_diff = max(abs(a - b) for a, b in zip(right, wrong_ple))
    if ple_diff <= 3 * tol:
        raise MarginError("%s: pad per-layer input changes logits by only %.3g x tolerance" % (name, ple_diff / tol))
    embed = text["bf16"].w.get(ref_gemma4.PREFIX + "embed_tokens.weight")
    pad_row = [v * text["bf16"].cfg.embed_scale for v in embed[G_PAD]]
    state = text["bf16"].new_state()
    for t in prefix:
        text["bf16"].step(state, t)
    wrong_h = text["bf16"].image_block(state, [pad_row for _ in rows], G_BOI, G_EOI, G_PAD)
    h_diff = max(abs(a - b) for a, b in zip(right, wrong_h))
    if h_diff <= 3 * tol:
        raise MarginError("%s: the vision rows change logits by only %.3g x tolerance" % (name, h_diff / tol))
    for image, _prefix in IMAGE_BLOCKS:
        bf_row, q8_row = raw[("bf16", image)], raw[("q8", image)]
        scale = GATE_REL_TOL * max(abs(v) for v in bf_row)
        diff = max(abs(a - b) for a, b in zip(bf_row, q8_row))
        if diff <= 3 * scale:
            raise MarginError("%s: q8 logits are too close to bf16 (%.3g x tolerance)" % (image, diff / scale))
    return files


def build(seed, family="qwen35"):
    if family == "qwen35":
        text = make_tiny.build_tensors(random.Random(make_tiny.SEED))
        tensors = {n: v for n, v in text.items() if not n.startswith(V)}
        g = make_tiny.Gen(random.Random(make_tiny.MTP_SEED))
        make_tiny.mtp_tensors(g.rng, g)
        tensors.update(g.t)
        rng = random.Random(seed)
        g = make_tiny.Gen(rng)
        vision_tensors(g)
        cfg, images, oracle = config(), IMAGES, rv
    else:
        text = make_tiny_gemma4.build_tensors(random.Random(make_tiny_gemma4.SEED))
        tensors = {n: v for n, v in text.items() if not n.startswith((rg.VT, rg.EV))}
        rng = random.Random(seed)
        g = make_tiny_gemma4.Gen(rng)
        gemma_vision_tensors(g)
        cfg, images, oracle = gemma_config(), G_IMAGES, rg
    tensors.update(g.t)
    files = {"config.json": make_tiny.jdump(cfg, indent=4),
             "model.safetensors": st.serialize(tensors, metadata={"format": "pt"})}
    for name, h, w in images:
        files[name] = ppm(rng, h, w)
    golden = {"bf16": [], "q8": []}
    info = {"seed": seed}
    with tempfile.TemporaryDirectory() as d:
        for name, data in files.items():
            with open(os.path.join(d, name), "wb") as f:
                f.write(data)
        files["model.q8.safetensors"] = quant_ref.quantize_checkpoint(
            d, os.path.join(d, "model.q8.safetensors"))
        bf = oracle.VisionModel.load(d)
        q8 = oracle.VisionModel.load(d, os.path.join(d, "model.q8.safetensors"))
        worst = None
        for name, h, w in images:
            rgb, pw, ph = rv.read_ppm(os.path.join(d, name))
            rb, rq = bf.encode(rgb, ph, pw), q8.encode(rgb, ph, pw)
            if family == "gemma4":
                bf.cfg.clipped = False
                ru = bf.encode(rgb, ph, pw)
                bf.cfg.clipped = True
                top = max(abs(v) for r in rb for v in r)
                bite = max(abs(a - b) for r1, r2 in zip(rb, ru) for a, b in zip(r1, r2)) / (GATE_REL_TOL * top)
                if bite <= 3:
                    raise MarginError("%s: the clamps change the rows by only %.3g x tolerance" % (name, bite))
            for label, rows in (("bf16", rb), ("q8", rq)):
                for i in range(len(rows)):
                    for j in range(i):
                        if max(abs(a - b) for a, b in zip(rows[i], rows[j])) < 1e-3:
                            raise MarginError("%s %s: rows %d and %d are equal" % (label, name, j, i))
                golden[label].append({"image": name, "height": ph, "width": pw, "rows": rnd(rows)})
            top = max(abs(v) for r in rb for v in r)
            diff = max(abs(a - b) for r1, r2 in zip(rb, rq) for a, b in zip(r1, r2))
            ratio = diff / (GATE_REL_TOL * top)
            if ratio <= 3:
                raise MarginError("%s: q8 rows are too close to bf16 (%.3g x tolerance)" % (name, ratio))
            worst = ratio if worst is None else min(worst, ratio)
            info[name] = {"rows": len(rb), "max_abs": top, "q8_vs_bf16_x_tol": ratio}
        if family == "gemma4":
            files.update(gemma_text_goldens(d, bf, q8))
    for label in ("bf16", "q8"):
        files["golden_vision_%s.json" % label] = make_tiny.jdump(
            {"images": golden[label]}, separators=(",", ":"))
    info["min_q8_vs_bf16_x_tol"] = worst
    return files, info


def default_dir(family):
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "tests", "fixtures", "%s-vl-tiny" % family)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--family", choices=("qwen35", "gemma4"), default="qwen35")
    ap.add_argument("--out")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--find-seed", nargs="?", const=0, type=int, metavar="START")
    a = ap.parse_args(argv)
    a.out = a.out or default_dir(a.family)
    seed = VISION_SEED if a.family == "qwen35" else GEMMA_VISION_SEED
    if a.find_seed is not None:
        s = a.find_seed
        while True:
            try:
                _, info = build(s, a.family)
            except MarginError as e:
                print("seed %d: %s" % (s, e), file=sys.stderr)
                s += 1
                continue
            print(json.dumps(info))
            return 0
    try:
        files, info = build(seed, a.family)
    except MarginError as e:
        print("make_tiny_vl: seed %d fails its margins: %s" % (seed, e), file=sys.stderr)
        return 1
    if a.check:
        bad = []
        for name, data in files.items():
            path = os.path.join(a.out, name)
            if not os.path.exists(path):
                bad.append("missing " + name)
            else:
                with open(path, "rb") as f:
                    msg = fixture_cmp.diff(name, f.read(), data)
                if msg:
                    bad.append("differs " + msg)
        if os.path.isdir(a.out):
            bad += ["unexpected " + n for n in sorted(os.listdir(a.out)) if n not in files]
        else:
            bad.append("missing directory " + a.out)
        if bad:
            print("make_tiny_vl --check FAILED: " + "; ".join(bad), file=sys.stderr)
            return 1
        print("make_tiny_vl --check OK: %s, %d files identical (seed %d)" % (a.family, len(files), seed))
        return 0
    os.makedirs(a.out, exist_ok=True)
    for name, data in files.items():
        with open(os.path.join(a.out, name), "wb") as f:
            f.write(data)
    print(json.dumps(info))
    return 0


if __name__ == "__main__":
    sys.exit(main())
