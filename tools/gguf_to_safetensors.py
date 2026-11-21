#!/usr/bin/env python3
"""Convert a llama.cpp BF16 GGUF of a Qwen3.5 fine-tune back to Hugging Face safetensors.

    gguf_to_safetensors.py --base DIR --gguf FILE --out DIR     convert
    gguf_to_safetensors.py --base DIR --out DIR --check          compare out against base

Standard library only. The base checkpoint (same architecture) supplies the HF names, shapes
and dtypes, and every tensor the GGUF lacks (vision tower, MTP) is copied from it byte for
byte. Tensors are streamed: the output is never held in memory.
"""

import argparse
import json
import math
import os
import re
import shutil
import struct
import sys

import safetensors as st

GGUF_MAGIC = b"GGUF"
GGUF_F32, GGUF_BF16 = 0, 30
GGUF_ELEM = {GGUF_F32: ("F32", 4), GGUF_BF16: ("BF16", 2)}
COPIED_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json",
                "chat_template.jinja", "preprocessor_config.json")
COPIED_PREFIXES = ("model.visual.", "mtp.")
CHECK_MIN_COSINE = 0.98
CHECK_SAMPLE = 1_000_000


class ConvertError(ValueError):
    pass


# --------------------------------------------------------------------------- GGUF reader

class Gguf:
    """GGUF v3 header: `meta` (key -> value), `tensors` (name -> (dims, type, abs offset))."""

    def __init__(self, path):
        self.path = path
        with open(path, "rb") as f:
            if f.read(4) != GGUF_MAGIC:
                raise ConvertError("%s: not a GGUF file" % path)
            (version,) = self._rd(f, "I")
            if version != 3:
                raise ConvertError("%s: GGUF version %d, only v3 is supported" % (path, version))
            n_tensors, n_kv = self._rd(f, "QQ")
            self.meta = {}
            for _ in range(n_kv):
                key = self._str(f)
                (vtype,) = self._rd(f, "I")
                self.meta[key] = self._value(f, vtype)
            infos = []
            for _ in range(n_tensors):
                name = self._str(f)
                (ndim,) = self._rd(f, "I")
                dims = self._rd(f, "%dQ" % ndim)
                ttype, off = self._rd(f, "IQ")
                infos.append((name, dims, ttype, off))
            align = self.meta.get("general.alignment", 32)
            self.data_start = (f.tell() + align - 1) // align * align
        self.tensors = {}
        for name, dims, ttype, off in infos:
            if ttype not in GGUF_ELEM:
                raise ConvertError("%s: tensor %r has unsupported GGUF type %d (only F32=0 and "
                                   "BF16=30 are accepted)" % (path, name, ttype))
            if name in self.tensors:
                raise ConvertError("%s: duplicate tensor %r" % (path, name))
            self.tensors[name] = (tuple(dims), ttype, self.data_start + off)

    @staticmethod
    def _rd(f, fmt):
        fmt = "<" + fmt
        data = f.read(struct.calcsize(fmt))
        if len(data) != struct.calcsize(fmt):
            raise ConvertError("truncated GGUF header")
        return struct.unpack(fmt, data)

    def _str(self, f):
        (n,) = self._rd(f, "Q")
        return f.read(n).decode("utf-8")

    def _value(self, f, vtype):
        scalar = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?",
                  10: "Q", 11: "q", 12: "d"}
        if vtype in scalar:
            return self._rd(f, scalar[vtype])[0]
        if vtype == 8:
            return self._str(f)
        if vtype == 9:
            (etype,) = self._rd(f, "I")
            (n,) = self._rd(f, "Q")
            return [self._value(f, etype) for _ in range(n)]
        raise ConvertError("unknown GGUF metadata type %d" % vtype)

    def hf_shape(self, name):
        return tuple(reversed(self.tensors[name][0]))  # GGUF lists the innermost dim first

    def read(self, name):
        dims, ttype, off = self.tensors[name]
        dtype, size = GGUF_ELEM[ttype]
        n = st.numel(dims) * size
        with open(self.path, "rb") as f:
            f.seek(off)
            raw = f.read(n)
        if len(raw) != n:
            raise ConvertError("%s: tensor %r runs past end of file" % (self.path, name))
        return dtype, raw


# --------------------------------------------------------------------------- name mapping

TOP_NAMES = {
    "token_embd.weight": "model.language_model.embed_tokens.weight",
    "output_norm.weight": "model.language_model.norm.weight",
}
BLOCK_NAMES = {
    "attn_norm.weight": "input_layernorm.weight",
    "post_attention_norm.weight": "post_attention_layernorm.weight",
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    "attn_qkv.weight": "linear_attn.in_proj_qkv.weight",
    "attn_gate.weight": "linear_attn.in_proj_z.weight",
    "ssm_alpha.weight": "linear_attn.in_proj_a.weight",
    "ssm_beta.weight": "linear_attn.in_proj_b.weight",
    "ssm_a": "linear_attn.A_log",
    "ssm_dt.bias": "linear_attn.dt_bias",
    "ssm_conv1d.weight": "linear_attn.conv1d.weight",
    "ssm_norm.weight": "linear_attn.norm.weight",
    "ssm_out.weight": "linear_attn.out_proj.weight",
}


def hf_name(gguf_name):
    if gguf_name in TOP_NAMES:
        return TOP_NAMES[gguf_name]
    m = re.fullmatch(r"blk\.(\d+)\.(.+)", gguf_name)
    if m and m.group(2) in BLOCK_NAMES:
        return "model.language_model.layers.%s.%s" % (m.group(1), BLOCK_NAMES[m.group(2)])
    raise ConvertError("no HF name for GGUF tensor %r" % gguf_name)


# --------------------------------------------------------------------------- inverse transforms

# What llama.cpp's Qwen3.5 / Qwen3-Next converter did to the HF weights, undone here. Each
# transform is keyed by the HF tensor suffix; `transform_of` names it for the --check table.
NORM_PLUS_ONE = ("input_layernorm.weight", "post_attention_layernorm.weight", "q_norm.weight",
                 "k_norm.weight", "language_model.norm.weight")
V_REORDER_ROWS = ("linear_attn.in_proj_z.weight", "linear_attn.in_proj_a.weight",
                  "linear_attn.in_proj_b.weight")


def transform_of(hf):
    if hf.endswith(NORM_PLUS_ONE):
        return "norm-1"
    if hf.endswith("linear_attn.A_log"):
        return "log(-a)+v-reorder"
    if hf.endswith("linear_attn.dt_bias"):
        return "v-reorder"
    if hf.endswith("linear_attn.conv1d.weight"):
        return "unsqueeze+v-reorder"
    if hf.endswith(V_REORDER_ROWS + ("linear_attn.in_proj_qkv.weight",)):
        return "v-reorder"
    if hf.endswith("linear_attn.out_proj.weight"):
        return "v-reorder(cols)"
    return "none"


def v_order(n_k, n_v):
    """Source head for each destination head: llama.cpp stores the value heads tiled
    ([ratio][k]); HF keeps them grouped by key head ([k][ratio])."""
    ratio = n_v // n_k
    return [r * n_k + k for k in range(n_k) for r in range(ratio)]


def reorder_units(raw, unit, order):
    return b"".join(raw[s * unit:(s + 1) * unit] for s in order)


def reorder_cols(raw, rows, unit, order):
    stride = len(raw) // rows
    return b"".join(reorder_units(raw[r * stride:(r + 1) * stride], unit, order)
                    for r in range(rows))


def reorder_qkv(raw, row, geo):
    """Rows are [q | k | v]; only the v rows (one head = v_dim rows) were reordered."""
    q_k = 2 * geo.n_k * geo.k_dim * row
    return raw[:q_k] + reorder_units(raw[q_k:], geo.v_dim * row, geo.order)


def to_floats(dtype, raw):
    return st.bf16_to_floats(raw) if dtype == "BF16" else st.f32_to_floats(raw)


def from_floats(dtype, values):
    return st.floats_to_bf16(values) if dtype == "BF16" else st.floats_to_f32(values)


class Geometry:
    def __init__(self, config):
        t = config.get("text_config", config)
        self.n_k = t["linear_num_key_heads"]
        self.n_v = t["linear_num_value_heads"]
        self.k_dim = t["linear_key_head_dim"]
        self.v_dim = t["linear_value_head_dim"]
        self.order = v_order(self.n_k, self.n_v)


def convert_tensor(hf, dtype, raw, shape, geo, out_dtype):
    """-> raw bytes in HF layout and `out_dtype`. `shape` is the HF shape of the GGUF tensor."""
    kind = transform_of(hf)
    if kind == "none":
        pass
    elif kind == "norm-1":
        raw, dtype = from_floats(out_dtype, [x - 1.0 for x in to_floats(dtype, raw)]), out_dtype
    elif hf.endswith("linear_attn.A_log"):
        vals = to_floats(dtype, raw)
        raw = from_floats(out_dtype, [math.log(-vals[s]) for s in geo.order])
        dtype = out_dtype
    elif hf.endswith("linear_attn.dt_bias"):
        raw = reorder_units(raw, st.DTYPE_BYTES[dtype], geo.order)
    elif hf.endswith("linear_attn.conv1d.weight"):
        raw = reorder_qkv(raw, shape[-1] * st.DTYPE_BYTES[dtype], geo)
    elif hf.endswith("linear_attn.in_proj_qkv.weight"):
        raw = reorder_qkv(raw, shape[-1] * st.DTYPE_BYTES[dtype], geo)
    elif hf.endswith("linear_attn.in_proj_z.weight"):
        raw = reorder_units(raw, geo.v_dim * shape[-1] * st.DTYPE_BYTES[dtype], geo.order)
    elif hf.endswith(("linear_attn.in_proj_a.weight", "linear_attn.in_proj_b.weight")):
        raw = reorder_units(raw, shape[-1] * st.DTYPE_BYTES[dtype], geo.order)
    elif hf.endswith("linear_attn.out_proj.weight"):
        raw = reorder_cols(raw, shape[0], geo.v_dim * st.DTYPE_BYTES[dtype], geo.order)
    if dtype != out_dtype:
        raw = from_floats(out_dtype, to_floats(dtype, raw))
    return raw


# --------------------------------------------------------------------------- tokenizer check

def check_tokenizer(gguf, base_dir):
    tokens = gguf.meta.get("tokenizer.ggml.tokens")
    if not tokens:
        raise ConvertError("GGUF has no tokenizer.ggml.tokens")
    with open(os.path.join(base_dir, "tokenizer.json"), encoding="utf-8") as f:
        tj = json.load(f)
    expect = dict((i, t) for t, i in tj["model"]["vocab"].items())
    expect.update((a["id"], a["content"]) for a in tj.get("added_tokens", []))
    with open(os.path.join(base_dir, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    vocab = cfg.get("text_config", cfg)["vocab_size"]
    if len(tokens) != vocab:
        raise ConvertError("tokenizer mismatch: GGUF has %d tokens, base vocab_size is %d"
                           % (len(tokens), vocab))
    bad = [i for i, t in expect.items() if i >= len(tokens) or tokens[i] != t]
    if bad:
        i = min(bad)
        raise ConvertError("tokenizer mismatch: %d of %d base tokens differ, first id %d: "
                           "base %r, GGUF %r" % (len(bad), len(expect), i, expect[i],
                                                 tokens[i] if i < len(tokens) else None))
    return len(expect)


# --------------------------------------------------------------------------- convert

def convert(base_dir, gguf_path, out_dir):
    gguf = Gguf(gguf_path)
    if gguf.meta.get("general.architecture") != "qwen35":
        raise ConvertError("GGUF architecture is %r, expected 'qwen35'"
                           % gguf.meta.get("general.architecture"))
    n_tok = check_tokenizer(gguf, base_dir)
    print("tokenizer: %d base tokens match the GGUF" % n_tok)
    with open(os.path.join(base_dir, "config.json"), encoding="utf-8") as f:
        geo = Geometry(json.load(f))
    base = st.open_checkpoint(base_dir)

    mapped = {}
    for name in gguf.tensors:
        hf = hf_name(name)
        if not base.has(hf):
            raise ConvertError("%s maps to %s, which the base checkpoint lacks" % (name, hf))
        shape = gguf.hf_shape(name)
        if transform_of(hf) == "unsqueeze+v-reorder":
            if len(shape) != 2:
                raise ConvertError("%s: expected a squeezed 2-D conv1d, got %r" % (name, shape))
            shape = (shape[0], 1, shape[1])
        if tuple(base.shape(hf)) != shape:
            raise ConvertError("%s: GGUF shape %r does not match base %r for %s"
                               % (name, shape, tuple(base.shape(hf)), hf))
        mapped[hf] = name
    copied = [n for n in base.names() if n not in mapped]
    stray = [n for n in copied if not n.startswith(COPIED_PREFIXES)]
    if stray:
        raise ConvertError("GGUF lacks %d text tensors, e.g. %s" % (len(stray), stray[0]))

    specs = {n: (base.dtype(n), tuple(base.shape(n))) for n in base.names()}
    os.makedirs(out_dir, exist_ok=True)
    for fname in COPIED_FILES:
        shutil.copyfile(os.path.join(base_dir, fname), os.path.join(out_dir, fname))
    target = os.path.join(out_dir, st.SINGLE_NAME)
    tmp = target + ".tmp"
    with open(tmp, "wb") as out:
        out.write(st.header_bytes(specs))
        for hf in sorted(specs):
            if hf in mapped:
                dtype, raw = gguf.read(mapped[hf])
                out.write(convert_tensor(hf, dtype, raw, gguf.hf_shape(mapped[hf]), geo,
                                         specs[hf][0]))
            else:
                out.write(base.view(hf))
    os.replace(tmp, target)
    print("converted %d tensors from the GGUF, copied %d from the base (%s)"
          % (len(mapped), len(copied),
             ", ".join("%d %s" % (sum(n.startswith(p) for n in copied), p.rstrip("."))
                       for p in COPIED_PREFIXES)))
    print("wrote %s (%d bytes)" % (target, os.path.getsize(target)))


# --------------------------------------------------------------------------- check

def sample_floats(ck, name):
    """Up to ~CHECK_SAMPLE evenly spaced elements as floats, plus the stride used."""
    dtype, shape = ck.dtype(name), ck.shape(name)
    step = max(1, st.numel(shape) // CHECK_SAMPLE)
    mv = ck.view(name).cast("H" if dtype == "BF16" else "f")[::step]
    if dtype == "BF16":
        bits = mv.tolist()
        return list(struct.unpack("<%df" % len(bits), struct.pack("<%dI" % len(bits),
                                                                   *[b << 16 for b in bits])))
    return mv.tolist()


def compare(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    cos = dot / (na * nb) if na and nb else (1.0 if na == nb else 0.0)
    peak = max(abs(y) for y in b) or 1.0
    return cos, max(abs(x - y) for x, y in zip(a, b)) / peak


def check(base_dir, out_dir):
    base = st.open_checkpoint(base_dir)
    out = st.open_checkpoint(out_dir)
    if out.names() != base.names():
        raise ConvertError("output and base have different tensor names")
    rows, failed = [], 0
    for name in out.names():
        if name.startswith(COPIED_PREFIXES):
            continue
        if (out.dtype(name), out.shape(name)) != (base.dtype(name), base.shape(name)):
            raise ConvertError("%s: dtype or shape differs from base" % name)
        cos, rel = compare(sample_floats(out, name), sample_floats(base, name))
        failed += cos < CHECK_MIN_COSINE
        rows.append((cos, rel, name))
    print("%-8s %-10s %-20s %s" % ("cosine", "max|d|/max", "transform", "tensor"))
    for cos, rel, name in sorted(rows):
        flag = "" if cos >= CHECK_MIN_COSINE else "  FAIL"
        print("%-8.4f %-10.4f %-20s %s%s" % (cos, rel, transform_of(name), name, flag))
    lowest = min(rows)
    print("checked %d text tensors, min cosine %.4f (%s), %d below %.2f"
          % (len(rows), lowest[0], lowest[2], failed, CHECK_MIN_COSINE))
    return failed == 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", required=True, help="base HF checkpoint dir (Qwen/Qwen3.5-4B)")
    ap.add_argument("--gguf", help="BF16 GGUF file to convert")
    ap.add_argument("--out", required=True, help="output dir")
    ap.add_argument("--check", action="store_true", help="compare --out against --base")
    args = ap.parse_args(argv)
    try:
        if args.check:
            return 0 if check(args.base, args.out) else 1
        if not args.gguf:
            ap.error("--gguf is required unless --check")
        convert(args.base, args.gguf, args.out)
    except (ConvertError, st.SafetensorsError) as e:
        print("error: %s" % e, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
