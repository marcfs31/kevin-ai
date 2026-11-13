#!/usr/bin/env python3
"""Reference q8 quantiser (the format owned by docs/PLAN.md; `bin/lm-quant` mirrors this).

CLI (same contract as lm-quant):  quant_ref.py --model DIR_OR_FILE --out FILE

WHICH TENSORS
  Only tensors whose name starts with `model.language_model.` are written. `model.visual.*`,
  `mtp.*` and anything else are DROPPED (text-only engine; this is what makes the real
  file about 4.75 GB). Of the kept tensors, those named `*.weight` with exactly 2 dims
  (including `embed_tokens.weight`) are quantised; all others (1-D norms, `A_log`,
  `dt_bias`, the 3-D `conv1d.weight`) are copied byte-for-byte with their own dtype/shape.
  Source dtype of a quantised matrix must be BF16 or F32. `in` must be a multiple of 32, else
  the quantiser fails loudly. A checkpoint with no `model.language_model.` tensor is an error.

QUANTISATION (row-major [out, in]; groups of 32 consecutive inputs inside one row)
  absmax = max |x| over the group, x = the stored value widened to double (bf16 -> exact)
  scale  = (float)(absmax / 127.0)    # division in double, rounded ONCE to f32 (nearest even)
  scale == 0 (after rounding) -> all 32 q = 0
  else q = clamp(round_half_even((double)x / (double)scale), -127, 127)
  Negative zero or a value that rounds to 0 gives q = 0 (int8 has no -0). Non-finite input
  is an error.
  Dequantised value (what the oracle uses) = (double)q * (double)scale, computed in double.

OUTPUT FILE (one safetensors file, byte layout from tools/safetensors.py)
  Per quantised matrix `<name>`: `<name>` dtype I8 shape [out, in] (row-major q) and
  `<name>.q8scale` dtype F32 shape [out, in/32] (row-major little-endian f32 scale per group).
  Every other kept tensor: same name/dtype/shape/bytes.
  Tensor order: ALL output names (including the `.q8scale` ones) in plain bytewise sort of
  the name string (ASCII; so `x.weight` < `x.weight.q8scale`, and e.g.
  `layers.10.` sorts before `layers.2.`). Data is stored in that same order, back to back,
  no padding or alignment between tensors, offsets relative to the data area.
  File = u64 LE header length N | header (N bytes) | data, with N a multiple of 8.
  Header = compact JSON (`,` and `:` separators, no spaces, ASCII, no newline), padded
  on the right with 0x20 spaces to a multiple of 8 bytes, exactly:
    {"__metadata__":{"format":"lm-q8","group":"32","source":"<hex>"},
     "<name>":{"dtype":"I8","shape":[out,in],"data_offsets":[a,b]}, ...}
  (written without the line break shown). Key order inside each tensor entry is dtype, shape,
  data_offsets; `__metadata__` key order is format, group, source; all three values are
  strings. `source` is the lowercase hex sha256 of `model.safetensors.index.json` (its
  bytes) when the input is a sharded directory, else of the single input file's bytes.
  Shape lists use no spaces, ints in plain decimal.
"""

import argparse
import struct
import sys

import safetensors as st

GROUP = 32
TEXT_PREFIX = "model.language_model."
SCALE_SUFFIX = ".q8scale"
QMAX = 127


class QuantError(ValueError):
    pass


def quantize_group(vals):
    """vals: <= any length list of finite floats (one group). -> (scale as f32-valued float, [q])."""
    absmax = max(abs(v) for v in vals)
    if absmax != absmax or absmax == float("inf"):
        raise QuantError("non-finite weight")
    scale = st.round_f32(absmax / 127.0)
    if scale == 0.0:
        return 0.0, [0] * len(vals)
    qs = []
    for v in vals:
        q = round(v / scale)  # Python round() on a float is round-half-even
        qs.append(max(-QMAX, min(QMAX, q)))
    return scale, qs


def quantize_matrix(values, rows, cols):
    """values: flat row-major list. -> (q list of ints [rows*cols], scales list [rows*cols/32])."""
    if cols % GROUP:
        raise QuantError("in dimension %d is not a multiple of %d" % (cols, GROUP))
    q, scales = [], []
    for r in range(rows):
        base = r * cols
        for g in range(0, cols, GROUP):
            s, qs = quantize_group(values[base + g:base + g + GROUP])
            scales.append(s)
            q.extend(qs)
    return q, scales


def dequantize_matrix(q, scales, rows, cols):
    """-> flat list of doubles, (double)q * (double)scale."""
    out = []
    ngroups = cols // GROUP
    for r in range(rows):
        for g in range(ngroups):
            s = scales[r * ngroups + g]
            base = r * cols + g * GROUP
            out.extend(float(x) * s for x in q[base:base + GROUP])
    return out


def is_quantised(name, shape):
    return name.endswith(".weight") and len(shape) == 2


def quantize_store(store, source_sha256):
    """store: Checkpoint-like (names/dtype/shape/raw/floats). Returns the q8 file bytes."""
    names = [n for n in store.names() if n.startswith(TEXT_PREFIX)]
    if not names:
        raise QuantError("no tensor starts with %r" % TEXT_PREFIX)
    out = {}
    for name in names:
        dt, shape = store.dtype(name), store.shape(name)
        if not is_quantised(name, shape):
            out[name] = (dt, shape, store.raw(name))
            continue
        if dt not in ("BF16", "F32"):
            raise QuantError("%s: cannot quantise dtype %s" % (name, dt))
        rows, cols = shape
        q, scales = quantize_matrix(store.floats(name), rows, cols)
        out[name] = ("I8", (rows, cols), struct.pack("<%db" % len(q), *q))
        out[name + SCALE_SUFFIX] = ("F32", (rows, cols // GROUP), st.floats_to_f32(scales))
    meta = {"format": "lm-q8", "group": str(GROUP), "source": source_sha256}
    return st.serialize(out, metadata=meta)


def quantize_checkpoint(model, out_path):
    store = st.open_checkpoint(model)
    data = quantize_store(store, store.source_sha256)
    with open(out_path, "wb") as f:
        f.write(data)
    return data


def load_q8(path_or_bytes):
    t = st.parse(path_or_bytes, "<q8>") if isinstance(path_or_bytes, (bytes, bytearray)) \
        else st.open_file(path_or_bytes)
    if t.metadata.get("format") != "lm-q8" or t.metadata.get("group") != str(GROUP):
        raise QuantError("not an lm-q8 group-%d file" % GROUP)
    return t


def main(argv=None):
    ap = argparse.ArgumentParser(description="write the q8 file (reference for lm-quant)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    data = quantize_checkpoint(a.model, a.out)
    print("wrote %s (%d bytes)" % (a.out, len(data)), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
