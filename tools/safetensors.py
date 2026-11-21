#!/usr/bin/env python3
"""Dependency-free safetensors reader and writer (standard library only).

Shadows the PyPI package name on purpose: run the tools from `tools/` and nothing else is
needed. Supports dtypes BF16, F32, I8 and U8 only; anything else is rejected.

Reading (`parse`, `open_checkpoint`):
  * a single file, or a sharded checkpoint directory described by
    `model.safetensors.index.json` (`weight_map` name -> shard file);
  * a directory without an index falls back to `model.safetensors`;
  * malformed headers are rejected with `SafetensorsError`: short file, header length past
    EOF or above 100 MiB, non-UTF-8 or non-object JSON, duplicate keys, unknown dtype, bad
    shape, bad `data_offsets`, extent that does not equal prod(shape) * itemsize, tensors
    that overlap, leave holes, or do not exactly cover the data area, non-string
    `__metadata__` values.

Writing (`serialize`, `write_file`): deterministic byte layout, the one the q8 file uses.
  * 8 bytes little-endian u64 header length N, then the N header bytes, then tensor data.
  * Header JSON is compact (`separators=(",", ":")`, `ensure_ascii=True`, no trailing
    newline): `__metadata__` first when present (keys in the caller's order, values strings),
    then tensors in the written order, each as {"dtype":D,"shape":[...],"data_offsets":[a,b]}
    with exactly those keys in that order. Header is right-padded with 0x20 so that N is a
    multiple of 8.
  * Tensor data follows in the same order as the header, back to back, with NO padding
    or alignment between tensors (the safetensors spec forbids holes). Offsets are relative
    to the start of the data area (file offset 8 + N). Readers must not assume that a tensor
    of a wider dtype is naturally aligned.
  * Default order is plain bytewise sort of the tensor names.
"""

import hashlib
import json
import mmap
import os
import struct
import sys

DTYPE_BYTES = {"BF16": 2, "F32": 4, "I8": 1, "U8": 1}
MAX_HEADER = 100 * 1024 * 1024
INDEX_NAME = "model.safetensors.index.json"
SINGLE_NAME = "model.safetensors"


class SafetensorsError(ValueError):
    pass


# --------------------------------------------------------------------------- numeric helpers

def bf16_to_floats(raw):
    """BF16 little-endian bytes -> list of floats. Exact (bf16 is the top half of an f32)."""
    raw = bytes(raw)
    if len(raw) % 2:
        raise SafetensorsError("bf16 buffer has odd length")
    n = len(raw) // 2
    buf = bytearray(4 * n)
    buf[2::4] = raw[0::2]
    buf[3::4] = raw[1::2]
    return list(struct.unpack("<%df" % n, bytes(buf)))


def float_to_bf16_bits(v):
    """Round a float to bf16 (round to nearest even, via f32). Returns the 16-bit pattern."""
    bits = struct.unpack("<I", struct.pack("<f", v))[0]
    if (bits & 0x7F800000) == 0x7F800000 and (bits & 0x007FFFFF):
        return ((bits >> 16) | 0x0040) & 0xFFFF  # quiet NaN
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) & 0xFFFF


def floats_to_bf16(values):
    return b"".join(struct.pack("<H", float_to_bf16_bits(v)) for v in values)


def round_bf16(v):
    return bf16_to_floats(floats_to_bf16([v]))[0]


def round_f32(v):
    return struct.unpack("<f", struct.pack("<f", v))[0]


def f32_to_floats(raw):
    raw = bytes(raw)
    if len(raw) % 4:
        raise SafetensorsError("f32 buffer length is not a multiple of 4")
    return list(struct.unpack("<%df" % (len(raw) // 4), raw))


def floats_to_f32(values):
    return struct.pack("<%df" % len(values), *values)


def numel(shape):
    n = 1
    for d in shape:
        n *= d
    return n


# --------------------------------------------------------------------------- parsing

def _no_duplicates(pairs):
    out = {}
    for k, v in pairs:
        if k in out:
            raise SafetensorsError("duplicate key in header: %r" % k)
        out[k] = v
    return out


class Tensors:
    """Parsed safetensors buffer. `buf` is bytes, bytearray, memoryview or mmap."""

    def __init__(self, buf, label="<buffer>"):
        self.label = label
        self.buf = memoryview(buf)
        total = len(self.buf)
        if total < 8:
            raise SafetensorsError("%s: shorter than 8 bytes" % label)
        (hlen,) = struct.unpack("<Q", self.buf[:8])
        if hlen > MAX_HEADER:
            raise SafetensorsError("%s: header length %d above limit" % (label, hlen))
        if 8 + hlen > total:
            raise SafetensorsError("%s: header length %d runs past end of file" % (label, hlen))
        try:
            text = bytes(self.buf[8:8 + hlen]).decode("utf-8")
            header = json.loads(text, object_pairs_hook=_no_duplicates)
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise SafetensorsError("%s: header is not valid JSON: %s" % (label, e))
        if not isinstance(header, dict):
            raise SafetensorsError("%s: header is not a JSON object" % label)
        self.data_start = 8 + hlen
        data_len = total - self.data_start
        meta = header.pop("__metadata__", {})
        if not isinstance(meta, dict) or not all(
                isinstance(k, str) and isinstance(v, str) for k, v in meta.items()):
            raise SafetensorsError("%s: __metadata__ must map strings to strings" % label)
        self.metadata = meta
        self.info = {}
        extents = []
        for name, t in header.items():
            if not isinstance(t, dict) or set(t) != {"dtype", "shape", "data_offsets"}:
                raise SafetensorsError("%s: bad entry for %r" % (label, name))
            dt, shape, off = t["dtype"], t["shape"], t["data_offsets"]
            if dt not in DTYPE_BYTES:
                raise SafetensorsError("%s: unsupported dtype %r for %r" % (label, dt, name))
            if (not isinstance(shape, list)
                    or not all(type(d) is int and d >= 0 for d in shape)):
                raise SafetensorsError("%s: bad shape for %r" % (label, name))
            if (not isinstance(off, list) or len(off) != 2
                    or not all(type(o) is int and o >= 0 for o in off) or off[0] > off[1]):
                raise SafetensorsError("%s: bad data_offsets for %r" % (label, name))
            if off[1] - off[0] != numel(shape) * DTYPE_BYTES[dt]:
                raise SafetensorsError("%s: extent of %r does not match shape and dtype"
                                       % (label, name))
            self.info[name] = (dt, tuple(shape), off[0], off[1])
            extents.append((off[0], off[1], name))
        extents.sort()
        pos = 0
        for lo, hi, name in extents:
            if lo != pos:
                kind = "overlaps" if lo < pos else "leaves a hole before"
                raise SafetensorsError("%s: data %s %r" % (label, kind, name))
            pos = hi
        if pos != data_len:
            raise SafetensorsError("%s: tensors cover %d of %d data bytes" % (label, pos, data_len))

    def names(self):
        return sorted(self.info)

    def has(self, name):
        return name in self.info

    def dtype(self, name):
        return self.info[name][0]

    def shape(self, name):
        return self.info[name][1]

    def view(self, name):
        _, _, lo, hi = self.info[name]
        return self.buf[self.data_start + lo:self.data_start + hi]

    def raw(self, name):
        return bytes(self.view(name))

    def floats(self, name):
        dt = self.dtype(name)
        if dt == "BF16":
            return bf16_to_floats(self.raw(name))
        if dt == "F32":
            return f32_to_floats(self.raw(name))
        raise SafetensorsError("%s: %r is %s, not a float tensor" % (self.label, name, dt))

    def ints(self, name):
        dt = self.dtype(name)
        raw = self.raw(name)
        if dt == "I8":
            return list(struct.unpack("<%db" % len(raw), raw))
        if dt == "U8":
            return list(raw)
        raise SafetensorsError("%s: %r is %s, not an integer tensor" % (self.label, name, dt))

    def source_sha256(self):
        return hashlib.sha256(self.buf).hexdigest()


def parse(buf, label="<buffer>"):
    return Tensors(buf, label)


def open_file(path):
    with open(path, "rb") as f:
        size = os.fstat(f.fileno()).st_size
        if size == 0:
            raise SafetensorsError("%s: empty file" % path)
        return Tensors(mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ), path)


class Checkpoint:
    """A set of tensors from one file or from the shards of an index.json."""

    def __init__(self, shards, weight_map, source_sha256):
        self._shards = shards          # shard name -> Tensors
        self._map = weight_map         # tensor name -> shard name
        self.source_sha256 = source_sha256
        self.metadata = next(iter(shards.values())).metadata if len(shards) == 1 else {}

    def names(self):
        return sorted(self._map)

    def has(self, name):
        return name in self._map

    def _t(self, name):
        return self._shards[self._map[name]]

    def dtype(self, name):
        return self._t(name).dtype(name)

    def shape(self, name):
        return self._t(name).shape(name)

    def view(self, name):
        return self._t(name).view(name)

    def raw(self, name):
        return self._t(name).raw(name)

    def floats(self, name):
        return self._t(name).floats(name)

    def ints(self, name):
        return self._t(name).ints(name)


def open_checkpoint(path):
    """`path` is a .safetensors file or a directory (index.json -> shards, else model.safetensors)."""
    if os.path.isdir(path):
        index = os.path.join(path, INDEX_NAME)
        if os.path.exists(index):
            with open(index, "rb") as f:
                raw = f.read()
            try:
                wm = json.loads(raw.decode("utf-8"), object_pairs_hook=_no_duplicates)["weight_map"]
            except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as e:
                raise SafetensorsError("%s: bad index: %s" % (index, e))
            if not isinstance(wm, dict) or not all(
                    isinstance(k, str) and isinstance(v, str) for k, v in wm.items()):
                raise SafetensorsError("%s: weight_map must map strings to strings" % index)
            shards = {}
            for shard in sorted(set(wm.values())):
                if os.path.basename(shard) != shard:
                    raise SafetensorsError("%s: shard name %r is not a plain file name"
                                           % (index, shard))
                shards[shard] = open_file(os.path.join(path, shard))
            for name, shard in wm.items():
                if not shards[shard].has(name):
                    raise SafetensorsError("%s: %r not in shard %s" % (index, name, shard))
            return Checkpoint(shards, dict(wm), hashlib.sha256(raw).hexdigest())
        path = os.path.join(path, SINGLE_NAME)
    t = open_file(path)
    return Checkpoint({"": t}, {n: "" for n in t.info}, t.source_sha256())


class MemoryCheckpoint(Checkpoint):
    """Same interface, over a dict name -> (dtype, shape, raw bytes) held in memory."""

    def __init__(self, tensors, source_sha256="memory"):
        t = parse(serialize(tensors), "<memory>")
        super().__init__({"": t}, {n: "" for n in t.info}, source_sha256)


# --------------------------------------------------------------------------- writing

def header_bytes(specs, metadata=None, order=None):
    """specs: dict name -> (dtype, shape). Returns the 8-byte length prefix plus the padded
    header, so a caller can stream the tensor bytes (in `order`) after it."""
    names = list(order) if order is not None else sorted(specs)
    if sorted(names) != sorted(specs):
        raise SafetensorsError("order does not match the tensor names")
    header = {}
    if metadata is not None:
        if not all(isinstance(k, str) and isinstance(v, str) for k, v in metadata.items()):
            raise SafetensorsError("metadata must map strings to strings")
        header["__metadata__"] = dict(metadata)
    offset = 0
    for name in names:
        dt, shape = specs[name]
        if dt not in DTYPE_BYTES:
            raise SafetensorsError("unsupported dtype %r for %r" % (dt, name))
        size = numel(shape) * DTYPE_BYTES[dt]
        header[name] = {"dtype": dt, "shape": list(shape), "data_offsets": [offset, offset + size]}
        offset += size
    h = json.dumps(header, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    h += b" " * ((-len(h)) % 8)
    return struct.pack("<Q", len(h)) + h


def serialize(tensors, metadata=None, order=None):
    """tensors: dict name -> (dtype, shape, raw bytes). Returns the file bytes.

    `order` defaults to sorted(names) (bytewise). `metadata` is an ordered dict of str -> str.
    """
    names = list(order) if order is not None else sorted(tensors)
    head = header_bytes({n: (t[0], t[1]) for n, t in tensors.items()}, metadata, names)
    blobs = []
    for name in names:
        dt, shape, raw = tensors[name]
        raw = bytes(raw)
        if len(raw) != numel(shape) * DTYPE_BYTES[dt]:
            raise SafetensorsError("size mismatch for %r" % name)
        blobs.append(raw)
    return head + b"".join(blobs)


def write_file(path, tensors, metadata=None, order=None):
    data = serialize(tensors, metadata, order)
    with open(path, "wb") as f:
        f.write(data)
    return data


if __name__ == "__main__":
    for p in sys.argv[1:]:
        c = open_checkpoint(p)
        for n in c.names():
            print(n, c.dtype(n), list(c.shape(n)))
