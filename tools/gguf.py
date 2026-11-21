#!/usr/bin/env python3
"""GGUF v3 reader and writer, and the ggml block quantisers / dequantisers the engine reads
(F32, F16, Q8_0, Q4_K, Q5_K, Q6_K). Standard library only.

The layouts are ggml's (ggml-common.h). The dequantisers evaluate each formula in float64,
which is what tools/ref_qwen2.py wants; the quantisers are plain min/max block fits, valid
encodings but not llama.cpp's iterative optimum (the fixture only needs bytes the engine and
the oracle decode identically).

    gguf.py FILE        print the header: metadata (arrays shortened) and the tensor table
"""

import struct
import sys

GGUF_MAGIC = b"GGUF"
F32, F16, Q8_0, Q4_K, Q5_K, Q6_K = 0, 1, 8, 12, 13, 14
BLOCK = {F32: (1, 4), F16: (1, 2), Q8_0: (32, 34), Q4_K: (256, 144), Q5_K: (256, 176), Q6_K: (256, 210)}
TYPE_NAME = {F32: "F32", F16: "F16", Q8_0: "Q8_0", Q4_K: "Q4_K", Q5_K: "Q5_K", Q6_K: "Q6_K"}

# metadata value types
U8, I8, U16, I16, U32, I32, FLT32, BOOL, STR, ARR, U64, I64, FLT64 = range(13)
_SCALAR = {U8: "B", I8: "b", U16: "H", I16: "h", U32: "I", I32: "i", FLT32: "f", BOOL: "?",
           U64: "Q", I64: "q", FLT64: "d"}


class GgufError(ValueError):
    pass


# --------------------------------------------------------------------------- reader

class Reader:
    def __init__(self, path):
        self.path = path
        with open(path, "rb") as f:
            self.buf = f.read()
        if self.buf[:4] != GGUF_MAGIC:
            raise GgufError("%s: not a GGUF file" % path)
        self.p = 4

    def take(self, fmt):
        n = struct.calcsize("<" + fmt)
        if self.p + n > len(self.buf):
            raise GgufError("%s: truncated" % self.path)
        v = struct.unpack_from("<" + fmt, self.buf, self.p)
        self.p += n
        return v

    def string(self):
        (n,) = self.take("Q")
        if self.p + n > len(self.buf):
            raise GgufError("%s: truncated string" % self.path)
        s = self.buf[self.p:self.p + n].decode("utf-8")
        self.p += n
        return s

    def value(self, t):
        if t in _SCALAR:
            return self.take(_SCALAR[t])[0]
        if t == STR:
            return self.string()
        if t == ARR:
            (et,) = self.take("I")
            (n,) = self.take("Q")
            return [self.value(et) for _ in range(n)]
        raise GgufError("unknown metadata type %d" % t)


def read_gguf(path):
    """-> (meta {key: value}, tensors {name: (dims, type, absolute offset)}, file bytes)."""
    r = Reader(path)
    (version,) = r.take("I")
    if version not in (2, 3):
        raise GgufError("GGUF version %d" % version)
    nt, nkv = r.take("QQ")
    meta = {}
    for _ in range(nkv):
        k = r.string()
        (t,) = r.take("I")
        meta[k] = r.value(t)
    infos = []
    for _ in range(nt):
        name = r.string()
        (nd,) = r.take("I")
        dims = r.take("%dQ" % nd)
        ty, off = r.take("IQ")
        infos.append((name, dims, ty, off))
    align = meta.get("general.alignment", 32)
    start = (r.p + align - 1) // align * align
    tensors = {name: (tuple(dims), ty, start + off) for name, dims, ty, off in infos}
    return meta, tensors, r.buf


def tensor_bytes(buf, dims, ty, off):
    n = 1
    for d in dims:
        n *= d
    be, bb = BLOCK[ty]
    return buf[off:off + n // be * bb]


# --------------------------------------------------------------------------- writer

def _pack_value(t, v):
    if t in _SCALAR:
        return struct.pack("<" + _SCALAR[t], v)
    if t == STR:
        b = v.encode("utf-8")
        return struct.pack("<Q", len(b)) + b
    raise GgufError("cannot pack metadata type %d" % t)


def write_gguf(path, meta, tensors, alignment=32):
    """meta: [(key, type, value)], an array as (key, ARR, (elem_type, [values])).
    tensors: [(name, dims innermost first, ggml type, bytes)]."""
    out = bytearray(GGUF_MAGIC + struct.pack("<IQQ", 3, len(tensors), len(meta) + 1))
    for key, t, v in [("general.alignment", U32, alignment)] + list(meta):
        kb = key.encode()
        out += struct.pack("<Q", len(kb)) + kb + struct.pack("<I", t)
        if t == ARR:
            et, vals = v
            out += struct.pack("<IQ", et, len(vals))
            for x in vals:
                out += _pack_value(et, x)
        else:
            out += _pack_value(t, v)
    off = 0
    offsets = []
    for name, dims, ty, data in tensors:
        be, bb = BLOCK[ty]
        n = 1
        for d in dims:
            n *= d
        if dims[0] % be or len(data) != n // be * bb:
            raise GgufError("tensor %s: %d bytes is not %s of %s" % (name, len(data), "x".join(map(str, dims)), TYPE_NAME[ty]))
        nb = name.encode()
        out += struct.pack("<Q", len(nb)) + nb + struct.pack("<I", len(dims)) + struct.pack("<%dQ" % len(dims), *dims)
        out += struct.pack("<IQ", ty, off)
        offsets.append(off)
        off += (len(data) + alignment - 1) // alignment * alignment
    out += b"\0" * ((-len(out)) % alignment)
    for name, dims, ty, data in tensors:
        out += data + b"\0" * ((-len(data)) % alignment)
    with open(path, "wb") as f:
        f.write(out)


# --------------------------------------------------------------------------- f16

def f16(x):
    """x rounded to half precision (round to nearest even), as a float."""
    return struct.unpack("<e", struct.pack("<e", x))[0]


def f16_bytes(x):
    return struct.pack("<e", x)


# --------------------------------------------------------------------------- Q8_0

def quant_q8_0(x):
    out = bytearray()
    for b in range(0, len(x), 32):
        blk = x[b:b + 32]
        amax = max(abs(v) for v in blk)
        d = f16(amax / 127.0)
        out += f16_bytes(d)
        for v in blk:
            q = 0 if d == 0 else max(-127, min(127, int(round(v / d))))
            out += struct.pack("<b", q)
    return bytes(out)


def dequant_q8_0(data, n):
    out = []
    for b in range(n // 32):
        blk = data[b * 34:(b + 1) * 34]
        d = struct.unpack_from("<e", blk, 0)[0]
        out += [d * q for q in struct.unpack_from("<32b", blk, 2)]
    return out


# --------------------------------------------------------------------------- K-quant scales

def _pack_scale_min(sc, mn):
    """8 six-bit scales and 8 six-bit mins into 12 bytes (ggml's K4 packing)."""
    q = bytearray(12)
    for j in range(4):
        q[j] = (sc[j] & 63) | ((sc[j + 4] >> 4) << 6)
        q[j + 4] = (mn[j] & 63) | ((mn[j + 4] >> 4) << 6)
    for j in range(4, 8):
        q[j + 4] = (sc[j] & 0xF) | ((mn[j] & 0xF) << 4)
    return bytes(q)


def _unpack_scale_min(q, j):
    if j < 4:
        return q[j] & 63, q[j + 4] & 63
    return (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4)


def _fit_sub_blocks(blk, levels):
    """Per sub-block of 32: (scale, min) with x ~ scale * q - min, q in 0..levels."""
    sc, mn = [], []
    for j in range(8):
        s = blk[32 * j:32 * j + 32]
        lo, hi = min(s), max(s)
        lo = min(lo, 0.0)
        sc.append((hi - lo) / levels if hi > lo else 0.0)
        mn.append(-lo)
    return sc, mn


def _quant_k(x, levels, five):
    out = bytearray()
    for b in range(0, len(x), 256):
        blk = x[b:b + 256]
        sc, mn = _fit_sub_blocks(blk, levels)
        d = f16(max(sc) / 63.0) if max(sc) > 0 else 0.0
        dmin = f16(max(mn) / 63.0) if max(mn) > 0 else 0.0
        isc = [0 if d == 0 else max(0, min(63, int(round(v / d)))) for v in sc]
        imn = [0 if dmin == 0 else max(0, min(63, int(round(v / dmin)))) for v in mn]
        qv = []
        for j in range(8):
            ds, dm = d * isc[j], dmin * imn[j]
            for v in blk[32 * j:32 * j + 32]:
                qv.append(0 if ds == 0 else max(0, min(levels, int(round((v + dm) / ds)))))
        qs = bytearray(128)
        qh = bytearray(32)
        for j in range(4):
            for l in range(32):
                lo, hi = qv[64 * j + l], qv[64 * j + 32 + l]
                qs[32 * j + l] = (lo & 0xF) | ((hi & 0xF) << 4)
                if five:
                    qh[l] |= (((lo >> 4) & 1) << (2 * j)) | (((hi >> 4) & 1) << (2 * j + 1))
        out += f16_bytes(d) + f16_bytes(dmin) + _pack_scale_min(isc, imn)
        if five:
            out += bytes(qh)
        out += bytes(qs)
    return bytes(out)


def quant_q4_k(x):
    return _quant_k(x, 15, False)


def quant_q5_k(x):
    return _quant_k(x, 31, True)


def _dequant_k(data, n, five):
    size = 176 if five else 144
    out = []
    for b in range(n // 256):
        blk = data[b * size:(b + 1) * size]
        d = struct.unpack_from("<e", blk, 0)[0]
        dmin = struct.unpack_from("<e", blk, 2)[0]
        sc12 = blk[4:16]
        qh = blk[16:48] if five else None
        qs = blk[48:176] if five else blk[16:144]
        for j in range(4):
            s1, m1 = _unpack_scale_min(sc12, 2 * j)
            s2, m2 = _unpack_scale_min(sc12, 2 * j + 1)
            for half, (s, m) in enumerate(((s1, m1), (s2, m2))):
                for l in range(32):
                    q = (qs[32 * j + l] >> (4 * half)) & 0xF
                    if five and qh[l] & (1 << (2 * j + half)):
                        q += 16
                    out.append((d * s) * q - dmin * m)
    return out


def dequant_q4_k(data, n):
    return _dequant_k(data, n, False)


def dequant_q5_k(data, n):
    return _dequant_k(data, n, True)


# --------------------------------------------------------------------------- Q6_K

def quant_q6_k(x):
    out = bytearray()
    for b in range(0, len(x), 256):
        blk = x[b:b + 256]
        sub = [blk[16 * j:16 * j + 16] for j in range(16)]
        sc = [max(abs(v) for v in s) / 32.0 for s in sub]
        d = f16(max(sc) / 127.0) if max(sc) > 0 else 0.0
        isc = [0 if d == 0 else max(-127, min(127, int(round(v / d)))) for v in sc]
        q = []
        for j in range(16):
            ds = d * isc[j]
            for v in sub[j]:
                q.append(32 if ds == 0 else max(0, min(63, int(round(v / ds)) + 32)))
        ql = bytearray(128)
        qh = bytearray(64)
        for h in range(2):
            for l in range(32):
                q1, q2, q3, q4 = (q[128 * h + l + 32 * k] for k in range(4))
                ql[64 * h + l] = (q1 & 0xF) | ((q3 & 0xF) << 4)
                ql[64 * h + l + 32] = (q2 & 0xF) | ((q4 & 0xF) << 4)
                qh[32 * h + l] = (q1 >> 4) | ((q2 >> 4) << 2) | ((q3 >> 4) << 4) | ((q4 >> 4) << 6)
        out += bytes(ql) + bytes(qh) + struct.pack("<16b", *isc) + f16_bytes(d)
    return bytes(out)


def dequant_q6_k(data, n):
    out = []
    for b in range(n // 256):
        blk = data[b * 210:(b + 1) * 210]
        ql, qh = blk[:128], blk[128:192]
        sc = struct.unpack_from("<16b", blk, 192)
        d = struct.unpack_from("<e", blk, 208)[0]
        y = [0.0] * 256
        for h in range(2):
            for l in range(32):
                is_ = l // 16
                qs = ((ql[64 * h + l] & 0xF) | (((qh[32 * h + l] >> 0) & 3) << 4),
                      (ql[64 * h + l + 32] & 0xF) | (((qh[32 * h + l] >> 2) & 3) << 4),
                      (ql[64 * h + l] >> 4) | (((qh[32 * h + l] >> 4) & 3) << 4),
                      (ql[64 * h + l + 32] >> 4) | (((qh[32 * h + l] >> 6) & 3) << 4))
                for k in range(4):
                    y[128 * h + l + 32 * k] = (d * sc[8 * h + is_ + 2 * k]) * (qs[k] - 32)
        out += y
    return out


# --------------------------------------------------------------------------- dispatch

def quantize(ty, x):
    if ty == F32:
        return struct.pack("<%df" % len(x), *x)
    if ty == F16:
        return b"".join(f16_bytes(v) for v in x)
    return {Q8_0: quant_q8_0, Q4_K: quant_q4_k, Q5_K: quant_q5_k, Q6_K: quant_q6_k}[ty](x)


def dequantize(ty, data, n):
    if ty == F32:
        return list(struct.unpack("<%df" % n, data[:4 * n]))
    if ty == F16:
        return list(struct.unpack("<%de" % n, data[:2 * n]))
    return {Q8_0: dequant_q8_0, Q4_K: dequant_q4_k, Q5_K: dequant_q5_k, Q6_K: dequant_q6_k}[ty](data, n)


def main(argv):
    if len(argv) != 2:
        print(__doc__)
        return 2
    meta, tensors, _ = read_gguf(argv[1])
    for k, v in meta.items():
        print("%s = %s" % (k, ("[%d items] %r..." % (len(v), v[:3])) if isinstance(v, list) else repr(v)[:200]))
    for name, (dims, ty, off) in tensors.items():
        print("%-32s %-18s %-5s @%d" % (name, dims, TYPE_NAME.get(ty, ty), off))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
