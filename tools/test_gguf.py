#!/usr/bin/env python3
"""Tests for tools/gguf.py, tools/ref_qwen2.py and the qwen2-tiny fixture generator.

    cd tools && python3 -m unittest test_gguf -v
"""

import os
import random
import struct
import sys
import tempfile
import unittest

TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS)

import gguf as gg  # noqa: E402
import make_tiny_qwen2 as mk  # noqa: E402

FIXTURE = os.path.join(TOOLS, "..", "tests", "fixtures", "qwen2-tiny")


class Codec(unittest.TestCase):
    def test_roundtrip_error_is_bounded(self):
        rng = random.Random(3)
        x = [rng.gauss(0, 0.05) for _ in range(512)]
        amax = max(abs(v) for v in x)
        bound = {gg.Q8_0: amax / 127, gg.Q4_K: amax / 5, gg.Q5_K: amax / 10, gg.Q6_K: amax / 20}
        for ty, tol in bound.items():
            y = gg.dequantize(ty, gg.quantize(ty, x), len(x))
            self.assertEqual(len(y), len(x))
            self.assertLessEqual(max(abs(a - b) for a, b in zip(x, y)), tol, gg.TYPE_NAME[ty])

    def test_block_sizes(self):
        x = [0.0] * 256
        for ty in (gg.Q4_K, gg.Q5_K, gg.Q6_K):
            self.assertEqual(len(gg.quantize(ty, x)), gg.BLOCK[ty][1])
        self.assertEqual(len(gg.quantize(gg.Q8_0, [0.0] * 32)), 34)

    def test_known_q8_block(self):
        blk = struct.pack("<e", 0.5) + bytes(range(32))
        self.assertEqual(gg.dequant_q8_0(blk, 32), [0.5 * q for q in range(32)])

    def test_writer_reader_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t.gguf")
            data = gg.quantize(gg.Q8_0, [float(i) for i in range(64)])
            gg.write_gguf(p, [("a.b", gg.U32, 7), ("s", gg.STR, "x"), ("arr", gg.ARR, (gg.STR, ["p", "q"]))],
                          [("w", (32, 2), gg.Q8_0, data)])
            meta, tensors, buf = gg.read_gguf(p)
            self.assertEqual((meta["a.b"], meta["s"], meta["arr"]), (7, "x", ["p", "q"]))
            dims, ty, off = tensors["w"]
            self.assertEqual((dims, ty, off % 32), ((32, 2), gg.Q8_0, 0))
            self.assertEqual(gg.tensor_bytes(buf, dims, ty, off), data)

    def test_writer_refuses_a_partial_block(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(gg.GgufError):
                gg.write_gguf(os.path.join(d, "t.gguf"), [], [("w", (100,), gg.Q4_K, b"")])


class Fixture(unittest.TestCase):
    def test_pretokenizer(self):
        self.assertEqual(mk.pretokenize("Hello, World! 123"), ["Hello", ",", " World", "!", " ", "1", "2", "3"])
        self.assertEqual(mk.pretokenize("don't I'LL"), ["don", "'t", " I", "'LL"])
        self.assertEqual(mk.pretokenize("a  \n\nb"), ["a", "  \n\n", "b"])

    def test_committed_fixture_is_reproducible(self):
        files, info = mk.build(mk.SEED)
        for name, data in files.items():
            with open(os.path.join(FIXTURE, name), "rb") as f:
                disk = f.read()
            if name == "golden.json":
                continue  # floats may differ in the last digit across libm (fixture_cmp tolerance)
            self.assertEqual(disk, data, name)
        self.assertGreater(info["min_gap"], mk.MIN_GAP)


if __name__ == "__main__":
    unittest.main()
