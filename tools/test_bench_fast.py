#!/usr/bin/env python3
"""Tests for tools/bench_fast.py: the comparison and parsing logic on synthetic logits, the
failure paths, and one end-to-end run against the tiny Qwen3.5 fixture with the real lm-run
(build it first: `make`). Run from anywhere:

    python3 -m unittest discover -s tools -p 'test_bench_fast.py' -v
"""

import array
import contextlib
import io
import os
import sys
import unittest
from unittest import mock

TOOLS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TOOLS)
sys.path.insert(0, TOOLS)

import bench_fast  # noqa: E402

V = 12  # vocabulary of the synthetic logits


def rows(*logit_rows):
    a = array.array("f")
    for r in logit_rows:
        assert len(r) == V
        a.extend(r)
    return a


def one_hot(tok, top=10.0):
    r = [0.0] * V
    r[tok] = top
    return r


class CompareTests(unittest.TestCase):
    def test_identical_logits_agree_everywhere(self):
        ex = rows(one_hot(3), one_hot(7))
        m = bench_fast.compare(ex, ex, V, 0, [3, 7])
        self.assertEqual((m["n"], m["top1"], m["top5"]), (2, 2, 2))
        self.assertEqual((m["err_mean"], m["err_worst"]), (0.0, 0.0))

    def test_a_flipped_top_token_is_counted_but_stays_in_top5(self):
        ex = rows(one_hot(3), one_hot(7))
        fast_row = one_hot(3)
        fast_row[5] = 10.5                      # fast prefers token 5 where exact picked 3
        fa = rows(fast_row, one_hot(7))
        m = bench_fast.compare(ex, fa, V, 0, [3, 7])
        self.assertEqual((m["top1"], m["top5"]), (1, 2))
        self.assertAlmostEqual(m["err_worst"], 10.5 / 10.0)  # exact has 0 at token 5, fast 10.5

    def test_the_exact_token_pushed_out_of_the_top5_is_counted(self):
        ex = rows(one_hot(3))
        fast_row = [5.0] * V
        fast_row[3] = 1.0                       # eleven tokens beat it
        m = bench_fast.compare(ex, rows(fast_row), V, 0, [3])
        self.assertEqual((m["top1"], m["top5"]), (0, 0))

    def test_ties_go_to_the_lowest_id_like_the_engine(self):
        ex = rows(one_hot(2))
        tie = [0.0] * V
        tie[2] = tie[6] = 4.0
        self.assertEqual(bench_fast.compare(ex, rows(tie), V, 0, [2])["top1"], 1)
        self.assertEqual(bench_fast.compare(ex, rows(tie), V, 0, [6])["top1"], 0)

    def test_row_offset_skips_the_prompt_rows(self):
        ex = rows(one_hot(1), one_hot(3), one_hot(7))
        m = bench_fast.compare(ex, ex, V, 1, [3, 7])
        self.assertEqual((m["n"], m["top1"]), (2, 2))

    def test_nan_in_the_fast_logits_fails(self):
        bad = one_hot(3)
        bad[4] = float("nan")
        with self.assertRaisesRegex(bench_fast.Failure, "NaN"):
            bench_fast.compare(rows(one_hot(3)), rows(bad), V, 0, [3])

    def test_a_token_outside_the_vocabulary_fails(self):
        with self.assertRaisesRegex(bench_fast.Failure, "outside the vocabulary"):
            bench_fast.compare(rows(one_hot(3)), rows(one_hot(3)), V, 0, [V])


class ParseAndRateTests(unittest.TestCase):
    def test_load_rows_splits_by_row_count(self):
        raw = rows(one_hot(1), one_hot(2)).tobytes()
        a, vocab = bench_fast.load_rows(raw, 2)
        self.assertEqual((len(a), vocab), (2 * V, V))

    def test_a_file_that_is_not_a_whole_number_of_rows_fails(self):
        with self.assertRaisesRegex(bench_fast.Failure, "whole number"):
            bench_fast.load_rows(b"\x00" * 13, 2)
        with self.assertRaisesRegex(bench_fast.Failure, "whole number"):
            bench_fast.load_rows(b"", 2)

    def test_rate_is_tokens_over_the_difference(self):
        self.assertAlmostEqual(bench_fast.rate(48, 40.0, 20.0, "x"), 2.4)

    def test_a_longer_run_that_is_not_slower_fails(self):
        for t_long in (10.0, 9.0):
            with self.assertRaisesRegex(bench_fast.Failure, "not slower"):
                bench_fast.rate(48, t_long, 10.0, "generation")


class VocabTests(unittest.TestCase):
    def test_reads_vocab_size_from_a_text_config_or_the_top_level(self):
        import json
        import tempfile
        for cfg, want in (({"vocab_size": 384}, 384), ({"text_config": {"vocab_size": 262144}}, 262144)):
            with tempfile.TemporaryDirectory() as d:
                with open(os.path.join(d, "config.json"), "w") as f:
                    json.dump(cfg, f)
                self.assertEqual(bench_fast.vocab_size(d), want)

    def test_a_missing_or_malformed_config_fails(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(bench_fast.Failure, "cannot read vocab_size"):
                bench_fast.vocab_size(d)
            with open(os.path.join(d, "config.json"), "w") as f:
                f.write('{"vocab_size": "big"}')
            with self.assertRaisesRegex(bench_fast.Failure, "not a usable integer"):
                bench_fast.vocab_size(d)


class EndToEndTests(unittest.TestCase):
    FIX = os.path.join(ROOT, "tests", "fixtures", "qwen35-tiny")
    RUN = os.path.join(ROOT, "bin", "lm-run")

    def setUp(self):
        for p, what in ((self.RUN, "bin/lm-run (run make)"), (self.FIX, "the tiny fixture")):
            self.assertTrue(os.path.exists(p), "%s not found at %s" % (what, p))

    def args(self, **kw):
        a = ["--run", self.RUN, "--model", self.FIX, "--weights", os.path.join(self.FIX, "model.q8.safetensors"),
             "--gen", "8", "--prompt", "4", "--repeats", "1"]
        for k, v in kw.items():
            a[a.index("--" + k) + 1] = v
        return a

    def test_runs_on_the_tiny_fixture_and_reports_every_figure(self):
        out = io.StringIO()
        # a 8-token run on a 64-wide model is faster than the timer's noise; the timing
        # arithmetic has its own tests above, so give it a fixed rate here
        with mock.patch.object(bench_fast, "rate", lambda n, tl, ts, what: float(n)), contextlib.redirect_stdout(out):
            self.assertEqual(bench_fast.main(self.args()), 0)
        text = out.getvalue()
        for needle in ("generation: exact", "prompt:     exact", "top token at", "top 5 at", "logit error vs exact"):
            self.assertIn(needle, text)
        top1 = int(text.split("top token at ")[1].split("/")[0])
        self.assertTrue(0 <= top1 <= 8)

    def test_a_long_prompt_stays_inside_the_tiny_vocabulary(self):
        # the fixture has 384 ids; 64 prompt tokens used to overrun it
        out = io.StringIO()
        with mock.patch.object(bench_fast, "rate", lambda n, tl, ts, what: float(n)), contextlib.redirect_stdout(out):
            self.assertEqual(bench_fast.main(self.args(prompt="64")), 0)
        self.assertIn("top token at", out.getvalue())

    def test_a_missing_weights_file_fails_closed(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            rc = bench_fast.main(self.args(weights="/nonexistent/model.q8.safetensors"))
        self.assertEqual(rc, 1)
        self.assertIn("bench_fast FAILED: weights file not found", err.getvalue())

    def test_fast_on_a_bf16_checkpoint_is_refused_by_the_engine(self):
        # lm-run --fast without a q8 file is an error; the bench must surface it, not report numbers
        err = io.StringIO()
        bf16 = os.path.join(self.FIX, "model.safetensors-00001-of-00002.safetensors")
        with contextlib.redirect_stderr(err):
            rc = bench_fast.main(self.args(weights=bf16))
        self.assertEqual(rc, 1)
        self.assertIn("bench_fast FAILED", err.getvalue())


if __name__ == "__main__":
    unittest.main()
