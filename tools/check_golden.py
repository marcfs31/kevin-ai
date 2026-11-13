#!/usr/bin/env python3
"""Gate 2 of docs/PLAN.md: compare the C engine against the oracle fixtures.

    check_golden.py --run BIN/lm-run --quant BIN/lm-quant --fixtures tests/fixtures/qwen35-tiny

For bf16 (weights from the fixture dir) and q8 (`--weights model.q8.safetensors`):
  1. `lm-run --model DIR [--weights F] --tokens <golden tokens> --logits-out TMP`: exit 0, empty
     stdout, TMP is exactly len(tokens) rows of vocab little-endian f32, all finite, and at every
     position every logit is within 2e-4 * max|golden row| of the golden one and the argmax
     (lowest id wins a tie) is the golden argmax;
  2. `lm-run ... --tokens <greedy prompt> --greedy N`: stdout is exactly N ids, one per line,
     equal to the golden `greedy.out`.
Then `lm-quant --model DIR --out TMP`: its file must be byte-identical to model.q8.safetensors.

Fails closed: a missing binary or fixture, non-zero exit, timeout, wrong row count, NaN/Inf,
or any violation stops at the FIRST failure, prints `check_golden FAILED: <what>` on stderr and
exits 1. Exit 0 only when every check passed (one summary line on stdout).
"""

import argparse
import json
import math
import os
import struct
import subprocess
import sys
import tempfile

import safetensors as st

REL_TOL = 2e-4


class Failure(Exception):
    pass


def need_file(path, what):
    if not os.path.isfile(path):
        raise Failure("%s not found: %s" % (what, path))


def need_exe(path, what):
    need_file(path, what)
    if not os.access(path, os.X_OK):
        raise Failure("%s is not executable: %s" % (what, path))


def run(cmd, timeout, label):
    try:
        p = subprocess.run(cmd, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise Failure("%s timed out after %ss" % (label, timeout))
    except OSError as e:
        raise Failure("%s could not be started: %s" % (label, e))
    if p.returncode != 0:
        tail = p.stderr.decode("utf-8", "replace").strip().splitlines()[-3:]
        raise Failure("%s exited with status %d: %s" % (label, p.returncode, " | ".join(tail)))
    return p.stdout.decode("utf-8", "replace")


def argmax(row):
    best = 0
    for i in range(1, len(row)):
        if row[i] > row[best]:
            best = i
    return best


def load_golden(path, vocab):
    need_file(path, "golden file")
    try:
        with open(path) as f:
            g = json.load(f)
        tokens, logits = g["tokens"], g["logits"]
        prompt, out = g["greedy"]["prompt"], g["greedy"]["out"]
    except (ValueError, KeyError, TypeError) as e:
        raise Failure("%s is malformed: %s" % (path, e))
    if not tokens or len(logits) != len(tokens) or any(len(r) != vocab for r in logits):
        raise Failure("%s: logits are not [%d tokens][%d vocab]" % (path, len(tokens), vocab))
    if not prompt or not out:
        raise Failure("%s: empty greedy prompt or out" % path)
    return tokens, logits, prompt, out


def check_variant(label, run_bin, model_dir, weights, golden_path, vocab, timeout, tmp):
    tokens, logits, prompt, out = load_golden(golden_path, vocab)
    base = [run_bin, "--model", model_dir]
    if weights:
        base += ["--weights", weights]
    csv = lambda ids: ",".join(str(t) for t in ids)

    # logits
    lpath = os.path.join(tmp, label + ".logits.f32")
    stdout = run(base + ["--tokens", csv(tokens), "--logits-out", lpath], timeout, label + " logits run")
    if stdout.strip():
        raise Failure("%s logits run wrote to stdout (contract: nothing but greedy ids): %r"
                      % (label, stdout[:80]))
    if not os.path.isfile(lpath):
        raise Failure("%s logits run did not write %s" % (label, lpath))
    with open(lpath, "rb") as f:
        raw = f.read()
    want = len(tokens) * vocab * 4
    if len(raw) != want:
        raise Failure("%s logits file has %d bytes, expected %d (%d rows of %d f32)"
                      % (label, len(raw), want, len(tokens), vocab))
    flat = struct.unpack("<%df" % (len(tokens) * vocab), raw)
    max_err = 0.0
    for t in range(len(tokens)):
        row = flat[t * vocab:(t + 1) * vocab]
        if not all(math.isfinite(v) for v in row):
            raise Failure("%s row %d contains NaN or Inf" % (label, t))
        ref = logits[t]
        tol = REL_TOL * max(abs(v) for v in ref)
        err = max(abs(a - b) for a, b in zip(row, ref))
        if err > tol:
            worst = max(range(vocab), key=lambda i: abs(row[i] - ref[i]))
            raise Failure("%s row %d (token %d): logit %d is %.6g, golden %.6g, |diff| %.3g > tol %.3g"
                          % (label, t, tokens[t], worst, row[worst], ref[worst], err, tol))
        if argmax(row) != argmax(ref):
            raise Failure("%s row %d: argmax %d, golden %d" % (label, t, argmax(row), argmax(ref)))
        max_err = max(max_err, err / tol if tol else 0.0)

    # greedy
    stdout = run(base + ["--tokens", csv(prompt), "--greedy", str(len(out))], timeout, label + " greedy run")
    lines = stdout.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    try:
        ids = [int(x) for x in lines]
    except ValueError:
        raise Failure("%s greedy stdout has a non-integer line: %r" % (label, stdout[:80]))
    if len(ids) != len(out):
        raise Failure("%s greedy printed %d ids, expected %d" % (label, len(ids), len(out)))
    for i, (a, b) in enumerate(zip(ids, out)):
        if a != b:
            raise Failure("%s greedy id %d is %d, golden %d" % (label, i, a, b))
    return "%s %d rows (worst error %.0f%% of tolerance) + %d greedy ids" % (
        label, len(tokens), 100 * max_err, len(ids))


def check_quantiser(quant_bin, model_dir, golden_q8, timeout, tmp):
    need_file(golden_q8, "q8 fixture")
    out = os.path.join(tmp, "quant.q8.safetensors")
    run([quant_bin, "--model", model_dir, "--out", out], timeout, "lm-quant")
    if not os.path.isfile(out):
        raise Failure("lm-quant did not write %s" % out)
    with open(out, "rb") as f1, open(golden_q8, "rb") as f2:
        a, b = f1.read(), f2.read()
    if a == b:
        return len(a)
    n = min(len(a), len(b))
    off = next((i for i in range(n) if a[i] != b[i]), n)
    where = "in the header" if off < 8 + struct.unpack("<Q", b[:8])[0] else "in tensor data"
    try:
        t = st.parse(b, "fixture")
        for name, (_, _, lo, hi) in sorted(t.info.items(), key=lambda kv: kv[1][2]):
            if t.data_start + lo <= off < t.data_start + hi:
                where = "in tensor %s" % name
    except st.SafetensorsError:
        pass
    raise Failure("lm-quant output differs from model.q8.safetensors at byte %d (%s); sizes %d vs %d"
                  % (off, where, len(a), len(b)))


def check(run_bin, quant_bin, fixtures, timeout):
    need_exe(run_bin, "lm-run")
    need_exe(quant_bin, "lm-quant")
    if not os.path.isdir(fixtures):
        raise Failure("fixtures directory not found: %s" % fixtures)
    cfg_path = os.path.join(fixtures, "config.json")
    need_file(cfg_path, "config.json")
    try:
        with open(cfg_path) as f:
            cfg = json.load(f)
        vocab = cfg.get("text_config", cfg)["vocab_size"]
    except (ValueError, KeyError, TypeError) as e:
        raise Failure("config.json is malformed: %s" % e)
    q8 = os.path.join(fixtures, "model.q8.safetensors")
    need_file(q8, "q8 fixture")
    parts = []
    with tempfile.TemporaryDirectory() as tmp:
        parts.append(check_variant("bf16", run_bin, fixtures, None,
                                   os.path.join(fixtures, "golden_bf16.json"), vocab, timeout, tmp))
        parts.append(check_variant("q8", run_bin, fixtures, q8,
                                   os.path.join(fixtures, "golden_q8.json"), vocab, timeout, tmp))
        size = check_quantiser(quant_bin, fixtures, q8, timeout, tmp)
    parts.append("quantiser byte-identical (%d bytes)" % size)
    return "check_golden OK: " + "; ".join(parts)


def main(argv=None):
    ap = argparse.ArgumentParser(description="gate 2: engine vs oracle fixtures")
    ap.add_argument("--run", required=True, help="path to lm-run")
    ap.add_argument("--quant", required=True, help="path to lm-quant")
    ap.add_argument("--fixtures", required=True, help="tests/fixtures/qwen35-tiny")
    ap.add_argument("--timeout", type=float, default=600.0, help="seconds per engine invocation")
    a = ap.parse_args(argv)
    try:
        print(check(a.run, a.quant, a.fixtures, a.timeout))
        return 0
    except Failure as e:
        print("check_golden FAILED: %s" % e, file=sys.stderr)
        return 1
    except Exception as e:  # fail closed on anything unexpected
        print("check_golden FAILED: unexpected %s: %s" % (type(e).__name__, e), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
