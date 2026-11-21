#!/usr/bin/env python3
"""Gate 2 of docs/PLAN.md: compare the C engine against the oracle fixtures.

    check_golden.py --run BIN/lm-run --quant BIN/lm-quant --fixtures tests/fixtures/qwen35-tiny
    check_golden.py --run BIN/lm-run --quant BIN/lm-quant --fixtures tests/fixtures/gemma4-tiny

The family comes from config.json (`model_type`); both fixtures use the same checks, the vocab and
row count come from the fixture, and a gemma4 fixture must also hold a single `model.safetensors`.

For bf16 (weights from the fixture dir) and q8 (`--weights model.q8.safetensors`):
  1. `lm-run --model DIR [--weights F] --tokens <golden tokens> --logits-out TMP`: exit 0, empty
     stdout, TMP is exactly len(tokens) rows of vocab little-endian f32, all finite, and at every
     position every logit is within 2e-4 * max|golden row| of the golden one and the argmax
     (lowest id wins a tie) is the golden argmax;
  2. `lm-run ... --tokens <greedy prompt> --greedy N`: stdout is exactly N ids, one per line,
     equal to the golden `greedy.out`.
When the golden holds "mtp" (Qwen3.5's multi-token-prediction head, one row per token position
but the last), also:
  3. `lm-run ... --tokens <golden tokens> --mtp --mtp-out TMP`: TMP is exactly len(tokens)-1 rows,
     each within the same tolerance of the golden head row with the same argmax;
  4. `lm-run ... --tokens <greedy prompt> --greedy N --mtp`: speculative greedy prints exactly the
     golden ids (the head only proposes; the main model decides every id).
When the fixture holds golden_vision_{bf16,q8}.json (the image encoder), also, for each variant
and each image listed:
  5. `lm-run ... --image IMG --vision-rows-out TMP`: TMP is exactly the golden number of rows of
     the golden width, all finite, every value within 2e-4 * max|golden rows of that image|.
When it holds golden_image_{bf16,q8}.json (logits after an image block), also, for each variant
and each block:
  6. `lm-run ... [--tokens PREFIX] --image IMG --logits-out TMP`: TMP is one vocab row, all finite,
     within 2e-4 * max|golden row| of the golden logits, same argmax.
The text checks 1-4 are required unless the fixture holds vision goldens.
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


def load_mtp(path, n_tokens, vocab):
    with open(path) as f:
        rows = json.load(f).get("mtp")
    if rows is None:
        return None
    if len(rows) != n_tokens - 1 or any(len(r) != vocab for r in rows):
        raise Failure("%s: mtp is not [%d positions][%d vocab]" % (path, n_tokens - 1, vocab))
    return rows


def compare_rows(label, path, golden, tokens, vocab):
    """The f32 rows in `path` against `golden` (2e-4 * max|golden row|, same argmax); returns the
    worst error as a fraction of the tolerance."""
    if not os.path.isfile(path):
        raise Failure("%s run did not write %s" % (label, path))
    with open(path, "rb") as f:
        raw = f.read()
    want = len(golden) * vocab * 4
    if len(raw) != want:
        raise Failure("%s file has %d bytes, expected %d (%d rows of %d f32)"
                      % (label, len(raw), want, len(golden), vocab))
    flat = struct.unpack("<%df" % (len(golden) * vocab), raw)
    max_err = 0.0
    for t in range(len(golden)):
        row = flat[t * vocab:(t + 1) * vocab]
        if not all(math.isfinite(v) for v in row):
            raise Failure("%s row %d contains NaN or Inf" % (label, t))
        ref = golden[t]
        tol = REL_TOL * max(abs(v) for v in ref)
        err = max(abs(a - b) for a, b in zip(row, ref))
        if err > tol:
            worst = max(range(vocab), key=lambda i: abs(row[i] - ref[i]))
            raise Failure("%s row %d (token %d): logit %d is %.6g, golden %.6g, |diff| %.3g > tol %.3g"
                          % (label, t, tokens[t], worst, row[worst], ref[worst], err, tol))
        if argmax(row) != argmax(ref):
            raise Failure("%s row %d: argmax %d, golden %d" % (label, t, argmax(row), argmax(ref)))
        max_err = max(max_err, err / tol if tol else 0.0)
    return max_err


def check_greedy(label, stdout, out):
    lines = stdout.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    try:
        ids = [int(x) for x in lines]
    except ValueError:
        raise Failure("%s stdout has a non-integer line: %r" % (label, stdout[:80]))
    if len(ids) != len(out):
        raise Failure("%s printed %d ids, expected %d" % (label, len(ids), len(out)))
    for i, (a, b) in enumerate(zip(ids, out)):
        if a != b:
            raise Failure("%s id %d is %d, golden %d" % (label, i, a, b))
    return len(ids)


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
    max_err = compare_rows(label + " logits", lpath, logits, tokens, vocab)

    # greedy
    stdout = run(base + ["--tokens", csv(prompt), "--greedy", str(len(out))], timeout, label + " greedy run")
    n_ids = check_greedy(label + " greedy", stdout, out)

    mtp = load_mtp(golden_path, len(tokens), vocab)
    if mtp is None:
        return "%s %d rows (worst error %.0f%% of tolerance) + %d greedy ids" % (
            label, len(tokens), 100 * max_err, n_ids)
    mpath = os.path.join(tmp, label + ".mtp.f32")
    stdout = run(base + ["--tokens", csv(tokens), "--mtp", "--mtp-out", mpath], timeout, label + " mtp run")
    if stdout.strip():
        raise Failure("%s mtp run wrote to stdout: %r" % (label, stdout[:80]))
    mtp_err = compare_rows(label + " mtp", mpath, mtp, tokens, vocab)
    stdout = run(base + ["--tokens", csv(prompt), "--greedy", str(len(out)), "--mtp"], timeout,
                 label + " mtp greedy run")
    check_greedy(label + " mtp greedy", stdout, out)
    return "%s %d rows (worst error %.0f%% of tolerance) + %d greedy ids + %d mtp rows (%.0f%%) and mtp greedy" % (
        label, len(tokens), 100 * max_err, n_ids, len(mtp), 100 * mtp_err)


def check_vision(label, run_bin, model_dir, weights, golden_path, timeout, tmp):
    need_file(golden_path, "golden file")
    with open(golden_path) as f:
        images = json.load(f)["images"]
    if not images:
        raise Failure("%s lists no images" % golden_path)
    worst, nrows = 0.0, 0
    for im in images:
        img = os.path.join(model_dir, im["image"])
        need_file(img, "test image")
        rows = im["rows"]
        width = len(rows[0])
        out = os.path.join(tmp, "%s.%s.f32" % (label, im["image"]))
        cmd = [run_bin, "--model", model_dir] + (["--weights", weights] if weights else [])
        stdout = run(cmd + ["--image", img, "--vision-rows-out", out], timeout, "%s vision run on %s" % (label, im["image"]))
        if stdout:
            raise Failure("%s vision run printed to stdout: %r" % (label, stdout[:80]))
        with open(out, "rb") as f:
            raw = f.read()
        if len(raw) != len(rows) * width * 4:
            raise Failure("%s %s: %d bytes of rows, expected %d rows of %d" % (label, im["image"], len(raw), len(rows), width))
        got = struct.unpack("<%df" % (len(rows) * width), raw)
        if not all(math.isfinite(v) for v in got):
            raise Failure("%s %s: rows contain NaN or Inf" % (label, im["image"]))
        flat = [v for r in rows for v in r]
        tol = REL_TOL * (max(abs(v) for v in flat) or 1.0)
        for i, (a, b) in enumerate(zip(got, flat)):
            if abs(a - b) > tol:
                raise Failure("%s %s: row %d value %d is %.9g, golden %.9g (tolerance %.3g)"
                              % (label, im["image"], i // width, i % width, a, b, tol))
        worst = max(worst, max(abs(a - b) for a, b in zip(got, flat)) / tol)
        nrows += len(rows)
    return "%s vision %d images, %d rows (worst error %.0f%% of tolerance)" % (label, len(images), nrows, 100 * worst)


def check_image_text(label, run_bin, model_dir, weights, golden_path, vocab, timeout, tmp):
    """Logits after one image block, and the M-RoPE (t, h, w) of every fed position, exactly."""
    need_file(golden_path, "image-text golden")
    with open(golden_path) as f:
        doc = json.load(f)
    prefix = doc["prefix"]
    worst, npos = 0.0, 0
    for case in doc["cases"]:
        img = os.path.join(model_dir, case["image"])
        need_file(img, "test image")
        lpath = os.path.join(tmp, "%s.%s.logits" % (label, case["image"]))
        ppath = os.path.join(tmp, "%s.%s.pos" % (label, case["image"]))
        cmd = [run_bin, "--model", model_dir] + (["--weights", weights] if weights else [])
        cmd += ["--tokens", ",".join(str(t) for t in prefix), "--image-block", img,
                "--logits-out", lpath, "--positions-out", ppath]
        stdout = run(cmd, timeout, "%s image block %s" % (label, case["image"]))
        if stdout:
            raise Failure("%s image block printed to stdout: %r" % (label, stdout[:80]))
        with open(lpath, "rb") as f:
            raw = f.read()
        if len(raw) != vocab * 4:
            raise Failure("%s %s: logit file is %d bytes, expected one row of %d" % (label, case["image"], len(raw), vocab))
        got = struct.unpack("<%df" % vocab, raw)
        ref = case["logits"]
        if len(ref) != vocab or not all(math.isfinite(v) for v in got):
            raise Failure("%s %s: logits are not one finite row of %d" % (label, case["image"], vocab))
        tol = REL_TOL * (max(abs(v) for v in ref) or 1.0)
        err = max(abs(a - b) for a, b in zip(got, ref))
        if err > tol:
            raise Failure("%s %s: logit error %.6g > tolerance %.3g" % (label, case["image"], err, tol))
        worst = max(worst, err / tol)
        want = case["positions"]
        with open(ppath, "rb") as f:
            praw = f.read()
        if len(praw) != len(want) * 3 * 4:
            raise Failure("%s %s: %d position bytes, expected %d triples" % (label, case["image"], len(praw), len(want)))
        ints = struct.unpack("<%di" % (len(want) * 3), praw)
        got_pos = [list(ints[i:i + 3]) for i in range(0, len(ints), 3)]
        if got_pos != want:
            raise Failure("%s %s: M-RoPE positions %s != %s" % (label, case["image"], got_pos, want))
        npos += len(want)
    return "%s image-text %d cases, %d positions exact (worst logit error %.0f%% of tolerance)" % (
        label, len(doc["cases"]), npos, 100 * worst)


def check_image_logits(label, run_bin, model_dir, weights, golden_path, vocab, timeout, tmp):
    need_file(golden_path, "golden file")
    with open(golden_path) as f:
        blocks = json.load(f)["blocks"]
    if not blocks:
        raise Failure("%s lists no image blocks" % golden_path)
    worst = 0.0
    for b in blocks:
        img = os.path.join(model_dir, b["image"])
        need_file(img, "test image")
        row = b["logits"]
        prefix = b.get("prefix") or []
        if len(row) != vocab or any(not isinstance(t, int) for t in prefix):
            raise Failure("%s %s: logits are not one row of %d, or the prefix is not ints" % (label, b["image"], vocab))
        out = os.path.join(tmp, "%s.%s.image.f32" % (label, b["image"]))
        cmd = [run_bin, "--model", model_dir] + (["--weights", weights] if weights else [])
        if prefix:
            cmd += ["--tokens", ",".join(str(t) for t in prefix)]
        stdout = run(cmd + ["--image", img, "--logits-out", out], timeout, "%s image-block run on %s" % (label, b["image"]))
        if stdout.strip():
            raise Failure("%s image-block run printed to stdout: %r" % (label, stdout[:80]))
        worst = max(worst, compare_rows("%s image block %s" % (label, b["image"]), out, [row], [prefix[-1] if prefix else 0], vocab))
    return "%s image-block logits %d blocks (worst error %.0f%% of tolerance)" % (label, len(blocks), 100 * worst)


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


def check_gguf(run_bin, fixtures, timeout):
    """A GGUF fixture (Qwen2): the same logits and greedy checks against golden.json, no q8 file
    and no lm-quant (the weights are the file's own block-quantised bytes)."""
    golden = os.path.join(fixtures, "golden.json")
    need_file(golden, "golden file")
    with open(golden) as f:
        vocab = len(json.load(f)["logits"][0])
    with tempfile.TemporaryDirectory() as tmp:
        part = check_variant("gguf", run_bin, fixtures, None, golden, vocab, timeout, tmp)
    return "check_golden OK [qwen2 gguf]: " + part


def check(run_bin, quant_bin, fixtures, timeout):
    need_exe(run_bin, "lm-run")
    need_exe(quant_bin, "lm-quant")
    if not os.path.isdir(fixtures):
        raise Failure("fixtures directory not found: %s" % fixtures)
    if os.path.isfile(os.path.join(fixtures, "model.gguf")):
        return check_gguf(run_bin, fixtures, timeout)
    cfg_path = os.path.join(fixtures, "config.json")
    need_file(cfg_path, "config.json")
    try:
        with open(cfg_path) as f:
            cfg = json.load(f)
        vocab = cfg.get("text_config", cfg)["vocab_size"]
        family = cfg.get("model_type", "")
    except (ValueError, KeyError, TypeError, AttributeError) as e:
        raise Failure("config.json is malformed: %s" % e)
    if family == "gemma4":  # single-file bf16 checkpoint (docs/PLAN.md phase 2); Qwen is sharded
        need_file(os.path.join(fixtures, "model.safetensors"), "model.safetensors")
    q8 = os.path.join(fixtures, "model.q8.safetensors")
    need_file(q8, "q8 fixture")
    parts = []
    vision = os.path.isfile(os.path.join(fixtures, "golden_vision_bf16.json"))
    text = not vision or os.path.isfile(os.path.join(fixtures, "golden_bf16.json"))
    with tempfile.TemporaryDirectory() as tmp:
        if text:
            parts.append(check_variant("bf16", run_bin, fixtures, None,
                                       os.path.join(fixtures, "golden_bf16.json"), vocab, timeout, tmp))
            parts.append(check_variant("q8", run_bin, fixtures, q8,
                                       os.path.join(fixtures, "golden_q8.json"), vocab, timeout, tmp))
        if vision:
            parts.append(check_vision("bf16", run_bin, fixtures, None,
                                      os.path.join(fixtures, "golden_vision_bf16.json"), timeout, tmp))
            parts.append(check_vision("q8", run_bin, fixtures, q8,
                                      os.path.join(fixtures, "golden_vision_q8.json"), timeout, tmp))
        if os.path.isfile(os.path.join(fixtures, "golden_text_bf16.json")):
            parts.append(check_image_text("bf16", run_bin, fixtures, None,
                                          os.path.join(fixtures, "golden_text_bf16.json"), vocab, timeout, tmp))
            parts.append(check_image_text("q8", run_bin, fixtures, q8,
                                          os.path.join(fixtures, "golden_text_q8.json"), vocab, timeout, tmp))
        image_logits = os.path.join(fixtures, "golden_image_bf16.json")
        if os.path.isfile(image_logits):
            parts.append(check_image_logits("bf16", run_bin, fixtures, None, image_logits, vocab, timeout, tmp))
            parts.append(check_image_logits("q8", run_bin, fixtures, q8,
                                            os.path.join(fixtures, "golden_image_q8.json"), vocab, timeout, tmp))
        size = check_quantiser(quant_bin, fixtures, q8, timeout, tmp)
    parts.append("quantiser byte-identical (%d bytes)" % size)
    tag = " [gemma4]" if family == "gemma4" else ""
    return "check_golden OK%s: " % tag + "; ".join(parts)


def report_fast(run_bin, fixtures, timeout):
    """REPORT (never a gate): the opt-in int8 fast path on the q8 fixture against the q8 oracle.

    The fast path is not bit-identical to the exact one and is not held to the 2e-4 gate; this
    prints how far it is, so the trade-off is a number. It fails only if the engine itself does
    (non-zero exit, wrong row count, NaN or Inf)."""
    need_exe(run_bin, "lm-run")
    cfg_path = os.path.join(fixtures, "config.json")
    need_file(cfg_path, "config.json")
    try:
        with open(cfg_path) as f:
            cfg = json.load(f)
        vocab = cfg.get("text_config", cfg)["vocab_size"]
    except (ValueError, KeyError, TypeError, AttributeError) as e:
        raise Failure("config.json is malformed: %s" % e)
    q8 = os.path.join(fixtures, "model.q8.safetensors")
    need_file(q8, "q8 fixture")
    tokens, logits, prompt, out = load_golden(os.path.join(fixtures, "golden_q8.json"), vocab)
    base = [run_bin, "--model", fixtures, "--weights", q8, "--fast"]
    csv = lambda ids: ",".join(str(t) for t in ids)
    with tempfile.TemporaryDirectory() as tmp:
        lpath = os.path.join(tmp, "fast.logits.f32")
        run(base + ["--tokens", csv(tokens), "--logits-out", lpath], timeout, "fast logits run")
        with open(lpath, "rb") as f:
            raw = f.read()
    if len(raw) != len(tokens) * vocab * 4:
        raise Failure("fast logits file has %d bytes, expected %d" % (len(raw), len(tokens) * vocab * 4))
    flat = struct.unpack("<%df" % (len(tokens) * vocab), raw)
    worst, argmax_ok = 0.0, 0
    for t in range(len(tokens)):
        row = flat[t * vocab:(t + 1) * vocab]
        if not all(math.isfinite(v) for v in row):
            raise Failure("fast row %d contains NaN or Inf" % t)
        ref = logits[t]
        scale = max(abs(v) for v in ref) or 1.0
        worst = max(worst, max(abs(a - b) for a, b in zip(row, ref)) / scale)
        argmax_ok += argmax(row) == argmax(ref)
    stdout = run(base + ["--tokens", csv(prompt), "--greedy", str(len(out))], timeout, "fast greedy run")
    try:
        ids = [int(x) for x in stdout.split()]
    except ValueError:
        raise Failure("fast greedy stdout has a non-integer line: %r" % stdout[:80])
    same = sum(1 for a, b in zip(ids, out) if a == b)
    return ("check_golden --fast REPORT (not a gate): %s: worst logit error %.1f%% of max|logit| "
            "(exact path: gate 0.02%%), argmax equal on %d/%d rows, greedy ids equal %d/%d"
            % (os.path.basename(os.path.normpath(fixtures)), 100 * worst, argmax_ok, len(tokens), same, len(out)))


def main(argv=None):
    ap = argparse.ArgumentParser(description="gate 2: engine vs oracle fixtures")
    ap.add_argument("--run", required=True, help="path to lm-run")
    ap.add_argument("--quant", required=True, help="path to lm-quant")
    ap.add_argument("--fixtures", required=True, help="tests/fixtures/qwen35-tiny or tests/fixtures/gemma4-tiny")
    ap.add_argument("--timeout", type=float, default=600.0, help="seconds per engine invocation")
    ap.add_argument("--fast", action="store_true",
                    help="REPORT how far the opt-in int8 fast path is from the q8 oracle instead of running the gate")
    a = ap.parse_args(argv)
    try:
        print(report_fast(a.run, a.fixtures, a.timeout) if a.fast else check(a.run, a.quant, a.fixtures, a.timeout))
        return 0
    except Failure as e:
        print("check_golden FAILED: %s" % e, file=sys.stderr)
        return 1
    except Exception as e:  # fail closed on anything unexpected
        print("check_golden FAILED: unexpected %s: %s" % (type(e).__name__, e), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
