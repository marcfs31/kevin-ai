#!/usr/bin/env python3
"""Measure the opt-in int8 fast path (lm-run --fast) against the exact q8 path on a real model.

    bench_fast.py --run bin/lm-run --model ~/models/qwen3.5-4b \\
                  --weights ~/models/qwen3.5-4b/model.q8.safetensors [--gen 96] [--repeats 3]

Speed: wall time of lm-run for a prompt and for prompt + N generated tokens, minimum over the
repeats; the difference removes the model load and the prompt, so generation tok/s is N over
that difference. Prompt tok/s is the same difference between a 2P-token and a P-token prompt,
measured twice: one token at a time (lm-run --no-prefill) and batched (lm_prefill, the default).

Accuracy: the exact path generates N tokens greedily; both paths are then fed that sequence
(teacher forcing, so one early disagreement cannot cascade) and, at each generated position,
the fast path's logits are compared with the exact ones: does it pick the same top token, is
the exact token in its top 5, and how far are the logits (as % of max |logit|).

Fails closed: a non-zero lm-run exit, a logits file of the wrong size, a NaN or a
non-positive timing difference stops with `bench_fast FAILED: <what>` and exit 1.
Pure Python 3, no dependencies. Reads and writes nothing outside a temporary directory."""

import argparse
import array
import json
import operator
import os
import subprocess
import sys
import tempfile
import time


class Failure(Exception):
    pass


def run(cmd, timeout):
    """Returns (stdout text, wall seconds)."""
    t0 = time.perf_counter()
    try:
        p = subprocess.run(cmd, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise Failure("%s timed out after %ss" % (" ".join(cmd[:3]), timeout))
    except OSError as e:
        raise Failure("could not start %s: %s" % (cmd[0], e))
    dt = time.perf_counter() - t0
    if p.returncode != 0:
        raise Failure("%s exited %d: %s" % (" ".join(cmd[:3]), p.returncode,
                                            p.stderr.decode(errors="replace").strip()[:300]))
    return p.stdout.decode(errors="replace"), dt


def rate(n_tokens, t_long, t_short, what):
    """Tokens per second for the n_tokens that separate the long run from the short one."""
    if t_long <= t_short:
        raise Failure("%s: the longer run (%.3fs) was not slower than the shorter (%.3fs); "
                      "too few tokens to measure, raise --gen" % (what, t_long, t_short))
    return n_tokens / (t_long - t_short)


def load_rows(raw, rows):
    """A little-endian f32 logits file of `rows` rows -> (array('f'), vocab)."""
    if rows <= 0 or len(raw) == 0 or len(raw) % (rows * 4) != 0:
        raise Failure("logits file has %d bytes, not a whole number of %d f32 rows" % (len(raw), rows))
    a = array.array("f")
    a.frombytes(raw)
    if sys.byteorder != "little":
        a.byteswap()
    return a, len(raw) // (rows * 4)


def compare(exact, fast, vocab, row0, ids):
    """Per generated position j (row row0 + j predicts ids[j]): fast vs exact logits.
    Returns a dict of counts and logit errors (fractions of max |exact logit|)."""
    agree = top5 = 0
    errs = []
    for j, tok in enumerate(ids):
        lo, hi = (row0 + j) * vocab, (row0 + j + 1) * vocab
        e, f = exact[lo:hi], fast[lo:hi]
        if any(v != v for v in f):
            raise Failure("the fast path produced a NaN at generated position %d" % j)
        if not 0 <= tok < vocab:
            raise Failure("token id %d is outside the vocabulary of %d" % (tok, vocab))
        best = max(f)
        agree += f.index(best) == tok  # lowest id wins a tie, as the engine's argmax does
        ftok = f[tok]
        top5 += sum(1 for v in f if v > ftok) < 5
        scale = max(abs(min(e)), abs(max(e))) or 1.0
        errs.append(max(map(abs, map(operator.sub, e, f))) / scale)
    n = len(ids)
    return {"n": n, "top1": agree, "top5": top5, "err_mean": sum(errs) / n, "err_worst": max(errs)}


def cpu_context():
    try:
        with open("/proc/cpuinfo") as f:
            text = f.read()
    except OSError:
        return "cpu: unknown (no /proc/cpuinfo)"
    name = next((l.split(":", 1)[1].strip() for l in text.splitlines() if l.startswith("model name")), "?")
    mhz = [float(l.split(":", 1)[1]) for l in text.splitlines() if l.startswith("cpu MHz")]
    return "cpu: %s, now %s MHz" % (name, "%.0f" % (sum(mhz) / len(mhz)) if mhz else "?")


def vocab_size(model_dir):
    path = os.path.join(model_dir, "config.json")
    try:
        with open(path) as f:
            cfg = json.load(f)
        v = cfg.get("text_config", cfg)["vocab_size"]
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as e:
        raise Failure("cannot read vocab_size from %s: %s" % (path, e))
    if not isinstance(v, int) or v < 2:
        raise Failure("%s: vocab_size %r is not a usable integer" % (path, v))
    return v


def bench(a):
    for path, what in ((a.run, "lm-run"), (a.weights, "weights file")):
        if not os.path.isfile(path):
            raise Failure("%s not found: %s" % (what, path))
    base = [a.run, "--model", a.model, "--weights", a.weights]
    if a.threads:
        base += ["--threads", str(a.threads)]
    csv = lambda ids: ",".join(str(t) for t in ids)
    span = min(vocab_size(a.model), 1000)  # ids from the common low end of any vocabulary
    prompt = [(7 * i + 13) % span for i in range(a.prompt)]
    prompt2 = prompt + [(7 * i + 29) % span for i in range(a.prompt)]

    def best(cmd):
        return min(run(cmd, a.timeout)[1] for _ in range(a.repeats))

    # the exact path's own greedy continuation is the text the accuracy is measured on
    out, _ = run(base + ["--tokens", csv(prompt), "--greedy", str(a.gen)], a.timeout)
    ids = [int(x) for x in out.split()]
    if len(ids) != a.gen:
        raise Failure("exact greedy printed %d ids, expected %d" % (len(ids), a.gen))

    speed = {}
    for mode, extra in (("exact", []), ("fast", ["--fast"])):
        t_p = best(base + extra + ["--tokens", csv(prompt)])
        t_pg = best(base + extra + ["--tokens", csv(prompt), "--greedy", str(a.gen)])
        t_2p = best(base + extra + ["--tokens", csv(prompt2)])
        t_pf = best(base + extra + ["--no-prefill", "--tokens", csv(prompt)])
        t_2pf = best(base + extra + ["--no-prefill", "--tokens", csv(prompt2)])
        speed[mode] = (rate(a.gen, t_pg, t_p, mode + " generation"),
                       rate(a.prompt, t_2pf, t_pf, mode + " prompt one token at a time"),
                       rate(a.prompt, t_2p, t_p, mode + " prompt batched"))

    full = prompt + ids
    with tempfile.TemporaryDirectory() as tmp:
        rows = {}
        for mode, extra in (("exact", []), ("fast", ["--fast"])):
            lp = os.path.join(tmp, mode + ".f32")
            run(base + extra + ["--tokens", csv(full), "--logits-out", lp], a.timeout)
            with open(lp, "rb") as f:
                rows[mode] = load_rows(f.read(), len(full))
    (ex, vocab), (fa, vocab2) = rows["exact"], rows["fast"]
    if vocab != vocab2:
        raise Failure("exact and fast logits have different vocabularies: %d and %d" % (vocab, vocab2))
    m = compare(ex, fa, vocab, len(prompt) - 1, ids)

    g_e, f_e, p_e = speed["exact"]
    g_f, f_f, p_f = speed["fast"]
    lines = [
        cpu_context(),
        "model: %s (%d prompt tokens, %d generated, %d repeats, threads %s)"
        % (a.model, a.prompt, a.gen, a.repeats, a.threads or "default"),
        "generation: exact %.2f tok/s, fast %.2f tok/s (%.2fx)" % (g_e, g_f, g_f / g_e),
        "prompt, one token at a time: exact %.2f tok/s, fast %.2f tok/s (%.2fx)" % (f_e, f_f, f_f / f_e),
        "prompt, batched (lm_prefill): exact %.2f tok/s (%.2fx), fast %.2f tok/s (%.2fx)"
        % (p_e, p_e / f_e, p_f, p_f / f_f),
        "accuracy over %d generated positions (teacher-forced on the exact path's text):" % m["n"],
        "  fast picks the exact path's top token at %d/%d positions (%.1f%%)"
        % (m["top1"], m["n"], 100.0 * m["top1"] / m["n"]),
        "  the exact token is in fast's top 5 at %d/%d positions (%.1f%%)"
        % (m["top5"], m["n"], 100.0 * m["top5"] / m["n"]),
        "  logit error vs exact: mean %.2f%%, worst %.2f%% of max |logit|" % (100 * m["err_mean"], 100 * m["err_worst"]),
    ]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description="speed and accuracy of lm-run --fast against the exact q8 path")
    ap.add_argument("--run", required=True, help="path to lm-run")
    ap.add_argument("--model", required=True, help="model directory (config.json, tokenizer)")
    ap.add_argument("--weights", required=True, help="the q8 weights file (--fast needs one)")
    ap.add_argument("--gen", type=int, default=96, help="tokens to generate (default 96)")
    ap.add_argument("--prompt", type=int, default=32, help="prompt tokens (default 32, one prefill chunk)")
    ap.add_argument("--repeats", type=int, default=3, help="timing repeats, the minimum counts (default 3)")
    ap.add_argument("--threads", type=int, default=0, help="lm-run --threads (default: its default)")
    ap.add_argument("--timeout", type=float, default=1800.0, help="seconds per lm-run invocation")
    a = ap.parse_args(argv)
    if a.gen < 1 or a.prompt < 1 or a.repeats < 1 or a.threads < 0:
        ap.error("--gen, --prompt and --repeats must be at least 1, --threads not negative")
    try:
        print(bench(a))
        return 0
    except Failure as e:
        print("bench_fast FAILED: %s" % e, file=sys.stderr)
        return 1
    except Exception as e:  # fail closed on anything unexpected
        print("bench_fast FAILED: unexpected %s: %s" % (type(e).__name__, e), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
