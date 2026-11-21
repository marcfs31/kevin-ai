#!/usr/bin/env python3
"""Deterministic generator for tests/fixtures/qwen2-tiny/ (see tests/fixtures/README.md).

    make_tiny_qwen2.py                 write the fixture with the committed SEED
    make_tiny_qwen2.py --check         regenerate in memory, fail unless identical to the files on disk
                                       (JSON floats within a few ulps, see fixture_cmp.py)
    make_tiny_qwen2.py --find-seed [N] search seeds N, N+1, ... until every margin holds

The fixture is a GGUF file in the layout of a Qwen2.5 model (architecture qwen2, QKV bias,
GQA 4 heads / 2 kv heads of 64, 2 layers, hidden 256, SwiGLU 512, untied output) holding every
matrix type the engine reads: Q4_K, Q5_K, Q6_K and Q8_0, plus F32 norms and biases, and a real
byte-level BPE tokenizer (256 byte tokens, 40 merges trained on a tiny corpus, three control
tokens) with the Qwen2 chat template. Everything derives from `random.Random(seed)`, the float64
oracle tools/ref_qwen2.py and tools/gguf.py, so the output is byte-for-byte reproducible.

Margins asserted: top-1 minus top-2 logit > MIN_GAP at every golden position and greedy step,
at least 8 distinct argmax ids among the golden positions, a non-degenerate greedy output, and
all four matrix types actually present in the file.

Files: model.gguf, golden.json (tokens, logits, greedy), tok_cases.txt ("ids|text" per line,
newline written \\n, from the independent Python tokenizer below).
"""

import argparse
import json
import math
import os
import random
import struct
import sys
import tempfile
import unicodedata

import fixture_cmp
import gguf as gg
import ref_qwen2 as ref

SEED = 0  # found with --find-seed
MIN_GAP = 3e-2
N_TOKENS, N_PROMPT, N_GREEDY = 24, 12, 8
HIDDEN, INTER, LAYERS, HEADS, KV_HEADS, VOCAB, CTX = 256, 512, 2, 4, 2, 320, 128
THETA, EPS = 1000000.0, 1e-6
N_MERGES = 40
CORPUS = ("the quick brown fox jumps over the lazy dog and the cat sat on the mat "
          "def main print hello world return the value 1 2 3 if x then the end").split()
TEMPLATE = ("{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}"
            "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + "
            "'<|im_end|>' + '\\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}")
SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]


# --------------------------------------------------------------------------- tokenizer

def bytes_to_unicode():
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


B2U = bytes_to_unicode()


def train_merges(n):
    words = [[B2U[b] for b in w.encode()] for w in CORPUS]
    words += [[B2U[b] for b in (" " + w).encode()] for w in CORPUS]
    merges = []
    for _ in range(n):
        counts = {}
        for w in words:
            for a, b in zip(w, w[1:]):
                counts[(a, b)] = counts.get((a, b), 0) + 1
        if not counts:
            break
        best = max(sorted(counts), key=lambda p: counts[p])
        merges.append(best)
        merged = best[0] + best[1]
        for wi, w in enumerate(words):
            out, i = [], 0
            while i < len(w):
                if i + 1 < len(w) and (w[i], w[i + 1]) == best:
                    out.append(merged)
                    i += 2
                else:
                    out.append(w[i])
                    i += 1
            words[wi] = out
    return merges


def pretokenize(text):
    """The Qwen2 split regex, written out for text without combining marks:
    (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"""
    L = lambda c: unicodedata.category(c).startswith("L")
    N = lambda c: unicodedata.category(c).startswith("N")
    S = lambda c: c.isspace()
    out, i, n = [], 0, len(text)
    while i < n:
        c = text[i]
        low = text[i:i + 3].lower()
        if c == "'" and (low[1:3] in ("re", "ve", "ll") or low[1:2] in ("s", "t", "m", "d")):
            j = i + (3 if low[1:3] in ("re", "ve", "ll") else 2)
        elif L(c) or (c not in "\r\n" and not L(c) and not N(c) and i + 1 < n and L(text[i + 1])):
            j = i + (0 if L(c) else 1)
            while j < n and L(text[j]):
                j += 1
        elif N(c):
            j = i + 1
        elif (not S(c) and not L(c) and not N(c)) or (c == " " and i + 1 < n and not S(text[i + 1]) and not L(text[i + 1]) and not N(text[i + 1])):
            j = i + (1 if c == " " else 0)
            while j < n and not S(text[j]) and not L(text[j]) and not N(text[j]):
                j += 1
            while j < n and text[j] in "\r\n":
                j += 1
        else:
            r = i
            while r < n and S(text[r]):
                r += 1
            nl = [k for k in range(i, r) if text[k] in "\r\n"]
            if nl:
                j = nl[-1] + 1
            else:
                j = r - 1 if r < n else r
                if j <= i:
                    j = i + 1
        out.append(text[i:j])
        i = j
    return out


def bpe(piece, ranks):
    syms = [B2U[b] for b in piece.encode()]
    while len(syms) > 1:
        best = min(((ranks.get((a, b), 1 << 30), k) for k, (a, b) in enumerate(zip(syms, syms[1:]))))
        if best[0] == 1 << 30:
            break
        k = best[1]
        syms[k:k + 2] = [syms[k] + syms[k + 1]]
    return syms


def tokenizer(merges):
    tokens = [B2U[b] for b in range(256)] + [a + b for a, b in merges]
    types = [1] * len(tokens)
    tokens += SPECIALS
    types += [3] * len(SPECIALS)
    while len(tokens) < VOCAB:
        tokens.append("[PAD%d]" % len(tokens))
        types.append(5)
    return tokens, types


def encode(text, tokens, ranks):
    ids = {t: i for i, t in enumerate(tokens)}
    out = []
    for p in pretokenize(unicodedata.normalize("NFC", text)):
        out += [ids[s] for s in bpe(p, ranks)]
    return out


TOK_CASES = ["the quick brown fox", "def main():\n    print(\"hello\")\n", "Hello, World! 123 and 4567",
             "don't they've it's I'LL", "  leading   spaces  \n\nand tabs\there", "x=1;y=22", "naïve café"]


# --------------------------------------------------------------------------- weights

def types_for(layer):
    a = [gg.Q4_K, gg.Q5_K, gg.Q6_K, gg.Q8_0, gg.Q4_K, gg.Q5_K, gg.Q6_K]
    return a[layer:] + a[:layer]


def build_tensors(rng):
    def mat(rows, cols, std):
        return [rng.gauss(0.0, std) for _ in range(rows * cols)]

    def vec(n, std, mean=0.0):
        return [mean + rng.gauss(0.0, std) for _ in range(n)]

    kvd, qd = KV_HEADS * (HIDDEN // HEADS), HIDDEN
    t = []
    q = lambda name, ty, cols, rows, std: t.append((name, (cols, rows), ty, gg.quantize(ty, mat(rows, cols, std))))
    f = lambda name, vals: t.append((name, (len(vals),), gg.F32, gg.quantize(gg.F32, vals)))
    q("token_embd.weight", gg.Q4_K, HIDDEN, VOCAB, 0.6)
    for i in range(LAYERS):
        ty = types_for(i)
        p = "blk.%d." % i
        f(p + "attn_norm.weight", vec(HIDDEN, 0.1, 1.0))
        q(p + "attn_q.weight", ty[0], HIDDEN, qd, 0.09)
        f(p + "attn_q.bias", vec(qd, 0.3))
        q(p + "attn_k.weight", ty[1], HIDDEN, kvd, 0.09)
        f(p + "attn_k.bias", vec(kvd, 0.3))
        q(p + "attn_v.weight", ty[2], HIDDEN, kvd, 0.09)
        f(p + "attn_v.bias", vec(kvd, 0.3))
        q(p + "attn_output.weight", ty[3], qd, HIDDEN, 0.09)
        f(p + "ffn_norm.weight", vec(HIDDEN, 0.1, 1.0))
        q(p + "ffn_gate.weight", ty[4], HIDDEN, INTER, 0.09)
        q(p + "ffn_up.weight", ty[5], HIDDEN, INTER, 0.09)
        q(p + "ffn_down.weight", ty[6], INTER, HIDDEN, 0.06)
    f("output_norm.weight", vec(HIDDEN, 0.1, 1.0))
    q("output.weight", gg.Q6_K, HIDDEN, VOCAB, 0.35)
    return t


# --------------------------------------------------------------------------- build

class MarginError(Exception):
    pass


def jdump(obj, **kw):
    return (json.dumps(obj, **kw) + "\n").encode("ascii")


def build(seed):
    rng = random.Random(seed)
    merges = train_merges(N_MERGES)
    tokens, types = tokenizer(merges)
    ranks = {m: i for i, m in enumerate(merges)}
    meta = [
        ("general.architecture", gg.STR, "qwen2"), ("general.type", gg.STR, "model"),
        ("general.name", gg.STR, "qwen2-tiny"), ("general.license", gg.STR, "apache-2.0"),
        ("qwen2.block_count", gg.U32, LAYERS), ("qwen2.context_length", gg.U32, CTX),
        ("qwen2.embedding_length", gg.U32, HIDDEN), ("qwen2.feed_forward_length", gg.U32, INTER),
        ("qwen2.attention.head_count", gg.U32, HEADS), ("qwen2.attention.head_count_kv", gg.U32, KV_HEADS),
        ("qwen2.rope.freq_base", gg.FLT32, THETA), ("qwen2.attention.layer_norm_rms_epsilon", gg.FLT32, EPS),
        ("tokenizer.ggml.model", gg.STR, "gpt2"), ("tokenizer.ggml.pre", gg.STR, "qwen2"),
        ("tokenizer.ggml.tokens", gg.ARR, (gg.STR, tokens)),
        ("tokenizer.ggml.token_type", gg.ARR, (gg.I32, types)),
        ("tokenizer.ggml.merges", gg.ARR, (gg.STR, [a + " " + b for a, b in merges])),
        ("tokenizer.ggml.eos_token_id", gg.U32, tokens.index("<|im_end|>")),
        ("tokenizer.ggml.add_bos_token", gg.BOOL, False),
        ("tokenizer.chat_template", gg.STR, TEMPLATE),
    ]
    tensors = build_tensors(rng)
    present = {ty for _, _, ty, _ in tensors}
    if present != {gg.F32, gg.Q4_K, gg.Q5_K, gg.Q6_K, gg.Q8_0}:
        raise MarginError("not every matrix type is present")
    tokens_in = [rng.randrange(256 + N_MERGES) for _ in range(N_TOKENS)]
    files = {}
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "model.gguf")
        gg.write_gguf(path, meta, tensors)
        with open(path, "rb") as fh:
            files["model.gguf"] = fh.read()
        model = ref.Model.load(path)
        prompt_rows, ids, steps = model.generate(tokens_in[:N_PROMPT], N_GREEDY)
        full = model.forward(tokens_in)
    gaps = [ref.top_gap(r) for r in full] + [ref.top_gap(r) for r in steps]
    if min(gaps) <= MIN_GAP:
        raise MarginError("top-1/top-2 gap %.3g <= %g" % (min(gaps), MIN_GAP))
    if len({ref.argmax(r) for r in full}) < 8:
        raise MarginError("fewer than 8 distinct argmax ids")
    if len(set(ids)) < 4:
        raise MarginError("greedy output is degenerate")
    rounded = [[float("%.9g" % v) for v in r] for r in full]
    files["golden.json"] = jdump({"tokens": tokens_in, "logits": rounded,
                                  "greedy": {"prompt": tokens_in[:N_PROMPT], "out": ids}}, separators=(",", ":"))
    lines = []
    for text in TOK_CASES:
        esc = text.replace("\n", "\\n").replace("\t", "\\t")
        lines.append(" ".join(map(str, encode(text, tokens, ranks))) + "|" + esc)
    files["tok_cases.txt"] = ("\n".join(lines) + "\n").encode("utf-8")
    info = {"seed": seed, "min_gap": min(gaps), "max_abs_logit": max(abs(v) for r in full for v in r),
            "greedy": ids, "gguf_bytes": len(files["model.gguf"])}
    return files, info


def default_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", "fixtures", "qwen2-tiny")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=default_dir())
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--find-seed", nargs="?", const=0, type=int, metavar="START")
    a = ap.parse_args(argv)
    if a.find_seed is not None:
        s = a.find_seed
        while True:
            try:
                _, info = build(s)
            except MarginError as e:
                print("seed %d: %s" % (s, e), file=sys.stderr)
                s += 1
                continue
            print(json.dumps(info))
            return 0
    try:
        files, info = build(SEED)
    except MarginError as e:
        print("make_tiny_qwen2: seed %d fails its margins: %s" % (SEED, e), file=sys.stderr)
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
            print("make_tiny_qwen2 --check FAILED: " + "; ".join(bad), file=sys.stderr)
            return 1
        print("make_tiny_qwen2 --check OK: %d files identical (seed %d, min gap %.4g)" % (len(files), SEED, info["min_gap"]))
        return 0
    os.makedirs(a.out, exist_ok=True)
    for name, data in files.items():
        with open(os.path.join(a.out, name), "wb") as f:
            f.write(data)
    print(json.dumps(info))
    return 0


if __name__ == "__main__":
    sys.exit(main())
