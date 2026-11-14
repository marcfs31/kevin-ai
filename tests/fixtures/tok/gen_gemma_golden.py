#!/usr/bin/env python3
"""Generate tests/fixtures/tok/gemma-golden.txt: expected ids and decodes for the synthetic
Gemma tokenizers, computed by the stdlib reference tools/ref_tok_gemma.py.

One case per line, tab separated:

    tok  mode  text_hex  ids  decoded_hex

  tok   g = gemma-tokenizer.json, n = gemma-tokenizer-nobyte.json
  mode  o  encode as ordinary text (added tokens are plain characters)
        s  encode with added tokens parsed (the Hugging Face behaviour)
        d  decode only: text_hex is empty, ids are the input
  ids   space separated (the encode result, or the decode input); may be empty
  decoded_hex  decode(ids) per the reference, UTF-8 as hex

Run from the repository root:
    python3 -I tests/fixtures/tok/gen_gemma_golden.py > tests/fixtures/tok/gemma-golden.txt
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "tools"))
from ref_tok_gemma import Tokenizer  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

TEXTS = [
    "", " ", "  ", "   ", "hello", "hello world", "Hello, World!", "the cat sat on the mat",
    "the quick brown fox jumps over the lazy dog", "aaa", "aaaa", "aaaaaaa", "baaab",
    "hell", "hello hello hello", "tokens and tokens", "  indented line", "    four spaces",
    "a   b", "a▁b", "▁▁x", "\n", "\n\n\nline\n\n", "line one\nline two\n",
    "a\tb", "\ttab", "Zebra {x}", "ABCDEF abcdef", "café niño", "straße",
    "世界", "日本語 text", "\U0001F600", "x\U0001F600y \U0001F600",
    "മലയാളം", "नमस्ते duniya",
    "12345 2024 0 9", "def fn(a, b): return a + b", "what? yes! no.", "quote \" and slash /",
    "é", "é́x", "\x00\x01\x7f", "tab\there", "\r\n",
    "<bos>hello<eos>", "<bos>", "<eos><eos>", "<|turn>user\nhello<turn|>\n<|turn>model\n",
    "<|turn>system\n<|think|>\nbe brief<turn|>\n<bos>", "<turn|><turn|>", "<|think|>", "<unk>", "x<mask>y",
    "<|image|><|image>", "<|image>", "<|image|>", "<|tool_call>call:f{}<tool_call|>",
    "<|turn", "<bos", "bos>", "<<bos>>", "<|channel>thought\nhmm<channel|>answer",
    "hello<bos> world", "<pad>", "<|\"|>quoted<|\"|>", "<|tool_response>x<tool_response|>",
    "hello" * 60, "a" * 200, "ab" * 100 + " " + "ba" * 100, "▁" * 40, " " * 70,
    "line\n" * 40 + "tail",
]



def lcg_texts():
    """Deterministic pseudo-random strings (own LCG, so no dependence on the random module):
    long words over a small alphabet exercise the merge heap, ties and stale entries."""
    state = 12345
    alpha = ["a", "b", "c", "e", "h", "l", "n", "o", "t", "▁", " ", "\n", "世", "é", "Z"]
    out = []
    for length in [5, 9, 17, 31, 64, 100, 150, 250, 400, 700, 1500, 3000]:
        for _ in range(3):
            s = []
            for _ in range(length):
                state = (state * 1103515245 + 12345) & 0x7FFFFFFF
                s.append(alpha[(state >> 8) % len(alpha)])
            out.append("".join(s))
    return out


TEXTS += lcg_texts()

# Decode-only cases are built from token strings so they survive a regenerated vocabulary.
DECODE = [
    ["<0xC3>", "<0xA9>"],                       # valid run: e acute
    ["<0xE2>", "<0x82>", "<0xAC>"],             # valid run: euro sign
    ["<0xE2>", "<0x82>"],                       # truncated run: two U+FFFD
    ["<0xFF>"],                                 # invalid byte
    ["<0xC3>", "<0xA9>", "<0xFF>"],             # a valid prefix does not save an invalid run
    ["<0xE2>", "<0x82>", "h", "<0xAC>"],        # a normal token ends the run
    ["<0xF0>", "<0x9F>", "<0x98>", "<0x80>"],   # valid 4-byte emoji
    ["<0xF0>", "<0x9F>", "<0x98>"],             # truncated emoji
    ["<0xC0>", "<0x80>"],                       # overlong NUL: invalid
    ["<0xED>", "<0xA0>", "<0x80>"],             # a surrogate: invalid
    ["▁hello", "▁world"],
    ["<bos>", "<|turn>", "user", "\n", "<turn|>"],
    ["<0x41>", "<0x42>"],
    ["<0x20>", "▁"],
]


def build(tok_tag, path):
    t = Tokenizer(path)
    lines = []
    for text in TEXTS:
        for mode in ("o", "s"):
            ids = t.encode(text, ordinary=(mode == "o"))
            dec = t.decode(ids)
            lines.append("\t".join([tok_tag, mode, text.encode("utf-8").hex(), " ".join(map(str, ids)),
                                    dec.encode("utf-8").hex()]))
    for toks in DECODE:
        ids = []
        for s in toks:
            ids.append(t.vocab[s] if s in t.vocab else t.added[s])
        lines.append("\t".join([tok_tag, "d", "", " ".join(map(str, ids)), t.decode(ids).encode("utf-8").hex()]))
    # ids the vocabulary does not define are skipped, and do not end a run of byte tokens
    ids = [t.vocab["<0xE2>"] if "<0xE2>" in t.vocab else 0, 999999]
    if "<0xE2>" in t.vocab:
        ids = [t.vocab["<0xE2>"], 999999, t.vocab["<0x82>"], t.vocab["<0xAC>"]]
        lines.append("\t".join([tok_tag, "d", "", " ".join(map(str, ids)), t.decode(ids).encode("utf-8").hex()]))
    return lines


def main():
    out = build("g", os.path.join(HERE, "gemma-tokenizer.json"))
    # the variant without byte tokens: unknown characters become <unk>, fused when adjacent
    nb = Tokenizer(os.path.join(HERE, "gemma-tokenizer-nobyte.json"))
    for text in ["hello", "世界", "a世界b", "\U0001F600\U0001F600 x", "tab\there", "世 界",
                 "<bos>世<eos>", "{}"]:
        for mode in ("o", "s"):
            ids = nb.encode(text, ordinary=(mode == "o"))
            out.append("\t".join(["n", mode, text.encode("utf-8").hex(), " ".join(map(str, ids)),
                                  nb.decode(ids).encode("utf-8").hex()]))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
