#!/usr/bin/env python3
"""Compare the C Gemma tokenizer with the Python reference on a real tokenizer.json.

    make build/tests/unit/test_tok_gemma
    python3 -I tools/check_tok_gemma.py --tokenizer /path/to/gemma/tokenizer.json \
        [--tool build/tests/unit/test_tok_gemma] [--fuzz 300]

The C side is `test_tok_gemma --tool TOKENIZER` (the unit-test binary in its tool mode: one
request per line on stdin). For every string it encodes in both modes (ordinary text, and
with added tokens parsed as Hugging Face does), compares the ids exactly, compares the C
decode and the streaming-path bytes with the reference decode and with the input, and
reports counts. A deterministic fuzz set of mixed-script strings follows, then the timing
of a 20 000 character single word. Exit status 0 only if every comparison matched.
Stdlib only.
"""
import argparse
import os
import random
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ref_tok_gemma import Tokenizer  # noqa: E402

CASES = [
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog. It's 2024, and e = mc^2.",
    "  leading and trailing  ",
    "   ",
    "multiple   spaces    in     a row",
    "def fib(n):\n    if n < 2:\n        return n\n    return fib(n - 1) + fib(n - 2)\n",
    "int main(void)\n{\n\tprintf(\"hi\\n\");\n\treturn 0;\n}\n",
    "line one\r\nline two\r\n\r\nline four",
    "\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n",
    "                                                                                ",
    "\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t",
    "今日はいい天気ですね。世界を見る。",
    "中文测试：我们在测试分词器。",
    "한국어 테스트 문장입니다.",
    "\U0001F600\U0001F680\U0001F30D \U0001F468‍\U0001F469‍\U0001F467‍\U0001F466 \U0001F1EA\U0001F1F8",
    "മലയാളം ഒരു ഭാഷയാണ്",
    "हिन्दी भाषा में परीक्षण वाक्य",
    "العربية نص للاختبار",
    "עברית בדיקה",
    "ภาษาไทยทดสอบ",
    "café naïve façade über straße señor",
    "é ä ô (decomposed accents)",
    "αβγ Δx ∑ ∫ √2 ≠ ≤ ± × ÷",
    "https://example.com/path?query=1&x=y#frag user@example.org",
    '{"a": [1, 2, {"b": null}], "c": "d\\n"}',
    "3.14159 2,718 1e-9 0x1F 007 1234567890",
    "\x00\x01\x02 control \x7f \x1b[31mred\x1b[0m",
    "a b c　d (odd spaces)",
    "▁literal-block-char",
    "<bos>",
    "<bos>Hello<eos>",
    "<|turn>user\nWhat is 2+2?<turn|>\n<|turn>model\n",
    "<bos><|turn>system\nYou are helpful.<turn|>\n<|turn>user\nHi<turn|>\n<|turn>model\n",
    "<bos><|turn>system\n<|think|>\nThink.<turn|>\n<|turn>user\nHi<turn|>\n<|turn>model\n"
    "<|channel>thought\nhmm<channel|>Hello!<turn|>\n",
    "<|turn>model\n<|tool_call>call:f{x:<|\"|>y<|\"|>}<tool_call|><|tool_response>",
    "<|image|><|image>x<image|><audio|><|audio><|audio|><|video|>",
    "<unk><mask><pad>",
    "<|turn><turn|><|turn><turn|>",
    "almost <|turn and <turn| and <bos",
    "x" * 5000,
    "word " * 3000,
]


def fuzz_cases(n):
    rnd = random.Random(20261114)
    alphabets = [
        "abcdefghijklmnopqrstuvwxyz     \n",
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 \t\n.,;:!?()[]{}<>=+-*/_\"'",
        "一二三四五六七八九十 ",
        "കഖഗാിു ം",
        "\U0001F600\U0001F601\U0001F602 a",
        "éèêë́̈ e",
        "    \n\t  x",
        "<|turn>user<turn|><bos><eos>model\n ",
    ]
    out = []
    for i in range(n):
        alpha = alphabets[i % len(alphabets)]
        length = rnd.choice([1, 2, 3, 5, 8, 13, 40, 120, 400])
        out.append("".join(rnd.choice(alpha) for _ in range(length)))
    return out


class CTool:
    def __init__(self, binary, tokenizer):
        self.p = subprocess.Popen([binary, "--tool", tokenizer], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  text=True, bufsize=1)

    def ask(self, line):
        self.p.stdin.write(line + "\n")
        self.p.stdin.flush()
        reply = self.p.stdout.readline()
        if not reply:
            raise RuntimeError("the C tool exited (see its stderr)")
        return reply.rstrip("\n")

    def encode(self, text, special):
        r = self.ask(("s " if special else "o ") + text.encode("utf-8").hex())
        ids, us = r.split("\t")
        return [int(x) for x in ids.split()], int(us)

    def decode(self, ids):
        return bytes.fromhex(self.ask("d " + " ".join(map(str, ids))))

    def pieces(self, ids):
        return bytes.fromhex(self.ask("p " + " ".join(map(str, ids))))

    def close(self):
        self.p.stdin.close()
        self.p.wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--tool", default="build/tests/unit/test_tok_gemma")
    ap.add_argument("--fuzz", type=int, default=300)
    a = ap.parse_args()

    t0 = time.time()
    ref = Tokenizer(a.tokenizer)
    print("reference loaded in %.1f s" % (time.time() - t0))
    t0 = time.time()
    c = CTool(a.tool, a.tokenizer)
    c.encode("warm up", False)
    print("C tokenizer loaded in %.1f s" % (time.time() - t0))

    stats = {"cases": 0, "enc_match": 0, "dec_match": 0, "stream_match": 0, "roundtrip": 0}
    bad = []

    def check(label, text):
        for special in (False, True):
            want = ref.encode(text, ordinary=not special)
            got, _ = c.encode(text, special)
            stats["cases"] += 1
            if got == want:
                stats["enc_match"] += 1
            else:
                bad.append("%s encode(%s): %r" % ("special" if special else "ordinary", label, text[:60]))
                continue
            want_dec = ref.decode(want)
            if c.decode(want).decode("utf-8") == want_dec:
                stats["dec_match"] += 1
            else:
                bad.append("decode(%s): %r" % (label, text[:60]))
            if c.pieces(want) == want_dec.encode("utf-8"):
                stats["stream_match"] += 1
            else:
                bad.append("pieces(%s): %r" % (label, text[:60]))
            # U+2581 in the input is indistinguishable from a space after normalisation
            if want_dec == text.replace("▁", " "):
                stats["roundtrip"] += 1
            else:
                bad.append("round trip(%s): %r" % (label, text[:60]))

    for i, s in enumerate(CASES):
        check("case %d" % i, s)
    n_named = stats["cases"]
    named = dict(stats)
    for i, s in enumerate(fuzz_cases(a.fuzz)):
        check("fuzz %d" % i, s)

    print("named strings : %d strings x 2 modes = %d encodes" % (len(CASES), n_named))
    print("  exact id match      %d / %d" % (named["enc_match"], n_named))
    print("  decode match        %d / %d" % (named["dec_match"], n_named))
    print("  piece-bytes match   %d / %d" % (named["stream_match"], n_named))
    print("  decode round trip   %d / %d" % (named["roundtrip"], n_named))
    nf = stats["cases"] - n_named
    print("fuzz strings  : %d strings x 2 modes = %d encodes" % (a.fuzz, nf))
    print("  exact id match      %d / %d" % (stats["enc_match"] - named["enc_match"], nf))
    print("  decode match        %d / %d" % (stats["dec_match"] - named["dec_match"], nf))
    print("  piece-bytes match   %d / %d" % (stats["stream_match"] - named["stream_match"], nf))
    print("  decode round trip   %d / %d" % (stats["roundtrip"] - named["roundtrip"], nf))

    # the decode of arbitrary id sequences, including byte-token runs and undefined ids
    rnd = random.Random(7)
    ids_pool = [ref.vocab["<0x%02X>" % b] for b in (0x41, 0xC3, 0xA9, 0xE2, 0x82, 0xAC, 0xFF, 0x80, 0xF0, 0x9F, 0x98)]
    ids_pool += [2, 105, 106, 107, 1000, 50000, 262143, 999999]
    dmatch = 0
    nd = 400
    for _ in range(nd):
        ids = [rnd.choice(ids_pool) for _ in range(rnd.choice([1, 2, 3, 4, 6, 10]))]
        if c.decode(ids) == ref.decode(ids).encode("utf-8"):
            dmatch += 1
        else:
            bad.append("decode of ids %r" % ids)
    print("random id decode (byte runs, specials, undefined ids): %d / %d" % (dmatch, nd))

    # one 20 000 character word: mixed words, spaces and newlines
    words = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "def", "return", "café",
             "世界", "\U0001F600", "tokenizer", "Gemma", "12345", "{", "}", "\n"]
    rnd = random.Random(99)
    parts, n = [], 0
    while n < 20000:
        w = rnd.choice(words)
        parts.append(w)
        n += len(w) + 1
    long_text = " ".join(parts)[:20000]
    t0 = time.time()
    want = ref.encode(long_text, ordinary=True)
    py_s = time.time() - t0
    got, us = c.encode(long_text, False)
    print("20000-character single word: %d ids, ids match: %s, C encode %.2f ms, Python reference %.2f s"
          % (len(got), got == want, us / 1000.0, py_s))
    if got != want:
        bad.append("long word")
    c.close()

    if bad:
        print("MISMATCHES: %d" % len(bad))
        for b in bad[:20]:
            print("  " + b)
        return 1
    print("OK: every comparison matched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
