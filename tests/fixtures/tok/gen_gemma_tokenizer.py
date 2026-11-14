#!/usr/bin/env python3
"""Generate the synthetic Gemma-style tokenizer fixtures (deterministic, stdlib only).

The schema is the real Gemma 4 tokenizer.json: normalizer Replace " " -> U+2581, pre_tokenizer
Split " " (MergedWithPrevious), model BPE with ignore_merges false, byte_fallback true,
fuse_unk true, unk "<unk>", merges as two-element arrays, 24 added tokens (all special, not
normalized) at the real ids, decoder Sequence[Replace, ByteFallback, Fuse], and a
post_processor that adds nothing. Ids 0-4 and 46-52, 98, 100, 101, 105, 106 and the high
multimodal ids keep the real values; the vocabulary in between is small:

  - 256 byte tokens <0x00>..<0xFF>
  - single characters: U+2581, newline, a-z, A-F, 0-9, a little punctuation, "e acute" and
    "n tilde". Deliberately ABSENT (so byte fallback runs): tab, "Z", "{", "}", the sharp s,
    CJK, emoji, and every other non-ASCII character.
  - tokens learned by a deterministic BPE trainer over the embedded corpus below (ties break
    on the pair's string), so merge order matters: e.g. "hello" can merge as (h,e)(l,l)... or
    (e,l)... depending on rank, and "aaa" shows the leftmost-first rule.

    python3 -I tests/fixtures/tok/gen_gemma_tokenizer.py            > tests/fixtures/tok/gemma-tokenizer.json
    python3 -I tests/fixtures/tok/gen_gemma_tokenizer.py --nobyte   > tests/fixtures/tok/gemma-tokenizer-nobyte.json

--nobyte emits a variant with byte_fallback false and no byte tokens, to exercise
unk / fuse_unk.
"""
import json
import sys

SP = "▁"
N_MERGES = 260

CORPUS = """
the quick brown fox jumps over the lazy dog
hello world hello there hello hello
the cat sat on the mat and the dog sat on the log
a man a plan a canal panama
tokenizers split text into tokens and tokens into ids
the model reads the tokens and the tokens read the model
  indented line with two leading spaces
    four spaces here and   three spaces there
aaa aaaa aaaaa bbb bbbb ccc cccc
line one
line two
line three
cafe café niño jalapeño
numbers 0 1 2 3 4 5 6 7 8 9 10 2024 12345
def fn(a, b): return a + b
what? yes! no. maybe: sure, ok - fine
the end of the story is the beginning of the end
chat template turn user model think channel
"""

ADDED = [  # (id, content): the real ids and the real order
    (0, "<pad>"), (1, "<eos>"), (2, "<bos>"), (3, "<unk>"), (4, "<mask>"),
    (46, "<|tool>"), (47, "<tool|>"), (48, "<|tool_call>"), (49, "<tool_call|>"),
    (50, "<|tool_response>"), (51, "<tool_response|>"), (52, "<|\"|>"),
    (98, "<|think|>"), (100, "<|channel>"), (101, "<channel|>"),
    (105, "<|turn>"), (106, "<turn|>"),
    (255999, "<|image>"), (256000, "<|audio>"), (258880, "<|image|>"), (258881, "<|audio|>"),
    (258882, "<image|>"), (258883, "<audio|>"), (258884, "<|video|>"),
]

CHARS = [SP, "\n"] + list("abcdefghijklmnopqrstuvwxy") + list("ABCDEF") + list("0123456789") \
    + list(".,!?'-:<>|()+=_/\"") + ["é", "ñ"]


def build(nobyte):
    reserved = {i for i, _ in ADDED}
    vocab = {c: i for i, c in ADDED}
    state = {"next": 5}

    def add(tok):
        if tok in vocab:
            return
        while state["next"] in reserved:
            state["next"] += 1
        vocab[tok] = state["next"]
        state["next"] += 1

    if not nobyte:
        for b in range(256):
            add("<0x%02X>" % b)
    for c in CHARS:
        add(c)

    # BPE training over "words" = space-separated pieces with the leading space as U+2581.
    words = {}
    for line in CORPUS.split("\n"):
        text = line.replace(" ", SP)
        if not text:
            continue
        # one word per line segment starting at each U+2581 (a space starts its word)
        cur = ""
        for ch in text:
            if ch == SP and cur and cur[-1] != SP:
                words[cur] = words.get(cur, 0) + 1
                cur = ""
            cur += ch
        if cur:
            words[cur] = words.get(cur, 0) + 1
        words["\n"] = words.get("\n", 0) + 1
    seqs = {w: [c if c in vocab else None for c in w] for w in words}
    seqs = {w: [c for c in s if c is not None] for w, s in seqs.items()}
    merges = []
    for _ in range(N_MERGES):
        counts = {}
        for w, s in seqs.items():
            for a, b in zip(s, s[1:]):
                counts[(a, b)] = counts.get((a, b), 0) + words[w]
        if not counts:
            break
        best = max(counts.items(), key=lambda kv: (kv[1], [-ord(x) for x in kv[0][0] + "\0" + kv[0][1]]))[0]
        a, b = best
        merges.append([a, b])
        add(a + b)
        for w, s in seqs.items():
            out, i = [], 0
            while i < len(s):
                if i + 1 < len(s) and s[i] == a and s[i + 1] == b:
                    out.append(a + b)
                    i += 2
                else:
                    out.append(s[i])
                    i += 1
            seqs[w] = out
    # Hand-placed merges that pin rank behaviour (appended after the trained ones, so the
    # trained ranks stay stable): runs of a repeated letter and of U+2581.
    for a, b in [("a", "a"), ("aa", "a"), (SP, SP), (SP + SP, SP), (SP + SP, SP + SP)]:
        add(a + b)
        merges.append([a, b])
    # Real files hold every merge result as a vocab entry; so does this one.
    for a, b in merges:
        assert a in vocab and b in vocab and a + b in vocab

    added_tokens = [{"id": i, "content": c, "single_word": False, "lstrip": False, "rstrip": False,
                     "normalized": False, "special": True} for i, c in ADDED]
    return {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": added_tokens,
        "normalizer": {"type": "Replace", "pattern": {"String": " "}, "content": SP},
        "pre_tokenizer": {"type": "Split", "pattern": {"String": " "}, "behavior": "MergedWithPrevious",
                          "invert": False},
        "post_processor": {"type": "TemplateProcessing",
                           "single": [{"Sequence": {"id": "A", "type_id": 0}}],
                           "pair": [{"Sequence": {"id": "A", "type_id": 0}},
                                    {"Sequence": {"id": "B", "type_id": 1}}],
                           "special_tokens": {}},
        "decoder": {"type": "Sequence", "decoders": [
            {"type": "Replace", "pattern": {"String": SP}, "content": " "},
            {"type": "ByteFallback"}, {"type": "Fuse"}]},
        "model": {"type": "BPE", "dropout": None, "unk_token": "<unk>", "continuing_subword_prefix": None,
                  "end_of_word_suffix": None, "fuse_unk": True, "byte_fallback": not nobyte,
                  "ignore_merges": False, "vocab": vocab, "merges": merges},
    }


def main(argv):
    nobyte = "--nobyte" in argv[1:]
    # Added tokens are part of the vocab in the real file; do the same.
    doc = build(nobyte)
    json.dump(doc, sys.stdout, ensure_ascii=False, indent=1)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
