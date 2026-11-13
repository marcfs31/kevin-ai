#!/usr/bin/env python3
"""Generate tests/fixtures/tok/qwen-tokenizer.json, a small synthetic tokenizer.json in
exactly the schema of Qwen3.5's real file:

  - normalizer {"type": "NFC"}
  - pre_tokenizer Sequence[Split(Qwen regex, Isolated), ByteLevel]
  - decoder ByteLevel, post_processor ByteLevel
  - model BPE, ignore_merges false, byte_fallback false, vocab = the 256 byte-level
    characters in GPT-2 order followed by merged tokens, merges as "left right" STRINGS
  - added_tokens with the real Qwen contents, in the real order, the real special flags
    (<|endoftext|>..<|video_pad|> special true; <tool_call>..<think>, </think> false)

Ids differ from the real file (the real added tokens start at 248044); the tests never
hardcode an id, they look tokens up by content. The vocabulary is a hand-picked word list
whose merge chains are built left to right, so BPE on each word is unambiguous.

    python3 -I tests/fixtures/tok/gen_qwen_tokenizer.py > tests/fixtures/tok/qwen-tokenizer.json
"""
import json
import sys

REGEX = ("(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}|"
         " ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+")


def bytes_to_unicode():
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs))), [chr(c) for c in cs]


B2U, ORDER = bytes_to_unicode()


def bl(text):
    return "".join(B2U[b] for b in text.encode("utf-8"))


vocab = {ch: i for i, ch in enumerate(ORDER)}
merges = []


def add_chain(word):
    """Merge the byte-level spelling of `word` left to right, one new token per step."""
    chars = list(bl(word))
    cur = chars[0]
    for ch in chars[1:]:
        nxt = cur + ch
        if nxt not in vocab:
            vocab[nxt] = len(vocab)
            merges.append(cur + " " + ch)
        cur = nxt


WORDS = [
    "Think", "think", " first", "Hello", " world", "hello", " the", "the", "The", "caf", "é", "你", "好",
    "'s", " is", " a", "ing", " test", "user", "system", "assistant", "Hi", " there",
    "What", "How", " are", " you", "OK", "Yes", "No", "Think", "think", "Sure", "answer",
    "Answer", "Reason", "reason", "first", "second", "Bye", "Thanks", "Done", "Ok",
    "\n\n", "20", "q" + chr(0x301),
]
for w in WORDS:
    add_chain(w)

added_contents = [
    ("<|endoftext|>", True), ("<|im_start|>", True), ("<|im_end|>", True),
    ("<|object_ref_start|>", True), ("<|object_ref_end|>", True), ("<|box_start|>", True),
    ("<|box_end|>", True), ("<|quad_start|>", True), ("<|quad_end|>", True),
    ("<|vision_start|>", True), ("<|vision_end|>", True), ("<|vision_pad|>", True),
    ("<|image_pad|>", True), ("<|video_pad|>", True),
    ("<tool_call>", False), ("</tool_call>", False), ("<|fim_prefix|>", False),
    ("<|fim_middle|>", False), ("<|fim_suffix|>", False), ("<|fim_pad|>", False),
    ("<|repo_name|>", False), ("<|file_sep|>", False), ("<tool_response>", False),
    ("</tool_response>", False), ("<think>", False), ("</think>", False),
]
first = len(vocab)
added = []
for i, (content, special) in enumerate(added_contents):
    entry = {"id": first + i, "content": content, "single_word": False, "lstrip": False,
             "rstrip": False, "normalized": False, "special": special}
    added.append(entry)

doc = {
    "version": "1.0",
    "truncation": None,
    "padding": None,
    "added_tokens": added,
    "normalizer": {"type": "NFC"},
    "pre_tokenizer": {
        "type": "Sequence",
        "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": REGEX}, "behavior": "Isolated", "invert": False},
            {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False, "use_regex": False},
        ],
    },
    "post_processor": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False, "use_regex": False},
    "decoder": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False, "use_regex": False},
    "model": {
        "type": "BPE", "dropout": None, "unk_token": None, "continuing_subword_prefix": "",
        "end_of_word_suffix": "", "fuse_unk": False, "byte_fallback": False,
        "ignore_merges": False, "vocab": vocab, "merges": merges,
    },
}
json.dump(doc, sys.stdout, ensure_ascii=False, indent=2)
sys.stdout.write("\n")
