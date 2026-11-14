#!/usr/bin/env python3
"""Stdlib reference for the Gemma 4 tokenizer.json (Hugging Face `tokenizers` pipeline).

Implements exactly this pipeline, nothing more, and refuses any other schema:

  encode:  added-token split -> normalizer Replace " " -> "▁" -> pre_tokenizer Split " "
           (MergedWithPrevious) -> BPE (ignore_merges false, byte_fallback, fuse_unk)
  decode:  Replace "▁" -> " ", ByteFallback, Fuse

The post_processor (TemplateProcessing with no special tokens) adds nothing: there is no BOS.

HF semantics relied on (tokenizers 0.2x, Rust sources `AddedVocabulary`, `models/bpe/model.rs`,
`models/bpe/word.rs`, `decoders/byte_fallback.rs`):

  * Added tokens are matched in the RAW text, before the normalizer (all 24 are
    normalized=false), by an Aho-Corasick automaton with leftmost-longest semantics: scan
    left to right, at each position take the longest added token that starts there. The
    text between matches is encoded on its own; an empty gap yields nothing. With
    `ordinary=True` this step is skipped (the chat code encodes user text that way, so that
    typing "<turn|>" cannot forge a control token).
  * The normalizer replaces every U+0020 by U+2581. Nothing else is touched (no NFC).
  * The Split pre-tokenizer on " " therefore never fires, so each gap is ONE word. It is
    still implemented (MergedWithPrevious: a space ends the piece it terminates).
  * BPE per word. The word is cut into Unicode characters. A character that is a vocab
    entry becomes that token. Otherwise, with byte_fallback, it becomes one `<0xXX>` token
    per UTF-8 byte, provided ALL its byte tokens exist in the vocab (otherwise the whole
    character is "unknown"). An unknown character becomes `unk_token` if one is configured
    (and is dropped when it is null); with fuse_unk a run of consecutive unknown
    characters is ONE unk token. A known token or byte-fallback sequence ends the run.
  * Merges: a pair (left token, right token) has the rank of its position in `merges`
    (lowest rank = highest priority); the merged token is the concatenation, which must be
    a vocab entry (HF refuses to load a file otherwise; so do we). A pair listed twice keeps
    its last rank (the real file has no duplicates, so this is unexercised). The pair with the lowest
    rank is merged first; among equal ranks the LEFTMOST pair wins (HF orders its queue by
    (rank, position)). After a merge the new neighbouring pairs are queued. Byte-fallback
    tokens are ordinary symbols: they merge only if the merge list contains a pair that
    names them (the real file has none).
  * Decode: every token string has U+2581 replaced by a space (this also applies to added
    tokens, which are emitted literally). ByteFallback turns each token that is exactly
    `<0xXX>` into a byte; a maximal run of such tokens is decoded as UTF-8, and if the run
    is not valid UTF-8 EVERY byte of it becomes one U+FFFD (not only the bad ones). Ids with
    no token are skipped. Fuse concatenates.

CLI:
    ref_tok_gemma.py encode TOKENIZER [--ordinary] [--hex] [TEXT | --file F]   -> ids, space separated
    ref_tok_gemma.py decode TOKENIZER ID [ID ...]                              -> text on stdout
With --hex the text is given as a hex string (so bytes that are awkward in a shell survive).
"""
import heapq
import json
import re
import sys

SPACE = "▁"


class Tokenizer:
    def __init__(self, path):
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        self._check_schema(d)
        m = d["model"]
        self.vocab = m["vocab"]
        self.id2tok = {i: t for t, i in self.vocab.items()}
        self.byte_fallback = bool(m.get("byte_fallback"))
        self.fuse_unk = bool(m.get("fuse_unk"))
        self.unk = m.get("unk_token")
        if self.unk is not None and self.unk not in self.vocab:
            raise ValueError("unk_token %r is not in the vocab" % self.unk)
        self.added = {a["content"]: a["id"] for a in d["added_tokens"]}
        for c, i in self.added.items():
            self.id2tok[i] = c
        self.ranks = {}  # (left_id, right_id) -> (rank, merged_id)
        for rank, pair in enumerate(m["merges"]):
            if isinstance(pair, str):
                a, b = pair.split(" ")
            else:
                a, b = pair
            if a not in self.vocab or b not in self.vocab or (a + b) not in self.vocab:
                raise ValueError("merge %d (%r, %r) names a token missing from the vocab" % (rank, a, b))
            # a duplicated pair keeps its LAST rank (HF collects the merge list into a map)
            self.ranks[(self.vocab[a], self.vocab[b])] = (rank, self.vocab[a + b])
        self.byte_ids = [self.vocab.get("<0x%02X>" % b) for b in range(256)]
        if self.added:
            alts = sorted(self.added, key=lambda s: -len(s))
            self.added_re = re.compile("|".join(re.escape(s) for s in alts))
        else:
            self.added_re = None
        self.split_str = d["pre_tokenizer"]["pattern"]["String"]

    @staticmethod
    def _check_schema(d):
        m = d.get("model", {})
        if m.get("type") != "BPE":
            raise ValueError("model.type %r is not BPE" % m.get("type"))
        nz = d.get("normalizer")
        if not (isinstance(nz, dict) and nz.get("type") == "Replace"
                and nz.get("pattern") == {"String": " "} and nz.get("content") == SPACE):
            raise ValueError("normalizer is not Replace ' ' -> U+2581")
        pt = d.get("pre_tokenizer")
        if not (isinstance(pt, dict) and pt.get("type") == "Split" and pt.get("behavior") == "MergedWithPrevious"
                and not pt.get("invert") and isinstance(pt.get("pattern"), dict) and "String" in pt["pattern"]):
            raise ValueError("pre_tokenizer is not a literal-string Split MergedWithPrevious")
        if m.get("ignore_merges"):
            raise ValueError("ignore_merges true is not supported")
        if m.get("continuing_subword_prefix") or m.get("end_of_word_suffix") or m.get("dropout"):
            raise ValueError("continuing_subword_prefix / end_of_word_suffix / dropout are not supported")
        dec = d.get("decoder", {})
        kinds = [x.get("type") for x in dec.get("decoders", [])] if dec.get("type") == "Sequence" else None
        if kinds != ["Replace", "ByteFallback", "Fuse"]:
            raise ValueError("decoder is not Sequence[Replace, ByteFallback, Fuse]")

    # ---- encode -----------------------------------------------------------------
    def _split_words(self, s):
        sp = self.split_str
        words, cur = [], 0
        if sp:
            while True:
                j = s.find(sp, cur)
                if j < 0:
                    break
                words.append(s[cur:j + len(sp)])
                cur = j + len(sp)
        if cur < len(s):
            words.append(s[cur:])
        return words

    def _bpe_word(self, word):
        """Symbols are [token_id, token_string]; returns the final id list."""
        syms = []
        pending_unk = False
        for ch in word:
            tid = self.vocab.get(ch)
            if tid is not None:
                pending_unk = False
                syms.append((tid, ch))
                continue
            if self.byte_fallback:
                bs = [self.byte_ids[b] for b in ch.encode("utf-8")]
                if all(b is not None for b in bs):
                    pending_unk = False
                    for b, raw in zip(bs, ch.encode("utf-8")):
                        syms.append((b, "<0x%02X>" % raw))
                    continue
            if self.unk is not None:
                if pending_unk and self.fuse_unk:
                    continue
                syms.append((self.vocab[self.unk], self.unk))
                pending_unk = True
        n = len(syms)
        if n < 2:
            return [s[0] for s in syms]
        ids = [s[0] for s in syms]
        toks = [s[1] for s in syms]
        prev = list(range(-1, n - 1))
        nxt = list(range(1, n + 1))
        nxt[-1] = -1
        alive = [True] * n
        heap = []

        def push(i):
            j = nxt[i]
            if j < 0:
                return
            r = self.ranks.get((ids[i], ids[j]))
            if r is not None:
                heapq.heappush(heap, (r[0], i, j, ids[i], ids[j], r[1]))

        for i in range(n - 1):
            push(i)
        while heap:
            rank, i, j, li, rj, new_id = heapq.heappop(heap)
            if not alive[i] or not alive[j] or nxt[i] != j or ids[i] != li or ids[j] != rj:
                continue  # stale entry
            ids[i] = new_id
            toks[i] = toks[i] + toks[j]
            alive[j] = False
            nxt[i] = nxt[j]
            if nxt[j] >= 0:
                prev[nxt[j]] = i
            if prev[i] >= 0:
                push(prev[i])
            push(i)
        out, i = [], 0
        while i >= 0 and i < n:
            out.append(ids[i])
            i = nxt[i]
        return out

    def _encode_plain(self, s):
        if not s:
            return []
        s = s.replace(" ", SPACE)
        out = []
        for w in self._split_words(s):
            out.extend(self._bpe_word(w))
        return out

    def encode(self, text, ordinary=False):
        if ordinary or self.added_re is None:
            return self._encode_plain(text)
        out, pos = [], 0
        for m in self.added_re.finditer(text):
            out.extend(self._encode_plain(text[pos:m.start()]))
            out.append(self.added[m.group(0)])
            pos = m.end()
        out.extend(self._encode_plain(text[pos:]))
        return out

    # ---- decode -----------------------------------------------------------------
    def decode(self, ids):
        out, run = [], bytearray()

        def flush():
            if run:
                try:
                    out.append(bytes(run).decode("utf-8"))
                except UnicodeDecodeError:
                    out.append("�" * len(run))
                run.clear()

        for i in ids:
            tok = self.id2tok.get(i)
            if tok is None:
                continue
            tok = tok.replace(SPACE, " ")
            if len(tok) == 6 and tok.startswith("<0x") and tok.endswith(">") \
                    and all(c in "0123456789abcdefABCDEF" for c in tok[3:5]):
                run.append(int(tok[3:5], 16))
                continue
            flush()
            out.append(tok)
        flush()
        return "".join(out)


def main(argv):
    if len(argv) < 3 or argv[1] not in ("encode", "decode"):
        sys.stderr.write(__doc__.split("CLI:")[1])
        return 2
    tok = Tokenizer(argv[2])
    if argv[1] == "decode":
        sys.stdout.write(tok.decode([int(x) for x in argv[3:]]))
        return 0
    ordinary = hexmode = False
    text = None
    rest = argv[3:]
    i = 0
    while i < len(rest):
        a = rest[i]
        if a == "--ordinary":
            ordinary = True
        elif a == "--hex":
            hexmode = True
        elif a == "--file":
            i += 1
            with open(rest[i], "rb") as f:
                text = f.read().decode("utf-8")
        else:
            text = a
        i += 1
    if text is None:
        text = sys.stdin.buffer.read().decode("utf-8")
    elif hexmode:
        text = bytes.fromhex(text).decode("utf-8")
    print(" ".join(str(x) for x in tok.encode(text, ordinary)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
