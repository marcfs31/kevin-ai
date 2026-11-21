/* lm_tok.h -- text <-> token ids for Qwen3.5, on top of the vendored third_party/tok.h.
 *
 * Adapted from the unreviewed gpt-oss wrapper src/tokenizer/oss_tok.[ch] of an earlier
 * worktree (renamed oss_ -> lm_), with these changes: Qwen3.5's pre-tokenizer family
 * (tok.h `qwen35`), merges as "a b" strings or [a, b] arrays, ignore_merges false, an NFC
 * normalizer (lm_nfc), a real destructor (tok_free), and the UTF-8 streaming decoder
 * made strict about overlong forms and surrogates.
 *
 * tok.h does the byte-level BPE; this layer adds what the chat code needs:
 *   - errors as messages, not exit(1): tok_load() exits the process on a missing file or a
 *     malformed document, so lm_tok_open() reads and validates the file first and only
 *     calls tok_load() on a document that has the schema tok.h assumes;
 *   - encoding of ORDINARY text only: a user typing the characters "<|im_start|>" gets the
 *     ids of those characters, never the control id (tok.h's default mode would turn it
 *     into one, which lets message content forge chat structure). The chat template
 *     inserts its own special ids explicitly;
 *   - special tokens looked up by their content string in added_tokens, so no id is
 *     hardcoded anywhere (the real ids live in the model's tokenizer.json);
 *   - a streaming-safe UTF-8 path: a character split across tokens is emitted only when
 *     it is complete.
 *
 * Two file families, told apart by the normalizer (no flag for the caller):
 *   Qwen3.5: normalizer NFC (or null), pre_tokenizer = Sequence[Split(Qwen regex), ByteLevel],
 *            model BPE, ByteLevel decoder.
 *   Gemma 4: normalizer Replace " " -> U+2581, pre_tokenizer Split " ", model BPE with
 *            byte_fallback / fuse_unk / unk_token, decoder Sequence[Replace, ByteFallback,
 *            Fuse]. Implemented here (a heap merge, O(n log n) on a long word) rather than
 *            in tok.h; with this family tok.h only loads the vocab, merges and added tokens.
 *            The result matches tools/ref_tok_gemma.py. Invalid UTF-8 in the input (which HF
 *            cannot receive) becomes one <0xXX> byte-fallback token per bad byte.
 *            lm_tok_vocab_id takes the token's literal spelling; lm_tok_piece returns the
 *            token's bytes (a <0xXX> token is its one byte, U+2581 is a space);
 *            lm_tok_decode_alloc follows HF: a run of byte tokens that is not valid UTF-8
 *            becomes one U+FFFD per byte, while the streaming path (lm_tok_piece + LmUtf8)
 *            emits one U+FFFD per invalid sequence. */
#ifndef LM_TOK_H
#define LM_TOK_H

#include <stddef.h>

#include "io/lm_gguf.h"

typedef struct LmTok LmTok;

/* Longest buffer lm_tok_piece() may need. */
#define LM_TOK_PIECE_MAX 1024

/* Returns NULL with a message in err on: missing/unreadable file, malformed JSON, an
 * unsupported normalizer / pre-tokenizer / decoder / model, or a malformed vocab, merge
 * list or added_tokens entry. */
LmTok *lm_tok_open(const char *path, char *err, size_t errlen);
void   lm_tok_close(LmTok *t);

/* The byte-level BPE tables of a GGUF file (tokenizer.ggml.*, model "gpt2", pre "qwen2") as a
 * tokenizer of the Qwen2 family: NFC, then the Qwen2 split (a digit is a piece on its own) and
 * BPE. Control tokens (type 3) are special added tokens, user-defined ones (4) added but not
 * special, unused placeholders (5) absent. Nothing refers to g afterwards. NULL with a message
 * in err for any other tokenizer model or pre-tokenizer name or a malformed table. */
LmTok *lm_tok_open_gguf(const LmGguf *g, char *err, size_t errlen);

/* The tokenizer of a model path as the CLIs name it: DIR/tokenizer.json, or the tables inside
 * a GGUF model (lm_gguf_locate). */
LmTok *lm_tok_open_model(const char *model, char *err, size_t errlen);

/* Size of the id -> string table: every id the file defines is < lm_tok_n_ids(). */
int lm_tok_n_ids(const LmTok *t);

/* Encode ordinary text: NFC (when the file's normalizer says so), then the Qwen split and
 * BPE. Never produces an added/special id. Returns a malloc'd array (free it) and stores
 * the count in *n; NULL only on allocation failure. A zero-length text gives n == 0 and a
 * non-NULL array. Invalid UTF-8 is tokenised byte-wise, never rejected. */
int *lm_tok_encode(const LmTok *t, const char *text, size_t len, int *n);

/* As lm_tok_encode, but added tokens spelled in the text become their ids (leftmost, then
 * longest), as the Hugging Face tokenizer does. For tests, tools and trusted text only: user
 * text must go through lm_tok_encode. */
int *lm_tok_encode_special(const LmTok *t, const char *text, size_t len, int *n);

/* Id of an added token by its content (e.g. "<|im_start|>"), -1 when absent. */
int lm_tok_special_id(const LmTok *t, const char *content);

/* As lm_tok_special_id, but absent is an error: 0 and *id set, or -1 with
 * "required special token <|x|> is missing from tokenizer.json" in err. */
int lm_tok_require(const LmTok *t, const char *content, int *id, char *err, size_t errlen);

/* 1 if id is an added token flagged "special": true. (<think> and <tool_call> are added
 * but NOT special in Qwen's file.) */
int lm_tok_is_special(const LmTok *t, int id);

/* Id of a vocabulary entry given in its byte-level spelling (e.g. "\xC4\xA0the" for " the"),
 * -1 when absent. Meant for tests. */
int lm_tok_vocab_id(const LmTok *t, const char *bytelevel);

/* Raw bytes of one token (added tokens: their literal content). Returns the byte count
 * (0 for an id the file does not define), or -1 if id is out of range or out is too
 * small (cap should be LM_TOK_PIECE_MAX). Not NUL-terminated. */
int lm_tok_piece(const LmTok *t, int id, char *out, size_t cap);

/* Whole sequence to a malloc'd, NUL-terminated byte string (special tokens appear as
 * their literal text); *len gets the byte count. For tests and debugging; for output to a
 * user use lm_tok_piece + LmUtf8. */
char *lm_tok_decode_alloc(const LmTok *t, const int *ids, int n, size_t *len);

/* ---- streaming-safe UTF-8 ------------------------------------------------------
 * Feed bytes in any chunking; complete characters come out, an incomplete tail is held
 * until the next push. Invalid sequences are replaced by U+FFFD (EF BF BD), one per
 * offending byte run start, so the output is always valid UTF-8. */
typedef struct {
    unsigned char pend[4];
    int pend_len;  /* bytes held */
    int need;      /* total length of the held sequence */
} LmUtf8;

/* Output capacity needed for lm_utf8_push of n bytes (plus anything already held). */
#define LM_UTF8_OUT_MAX(n) (3 * ((size_t)(n) + 4))

void   lm_utf8_init(LmUtf8 *u);
size_t lm_utf8_push(LmUtf8 *u, const char *in, size_t n, char *out);
/* End of stream: a held incomplete sequence becomes one U+FFFD. Needs 3 bytes. */
size_t lm_utf8_flush(LmUtf8 *u, char *out);

#endif
