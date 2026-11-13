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
 * Expected file: Hugging Face tokenizers JSON, normalizer NFC (or null), pre_tokenizer =
 * Sequence[Split(Qwen regex), ByteLevel], model.type BPE, ByteLevel decoder. */
#ifndef LM_TOK_H
#define LM_TOK_H

#include <stddef.h>

typedef struct LmTok LmTok;

/* Longest buffer lm_tok_piece() may need. */
#define LM_TOK_PIECE_MAX 1024

/* Returns NULL with a message in err on: missing/unreadable file, malformed JSON, an
 * unsupported normalizer / pre-tokenizer / decoder / model, or a malformed vocab, merge
 * list or added_tokens entry. */
LmTok *lm_tok_open(const char *path, char *err, size_t errlen);
void   lm_tok_close(LmTok *t);

/* Size of the id -> string table: every id the file defines is < lm_tok_n_ids(). */
int lm_tok_n_ids(const LmTok *t);

/* Encode ordinary text: NFC (when the file's normalizer says so), then the Qwen split and
 * BPE. Never produces an added/special id. Returns a malloc'd array (free it) and stores
 * the count in *n; NULL only on allocation failure. A zero-length text gives n == 0 and a
 * non-NULL array. Invalid UTF-8 is tokenised byte-wise, never rejected. */
int *lm_tok_encode(const LmTok *t, const char *text, size_t len, int *n);

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
