/* lm_nfc.h -- Unicode NFC (canonical decomposition, canonical reordering, canonical
 * composition, UAX #15) for the tokenizer's normalizer. Tables: third_party/
 * tok_unicode_nfc.h, generated from Python's unicodedata (version in NFC_UNICODE_VERSION).
 * Original to this repository. */
#ifndef LM_NFC_H
#define LM_NFC_H

#include <stddef.h>

/* NFC of the UTF-8 text s[0..n). Returns a malloc'd, NUL-terminated buffer and stores its
 * length in *out_n (NULL on allocation failure). Pure ASCII is copied unchanged. Bytes
 * that are not valid UTF-8 are passed through untouched (they never compose and never
 * reorder), so a caller feeding raw bytes loses nothing. */
char *lm_nfc(const char *s, size_t n, size_t *out_n);

#endif
