#!/usr/bin/env bash
# test_lm_run_gemma_image.sh - Gemma image tokens through lm-run.
# --image --logits-out writes one vocab row after the block (optional text prefix).
# --image --vision-rows-out stays the encoder and still refuses --logits-out.
# Run from the repository root after `make`.
set -u
RUN=bin/lm-run FIX=tests/fixtures/gemma4-vl-tiny
[ -x "$RUN" ] || { echo "  FAIL $RUN not found (run make from the repository root)"; exit 1; }
[ -f "$FIX/config.json" ] || { echo "  FAIL fixture $FIX not found (run from the repository root)"; exit 1; }
tmp=$(mktemp -d) || exit 1
trap 'find "$tmp" -delete' EXIT
checks=0; fails=0
check() { checks=$((checks + 1)); if ! eval "$1"; then fails=$((fails + 1)); echo "  FAIL $2"; fi; }

"$RUN" --model "$FIX" --tokens 3,15,17,2 --image "$FIX/img_96x144.ppm" --logits-out "$tmp/logits" \
    > "$tmp/out" 2> "$tmp/err"; rc=$?
check '[ $rc -eq 0 ]' "image logits exited $rc: $(cat "$tmp/err")"
check '[ ! -s "$tmp/out" ]' "image logits wrote to stdout"
check '[ "$(wc -c < "$tmp/logits")" -eq $((384 * 4)) ]' "logits file is not one row of 384 f32"

# one image token, no prefix
"$RUN" --model "$FIX" --image "$FIX/img_48x48.ppm" --logits-out "$tmp/one" > "$tmp/out" 2> "$tmp/err"; rc=$?
check '[ $rc -eq 0 ]' "single-token image exited $rc: $(cat "$tmp/err")"
check '[ "$(wc -c < "$tmp/one")" -eq $((384 * 4)) ]' "single-token logits file is not one row of 384 f32"

# encoder-only still refuses the logits file
"$RUN" --model "$FIX" --image "$FIX/img_48x48.ppm" --vision-rows-out "$tmp/rows" --logits-out "$tmp/no" \
    > /dev/null 2>&1; rc=$?
check '[ $rc -eq 2 ]' "--image --vision-rows-out --logits-out exited $rc, not the usage refusal"

# a text model has no vision tensors
"$RUN" --model tests/fixtures/gemma4-tiny --image "$FIX/img_48x48.ppm" --logits-out "$tmp/no" \
    > /dev/null 2> "$tmp/err"; rc=$?
check '[ $rc -eq 1 ]' "text fixture image logits exited $rc, expected 1"
check 'grep -q "vision_tower" "$tmp/err"' "text fixture did not refuse its vision stub"

if [ "$fails" -ne 0 ]; then echo "test_lm_run_gemma_image: $fails of $checks checks FAILED"; exit 1; fi
echo "test_lm_run_gemma_image: $checks checks passed"
