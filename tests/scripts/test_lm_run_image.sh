#!/usr/bin/env bash
# test_lm_run_image.sh - lm-run --image on the tiny Qwen3.5 VL fixture: it writes one f32 row per
# image token, and every flag the image path does not honour is refused (exit 2), never
# silently ignored. Run from the repository root after `make`.
set -u
RUN=bin/lm-run FIX=tests/fixtures/qwen35-vl-tiny
[ -x "$RUN" ] || { echo "  FAIL $RUN not found (run make from the repository root)"; exit 1; }
[ -f "$FIX/config.json" ] || { echo "  FAIL fixture $FIX not found (run from the repository root)"; exit 1; }
tmp=$(mktemp -d) || exit 1
trap 'find "$tmp" -delete' EXIT
checks=0; fails=0
check() { checks=$((checks + 1)); if ! eval "$1"; then fails=$((fails + 1)); echo "  FAIL $2"; fi; }

args=(--model "$FIX" --image "$FIX/img_64x96.ppm" --vision-rows-out "$tmp/rows")
"$RUN" "${args[@]}" > "$tmp/out" 2> "$tmp/err"; rc=$?
check '[ $rc -eq 0 ]' "lm-run --image exited $rc: $(cat "$tmp/err")"
check '[ "$(wc -c < "$tmp/rows")" -eq $((6 * 64 * 4)) ]' "rows file is not 6 rows of 64 f32"
for extra in "--threads 1" "--ctx 64" "--fast" "--no-prefill" "--profile" "--mtp" "--mtp-out $tmp/m" \
             "--greedy 1" "--tokens 1,2" "--logits-out $tmp/l"; do
    read -ra more <<< "$extra"
    "$RUN" "${args[@]}" "${more[@]}" > /dev/null 2>&1; rc=$?
    check '[ $rc -eq 2 ]' "--image with $extra exited $rc, not the usage refusal"
done
# NEGATIVE: --image without --vision-rows-out is refused too
"$RUN" --model "$FIX" --image "$FIX/img_64x96.ppm" > /dev/null 2>&1; rc=$?
check '[ $rc -eq 2 ]' "--image without --vision-rows-out exited $rc"

# an image prompt: img placeholders must match the --image files, one logits row per ITEM
img=$FIX/img_32x32.ppm
"$RUN" --model "$FIX" --tokens 5,6,img,7 --image "$img" --logits-out "$tmp/l1" --greedy 2 > "$tmp/ids" 2> "$tmp/err"; rc=$?
check '[ $rc -eq 0 ]' "image prompt exited $rc: $(cat "$tmp/err")"
check '[ "$(wc -c < "$tmp/l1")" -eq $((3 * 384 * 4)) ]' "logits are not 3 items (ids, image, ids) of 384 f32"
check '[ "$(wc -l < "$tmp/ids")" -eq 2 ]' "no 2 greedy ids"
# NEGATIVE: the text-only fixture cannot take an image, and says why
"$RUN" --model tests/fixtures/qwen35-tiny --tokens 5,img --image "$img" > /dev/null 2> "$tmp/err2"; rc=$?
check '[ $rc -eq 1 ] && grep -q "no image encoder" "$tmp/err2"' "text-only model: exit $rc: $(cat "$tmp/err2")"
# NEGATIVE: placeholder and file counts must agree; --no-prefill does not apply
"$RUN" --model "$FIX" --tokens 5,img,img --image "$img" > /dev/null 2>&1; rc=$?
check '[ $rc -eq 2 ]' "2 placeholders with 1 --image exited $rc"
"$RUN" --model "$FIX" --tokens 5,6 --image "$img" > /dev/null 2>&1; rc=$?
check '[ $rc -eq 2 ]' "an --image with no placeholder exited $rc"
"$RUN" --model "$FIX" --tokens 5,img --image "$img" --no-prefill > /dev/null 2>&1; rc=$?
check '[ $rc -eq 2 ]' "--no-prefill with an image prompt exited $rc"

if [ "$fails" -ne 0 ]; then echo "test_lm_run_image: $fails of $checks checks FAILED"; exit 1; fi
echo "test_lm_run_image: $checks checks passed"
