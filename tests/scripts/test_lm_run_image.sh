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

if [ "$fails" -ne 0 ]; then echo "test_lm_run_image: $fails of $checks checks FAILED"; exit 1; fi
echo "test_lm_run_image: $checks checks passed"
