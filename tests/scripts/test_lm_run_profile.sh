#!/usr/bin/env bash
# test_lm_run_profile.sh - lm-run --profile on the tiny Qwen3.5 fixture: the ids on stdout are
# the same with and without it, it prints a prompt and a generation line on stderr, and
# without it stderr stays empty. Run from the repository root after `make`.
set -u
RUN=bin/lm-run FIX=tests/fixtures/qwen35-tiny
[ -x "$RUN" ] || { echo "  FAIL $RUN not found (run make from the repository root)"; exit 1; }
[ -f "$FIX/config.json" ] || { echo "  FAIL fixture $FIX not found (run from the repository root)"; exit 1; }
tmp=$(mktemp -d) || exit 1
trap 'find "$tmp" -delete' EXIT
checks=0; fails=0
check() { checks=$((checks + 1)); if ! eval "$1"; then fails=$((fails + 1)); echo "  FAIL $2"; fi; }

args=(--model "$FIX" --weights "$FIX/model.q8.safetensors" --tokens 3,1,4,1,5,9,2,6,5,3 --greedy 4)
"$RUN" "${args[@]}" > "$tmp/plain.out" 2> "$tmp/plain.err"; rc_plain=$?
"$RUN" "${args[@]}" --profile > "$tmp/prof.out" 2> "$tmp/prof.err"; rc_prof=$?

check '[ $rc_plain -eq 0 ] && [ $rc_prof -eq 0 ]' "lm-run exited $rc_plain without and $rc_prof with --profile"
check 'cmp -s "$tmp/plain.out" "$tmp/prof.out"' "--profile changed the ids on stdout"
check '[ "$(wc -l < "$tmp/prof.out")" -eq 4 ]' "stdout is not exactly the 4 ids"
check 'grep -q "^profile prompt: 10 tokens in .* matmul .* linear attention .* attention .* rest " "$tmp/prof.err"' \
      "no prompt line on stderr: $(cat "$tmp/prof.err")"
check 'grep -q "^profile generation: 3 tokens in " "$tmp/prof.err"' "no generation line for the 3 fed ids"
# NEGATIVE: without the flag nothing is printed on stderr
check '[ ! -s "$tmp/plain.err" ]' "stderr without --profile is not empty: $(cat "$tmp/plain.err")"
# NEGATIVE: one generated id feeds nothing back, so there is no generation line
"$RUN" --model "$FIX" --tokens 3,1,4 --greedy 1 --profile > /dev/null 2> "$tmp/one.err"
check '! grep -q "^profile generation" "$tmp/one.err"' "a generation line for a run that fed no generated token"

if [ "$fails" -ne 0 ]; then echo "test_lm_run_profile: $fails of $checks checks FAILED"; exit 1; fi
echo "test_lm_run_profile: $checks checks passed"
