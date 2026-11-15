#!/usr/bin/env bash
# test_bench_fast.sh - runs tools/test_bench_fast.py (needs bin/lm-run: `make` builds it).
# Run from the repository root.
set -u
[ -x bin/lm-run ] || { echo "  FAIL bin/lm-run not built (run make)"; exit 1; }
python3 -m unittest discover -s tools -p 'test_bench_fast.py' 2>&1 | tail -4
exit "${PIPESTATUS[0]}"
