#!/usr/bin/env bash
# publish-docs-pages.sh: argument checks only (no network push).
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
script=$root/scripts/publish-docs-pages.sh
checks=0; fails=0
check() { checks=$((checks+1)); if eval "$1"; then :; else echo "FAIL: $2"; fails=$((fails+1)); fi; }

out=$(bash "$script" --help 2>&1) || true
check '[[ "$out" == *"usage: scripts/publish-docs-pages.sh"* ]]' "help text missing, got: $out"

set +e
out=$(bash "$script" --nope 2>&1); code=$?
set -e
check '[[ "$code" -eq 2 ]]' "unknown arg exit not 2, got $code"
check '[[ "$out" == *"unknown argument"* ]]' "unknown arg message missing, got: $out"

check '[[ -x "$script" ]]' "script not executable"

if [ "$fails" -ne 0 ]; then echo "test_publish_docs_pages: $fails of $checks checks FAILED"; exit 1; fi
echo "test_publish_docs_pages: $checks checks passed"
