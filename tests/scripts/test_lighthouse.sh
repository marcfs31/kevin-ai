#!/usr/bin/env bash
# lighthouse.sh: argument handling and what it demands. It is not run here (needs Chrome and the network).
set -u
root=$(cd "$(dirname "$0")/../.." && pwd)
script=$root/scripts/lighthouse.sh
fails=0; checks=0
check() { checks=$((checks + 1)); if ! eval "$1"; then fails=$((fails + 1)); echo "  FAIL $2"; fi; }

out=$(bash "$script" --help 2>&1); rc=$?
check '[ "$rc" -eq 0 ] && [[ "$out" == *"usage: scripts/lighthouse.sh"* ]]' "--help should print usage and exit 0, got $rc: $out"
out=$(bash "$script" --nope 2>&1); rc=$?
check '[ "$rc" -eq 2 ] && [[ "$out" == *"unknown argument"* ]]' "an unknown argument should exit 2, got $rc: $out"
check 'grep -q "lighthouse@[0-9]*\.[0-9]*\.[0-9]*" "$script"' "Lighthouse must be pinned to an exact version"
check 'grep -q "tools/chat_server.py" "$script" && ! grep -q "vite --\|npm.*run dev" "$script"' "it must measure the production server, not the dev server"
check 'grep -q "desktop" "$script" && grep -q "mobile" "$script"' "both form factors"
check 'grep -q "== 100" "$script"' "every category must be 100"

if [ "$fails" -ne 0 ]; then echo "test_lighthouse: $fails of $checks checks FAILED"; exit 1; fi
echo "test_lighthouse: $checks checks passed"
