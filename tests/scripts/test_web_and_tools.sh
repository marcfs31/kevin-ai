#!/usr/bin/env bash
# The web chat and the tool service are part of the gate: `make test` (and so the CI job) runs their
# tests. Before this, only the C tests ran, so the web and Python code could break with every check green.
# Fails when Node is missing instead of skipping: a gate that can silently skip is not a gate.
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
cd "$root"

echo "tool service and proxy (Python unittest)"
python3 -m unittest discover -s tools -p 'test_chat_*.py'

command -v npm >/dev/null 2>&1 || { echo "test_web_and_tools: npm not found; the web tests are part of the gate" >&2; exit 1; }
# A fresh checkout (CI) has no node_modules: install exactly what package-lock.json says, without install scripts.
[ -d web/node_modules ] || npm ci --prefix web --ignore-scripts --no-audit --no-fund
echo "web chat (eslint, tsc, vitest, build)"
npm --prefix web run lint
npm --prefix web run typecheck
npm --prefix web test
npm --prefix web run build
echo "test_web_and_tools: ok"
