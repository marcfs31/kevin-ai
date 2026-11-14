#!/usr/bin/env bash
# Heavier checks kept out of the per-pull CI: run by .githooks/pre-push and by the weekly
# ci-heavy workflow. Fails on the first error.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
make clean
make sanitize
make portable
make clean
echo "ci-heavy OK"
