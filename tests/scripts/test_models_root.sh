#!/usr/bin/env bash
# test_models_root.sh - scripts/models-root.sh prints LM_MODELS_DIR when set, else $HOME/models
# (never a second disk by itself), and the downloader, doctor and installer default to it.
# Run from the repository root.
set -u
ROOT=scripts/models-root.sh
[ -x "$ROOT" ] || { echo "  FAIL $ROOT not found or not executable (run from the repository root)"; exit 1; }
tmp=$(mktemp -d) || exit 1
trap 'find "$tmp" -delete' EXIT
checks=0; fails=0
check() { # check LABEL EXPECTED ACTUAL
  checks=$((checks + 1))
  [ "$2" = "$3" ] || { fails=$((fails + 1)); echo "  FAIL $1: expected $2, got $3"; }
}
mkdir -p "$tmp/home"

check "LM_MODELS_DIR wins" "/x/y" "$(LM_MODELS_DIR=/x/y HOME="$tmp/home" "$ROOT")"
check "default is HOME/models" "$tmp/home/models" "$(env -u LM_MODELS_DIR HOME="$tmp/home" "$ROOT")"
check "an empty LM_MODELS_DIR is unset" "$tmp/home/models" "$(LM_MODELS_DIR= HOME="$tmp/home" "$ROOT")"

# the scripts that default to it
out=$(env -u LM_MODELS_DIR HOME="$tmp/home" scripts/doctor.sh 2>&1)
case "$out" in *"$tmp/home/models/qwen3.5-4b"*) ok=yes ;; *) ok=no ;; esac
check "doctor defaults to ROOT/qwen3.5-4b" yes "$ok"
out=$(env -u LM_MODELS_DIR HOME="$tmp/home" scripts/download-model.sh --help 2>&1)
case "$out" in *'scripts/models-root.sh'*) ok=yes ;; *) ok=no ;; esac
check "download-model --help names the default" yes "$ok"
out=$(env -u LM_MODELS_DIR HOME="$tmp/home" scripts/install.sh --help 2>&1)
case "$out" in *'scripts/models-root.sh'*) ok=yes ;; *) ok=no ;; esac
check "install --help names the default" yes "$ok"

if [ "$fails" -ne 0 ]; then echo "test_models_root: $fails of $checks checks FAILED"; exit 1; fi
echo "test_models_root: $checks checks passed"
