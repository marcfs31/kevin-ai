#!/usr/bin/env bash
# test_models_root.sh - scripts/models-root.sh picks LM_MODELS_DIR, then a writable models
# SSD mount (LM_MODELS_MOUNT stands in for /mnt/models), then $HOME/models; and the
# downloader and doctor default to it. Run from the repository root.
set -u
ROOT=scripts/models-root.sh
[ -x "$ROOT" ] || { echo "  FAIL $ROOT not found or not executable (run from the repository root)"; exit 1; }
tmp=$(mktemp -d) || exit 1
trap 'chmod -R u+w "$tmp"; find "$tmp" -delete' EXIT
checks=0; fails=0
check() { # check LABEL EXPECTED ACTUAL
  checks=$((checks + 1))
  [ "$2" = "$3" ] || { fails=$((fails + 1)); echo "  FAIL $1: expected $2, got $3"; }
}
mkdir -p "$tmp/home" "$tmp/ssd" "$tmp/ro"
chmod 555 "$tmp/ro"

check "LM_MODELS_DIR wins" "/x/y" "$(LM_MODELS_DIR=/x/y LM_MODELS_MOUNT="$tmp/ssd" HOME="$tmp/home" "$ROOT")"
check "a writable mount" "$tmp/ssd" "$(env -u LM_MODELS_DIR LM_MODELS_MOUNT="$tmp/ssd" HOME="$tmp/home" "$ROOT")"
if [ "$(id -u)" != 0 ]; then   # root can write anywhere
  check "a read-only mount point (disk not mounted)" "$tmp/home/models" \
        "$(env -u LM_MODELS_DIR LM_MODELS_MOUNT="$tmp/ro" HOME="$tmp/home" "$ROOT")"
fi
check "no mount point" "$tmp/home/models" "$(env -u LM_MODELS_DIR LM_MODELS_MOUNT="$tmp/none" HOME="$tmp/home" "$ROOT")"
check "an empty LM_MODELS_DIR is unset" "$tmp/ssd" "$(LM_MODELS_DIR= LM_MODELS_MOUNT="$tmp/ssd" HOME="$tmp/home" "$ROOT")"

# the scripts that default to it
out=$(env -u LM_MODELS_DIR LM_MODELS_MOUNT="$tmp/ssd" HOME="$tmp/home" scripts/doctor.sh 2>&1)
case "$out" in *"$tmp/ssd/qwen3.5-4b"*) ok=yes ;; *) ok=no ;; esac
check "doctor defaults to ROOT/qwen3.5-4b" yes "$ok"
out=$(env -u LM_MODELS_DIR LM_MODELS_MOUNT="$tmp/ssd" HOME="$tmp/home" scripts/download-model.sh --help 2>&1)
case "$out" in *'scripts/models-root.sh'*) ok=yes ;; *) ok=no ;; esac
check "download-model --help names the default" yes "$ok"
out=$(env -u LM_MODELS_DIR LM_MODELS_MOUNT="$tmp/ssd" HOME="$tmp/home" scripts/install.sh --help 2>&1)
case "$out" in *'scripts/models-root.sh'*) ok=yes ;; *) ok=no ;; esac
check "install --help names the default" yes "$ok"

if [ "$fails" -ne 0 ]; then echo "test_models_root: $fails of $checks checks FAILED"; exit 1; fi
echo "test_models_root: $checks checks passed"
