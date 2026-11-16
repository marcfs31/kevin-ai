#!/usr/bin/env bash
# Download the pinned checkpoint files for one supported model into MODEL_DIR.
set -euo pipefail

DEFAULT_MODEL=qwen3.5-4b
ONE_GB=1000000000   # decimal; free-space margin on top of the checkpoint size

usage() {
  cat <<'EOF'
Usage: scripts/download-model.sh [--model NAME] [--allow-hdd] [--help] [MODEL_DIR]

Downloads the pinned checkpoint files for one model into MODEL_DIR.

  --model NAME  qwen3.5-4b (default) or gemma-4-e4b
  --allow-hdd   Continue when MODEL_DIR is on a rotational disk. Not recommended: the
                engine reads the weights from this directory and an HDD is far too slow.
  --help        Show this help and exit.

  MODEL_DIR     default: <root>/<NAME>, where <root> is what scripts/models-root.sh prints
                ($LM_MODELS_DIR, else $HOME/models)

qwen3.5-4b (phase 1, Apache-2.0)
  Qwen/Qwen3.5-4B at revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
    model.safetensors-00001-of-00002.safetensors  5329398688 bytes, sha256 pinned
    model.safetensors-00002-of-00002.safetensors  3990429408 bytes, sha256 pinned
    tokenizer.json                                12807982 bytes, sha256 pinned
    config.json, model.safetensors.index.json, tokenizer_config.json, chat_template.jinja
                                                  no pinned size or hash (presence checked)

gemma-4-e4b (phase 2, Apache-2.0)
  google/gemma-4-E4B-it at revision 4ecc2ba63bcac23ba4a767fc87025b1450f84c9c
    model.safetensors   15992595884 bytes, sha256 pinned
    tokenizer.json      32156428 bytes, sha256 pinned
    config.json, tokenizer_config.json, chat_template.jinja, generation_config.json
                        no pinned size or hash (presence checked)

Behaviour:
  - Requires curl and sha256sum, and free space of the checkpoint size plus 1 GB.
  - Refuses a rotational disk unless --allow-hdd is given. If the disk type cannot be
    determined, a warning is printed and the script continues.
  - Downloads are resumable (curl -C -): re-run the script after an interruption.
  - A pinned file with the expected size and sha256 is skipped. A pinned file with the
    expected size but the wrong sha256 stops the script.
  - After a download the size and sha256 are checked. On mismatch the script exits
    non-zero and leaves the file in place for inspection.
  - Never deletes or overwrites a file, and never uses sudo.
EOF
}

die() { echo "error: $*" >&2; exit 1; }
usage_error() { echo "error: $*" >&2; echo "see: scripts/download-model.sh --help" >&2; exit 2; }
warn() { echo "warning: $*" >&2; }

MODEL="$DEFAULT_MODEL"
ALLOW_HDD=0
MODEL_DIR_ARG=""
while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --model)
      [ $# -ge 2 ] || usage_error "--model needs a value"
      MODEL="$2"; shift ;;
    --model=*) MODEL="${1#--model=}" ;;
    --allow-hdd) ALLOW_HDD=1 ;;
    -*) usage_error "unknown option: $1" ;;
    *)
      if [ -n "$MODEL_DIR_ARG" ]; then usage_error "more than one MODEL_DIR given"; fi
      MODEL_DIR_ARG="$1"
      ;;
  esac
  shift
done

# Model table. Each entry is "name|expected_size|expected_sha256"; an empty size or
# sha means the file is not pinned. Every file lives at the root of its revision.
case "$MODEL" in
  qwen3.5-4b)
    BASE_URL="https://huggingface.co/Qwen/Qwen3.5-4B/resolve/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
    CHECKPOINT_BYTES=9319828096   # both weight shards
    FILES=(
      "config.json||"
      "model.safetensors.index.json||"
      "model.safetensors-00001-of-00002.safetensors|5329398688|26a93f066e1916adb13453dae5a0c707c0fbc71299ed98779571a907b8e74c61"
      "model.safetensors-00002-of-00002.safetensors|3990429408|cb544bd9bfae93dc59b0f22b292f5933573854a7f9b97835c67060d7d910e188"
      "tokenizer.json|12807982|5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"
      "tokenizer_config.json||"
      "chat_template.jinja||"
    )
    ;;
  gemma-4-e4b)
    BASE_URL="https://huggingface.co/google/gemma-4-E4B-it/resolve/4ecc2ba63bcac23ba4a767fc87025b1450f84c9c"
    CHECKPOINT_BYTES=15992595884
    FILES=(
      "config.json||"
      "model.safetensors|15992595884|cfbd3d2f1cd71bd471c37fe2bf8546d5028d41e5736f64e1ca6c6b8893125503"
      "tokenizer.json|32156428|91d9cf86906f78e197021fda88214d448e5c120665d0243a8e776e046986c6bc"
      "tokenizer_config.json||"
      "chat_template.jinja||"
      "generation_config.json||"
    )
    ;;
  *)
    usage_error "unknown model: $MODEL (use qwen3.5-4b or gemma-4-e4b)"
    ;;
esac

MODEL_DIR="${MODEL_DIR_ARG:-$("$(dirname "${BASH_SOURCE[0]}")/models-root.sh")/$MODEL}"
MIN_FREE_BYTES=$((CHECKPOINT_BYTES + ONE_GB))

for cmd in curl sha256sum; do
  command -v "$cmd" >/dev/null 2>&1 || die "required command not found: $cmd"
done

mkdir -p "$MODEL_DIR" || die "cannot create $MODEL_DIR"

file_size() { wc -c < "$1" | tr -d '[:space:]'; }
sha256_of() { sha256sum "$1" | awk '{print $1}'; }

# Prints 0 (non-rotational), 1 (rotational) or unknown for the filesystem holding $1.
# Sets rot_src to the block device it resolved.
rot_src=""
rotational_status() {
  local p="$1" src="" flags name f
  rot_src=""
  if command -v findmnt >/dev/null 2>&1; then
    src=$(findmnt -n -o SOURCE --target "$p" 2>/dev/null | awk 'NR==1 {print; exit}') || src=""
  fi
  if [ -z "$src" ]; then
    src=$(df -P "$p" 2>/dev/null | awk 'NR==2 {print $1}') || src=""
  fi
  src=${src%%\[*}          # strip btrfs subvolume suffix, e.g. /dev/sda2[/@]
  rot_src="$src"
  case "$src" in
    /dev/*) ;;
    *) echo unknown; return 0 ;;
  esac
  if command -v lsblk >/dev/null 2>&1; then
    # -s lists the device and its parents (LVM, LUKS, RAID); any rotational parent counts.
    flags=$(lsblk -sn -o ROTA "$src" 2>/dev/null | tr -d ' ' | awk 'NF') || flags=""
    if [ -n "$flags" ]; then
      if grep -qx 1 <<< "$flags"; then echo 1
      elif grep -qvx 0 <<< "$flags"; then echo unknown
      else echo 0
      fi
      return 0
    fi
  fi
  name=${src#/dev/}
  for f in "/sys/class/block/$name/queue/rotational" "/sys/class/block/$name/../queue/rotational"; do
    if [ -r "$f" ]; then
      flags=$(tr -d ' \n' < "$f")
      case "$flags" in
        0) echo 0 ;;
        1) echo 1 ;;
        *) echo unknown ;;
      esac
      return 0
    fi
  done
  echo unknown
}

# Free space on the filesystem that will hold the files.
avail_kb=$(df -Pk "$MODEL_DIR" | awk 'NR==2 {print $4}')
case "$avail_kb" in
  ''|*[!0-9]*) die "cannot determine free space on $MODEL_DIR" ;;
esac
avail_bytes=$((avail_kb * 1024))
if [ "$avail_bytes" -lt "$MIN_FREE_BYTES" ]; then
  die "need at least $((MIN_FREE_BYTES / ONE_GB)) GB free on the filesystem holding $MODEL_DIR; about $((avail_bytes / ONE_GB)) GB free"
fi
echo "free space: about $((avail_bytes / ONE_GB)) GB on the filesystem holding $MODEL_DIR"

rot=$(rotational_status "$MODEL_DIR")
case "$rot" in
  1)
    if [ "$ALLOW_HDD" -eq 1 ]; then
      warn "$MODEL_DIR is on a rotational disk ($rot_src); continuing because --allow-hdd was given"
    else
      die "$MODEL_DIR is on a rotational disk ($rot_src). Weight reads from an HDD are far too slow. Use a directory on the SSD, or pass --allow-hdd to override."
    fi
    ;;
  0) echo "storage: non-rotational ($rot_src)" ;;
  *) warn "could not determine whether $MODEL_DIR is on a rotational disk (device: ${rot_src:-unknown}); continuing" ;;
esac

# fetch NAME EXPECTED_SIZE EXPECTED_SHA256   (size and sha may be empty = not pinned)
fetch() {
  local name="$1" size="$2" sha="$3"
  local dest="$MODEL_DIR/$name" have got h

  if [ -f "$dest" ]; then
    if [ -z "$size" ]; then
      if [ -s "$dest" ]; then
        echo "skip $name: present (no pinned size or hash; delete it to force a re-fetch)"
        return 0
      fi
    else
      have=$(file_size "$dest")
      if [ "$have" = "$size" ]; then
        h=$(sha256_of "$dest")
        if [ "$h" = "$sha" ]; then
          echo "skip $name: already present and verified"
          return 0
        fi
        die "$dest has the expected size but the wrong sha256 ($h). The script does not delete files; move it aside and re-run."
      fi
      if [ "$have" -gt "$size" ]; then
        die "$dest is $have bytes, larger than the expected $size. The script does not delete files; move it aside and re-run."
      fi
    fi
  fi

  echo "download $name (${size:-size not pinned} bytes)"
  curl -fL -C - --retry 3 --retry-delay 5 -o "$dest" "$BASE_URL/$name" \
    || die "download failed for $name; re-run the script to resume"

  got=$(file_size "$dest")
  if [ -z "$size" ]; then
    [ "$got" -gt 0 ] || die "$name is empty after download"
  else
    if [ "$got" != "$size" ]; then
      die "$name: size is $got bytes, expected $size (left in place for inspection)"
    fi
    h=$(sha256_of "$dest")
    if [ "$h" != "$sha" ]; then
      die "$name: sha256 is $h, expected $sha (left in place for inspection)"
    fi
  fi
  echo "ok $name"
}

for entry in "${FILES[@]}"; do
  IFS='|' read -r name size sha <<< "$entry"
  fetch "$name" "$size" "$sha"
done

echo "done: $MODEL files are in $MODEL_DIR"
