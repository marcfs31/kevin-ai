#!/usr/bin/env bash
# One-shot setup of small-llm-in-c on an Ubuntu Server: check tools, build, test, download
# the model checkpoint, make the q8 file, and print the chat command.
# Run it from a checkout of the repository. It never uses sudo itself.
set -euo pipefail

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

usage() {
  cat <<'EOF'
Usage: scripts/install.sh [--model NAME] [--model-dir DIR] [--skip-tests] [--allow-hdd] [--help]

Steps, in order, each stopping the script on failure:
  1. check build tools (gcc or cc, make, python3, curl, sha256sum) and print the exact
     `sudo apt install` line if something is missing (this script never runs sudo)
  2. make          build bin/lm-chat, bin/lm-run, bin/lm-quant (OpenMP on Linux)
  3. make test, make gate   unit tests and the golden gates for both model families
  4. scripts/download-model.sh   pinned, resumable, sha256-checked
  5. bin/lm-quant   write MODEL_DIR/model.q8.safetensors
  6. scripts/doctor.sh   read-only pre-flight report, then the chat command

  --model NAME    qwen3.5-4b (default), gemma-4-e4b, both, or none (build and test only)
  --model-dir DIR where model folders go: DIR/<NAME>. Default: $HOME/models
  --skip-tests    skip step 3 (not recommended on the first run: it is the first AVX2 run)
  --allow-hdd     pass through to the downloader. Not recommended.
  --help          show this help

Disk needed: qwen3.5-4b about 10.3 GB download + 4.8 GB q8; gemma-4-e4b about 17 GB
download + 8.5 GB q8. Put --model-dir on the NVMe SSD, not the 5400 rpm HDD.
Re-running is safe: finished downloads and an existing q8 file are skipped.
EOF
}

die() { echo "error: $*" >&2; exit 1; }
step() { echo; echo "==> $*"; }

MODEL=qwen3.5-4b
MODELS_ROOT="$HOME/models"
SKIP_TESTS=0
ALLOW_HDD=""
while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --model) [ $# -ge 2 ] || die "--model needs a value"; MODEL=$2; shift ;;
    --model-dir) [ $# -ge 2 ] || die "--model-dir needs a value"; MODELS_ROOT=$2; shift ;;
    --skip-tests) SKIP_TESTS=1 ;;
    --allow-hdd) ALLOW_HDD=--allow-hdd ;;
    *) die "unknown argument: $1 (see --help)" ;;
  esac
  shift
done

case "$MODEL" in
  qwen3.5-4b) TARGETS=(qwen3.5-4b) ;;
  gemma-4-e4b) TARGETS=(gemma-4-e4b) ;;
  both) TARGETS=(qwen3.5-4b gemma-4-e4b) ;;
  none) TARGETS=() ;;
  *) die "unknown --model: $MODEL (use qwen3.5-4b, gemma-4-e4b, both or none)" ;;
esac

# --- 1. tools --------------------------------------------------------------------
step "Checking the system"
[ "$(uname -s)" = Linux ] || echo "warning: this is not Linux ($(uname -s)); the script is meant for Ubuntu Server" >&2
if [ -r /etc/os-release ]; then ( . /etc/os-release; echo "OS: ${PRETTY_NAME:-unknown}" ); fi
echo "kernel: $(uname -r)  arch: $(uname -m)"
if [ "$(uname -m)" = x86_64 ] && [ -r /proc/cpuinfo ] && ! grep -qw avx2 /proc/cpuinfo; then
  die "this CPU has no AVX2; the engine's fast path needs it"
fi

missing=()
command -v make >/dev/null || missing+=(make)
command -v gcc >/dev/null || command -v cc >/dev/null || missing+=(build-essential)
command -v python3 >/dev/null || missing+=(python3)
command -v curl >/dev/null || missing+=(curl)
command -v sha256sum >/dev/null || missing+=(coreutils)
if [ ${#missing[@]} -gt 0 ]; then
  echo "Missing tools: ${missing[*]}" >&2
  echo "Run this yourself, then re-run this script:" >&2
  echo "  sudo apt install build-essential python3 curl" >&2
  exit 1
fi
echo "tools OK: $(${CC:-cc} --version | head -1)"

free_gb() { df -Pk "$1" | awk 'NR==2 {printf "%d", $4 * 1024 / 1000000000}'; }
mkdir -p "$MODELS_ROOT"
need_gb=0
for m in ${TARGETS[@]+"${TARGETS[@]}"}; do
  case "$m" in qwen3.5-4b) need_gb=$((need_gb + 16)) ;; gemma-4-e4b) need_gb=$((need_gb + 27)) ;; esac
done
have_gb=$(free_gb "$MODELS_ROOT")
echo "free space at $MODELS_ROOT: ${have_gb} GB (needed for the chosen models, about ${need_gb} GB)"
[ "$have_gb" -ge "$need_gb" ] || die "not enough free space at $MODELS_ROOT"

# --- 2. build --------------------------------------------------------------------
cd "$REPO_DIR"
step "Building"
make

# --- 3. tests ---------------------------------------------------------------------
if [ "$SKIP_TESTS" -eq 0 ]; then
  step "Unit tests"
  make test
  step "Golden gates (Qwen3.5 and Gemma 4 tiny models vs the float64 oracles)"
  make gate
else
  echo "skipping tests (--skip-tests)"
fi

# --- 4-5. model download and q8 ----------------------------------------------------
for m in ${TARGETS[@]+"${TARGETS[@]}"}; do
  dir="$MODELS_ROOT/$m"
  step "Downloading $m to $dir"
  scripts/download-model.sh --model "$m" ${ALLOW_HDD:+"$ALLOW_HDD"} "$dir"
  q8="$dir/model.q8.safetensors"
  if [ -s "$q8" ]; then
    echo "q8 file already present: $q8 (not rewritten)"
  else
    step "Writing the q8 file for $m"
    bin/lm-quant --model "$dir" --out "$q8"
  fi
done

# --- 6. report ----------------------------------------------------------------------
for m in ${TARGETS[@]+"${TARGETS[@]}"}; do
  step "Pre-flight report for $m"
  scripts/doctor.sh "$MODELS_ROOT/$m" || echo "doctor reported FAIL lines above; read them before chatting" >&2
done

echo
echo "Done. Chat with:"
for m in ${TARGETS[@]+"${TARGETS[@]}"}; do
  echo "  $REPO_DIR/bin/lm-chat --model $MODELS_ROOT/$m --weights $MODELS_ROOT/$m/model.q8.safetensors --stats"
done
[ ${#TARGETS[@]} -gt 0 ] || echo "  (no model chosen; re-run with --model qwen3.5-4b or gemma-4-e4b)"
echo "The chat command prints tokens/s (--stats). If the doctor report above has a 'cpu clock' NOTE, fix that cap first: see README Troubleshooting."
