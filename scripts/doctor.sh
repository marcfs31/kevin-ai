#!/usr/bin/env bash
# Read-only pre-flight check for small-llm-in-c. Prints one line per check: OK, FAIL,
# UNKNOWN (could not be evaluated on this system) or NOTE (information, neither pass
# nor fail). Exits 1 if any check printed FAIL. Changes nothing on the system.
set -uo pipefail

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
GIB=1073741824
TWO_GIB=$((2 * GIB))
Q8_NAME=model.q8.safetensors
Q8_MIN_AVAIL_KIB=$((21 * 262144))     # q8 Qwen: 5.25 GiB (peak RSS measured 5.06 GiB, 2026-11-16)
GEMMA_Q8_MIN_AVAIL_KIB=$((6 * 1048576))  # q8 Gemma: 6 GiB (peak RSS measured 5.73 GiB, 2026-11-16)
BF16_QWEN_BYTES=9319828096            # bf16 Qwen checkpoint, both shards

usage() {
  cat <<'EOF'
Usage: scripts/doctor.sh [--read-speed] [--help] [MODEL_DIR]

Read-only checks: CPU model, cores, avx2 and fma flags, CPU clock cap, memory and swap, compiler,
the model type named in MODEL_DIR/config.json (qwen3.5-4b or gemma-4-e4b), that
model's files and sizes, the q8 file, whether MODEL_DIR is on a rotational disk, and
whether the engine binaries are built.

  MODEL_DIR      default: <root>/qwen3.5-4b, where <root> is what scripts/models-root.sh
                 prints ($LM_MODELS_DIR, else /mnt/models when you can write it, else
                 $HOME/models)
  --read-speed   time a direct read of the first 2 GiB of the q8 file (or of the first
                 bf16 weight file when there is no q8 file yet). Linux, dd iflag=direct.
  --help         show this help

Each check prints OK, FAIL, UNKNOWN or NOTE. UNKNOWN means the check could not be
evaluated on this system. NOTE is information, for example a q8 file not made yet.
The script exits 1 if any check printed FAIL.
EOF
}

READ_SPEED=0
MODEL_DIR_ARG=""
while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --read-speed) READ_SPEED=1 ;;
    -*) echo "error: unknown option: $1" >&2; usage >&2; exit 2 ;;
    *)
      if [ -n "$MODEL_DIR_ARG" ]; then
        echo "error: more than one MODEL_DIR given" >&2; usage >&2; exit 2
      fi
      MODEL_DIR_ARG="$1"
      ;;
  esac
  shift
done
MODEL_DIR="${MODEL_DIR_ARG:-$("$REPO_DIR/scripts/models-root.sh")/qwen3.5-4b}"
# Where the CPU facts are read from; the tests point these at fixture files.
CPUINFO="${LM_DOCTOR_CPUINFO:-/proc/cpuinfo}"
CPUFREQ_DIR="${LM_DOCTOR_CPUFREQ_DIR:-/sys/devices/system/cpu/cpu0/cpufreq}"

# Expected files per model: "name|size", empty size = not pinned (presence checked).
# These match the pins in scripts/download-model.sh.
QWEN_FILES=(
  "config.json|"
  "model.safetensors.index.json|"
  "model.safetensors-00001-of-00002.safetensors|5329398688"
  "model.safetensors-00002-of-00002.safetensors|3990429408"
  "tokenizer.json|12807982"
  "tokenizer_config.json|"
  "chat_template.jinja|"
)
GEMMA_FILES=(
  "config.json|"
  "model.safetensors|15992595884"
  "tokenizer.json|32156428"
  "tokenizer_config.json|"
  "chat_template.jinja|"
  "generation_config.json|"
)

FAILS=0
report() { # report STATUS LABEL DETAIL
  printf '[%-7s] %-24s %s\n' "$1" "$2" "${3:-}"
  if [ "$1" = FAIL ]; then FAILS=$((FAILS + 1)); fi
}

is_uint() { case "$1" in ''|*[!0-9]*) return 1 ;; *) return 0 ;; esac; }
file_size() { wc -c < "$1" | tr -d '[:space:]'; }
kb_to_gib() { awk -v k="$1" 'BEGIN { printf "%.1f GiB", k / 1048576 }'; }

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

# --- CPU -------------------------------------------------------------------------
if [ -r "$CPUINFO" ]; then
  cpu_model=$(awk -F': *' '/^model name/ {print $2; exit}' "$CPUINFO")
  if [ -n "$cpu_model" ]; then
    report OK "cpu model" "$cpu_model"
  else
    report UNKNOWN "cpu model" "no 'model name' in /proc/cpuinfo"
  fi

  flags=$(awk -F': *' '/^flags/ {print " " $2 " "; exit}' "$CPUINFO")
  for f in avx2 fma; do
    if [ -z "$flags" ]; then
      report UNKNOWN "cpu flag $f" "no 'flags' line in /proc/cpuinfo"
    elif [[ "$flags" == *" $f "* ]]; then
      report OK "cpu flag $f" "present"
    else
      report FAIL "cpu flag $f" "not in /proc/cpuinfo; the AVX2 build needs it"
    fi
  done
else
  report UNKNOWN "cpu model" "/proc/cpuinfo not readable"
  report UNKNOWN "cpu flag avx2" "/proc/cpuinfo not readable"
  report UNKNOWN "cpu flag fma" "/proc/cpuinfo not readable"
fi

cores=""
if command -v nproc >/dev/null 2>&1; then cores=$(nproc 2>/dev/null) || cores=""; fi
if ! is_uint "$cores" && [ -r "$CPUINFO" ]; then
  cores=$(awk '/^processor/ {n++} END {print n + 0}' "$CPUINFO")
fi
if is_uint "$cores" && [ "$cores" -gt 0 ]; then
  report OK "cores" "$cores"
else
  report UNKNOWN "cores" "cannot determine the core count"
fi

# --- CPU clock -------------------------------------------------------------------
# A CPU held far below the speed in its model name is held by firmware (BIOS power or
# thermal setting, or an unrecognised power adapter); the engine then runs at that clock.
# NOTE, not FAIL: a low clock is slow, not broken.
max_khz=""
[ -r "$CPUFREQ_DIR/cpuinfo_max_freq" ] && max_khz=$(tr -d '[:space:]' < "$CPUFREQ_DIR/cpuinfo_max_freq" 2>/dev/null)
if is_uint "$max_khz" && [ "$max_khz" -gt 0 ]; then
  max_mhz=$((max_khz / 1000))
  cap_khz=""
  [ -r "$CPUFREQ_DIR/scaling_max_freq" ] && cap_khz=$(tr -d '[:space:]' < "$CPUFREQ_DIR/scaling_max_freq" 2>/dev/null)
  gov=""
  [ -r "$CPUFREQ_DIR/scaling_governor" ] && gov=$(tr -d '[:space:]' < "$CPUFREQ_DIR/scaling_governor" 2>/dev/null)
  nominal_mhz=""
  if [[ "${cpu_model:-}" =~ @[[:space:]]*([0-9]+(\.[0-9]+)?)GHz ]]; then
    nominal_mhz=$(awk -v g="${BASH_REMATCH[1]}" 'BEGIN { printf "%d", g * 1000 }')
  fi
  if [ -n "$nominal_mhz" ] && [ $((max_mhz * 10)) -lt $((nominal_mhz * 8)) ]; then
    report NOTE "cpu clock" "capped at $max_mhz MHz, the model name says $nominal_mhz MHz: firmware is limiting the CPU. See README Troubleshooting (power adapter, BIOS)"
  elif is_uint "$cap_khz" && [ "$cap_khz" -lt "$max_khz" ]; then
    report NOTE "cpu clock" "scaling_max_freq is $((cap_khz / 1000)) MHz, below the hardware maximum of $max_mhz MHz: a software limit"
  else
    report OK "cpu clock" "hardware maximum $max_mhz MHz${nominal_mhz:+ (model name: $nominal_mhz MHz)}, governor ${gov:-unknown}"
  fi
else
  report UNKNOWN "cpu clock" "cpufreq is not exposed here (virtual machine, or no frequency driver)"
fi

# --- Memory ----------------------------------------------------------------------
mem_total_kb=""; mem_avail_kb=""; swap_total_kb=""; swap_free_kb=""
if [ -r /proc/meminfo ]; then
  mem_total_kb=$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)
  mem_avail_kb=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
  swap_total_kb=$(awk '/^SwapTotal:/ {print $2}' /proc/meminfo)
  swap_free_kb=$(awk '/^SwapFree:/ {print $2}' /proc/meminfo)
fi

if is_uint "$mem_total_kb"; then
  report OK "MemTotal" "$(kb_to_gib "$mem_total_kb")"
else
  report UNKNOWN "MemTotal" "not available in /proc/meminfo"
fi
if is_uint "$mem_avail_kb"; then
  report OK "MemAvailable" "$(kb_to_gib "$mem_avail_kb")"
else
  report UNKNOWN "MemAvailable" "not available in /proc/meminfo"
fi
if is_uint "$swap_total_kb" && is_uint "$swap_free_kb"; then
  report OK "swap" "total $(kb_to_gib "$swap_total_kb"), free $(kb_to_gib "$swap_free_kb")"
else
  report UNKNOWN "swap" "not available in /proc/meminfo"
fi

# --- Compiler --------------------------------------------------------------------
cc_bin="${CC:-cc}"
if command -v "$cc_bin" >/dev/null 2>&1; then
  cc_ver=$("$cc_bin" --version 2>/dev/null | awk 'NR == 1 {print; exit}')
  report OK "compiler" "$cc_bin: ${cc_ver:-version not reported}"
else
  report FAIL "compiler" "$cc_bin not found; install build-essential"
fi

# --- Model type ------------------------------------------------------------------
CONFIG="$MODEL_DIR/config.json"
model_id=unknown
if [ ! -d "$MODEL_DIR" ]; then
  report FAIL "MODEL_DIR" "$MODEL_DIR is not a directory (run scripts/download-model.sh)"
elif [ ! -e "$CONFIG" ]; then
  report FAIL "config.json" "not found in $MODEL_DIR (run scripts/download-model.sh)"
elif [ ! -r "$CONFIG" ]; then
  report UNKNOWN "config.json" "exists but is not readable: $CONFIG"
else
  if grep -Eq '"model_type"[[:space:]]*:[[:space:]]*"qwen3_5(_text)?"' "$CONFIG"; then
    model_id=qwen3.5-4b
  elif grep -Eq '"model_type"[[:space:]]*:[[:space:]]*"gemma4(_text)?"' "$CONFIG"; then
    model_id=gemma-4-e4b
  fi
  if [ "$model_id" = unknown ]; then
    report FAIL "model type" "config.json model_type is neither qwen3_5 nor gemma4"
  else
    report OK "model type" "$model_id (from config.json)"
  fi
fi

# --- Model files -----------------------------------------------------------------
SPEED_WEIGHTS=""
case "$model_id" in
  qwen3.5-4b)  MODEL_FILES=("${QWEN_FILES[@]}");  SPEED_WEIGHTS=model.safetensors-00001-of-00002.safetensors ;;
  gemma-4-e4b) MODEL_FILES=("${GEMMA_FILES[@]}"); SPEED_WEIGHTS=model.safetensors ;;
  *)           MODEL_FILES=() ;;
esac

check_model_file() { # check_model_file NAME EXPECTED_SIZE_OR_EMPTY
  local path="$MODEL_DIR/$1" size
  if [ ! -e "$path" ]; then
    report FAIL "$1" "not found in $MODEL_DIR (run scripts/download-model.sh)"
  elif [ ! -r "$path" ]; then
    report UNKNOWN "$1" "exists but is not readable: $path"
  else
    size=$(file_size "$path")
    if [ -z "$2" ]; then
      if [ "$size" -gt 0 ]; then
        report OK "$1" "$size bytes (size not pinned)"
      else
        report FAIL "$1" "empty file"
      fi
    elif [ "$size" = "$2" ]; then
      report OK "$1" "$size bytes"
    else
      report FAIL "$1" "$size bytes, expected $2"
    fi
  fi
}

if [ "$model_id" = unknown ]; then
  report UNKNOWN "model files" "model type not known; no file list to check"
else
  for entry in "${MODEL_FILES[@]}"; do
    IFS='|' read -r name size <<< "$entry"
    check_model_file "$name" "$size"
  done
fi

# --- q8 file ---------------------------------------------------------------------
Q8="$MODEL_DIR/$Q8_NAME"
case "$model_id" in
  qwen3.5-4b|gemma-4-e4b)
    if [ -e "$Q8" ]; then
      q8_size=$(file_size "$Q8")
      if [ "$q8_size" -gt 0 ]; then
        report OK "q8 file" "$Q8_NAME, $q8_size bytes"
      else
        report FAIL "q8 file" "$Q8 is empty; move it aside and re-run bin/lm-quant"
      fi
    else
      report NOTE "q8 file" "not made yet. From the repo root (after make): bin/lm-quant --model $MODEL_DIR --out $Q8"
    fi
    ;;
  *)
    report UNKNOWN "q8 file" "model type not known"
    ;;
esac

# --- RAM -------------------------------------------------------------------------
case "$model_id" in
  qwen3.5-4b)
    if is_uint "$mem_avail_kb"; then
      if [ "$mem_avail_kb" -ge "$Q8_MIN_AVAIL_KIB" ]; then
        report OK "RAM for q8" "$(kb_to_gib "$mem_avail_kb") available; q8 Qwen needs about 5.1 GiB"
      else
        report FAIL "RAM for q8" "only $(kb_to_gib "$mem_avail_kb") available; q8 Qwen needs about 5.1 GiB. Stop other programs, then retry"
      fi
    else
      report UNKNOWN "RAM for q8" "MemAvailable unknown"
    fi
    if is_uint "$mem_total_kb"; then
      if [ $((mem_total_kb * 1024)) -ge "$BF16_QWEN_BYTES" ]; then
        report OK "RAM for bf16" "bf16 Qwen (9.3 GB) fits in $(kb_to_gib "$mem_total_kb")"
      else
        report NOTE "RAM for bf16" "bf16 Qwen is 9.3 GB and MemTotal is $(kb_to_gib "$mem_total_kb"); bf16 (exact) mode streams from disk and is slow"
      fi
    else
      report UNKNOWN "RAM for bf16" "MemTotal unknown"
    fi
    ;;
  gemma-4-e4b)
    if is_uint "$mem_avail_kb"; then
      if [ "$mem_avail_kb" -ge "$GEMMA_Q8_MIN_AVAIL_KIB" ]; then
        report OK "RAM for q8" "$(kb_to_gib "$mem_avail_kb") available; q8 Gemma needs about 5.8 GiB"
      else
        report FAIL "RAM for q8" "only $(kb_to_gib "$mem_avail_kb") available; q8 Gemma needs about 5.8 GiB. Stop other programs, then retry"
      fi
    else
      report UNKNOWN "RAM for q8" "MemAvailable unknown"
    fi
    report NOTE "RAM for bf16" "bf16 Gemma is 16 GB; bf16 (exact) mode streams from disk and is very slow on this machine"
    ;;
  *)
    report UNKNOWN "RAM for model" "model type not known"
    ;;
esac

# --- Engine binaries -------------------------------------------------------------
missing_bins=""
for b in lm-run lm-chat lm-quant; do
  [ -x "$REPO_DIR/bin/$b" ] || missing_bins="$missing_bins $b"
done
if [ -z "$missing_bins" ]; then
  report OK "engine binaries" "bin/lm-run, bin/lm-chat, bin/lm-quant present"
else
  report NOTE "engine binaries" "not built:$missing_bins. From the repo root run: make"
fi

# --- Storage ---------------------------------------------------------------------
probe="$MODEL_DIR"
while [ ! -e "$probe" ] && [ "$probe" != "/" ]; do probe=$(dirname "$probe"); done
rot=$(rotational_status "$probe")
case "$rot" in
  1) report FAIL "model storage" "$MODEL_DIR is on a rotational disk ($rot_src); weight reads need the SSD" ;;
  0) report OK "model storage" "non-rotational device ($rot_src)" ;;
  *) report UNKNOWN "model storage" "cannot determine whether $MODEL_DIR is rotational (device: ${rot_src:-unknown})" ;;
esac

# --- Read speed (optional) -------------------------------------------------------
if [ "$READ_SPEED" -eq 1 ]; then
  if [ "$model_id" = unknown ]; then
    report UNKNOWN "read speed" "model type not known"
  else
    f="$MODEL_DIR/$SPEED_WEIGHTS"
    if [ "$model_id" = qwen3.5-4b ] && [ -f "$Q8" ]; then f="$Q8"; fi
    if [ ! -r "$f" ]; then
      report UNKNOWN "read speed" "not readable: $f"
    elif [ "$(file_size "$f")" -lt "$TWO_GIB" ]; then
      report UNKNOWN "read speed" "$(basename "$f") is smaller than 2 GiB"
    elif ! command -v dd >/dev/null 2>&1; then
      report UNKNOWN "read speed" "dd not found"
    else
      out=$(dd if="$f" of=/dev/null bs=1M count=2048 iflag=direct 2>&1)
      dd_rc=$?
      last=$(printf '%s\n' "$out" | awk 'NF {l = $0} END {print l}')
      if [ "$dd_rc" -ne 0 ]; then
        report UNKNOWN "read speed" "dd failed (iflag=direct may be unsupported here): $last"
      else
        mbps=$(printf '%s\n' "$last" | awk '
          /copied/ {
            rate = $(NF - 1); unit = $NF
            if (unit == "GB/s") rate *= 1000
            else if (unit == "kB/s") rate /= 1000
            else if (unit == "B/s") rate /= 1000000
            printf "%.1f", rate
          }')
        if [ -n "$mbps" ]; then
          report OK "read speed" "$mbps MB/s from $(basename "$f") (first 2 GiB, direct I/O; dd: $last)"
        else
          report UNKNOWN "read speed" "could not parse dd output: $last"
        fi
      fi
    fi
  fi
fi

if [ "$FAILS" -gt 0 ]; then
  echo "doctor: $FAILS check(s) FAILED"
  exit 1
fi
echo "doctor: no FAIL checks"
exit 0
