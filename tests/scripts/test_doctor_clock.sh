#!/usr/bin/env bash
# test_doctor_clock.sh - the `cpu clock` check of scripts/doctor.sh, against fixture cpuinfo
# and cpufreq files (LM_DOCTOR_CPUINFO, LM_DOCTOR_CPUFREQ_DIR). Run from the repository root.
set -u
DOCTOR=scripts/doctor.sh
[ -x "$DOCTOR" ] || { echo "  FAIL $DOCTOR not found or not executable (run from the repository root)"; exit 1; }
tmp=$(mktemp -d) || exit 1
trap 'find "$tmp" -delete' EXIT
checks=0; fails=0

# line CPU_NAME MAX_KHZ [SCALING_MAX_KHZ]: prints doctor's "cpu clock" line
line() {
  local name="$1" max="$2" cap="${3:-$2}" d="$tmp/freq$RANDOM"
  mkdir -p "$d"
  printf 'model name\t: %s\nflags\t\t: fpu avx2 fma\nprocessor\t: 0\n' "$name" > "$d/cpuinfo"
  if [ "$max" != none ]; then
    echo "$max" > "$d/cpuinfo_max_freq"; echo "$cap" > "$d/scaling_max_freq"; echo powersave > "$d/scaling_governor"
  fi
  LM_DOCTOR_CPUINFO="$d/cpuinfo" LM_DOCTOR_CPUFREQ_DIR="$d" "$DOCTOR" /nonexistent 2>&1 | grep ' cpu clock '
}
expect() { # expect STATUS NEEDLE LABEL OUTPUT
  checks=$((checks + 1))
  case "$4" in
    "[$1"*"$2"*) ;;
    *) fails=$((fails + 1)); echo "  FAIL $3: expected [$1] containing \"$2\", got: $4" ;;
  esac
}
I5='Intel(R) Core(TM) i5-6500T CPU @ 2.50GHz'

expect NOTE "capped at 800 MHz" "the real case: 800 MHz on a 2.50 GHz CPU" "$(line "$I5" 800000)"
expect OK "hardware maximum 3100 MHz" "turbo CPU above its rating" "$(line "$I5" 3100000)"
expect OK "hardware maximum 2000 MHz" "exactly 80% of nominal is not capped" "$(line "$I5" 2000000)"
expect NOTE "capped at 1999 MHz" "just under 80% of nominal is capped" "$(line "$I5" 1999000)"
expect NOTE "scaling_max_freq is 800 MHz" "a software limit below the hardware maximum" "$(line "$I5" 3100000 800000)"
expect UNKNOWN "cpufreq is not exposed" "no cpufreq directory (virtual machine)" "$(line "$I5" none)"
# NEGATIVE: no rated speed in the model name means no basis to call 800 MHz a cap
expect OK "hardware maximum 800 MHz" "no GHz in the model name: no false alarm" "$(line 'Some ARM Cortex-A53' 800000)"

if [ "$fails" -ne 0 ]; then echo "test_doctor_clock: $fails of $checks checks FAILED"; exit 1; fi
echo "test_doctor_clock: $checks checks passed"
