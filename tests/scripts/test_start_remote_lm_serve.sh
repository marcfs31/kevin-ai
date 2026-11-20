#!/usr/bin/env bash
# Quoting for start-remote-lm-serve.sh. Arguments must survive the remote login
# shell as separate words. No SSH connection is opened.
set -u
root=$(cd "$(dirname "$0")/../.." && pwd)
script=$root/scripts/start-remote-lm-serve.sh
fails=0
checks=0
check() {
  checks=$((checks + 1))
  if ! eval "$1"; then
    fails=$((fails + 1))
    echo "  FAIL $2"
  fi
}

# shellcheck disable=SC1090
source "$script"

q=$(quote_for_remote --name 'hello world' 'a;b' '$HOME/models')
got=$(bash -c "set -- $q; printf '%s\n' \"\$@\"")
want=$(printf '%s\n' --name 'hello world' 'a;b' '$HOME/models')
check '[ "$got" = "$want" ]' "quoted arguments did not round-trip: [$got]"

check '! grep -q "remote_cmd=\${remote_cmd" "$script"' "the script still splices with a pattern substitution"

out=$(bash "$script" 2>&1 || true)
check '[[ "$out" == *"must be qwen or gemma"* ]]' "no arguments should refuse before ssh, got: $out"

out=$(bash "$script" --id qwen --host 0.0.0.0 2>&1 || true)
check '[[ "$out" == *refusing*--host* ]]' "the script accepted --host: $out"

if [[ "$fails" -ne 0 ]]; then
  echo "test_start_remote_lm_serve: $fails of $checks checks FAILED"
  exit 1
fi
echo "test_start_remote_lm_serve: $checks checks passed"
