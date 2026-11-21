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
check '[[ "$out" == *"must be qwen, qwen-uncensored or gemma"* ]]' "no arguments should refuse before ssh, got: $out"

out=$(bash "$script" --id qwen --host 0.0.0.0 2>&1 || true)
check '[[ "$out" == *refusing*--host* ]]' "the script accepted --host: $out"

out=$(bash "$script" --id llama 2>&1 || true)
check '[[ "$out" == *"must be qwen, qwen-uncensored or gemma"* ]]' "an unknown id should be refused, got: $out"

# Resolve each id to its remote directory with a fake ssh that prints the arguments
# it would send (the directory name is the third positional word of the remote command).
fakebin=$(mktemp -d)
cat >"$fakebin/ssh" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "$*"
EOF
chmod +x "$fakebin/ssh"
for pair in qwen:qwen3.5-4b qwen-uncensored:qwen3.5-4b-uncensored gemma:gemma-4-e4b; do
  id=${pair%%:*}
  want_dir=${pair#*:}
  out=$(PATH="$fakebin:$PATH" bash "$script" --id "$id" 2>&1 || true)
  check '[[ "$out" == *" $want_dir "* ]]' "--id $id should resolve to $want_dir, got: $out"
done
out=$(PATH="$fakebin:$PATH" bash "$script" --id qwen 2>&1 || true)
check '[[ "$out" != *"qwen3.5-4b-uncensored"* ]]' "--id qwen must not resolve to the uncensored dir"
rm -rf "$fakebin"

if [[ "$fails" -ne 0 ]]; then
  echo "test_start_remote_lm_serve: $fails of $checks checks FAILED"
  exit 1
fi
echo "test_start_remote_lm_serve: $checks checks passed"
