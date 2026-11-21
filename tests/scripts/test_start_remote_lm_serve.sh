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
check '[[ "$out" == *"must be qwen, qwen-uncensored, gemma, whiterabbitneo or gemma-uncensored"* ]]' "no arguments should refuse before ssh, got: $out"

out=$(bash "$script" --id qwen --host 0.0.0.0 2>&1 || true)
check '[[ "$out" == *refusing*--host* ]]' "the script accepted --host: $out"

out=$(bash "$script" --id llama 2>&1 || true)
check '[[ "$out" == *"must be qwen, qwen-uncensored, gemma, whiterabbitneo or gemma-uncensored"* ]]' "an unknown id should be refused, got: $out"

# Resolve each id to its remote directory with a fake ssh that prints the arguments
# it would send (the directory name is the third positional word of the remote command).
fakebin=$(mktemp -d)
cat >"$fakebin/ssh" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "$*"
EOF
chmod +x "$fakebin/ssh"
for pair in qwen:qwen3.5-4b qwen-uncensored:qwen3.5-4b-uncensored gemma:gemma-4-e4b whiterabbitneo:whiterabbitneo-2.5-7b gemma-uncensored:gemma-4-e4b-uncensored; do
  id=${pair%%:*}
  want_dir=${pair#*:}
  out=$(PATH="$fakebin:$PATH" bash "$script" --id "$id" 2>&1 || true)
  check '[[ "$out" == *" $want_dir "* ]]' "--id $id should resolve to $want_dir, got: $out"
done
out=$(PATH="$fakebin:$PATH" bash "$script" --id qwen 2>&1 || true)
check '[[ "$out" != *"qwen3.5-4b-uncensored"* ]]' "--id qwen must not resolve to the uncensored dir"
# whiterabbitneo runs on llama-server: its binary, the GGUF, translated flags, a default window
out=$(PATH="$fakebin:$PATH" bash "$script" --id whiterabbitneo 2>&1 || true)
check '[[ "$out" == *"llama-server "*"WhiteRabbitNeo-2.5-Qwen-2.5-Coder-7B-Q8_0.gguf"* ]]' "whiterabbitneo should start llama-server on the GGUF, got: $out"
check '[[ "$out" == *" -c 8192 --alias whiterabbitneo-2.5-7b --jinja"* ]]' "whiterabbitneo default flags missing, got: $out"
out=$(PATH="$fakebin:$PATH" bash "$script" --id whiterabbitneo --ctx 4096 --threads 3 --name wr 2>&1 || true)
check '[[ "$out" == *" -t 3 -c 4096 --alias wr --jinja"* ]]' "lm-serve flags were not translated, got: $out"
# NEGATIVE: lm-serve-only flags are refused for llama-server, before ssh
for bad in --fast "--cache-mb 64"; do
  read -ra words <<< "$bad"
  out=$(PATH="$fakebin:$PATH" bash "$script" --id whiterabbitneo "${words[@]}" 2>&1 || true)
  check '[[ "$out" == *"not supported for the llama-server model"* ]]' "--id whiterabbitneo $bad should be refused, got: $out"
done
# NEGATIVE: the lm-serve models never get the llama-server binary
out=$(PATH="$fakebin:$PATH" bash "$script" --id qwen --ctx 4096 2>&1 || true)
check '[[ "$out" == *"/bin/lm-serve "*" qwen3.5-4b lm-serve model.q8.safetensors '"''"' --ctx 4096"* ]]' "qwen should keep lm-serve and its flags, got: $out"
# gemma-uncensored: llama-server on the Q6_K_P GGUF, with the vision projector as its own word
out=$(PATH="$fakebin:$PATH" bash "$script" --id gemma-uncensored 2>&1 || true)
check '[[ "$out" == *"llama-server "*" gemma-4-e4b-uncensored llama-server Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q6_K_P.gguf mmproj-Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-f16.gguf -c 8192 --alias gemma-4-e4b-uncensored --jinja"* ]]' "gemma-uncensored should start llama-server on the GGUF with its projector, got: $out"
out=$(PATH="$fakebin:$PATH" bash "$script" --id gemma-uncensored --ctx 16384 --name g 2>&1 || true)
check '[[ "$out" == *" -c 16384 --alias g --jinja"* ]]' "gemma-uncensored flags were not translated, got: $out"
# NEGATIVE: lm-serve-only flags are refused for it too; the other llama-server model has no projector
out=$(PATH="$fakebin:$PATH" bash "$script" --id gemma-uncensored --fast 2>&1 || true)
check '[[ "$out" == *"not supported for the llama-server model"* ]]' "--id gemma-uncensored --fast should be refused, got: $out"
# --think / --no-think become --reasoning on / off
out=$(PATH="$fakebin:$PATH" bash "$script" --id gemma-uncensored --no-think --ctx 4096 2>&1 || true)
check '[[ "$out" == *" --reasoning off -c 4096 --alias gemma-4-e4b-uncensored --jinja"* ]]' "--no-think was not translated, got: $out"
out=$(PATH="$fakebin:$PATH" bash "$script" --id gemma-uncensored --think 2>&1 || true)
check '[[ "$out" == *" --reasoning on -c 8192 "* ]]' "--think was not translated, got: $out"
out=$(PATH="$fakebin:$PATH" bash "$script" --id whiterabbitneo 2>&1 || true)
check '[[ "$out" != *mmproj* ]]' "whiterabbitneo must not get a projector, got: $out"
rm -rf "$fakebin"

if [[ "$fails" -ne 0 ]]; then
  echo "test_start_remote_lm_serve: $fails of $checks checks FAILED"
  exit 1
fi
echo "test_start_remote_lm_serve: $checks checks passed"
