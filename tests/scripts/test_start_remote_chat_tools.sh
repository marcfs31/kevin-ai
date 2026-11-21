#!/usr/bin/env bash
# start-remote-chat-tools.sh: argument checks and the command it would send. No SSH connection is opened.
set -u
root=$(cd "$(dirname "$0")/../.." && pwd)
script=$root/scripts/start-remote-chat-tools.sh
fails=0; checks=0
check() { checks=$((checks + 1)); if ! eval "$1"; then fails=$((fails + 1)); echo "  FAIL $2"; fi; }

fakebin=$(mktemp -d)
trap 'find "$fakebin" -delete' EXIT
cat >"$fakebin/ssh" <<'EOS'
#!/usr/bin/env bash
printf '%s\n' "$*"
EOS
chmod +x "$fakebin/ssh"
run() { PATH="$fakebin:$PATH" bash "$script" "$@" 2>&1 || true; }

out=$(run)
check '[[ "$out" == *"-o BatchMode=yes"*"toor@192.168.1.96"*"bash -s --"* ]]' "default target missing, got: $out"
check '[[ "$out" == *"bash -s -- "*"small-llm-in-c "*"opt "*"8081 0"* ]]' "defaults should be the repo, ~/opt, port 8081 and no update, got: $out"
out=$(run --update --port 8123)
check '[[ "$out" == *" 8123 1"* ]]' "--update --port 8123 not passed on, got: $out"
out=$(LM_TOOLS_SSH=me@host run)
check '[[ "$out" == *"me@host"* ]]' "LM_TOOLS_SSH ignored, got: $out"
# NEGATIVE: bad ports and unknown arguments are refused before ssh
for bad in "--port" "--port abc" "--port 80" "--port 70000" "--frobnicate"; do
  read -ra words <<< "$bad"
  out=$(run "${words[@]}")
  check '[[ "$out" != *"BatchMode"* ]]' "$bad should be refused before ssh, got: $out"
done
# the remote script keeps the old service from surviving and never uses sudo
check 'grep -q "pkill -f" "$script" && ! grep -q "sudo" "$script"' "the script should replace the running service without sudo"

if [ "$fails" -ne 0 ]; then echo "test_start_remote_chat_tools: $fails of $checks checks FAILED"; exit 1; fi
echo "test_start_remote_chat_tools: $checks checks passed"
