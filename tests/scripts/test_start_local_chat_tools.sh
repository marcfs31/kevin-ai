#!/usr/bin/env bash
# start-local-chat-tools.sh: arguments and the command it would run (--dry-run). No process is started.
set -u
root=$(cd "$(dirname "$0")/../.." && pwd)
script=$root/scripts/start-local-chat-tools.sh
fails=0; checks=0
check() { checks=$((checks + 1)); if ! eval "$1"; then fails=$((fails + 1)); echo "  FAIL $2"; fi; }

home=$(mktemp -d)
trap 'find "$home" -delete' EXIT
run() { HOME=$home bash "$script" "$@" 2>&1 || true; }

out=$(run --dry-run)
check '[[ "$out" == *"tools/chat_tools.py"*"--host 127.0.0.1"*"--port 8082"* ]]' "default is loopback 8082, got: $out"
check '[[ "$out" == *"--config $home/.config/kevin-ai/mcp-local.json"* ]]' "own config path, not the shared lm-chat one, got: $out"
check '[[ "$out" == *"--state-dir $home/.local/share/kevin-ai"* ]]' "state dir, got: $out"
check '[[ "$out" != *"lm-chat"* ]]' "must not touch ~/.config/lm-chat, got: $out"
check '[[ "$out" == *"--projects-base $home/KevinProjects --projects-registry $home/.config/kevin-ai/projects-local.json"* ]]' "projects base and its own registry, got: $out"
out=$(LM_LOCAL_PROJECTS_BASE= run --dry-run)
check '[[ "$out" != *"--projects-base"* ]]' "an empty base should turn projects off, got: $out"
out=$(run --dry-run --port 8123)
check '[[ "$out" == *"--port 8123"* ]]' "--port not passed on, got: $out"
out=$(LM_LOCAL_TOOLS_CONFIG=/x/c.json LM_LOCAL_TOOLS_STATE=/x/s run --dry-run)
check '[[ "$out" == *"--config /x/c.json"*"--state-dir /x/s"* ]]' "environment overrides ignored, got: $out"
out=$(run --dry-run --stop)
check '[[ "$out" == "stop: pid file $home/.local/share/kevin-ai/chat-tools-local.pid" ]]' "--stop should name the pid file, got: $out"
# NEGATIVE: never listens beyond loopback, bad ports and unknown arguments are refused, dry-run creates nothing
check '[[ "$(run --dry-run)" != *"0.0.0.0"* ]]' "must stay on loopback"
for bad in "--port" "--port abc" "--port 80" "--port 70000" "--frobnicate"; do
  read -ra words <<< "$bad"
  out=$(run --dry-run "${words[@]}")
  check '[[ "$out" != *"chat_tools.py"* && "$out" == *"usage"* || "$out" == *"--port needs"* ]]' "$bad should be refused, got: $out"
done
check '[ -z "$(ls -A "$home")" ]' "dry-run must not create files"
# --stop must only kill a process whose command line is the service, never an unrelated pid from a stale file
check 'grep -q "grep -q \"tools/chat_tools.py\"" "$script" && ! grep -q "sudo\|pkill\|killall" "$script"' "stop should verify the pid and not use pkill or sudo"

if [ "$fails" -ne 0 ]; then echo "test_start_local_chat_tools: $fails of $checks checks FAILED"; exit 1; fi
echo "test_start_local_chat_tools: $checks checks passed"
