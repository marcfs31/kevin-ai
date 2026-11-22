#!/usr/bin/env bash
# Start a second copy of the tool service (tools/chat_tools.py) on this Mac, on 127.0.0.1:8082, for the
# projects and tools that must run here instead of on the model server. The chat reaches it at /tools-local
# (dev proxy: LM_TOOLS_LOCAL_URL; tools/chat_server.py: --tools-local-upstream http://127.0.0.1:8082).
# It has its own config and state, because ~/.config/lm-chat/mcp.json is shared with other tools.
set -euo pipefail

usage() {
  cat <<'EOF2' >&2
usage: start-local-chat-tools.sh [--port N] [--dry-run]
       start-local-chat-tools.sh --stop

  (none)     start the service, replacing one this script started earlier
  --stop     stop the one this script started (pid file), nothing else
  --port N   loopback port (default 8082)
  --dry-run  print the command that would run and exit; nothing is started or stopped

Environment (optional):
  LM_LOCAL_TOOLS_CONFIG  default ~/.config/kevin-ai/mcp-local.json
  LM_LOCAL_TOOLS_STATE   default ~/.local/share/kevin-ai (pid file chat-tools-local.pid, log chat-tools-local.log)
EOF2
  exit 2
}

main() {
  local stop=0 dry=0 port=8082
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --stop) stop=1; shift ;;
      --dry-run) dry=1; shift ;;
      --port)
        [[ $# -ge 2 && "$2" =~ ^[0-9]+$ && "$2" -ge 1024 && "$2" -le 65535 ]] || { echo "start-local-chat-tools.sh: --port needs a number from 1024 to 65535" >&2; exit 2; }
        port=$2; shift 2 ;;
      --help|-h) usage ;;
      *) echo "start-local-chat-tools.sh: unknown argument $1" >&2; usage ;;
    esac
  done
  local root config state pidfile log
  root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
  config=${LM_LOCAL_TOOLS_CONFIG:-$HOME/.config/kevin-ai/mcp-local.json}
  state=${LM_LOCAL_TOOLS_STATE:-$HOME/.local/share/kevin-ai}
  pidfile=$state/chat-tools-local.pid
  log=$state/chat-tools-local.log
  local cmd=(python3 "$root/tools/chat_tools.py" --host 127.0.0.1 --port "$port" --config "$config" --state-dir "$state")

  if [[ "$dry" == 1 ]]; then
    if [[ "$stop" == 1 ]]; then echo "stop: pid file $pidfile"; else printf '%q ' "${cmd[@]}"; echo; fi
    return 0
  fi

  stop_old() {
    [[ -f "$pidfile" ]] || return 0
    local pid
    pid=$(<"$pidfile")
    # Only a process that really is this service: a stale pid file may name an unrelated process.
    if [[ "$pid" =~ ^[0-9]+$ ]] && ps -p "$pid" -o command= 2>/dev/null | grep -q "tools/chat_tools.py"; then
      kill "$pid" 2>/dev/null || true
      for _ in $(seq 1 50); do kill -0 "$pid" 2>/dev/null || break; sleep 0.2; done
      if kill -0 "$pid" 2>/dev/null; then echo "the old local tool service (pid $pid) did not stop" >&2; exit 1; fi
    fi
    rm -f "$pidfile"
  }

  if [[ "$stop" == 1 ]]; then stop_old; echo "local tool service stopped"; return 0; fi

  mkdir -p "$state" "$(dirname "$config")"
  [[ -f "$root/tools/chat_tools.py" ]] || { echo "missing $root/tools/chat_tools.py" >&2; exit 1; }
  stop_old
  nohup "${cmd[@]}" >"$log" 2>&1 &
  local pid=$!
  echo "$pid" >"$pidfile"
  for _ in $(seq 1 50); do
    if curl -sf --connect-timeout 1 "http://127.0.0.1:$port/tools/status" >/dev/null; then
      echo "local tool service up on 127.0.0.1:$port (pid $pid, log $log)"
      return 0
    fi
    kill -0 "$pid" 2>/dev/null || { echo "chat_tools.py exited:" >&2; tail -n 20 "$log" >&2 || true; rm -f "$pidfile"; exit 1; }
    sleep 0.2
  done
  echo "chat_tools.py did not answer in time:" >&2; tail -n 20 "$log" >&2 || true
  exit 1
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
