#!/usr/bin/env bash
# Start the tool service (tools/chat_tools.py) on the machine that runs the model, replacing a
# running one. It listens on 127.0.0.1:8081 there; reach it from this Mac through a second SSH
# forward, next to the one for the model (the existing tunnel is left alone):
#     ssh -N -L 8081:127.0.0.1:8081 toor@192.168.1.96
# Node and uv for the catalog servers come from scripts/install-tool-runtimes.sh (~/opt/tool-bin).
set -euo pipefail

usage() {
  cat <<'EOF2' >&2
usage: start-remote-chat-tools.sh [--update] [--port N]

  --update   git pull --ff-only in the repository on the remote host first
  --port N   port on the remote host (default 8081; the tunnel must forward the same one)

Environment (optional):
  LM_TOOLS_SSH     default toor@192.168.1.96
  LM_TOOLS_REPO    default ~/small-llm-in-c on the remote host
  LM_TOOLS_OPT     where install-tool-runtimes.sh put the runtimes, default ~/opt
EOF2
  exit 2
}

quote_for_remote() {
  local a out=
  for a in "$@"; do out+=$(printf '%q ' "$a"); done
  printf '%s' "$out"
}

main() {
  local update=0 port=8081
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --update) update=1; shift ;;
      --port)
        [[ $# -ge 2 && "$2" =~ ^[0-9]+$ && "$2" -ge 1024 && "$2" -le 65535 ]] || { echo "start-remote-chat-tools.sh: --port needs a number from 1024 to 65535" >&2; exit 2; }
        port=$2; shift 2 ;;
      --help|-h) usage ;;
      *) echo "start-remote-chat-tools.sh: unknown argument $1" >&2; usage ;;
    esac
  done
  local target=${LM_TOOLS_SSH:-toor@192.168.1.96}
  local repo=${LM_TOOLS_REPO:-'$HOME/small-llm-in-c'}
  local opt=${LM_TOOLS_OPT:-'$HOME/opt'}

  local remote_script
  remote_script=$(cat <<'EOS'
set -euo pipefail
expand_home() { local v=$1; if [[ "$v" == '$HOME/'* ]]; then printf '%s\n' "$HOME/${v#'$HOME/'}"; else printf '%s\n' "$v"; fi; }
REPO=$(expand_home "$1"); OPT=$(expand_home "$2"); PORT=$3; UPDATE=$4
LOG=$HOME/logs/chat-tools.log
mkdir -p "$(dirname "$LOG")"
if [[ "$UPDATE" == 1 ]]; then git -C "$REPO" pull --ff-only -q; fi
[[ -f "$REPO/tools/chat_tools.py" ]] || { echo "missing $REPO/tools/chat_tools.py (run with --update, or merge the tool service first)" >&2; exit 1; }
if pgrep -f "tools/chat_tools.py" >/dev/null; then
  pkill -f "tools/chat_tools.py" || true
  for _ in $(seq 1 50); do pgrep -f "tools/chat_tools.py" >/dev/null || break; sleep 0.2; done
  if pgrep -f "tools/chat_tools.py" >/dev/null; then echo "the old tool service did not stop" >&2; exit 1; fi
fi
PATH="$OPT/tool-bin:$PATH" nohup python3 "$REPO/tools/chat_tools.py" --host 127.0.0.1 --port "$PORT" >"$LOG" 2>&1 &
ok=0
for _ in $(seq 1 50); do
  if curl -sf --connect-timeout 1 "http://127.0.0.1:$PORT/tools/status" >/dev/null; then ok=1; break; fi
  pgrep -f "tools/chat_tools.py" >/dev/null || { echo "chat_tools.py exited:" >&2; tail -n 20 "$LOG" >&2 || true; exit 1; }
  sleep 0.2
done
[[ "$ok" == 1 ]] || { echo "chat_tools.py did not answer in time:" >&2; tail -n 20 "$LOG" >&2 || true; exit 1; }
echo "tool service up on 127.0.0.1:$PORT (log $LOG)"
EOS
)
  local quoted
  quoted=$(quote_for_remote "$repo" "$opt" "$port" "$update")
  exec ssh -o BatchMode=yes -o ConnectTimeout=10 "$target" "bash -s -- $quoted" <<<"$remote_script"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
