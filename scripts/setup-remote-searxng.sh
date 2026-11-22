#!/usr/bin/env bash
# Install and start a SearXNG instance on the machine that runs the model, for the chat's web search.
# It listens on 127.0.0.1 only and answers the JSON format the tool service asks for; queries go from
# it to the search engines and nowhere else. Needs scripts/install-tool-runtimes.sh first (uv).
# In Settings > Tools and internet the search address is then http://127.0.0.1:<port> (the address
# is used by the tool service, which runs on the same machine).
# The packages come from scripts/searxng-requirements.txt (exact versions, from the environment that was
# tested); regenerate it with `uv pip freeze` in ~/opt/searxng-venv when the pinned commit moves.
set -euo pipefail

SEARXNG_COMMIT=c6ee3e9fd0235004a694e63720efc869bb1e3594

usage() {
  cat <<'EOF2' >&2
usage: setup-remote-searxng.sh [--port N]

  --port N   port on the remote host (default 8888)

Environment (optional):
  LM_SEARXNG_SSH   default toor@192.168.1.96
  LM_TOOLS_OPT     where install-tool-runtimes.sh put uv, default ~/opt

Clones github.com/searxng/searxng at a pinned commit into ~/opt/searxng, installs it into
~/opt/searxng-venv with uv (Python 3.12) from the pinned list scripts/searxng-requirements.txt, writes ~/.config/searxng/settings.yml once (loopback,
a random secret key, no limiter, JSON enabled), and (re)starts it; the log is ~/logs/searxng.log.
EOF2
  exit 2
}

main() {
  local port=8888
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --port)
        [[ $# -ge 2 && "$2" =~ ^[0-9]+$ && "$2" -ge 1024 && "$2" -le 65535 ]] || { echo "setup-remote-searxng.sh: --port needs a number from 1024 to 65535" >&2; exit 2; }
        port=$2; shift 2 ;;
      --help|-h) usage ;;
      *) echo "setup-remote-searxng.sh: unknown argument $1" >&2; usage ;;
    esac
  done
  local target=${LM_SEARXNG_SSH:-toor@192.168.1.96}
  local opt=${LM_TOOLS_OPT:-'$HOME/opt'}
  local remote_script
  remote_script=$(cat <<'EOS'
set -euo pipefail
expand_home() { local v=$1; if [[ "$v" == '$HOME/'* ]]; then printf '%s\n' "$HOME/${v#'$HOME/'}"; else printf '%s\n' "$v"; fi; }
OPT=$(expand_home "$1"); PORT=$2; COMMIT=$3; LOCK=$4
export PATH="$OPT/tool-bin:$PATH"
command -v uv >/dev/null || { echo "uv not found in $OPT/tool-bin: run scripts/install-tool-runtimes.sh first" >&2; exit 1; }
mkdir -p "$HOME/.config/searxng" "$HOME/logs"
[[ -d "$OPT/searxng/.git" ]] || git clone -q https://github.com/searxng/searxng.git "$OPT/searxng"
git -C "$OPT/searxng" fetch -q origin "$COMMIT" 2>/dev/null || true
git -C "$OPT/searxng" checkout -q "$COMMIT"
uv venv -q --python 3.12 "$OPT/searxng-venv"
PY="$OPT/searxng-venv/bin/python"
LOCKFILE=$(mktemp)
printf '%s\n' "$LOCK" >"$LOCKFILE"
uv pip install -q --python "$PY" -r "$LOCKFILE"      # every package at the exact version that was tested
(cd "$OPT/searxng" && uv pip install -q --python "$PY" --no-deps --no-build-isolation -e .)
rm -f "$LOCKFILE"
CFG="$HOME/.config/searxng/settings.yml"
if [[ ! -f "$CFG" ]]; then
  KEY=$(python3 -c 'import secrets; print(secrets.token_hex(32))')
  umask 077
  cat >"$CFG" <<EOF3
use_default_settings: true
general:
  instance_name: "local"
  enable_metrics: false
server:
  bind_address: "127.0.0.1"
  port: $PORT
  secret_key: "$KEY"
  limiter: false
  public_instance: false
search:
  formats:
    - html
    - json
EOF3
fi
pkill -f "searx.webapp" || true
for _ in $(seq 1 25); do pgrep -f "searx.webapp" >/dev/null || break; sleep 0.2; done
cd "$OPT/searxng"
SEARXNG_SETTINGS_PATH="$CFG" nohup "$PY" -m searx.webapp >"$HOME/logs/searxng.log" 2>&1 </dev/null &
disown
for _ in $(seq 1 60); do
  if curl -sf --connect-timeout 1 "http://127.0.0.1:$PORT/healthz" >/dev/null; then echo "SearXNG up on 127.0.0.1:$PORT (log ~/logs/searxng.log)"; exit 0; fi
  pgrep -f "searx.webapp" >/dev/null || { echo "searxng exited:" >&2; tail -n 20 "$HOME/logs/searxng.log" >&2; exit 1; }
  sleep 1
done
echo "searxng did not answer in time" >&2; tail -n 20 "$HOME/logs/searxng.log" >&2; exit 1
EOS
)
  local quoted
  local lock_file; lock_file="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/searxng-requirements.txt"
  [[ -f "$lock_file" ]] || { echo "setup-remote-searxng.sh: missing $lock_file" >&2; exit 1; }
  quoted=$(printf '%q %q %q %q' "$opt" "$port" "$SEARXNG_COMMIT" "$(cat "$lock_file")")
  exec ssh -o BatchMode=yes -o ConnectTimeout=10 "$target" "bash -s -- $quoted" <<<"$remote_script"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
