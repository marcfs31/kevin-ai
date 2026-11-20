#!/usr/bin/env bash
# Stop the remote lm-serve (if any) and start one for Qwen or Gemma.
# Extra flags are forwarded to lm-serve as-is; omit a flag to keep lm-serve's own default.
# The local SSH tunnel to port 8080 is left alone — only the remote process is replaced.
set -euo pipefail

usage() {
  cat <<'EOF' >&2
usage: start-remote-lm-serve.sh --id qwen|gemma [lm-serve flags…]

  --id qwen|gemma   which checkpoint under ~/models on the remote host
  Remaining arguments are passed to lm-serve (for example --ctx 8192 --think).
  --host and --port are always 127.0.0.1 and 8080 so the local tunnel keeps working.

Environment (optional):
  LM_SERVE_SSH          default toor@192.168.1.96
  LM_SERVE_REMOTE_BIN   default ~/small-llm-in-c/bin/lm-serve
  LM_SERVE_REMOTE_ROOT  default ~/models
EOF
  exit 2
}

ID=""
EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --id)
      [[ $# -ge 2 ]] || usage
      ID=$2
      shift 2
      ;;
    --host|--port)
      echo "start-remote-lm-serve.sh: refusing $1 (tunnel expects 127.0.0.1:8080)" >&2
      exit 2
      ;;
    --help|-h) usage ;;
    --)
      shift
      EXTRA+=("$@")
      break
      ;;
    *)
      EXTRA+=("$1")
      shift
      ;;
  esac
done

case "$ID" in
  qwen) DIR_NAME=qwen3.5-4b ;;
  gemma) DIR_NAME=gemma-4-e4b ;;
  *) echo "start-remote-lm-serve.sh: --id must be qwen or gemma" >&2; usage ;;
esac

SSH_TARGET=${LM_SERVE_SSH:-toor@192.168.1.96}
REMOTE_BIN=${LM_SERVE_REMOTE_BIN:-'$HOME/small-llm-in-c/bin/lm-serve'}
REMOTE_ROOT=${LM_SERVE_REMOTE_ROOT:-'$HOME/models'}

# Build a remote argv array as a NUL-safe bash -c script.
# shellcheck disable=SC2016
remote_cmd=$(cat <<'EOS'
set -euo pipefail
BIN=__BIN__
ROOT=__ROOT__
DIR="$ROOT/__DIR__"
WEIGHTS="$DIR/model.q8.safetensors"
LOG=${HOME}/logs/lm-serve.log
mkdir -p "$(dirname "$LOG")"
if [[ ! -x "$BIN" ]]; then echo "missing lm-serve at $BIN" >&2; exit 1; fi
if [[ ! -d "$DIR" ]]; then echo "missing model dir $DIR" >&2; exit 1; fi
if [[ ! -f "$WEIGHTS" ]]; then echo "missing weights $WEIGHTS" >&2; exit 1; fi
# Stop any previous serve before starting the new one.
if pgrep -x lm-serve >/dev/null; then
  pkill -x lm-serve || true
  for _ in $(seq 1 30); do
    pgrep -x lm-serve >/dev/null || break
    sleep 0.2
  done
  if pgrep -x lm-serve >/dev/null; then
    echo "lm-serve did not stop" >&2
    exit 1
  fi
fi
nohup "$BIN" --model "$DIR" --weights "$WEIGHTS" --host 127.0.0.1 --port 8080 __EXTRA__ >"$LOG" 2>&1 &
# Wait until /health answers (model load can take a while).
ok=0
for _ in $(seq 1 120); do
  if curl -sf --connect-timeout 1 http://127.0.0.1:8080/health >/dev/null; then
    ok=1
    break
  fi
  # If the process died, surface the log.
  if ! pgrep -x lm-serve >/dev/null; then
    echo "lm-serve exited before becoming healthy:" >&2
    tail -n 40 "$LOG" >&2 || true
    exit 1
  fi
  sleep 1
done
if [[ "$ok" -ne 1 ]]; then
  echo "lm-serve did not become healthy in time:" >&2
  tail -n 40 "$LOG" >&2 || true
  exit 1
fi
curl -sf http://127.0.0.1:8080/health
echo
EOS
)

# Quote each extra flag for remote bash.
extra_q=
for a in "${EXTRA[@]+"${EXTRA[@]}"}"; do
  extra_q+=$(printf '%q ' "$a")
done

remote_cmd=${remote_cmd//__BIN__/$REMOTE_BIN}
remote_cmd=${remote_cmd//__ROOT__/$REMOTE_ROOT}
remote_cmd=${remote_cmd//__DIR__/$DIR_NAME}
remote_cmd=${remote_cmd//__EXTRA__/$extra_q}

exec ssh -o BatchMode=yes -o ConnectTimeout=10 "$SSH_TARGET" "bash -s" <<<"$remote_cmd"
