#!/usr/bin/env bash
# Stop the remote lm-serve (if any) and start one for Qwen, Qwen uncensored or Gemma.
# Extra flags are forwarded to lm-serve as-is; omit a flag to keep lm-serve's own default.
# The local SSH tunnel to port 8080 is left alone — only the remote process is replaced.
set -euo pipefail

usage() {
  cat <<'EOF' >&2
usage: start-remote-lm-serve.sh --id qwen|qwen-uncensored|gemma [lm-serve flags…]

  --id qwen|qwen-uncensored|gemma
                    which checkpoint under ~/models on the remote host
  Remaining arguments are passed to lm-serve (for example --ctx 8192 --think).
  --host and --port are always 127.0.0.1 and 8080 so the local tunnel keeps working.

Environment (optional):
  LM_SERVE_SSH          default toor@192.168.1.96
  LM_SERVE_REMOTE_BIN   default ~/small-llm-in-c/bin/lm-serve
  LM_SERVE_REMOTE_ROOT  default ~/models
EOF
  exit 2
}

# One shell word per argument, for the remote login shell. The result is parsed
# once, as the remote command, and is not run through another substitution.
quote_for_remote() {
  local a out=
  for a in "$@"; do
    out+=$(printf '%q ' "$a")
  done
  printf '%s' "$out"
}

main() {
  local ID="" DIR_NAME SSH_TARGET REMOTE_BIN REMOTE_ROOT quoted
  local -a EXTRA=() args=()
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
    qwen-uncensored) DIR_NAME=qwen3.5-4b-uncensored ;;
    gemma) DIR_NAME=gemma-4-e4b ;;
    *) echo "start-remote-lm-serve.sh: --id must be qwen, qwen-uncensored or gemma" >&2; usage ;;
  esac

  SSH_TARGET=${LM_SERVE_SSH:-toor@192.168.1.96}
  REMOTE_BIN=${LM_SERVE_REMOTE_BIN:-'$HOME/small-llm-in-c/bin/lm-serve'}
  REMOTE_ROOT=${LM_SERVE_REMOTE_ROOT:-'$HOME/models'}

  # Values travel as positional parameters. A leading $HOME/ is expanded on the
  # remote host, after the login shell has parsed each quoted word.
  local remote_script
  remote_script=$(cat <<'EOS'
set -euo pipefail
expand_home() {
  local v=$1 prefix='$HOME/'
  if [[ "$v" == "$prefix"* ]]; then
    printf '%s\n' "$HOME/${v#"$prefix"}"
  else
    printf '%s\n' "$v"
  fi
}
BIN=$(expand_home "$1")
ROOT=$(expand_home "$2")
DIR_NAME=$3
shift 3
DIR="$ROOT/$DIR_NAME"
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
nohup "$BIN" --model "$DIR" --weights "$WEIGHTS" --host 127.0.0.1 --port 8080 "$@" >"$LOG" 2>&1 &
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

  args=("$REMOTE_BIN" "$REMOTE_ROOT" "$DIR_NAME")
  if [[ ${#EXTRA[@]} -gt 0 ]]; then
    args+=("${EXTRA[@]}")
  fi
  quoted=$(quote_for_remote "${args[@]}")

  exec ssh -o BatchMode=yes -o ConnectTimeout=10 "$SSH_TARGET" "bash -s -- $quoted" <<<"$remote_script"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
