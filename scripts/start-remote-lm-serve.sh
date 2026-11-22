#!/usr/bin/env bash
# Stop the remote model server (if any) and start one model: Qwen, Qwen uncensored or Gemma on
# lm-serve, or a GGUF the engine cannot run on llama.cpp's llama-server: WhiteRabbitNeo 2.5
# (Qwen2.5-Coder-7B) and Gemma 4 E4B Uncensored (with its vision projector). One model at a time: the 11 GB host cannot hold two. For lm-serve, extra flags
# are forwarded as-is; for llama-server the lm-serve flags it has an equivalent for are
# translated and the others refused. Omit a flag to keep the server's own default.
# The local SSH tunnel to port 8080 is left alone — only the remote process is replaced.
set -euo pipefail

usage() {
  cat <<'EOF' >&2
usage: start-remote-lm-serve.sh --id qwen|qwen-uncensored|gemma|whiterabbitneo|gemma-uncensored [flags…]

  --id qwen|qwen-uncensored|gemma|whiterabbitneo|gemma-uncensored
                    which model under ~/models on the remote host
  Remaining arguments are lm-serve flags (for example --ctx 8192 --think).
  whiterabbitneo and gemma-uncensored run on llama-server and take only --ctx (default 8192),
  --threads, --seed, --max-tokens, --timeout, --name and --think / --no-think (--reasoning on / off;
  Gemma thinks before it answers unless told not to).
  --host and --port are always 127.0.0.1 and 8080 so the local tunnel keeps working.

Environment (optional):
  LM_SERVE_SSH              default toor@192.168.1.96
  LM_SERVE_REMOTE_BIN       default ~/kevin-ai/bin/lm-serve
  LM_SERVE_REMOTE_LLAMA_BIN default ~/opt/llama.cpp-v1.0.0/build/bin/llama-server
  LM_SERVE_REMOTE_ROOT      default ~/models
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

# lm-serve flags -> llama-server flags, into the array LLAMA_ARGS. Unknown or unsupported
# flags are refused rather than dropped. 8 GB of Q8_0 weights leave room for an 8192 window.
LLAMA_DEFAULT_CTX=8192
translate_for_llama() {
  local ctx="" name="" v
  LLAMA_ARGS=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --ctx|--threads|--seed|--max-tokens|--timeout|--name)
        [[ $# -ge 2 ]] || { echo "start-remote-lm-serve.sh: $1 needs a value" >&2; exit 2; }
        v=$2
        case "$1" in
          --ctx) ctx=$v ;;
          --threads) LLAMA_ARGS+=(-t "$v") ;;
          --seed) LLAMA_ARGS+=(--seed "$v") ;;
          --max-tokens) LLAMA_ARGS+=(-n "$v") ;;
          --timeout) LLAMA_ARGS+=(--timeout "$v") ;;
          --name) name=$v ;;
        esac
        shift 2
        ;;
      --think) LLAMA_ARGS+=(--reasoning on); shift ;;
      --no-think) LLAMA_ARGS+=(--reasoning off); shift ;;
      *)
        echo "start-remote-lm-serve.sh: $1 is not supported for the llama-server model" >&2
        exit 2
        ;;
    esac
  done
  LLAMA_ARGS+=(-c "${ctx:-$LLAMA_DEFAULT_CTX}" --alias "${name:-$DIR_NAME}" --jinja)
}

main() {
  local ID="" DIR_NAME RUNTIME WEIGHTS MMPROJ="" SSH_TARGET REMOTE_BIN REMOTE_ROOT quoted
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

  RUNTIME=lm-serve
  WEIGHTS=model.q8.safetensors
  case "$ID" in
    qwen) DIR_NAME=qwen3.5-4b ;;
    qwen-uncensored) DIR_NAME=qwen3.5-4b-uncensored ;;
    gemma) DIR_NAME=gemma-4-e4b ;;
    whiterabbitneo)
      DIR_NAME=whiterabbitneo-2.5-7b
      RUNTIME=llama-server
      WEIGHTS=WhiteRabbitNeo-2.5-Qwen-2.5-Coder-7B-Q8_0.gguf
      ;;
    gemma-uncensored)
      # Q6_K_P: about Q8 quality in 5.8 GB, so the window and the 1 GB projector fit in 11 GB.
      DIR_NAME=gemma-4-e4b-uncensored
      RUNTIME=llama-server
      WEIGHTS=Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q6_K_P.gguf
      MMPROJ=mmproj-Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-f16.gguf
      ;;
    *) echo "start-remote-lm-serve.sh: --id must be qwen, qwen-uncensored, gemma, whiterabbitneo or gemma-uncensored" >&2; usage ;;
  esac
  if [[ "$RUNTIME" == llama-server ]]; then
    translate_for_llama ${EXTRA[@]+"${EXTRA[@]}"}
    EXTRA=("${LLAMA_ARGS[@]}")
  fi

  SSH_TARGET=${LM_SERVE_SSH:-toor@192.168.1.96}
  if [[ "$RUNTIME" == llama-server ]]; then
    REMOTE_BIN=${LM_SERVE_REMOTE_LLAMA_BIN:-'$HOME/opt/llama.cpp-v1.0.0/build/bin/llama-server'}
  else
    REMOTE_BIN=${LM_SERVE_REMOTE_BIN:-'$HOME/kevin-ai/bin/lm-serve'}
  fi
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
RUNTIME=$4
WEIGHTS_NAME=$5
MMPROJ_NAME=$6
shift 6
DIR="$ROOT/$DIR_NAME"
WEIGHTS="$DIR/$WEIGHTS_NAME"
LOG=${HOME}/logs/$RUNTIME.log
mkdir -p "$(dirname "$LOG")"
if [[ ! -x "$BIN" ]]; then echo "missing $RUNTIME at $BIN" >&2; exit 1; fi
if [[ ! -d "$DIR" ]]; then echo "missing model dir $DIR" >&2; exit 1; fi
if [[ ! -f "$WEIGHTS" ]]; then echo "missing weights $WEIGHTS" >&2; exit 1; fi
MM=()
if [[ -n "$MMPROJ_NAME" ]]; then
  if [[ ! -f "$DIR/$MMPROJ_NAME" ]]; then echo "missing vision projector $DIR/$MMPROJ_NAME" >&2; exit 1; fi
  MM=(--mmproj "$DIR/$MMPROJ_NAME")
fi
# Stop whichever server holds port 8080 (and its memory) before starting the new one.
for proc in lm-serve llama-server; do
  if pgrep -x "$proc" >/dev/null; then
    pkill -x "$proc" || true
    for _ in $(seq 1 50); do
      pgrep -x "$proc" >/dev/null || break
      sleep 0.2
    done
    if pgrep -x "$proc" >/dev/null; then
      echo "$proc did not stop" >&2
      exit 1
    fi
  fi
done
if [[ "$RUNTIME" == llama-server ]]; then
  nohup "$BIN" -m "$WEIGHTS" ${MM[@]+"${MM[@]}"} --host 127.0.0.1 --port 8080 "$@" >"$LOG" 2>&1 &
else
  nohup "$BIN" --model "$DIR" --weights "$WEIGHTS" --host 127.0.0.1 --port 8080 "$@" >"$LOG" 2>&1 &
fi
# Wait until /health answers (model load can take a while; llama-server answers 503 until loaded).
ok=0
for _ in $(seq 1 180); do
  if curl -sf --connect-timeout 1 http://127.0.0.1:8080/health >/dev/null; then
    ok=1
    break
  fi
  # If the process died, surface the log.
  if ! pgrep -x "$RUNTIME" >/dev/null; then
    echo "$RUNTIME exited before becoming healthy:" >&2
    tail -n 40 "$LOG" >&2 || true
    exit 1
  fi
  sleep 1
done
if [[ "$ok" -ne 1 ]]; then
  echo "$RUNTIME did not become healthy in time:" >&2
  tail -n 40 "$LOG" >&2 || true
  exit 1
fi
curl -sf http://127.0.0.1:8080/health
echo
EOS
)

  args=("$REMOTE_BIN" "$REMOTE_ROOT" "$DIR_NAME" "$RUNTIME" "$WEIGHTS" "$MMPROJ")
  if [[ ${#EXTRA[@]} -gt 0 ]]; then
    args+=("${EXTRA[@]}")
  fi
  quoted=$(quote_for_remote "${args[@]}")

  exec ssh -o BatchMode=yes -o ConnectTimeout=10 "$SSH_TARGET" "bash -s -- $quoted" <<<"$remote_script"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
