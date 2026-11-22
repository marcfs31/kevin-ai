#!/usr/bin/env bash
# Lighthouse on the production build of the web chat, the way it is really served (tools/chat_server.py,
# strict CSP, gzip): builds web/, serves it on a free local port, runs Lighthouse for desktop and mobile
# and exits 1 unless every category scores 100. Needs Node, python3 and Google Chrome (CHROME to point
# at another binary). Measure this, not the Vite dev server: that one sends 78 unbundled files.
set -euo pipefail

LIGHTHOUSE=lighthouse@13.6.0
case "${1:-}" in
  -h|--help) echo "usage: scripts/lighthouse.sh   (env: CHROME=/path/to/chrome, PORT=8130)"; exit 0 ;;
  "") ;;
  *) echo "lighthouse.sh: unknown argument $1 (see --help)" >&2; exit 2 ;;
esac
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
chrome=${CHROME:-"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"}
[[ -x "$chrome" ]] || chrome=$(command -v google-chrome || command -v chromium || true)
[[ -n "$chrome" && -x "$chrome" ]] || { echo "lighthouse.sh: no Chrome found; set CHROME" >&2; exit 1; }
port=${PORT:-8130}
work=$(mktemp -d)
cleanup() {
  if [[ -n "${server:-}" ]]; then kill "$server" 2>/dev/null || true; wait "$server" 2>/dev/null || true; fi
  find "$work" -delete
}
trap cleanup EXIT

[[ -d "$root/web/node_modules" ]] || npm ci --prefix "$root/web" --ignore-scripts --no-audit --no-fund
npm --prefix "$root/web" run build >/dev/null
python3 "$root/tools/chat_server.py" --port "$port" --dist "$root/web/dist" >"$work/server.log" 2>&1 &
server=$!
for _ in $(seq 1 25); do curl -sf -o /dev/null "http://127.0.0.1:$port/" && break; sleep 0.2; done

status=0
for mode in desktop mobile; do
  flag=--form-factor=mobile; [[ $mode == desktop ]] && flag=--preset=desktop
  npx -y "$LIGHTHOUSE" "http://127.0.0.1:$port/" "$flag" --chrome-path="$chrome" \
    --chrome-flags="--headless=new --no-sandbox" --output=json --output-path="$work/$mode.json" --quiet >/dev/null 2>&1
  python3 - "$work/$mode.json" "$mode" <<'PYEOF' || status=1
import json, sys
d = json.load(open(sys.argv[1]))
scores = {c["title"]: round((c["score"] or 0) * 100) for c in d["categories"].values()}
print(sys.argv[2] + ": " + ", ".join(f"{k} {v}" for k, v in scores.items()))
sys.exit(0 if all(v == 100 for v in scores.values()) else 1)
PYEOF
done
exit $status
