#!/usr/bin/env bash
# Build the chat web app on this machine and copy web/dist to the server.
#   scripts/deploy-web.sh
# Overridable: DEPLOY_HOST (toor@192.168.1.96), DEPLOY_PATH (~/small-llm-in-c/web/dist/).
# The server needs only Python 3 to run tools/chat_server.py; Node stays on this machine.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

host="${DEPLOY_HOST:-toor@192.168.1.96}"
path="${DEPLOY_PATH:-~/small-llm-in-c/web/dist/}"

(cd web && npm ci && npm run build)
test -f web/dist/index.html || { echo "deploy-web: web/dist/index.html missing after the build" >&2; exit 1; }

# The remote path is expanded by the remote shell (~), so it is passed unquoted on purpose.
ssh "$host" "mkdir -p $path"
echo "deploy-web: copying web/dist/ to $host:$path"
rsync -av --delete --itemize-changes web/dist/ "$host:$path"
echo "deploy-web: done. On the server: python3 tools/chat_server.py --upstream http://127.0.0.1:8080"
