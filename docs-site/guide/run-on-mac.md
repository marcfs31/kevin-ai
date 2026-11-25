# Run the demo on the Mac

Checkout: typically `/Users/marcfors/Documents/Kevin AI`, branch `main`, current tag.

## Prerequisites

- Node 24+ (`web/.nvmrc`)
- Python 3
- SSH to the model host (`toor@192.168.1.96` by default)
- Model already loaded or startable via `scripts/start-remote-lm-serve.sh`

## Ports

| Port | Role |
| --- | --- |
| 8080 | Model (SSH tunnel) |
| 8081 | Remote tools (SSH tunnel) |
| 8082 | Local tools (`start-local-chat-tools.sh`) |
| 5173 | Vite (dev UI) |
| 8000 | `chat_server.py` (prod UI) |

## Start sequence

```bash
cd "/Users/marcfors/Documents/Kevin AI"
git fetch origin && git checkout main && git pull --ff-only origin main

# 1. Model tunnel (if not already up) — do not kill a healthy one
ssh -N -o ExitOnForwardFailure=yes -L 8080:127.0.0.1:8080 toor@192.168.1.96 &

# 2. Tools tunnel + remote tool service
ssh -N -o ExitOnForwardFailure=yes -L 8081:127.0.0.1:8081 toor@192.168.1.96 &
./scripts/start-remote-chat-tools.sh --update

# 3. Local tools (optional, for “(on this Mac)” projects)
./scripts/start-local-chat-tools.sh

# 4a. Vite (dev — can start models from Settings)
cd web && npm ci && npm run dev
# open http://127.0.0.1:5173/

# 4b. Or production UI (separate terminal; no single script starts both)
cd web && npm ci && npm run build && cd ..
python3 tools/chat_server.py \
  --host 127.0.0.1 --port 8000 --dist web/dist \
  --upstream http://127.0.0.1:8080 \
  --tools-upstream http://127.0.0.1:8081 \
  --tools-local-upstream http://127.0.0.1:8082
# open http://127.0.0.1:8000/
```

**There is no single Makefile target that starts Vite and chat_server together.** Use the two commands above (4a and/or 4b).

## This docs site

```bash
cd docs-site
npm ci
npm run docs:dev    # http://127.0.0.1:5173 by default — use another port if chat Vite already owns 5173
```

If the chat Vite is already on 5173, run `npx vitepress dev --port 5174` (or set it in config).

## Stop without touching the model

```bash
# UI
# Ctrl-C Vite / chat_server, or kill the tmux sessions that hold them

./scripts/start-local-chat-tools.sh --stop
# Optional: stop only the 8081 tunnel, not 8080
pkill -f '8081:127.0.0.1:8081'
```

Leave `8080` and the loaded `lm-serve` / `llama-server` alone unless you intend to change models.
