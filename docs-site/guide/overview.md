# System overview

Kevin AI is split so the heavy model stays on a small Linux box while the Mac only runs the browser UI and optional local tools.

## Big picture

```mermaid
flowchart TB
  subgraph Mac["Mac (browser + proxies)"]
    browser["Browser"]
    vite["Vite :5173<br/>dev UI + proxy"]
    chatserver["chat_server.py :8000<br/>prod UI + proxy"]
    localtools["chat_tools.py :8082<br/>projects on this Mac"]
    t8080["SSH -L 8080"]
    t8081["SSH -L 8081"]
  end

  subgraph Host["Model host (OptiPlex)"]
    serve["lm-serve or llama-server<br/>:8080"]
    remotetools["chat_tools.py<br/>:8081"]
    models["~/models/*"]
  end

  browser --> vite
  browser --> chatserver
  vite --> t8080
  vite --> t8081
  vite --> localtools
  chatserver --> t8080
  chatserver --> t8081
  chatserver --> localtools
  t8080 --> serve
  t8081 --> remotetools
  serve --> models
```

## What talks to what

| From | Path on the page origin | Upstream |
| --- | --- | --- |
| Chat UI | `/v1/*`, `/health` | Model on `127.0.0.1:8080` (tunnel) |
| Chat UI | `/tools/*` | Remote tools on `127.0.0.1:8081` (tunnel) |
| Chat UI | `/tools-local/*` | Local tools on `127.0.0.1:8082` (rewritten to `/tools/`) |

The browser never opens those upstream ports directly. Vite or `chat_server.py` proxy same-origin so the CSP can stay `'self'`.

## Repository map (where to edit)

| Change | Where |
| --- | --- |
| Chat UI, styles, i18n | `web/src` |
| Tools, MCP, search, projects | `tools/chat_tools.py`, `tools/chat_projects.py`, … |
| Static proxy + CSP + gzip | `tools/chat_server.py` |
| Start model / tools | `scripts/start-remote-lm-serve.sh`, `start-*-chat-tools.sh` |
| C engine | `src/`, `bin/lm-serve` |

GitDiagram of the tree: [marcfs31/kevin-ai](https://gitdiagram.com/marcfs31/kevin-ai).
