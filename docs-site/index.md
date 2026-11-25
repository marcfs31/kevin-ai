# What Kevin AI is

Kevin AI is a **private, local-first** AI assistant: the chat UI runs on your machine, conversations stay in the browser, and the language model runs on a small home server (an OptiPlex) reached through SSH tunnels. There is no telemetry and no third-party script in the page — the Content-Security-Policy is `'self'` only.

## Three parts

| Part | What it is | Where it lives |
| --- | --- | --- |
| **Engine** | C99 inference (`lm-serve`) or llama.cpp (`llama-server`) | Model host, port **8080** |
| **Tools** | MCP servers, web search, projects | `tools/chat_tools.py` on **8081** (host) and optionally **8082** (Mac) |
| **Chat UI** | React app | Vite in **dev** (`5173`) or `tools/chat_server.py` in **prod** (`8000`) |

Interactive map of the repository: [gitdiagram.com/marcfs31/kevin-ai](https://gitdiagram.com/marcfs31/kevin-ai) (GitDiagram — AI overview; verify details against this guide and the source).

## Read next

1. [System overview](/guide/overview) — how the pieces connect  
2. [Vite vs chat_server](/guide/vite-and-chat-server) — which UI server to open  
3. [How the APIs are built](/guide/apis) — no framework; three hand-rolled HTTP layers  
4. [How models run](/guide/models) — engines, IDs, tunnels  
5. [Run the demo on the Mac](/guide/run-on-mac) — ports and start commands  

The long-form product and install story remains in the [root README](https://github.com/marcfs31/kevin-ai/blob/main/README.md). Engine roadmap: [`docs/PLAN.md`](https://github.com/marcfs31/kevin-ai/blob/main/docs/PLAN.md).
