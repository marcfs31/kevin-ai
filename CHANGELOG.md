# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Tool service (`tools/chat_tools.py`): an MCP client (stdio and Streamable HTTP) on the model server with a pinned catalog (filesystem, memory, sequential thinking, time, git, fetch), lazy start, idle stop, a cap on running servers and a memory floor, and an optional web search through the user's own SearXNG. `tools/chat_server.py` forwards `/tools/*` to it.
- Web chat: MCP tools with a token budget and `find_tools`, approval cards (Allow once, Always allow, Deny), a Tools and internet panel in Settings (servers, per-tool switches, budget with the number of tools that fit, approval mode, SearXNG address with a test), and a header badge while the internet is on.
- Web chat: effort (thinking and answer length) and speed (image size, when the chat is compacted) per chat, next to the model picker; redirecting a reply while it streams.
- Web chat: the context bar shows the tool definitions sent and which ones; previews of attached files before and after sending; the model is told when to search and when to answer from its own knowledge.
- `scripts/install-tool-runtimes.sh`, `scripts/start-remote-chat-tools.sh` and `scripts/setup-remote-searxng.sh` (SearXNG installed from `scripts/searxng-requirements.txt`).
- `make test` runs the tool service's Python tests and the web chat's lint, typecheck, tests and build.

### Changed
- Web chat defaults: effort High and speed Thorough (thinking on, answers up to 4,096 tokens, images at 1,024 px, compaction at 90% of the window); temperature 0.8, image size 1,024 px, tools on; at most 50 tool steps per message (default 20, was 8).
- The tool budget counts only the tools beyond the always-sent ones; the per-step cap is 30 tools.
- A read-only tool that reaches the network (fetch) now asks first; "always allow" is stored as `server/tool` (entries saved under the old names ask again).
- The tool service lists a stopped server's tools from the list kept at its last start, so opening the chat no longer starts every enabled server.

### Fixed
- The reading countdown counts only what the server has not cached, and only for the same chat, model and tools.
- A cross-site GET can no longer start a tool server; tool requests work when the chat is served on a LAN address; web search no longer accepts the service itself or an unlisted private address as its target (`--search-allow`).
- The approval card takes focus itself, not the Allow once button, and cuts each argument on its own so no key is hidden.
- The Settings backdrop no longer turns opaque under the pointer.

## [0.2.0] - 2026-11-21

### Added
- Web chat: live token counter (tokens read and written per reply and per chat, estimates marked until the server's usage arrives).
- Web chat: compacting of older turns into a model-written summary, automatically near the context limit (also between tool steps) or with the Compact button.
- Web chat: per-conversation model picker over `/v1/models`, showing each context length.
- Web chat: optional tool loop (calculator, current time) with a collapsible trace, a step cap (default 8) and Stop.

## [0.1.0] - 2026-11-21

### Added
- Web chat app for lm-serve (Vite, React, TypeScript) with streaming, attachments, English and Spanish, and the `tools/chat_server.py` proxy.
