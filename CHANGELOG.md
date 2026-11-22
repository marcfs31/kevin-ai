# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-11-22

### Fixed
- The thinking text is drawn as Markdown like the answer (bold, italics, lists, maths), not as raw text with asterisks. There a single-asterisk span (*Total weight:*) is drawn bold, as models mean it; the answer keeps the standard italics. LaTeX written as \\( ... \\) and \\[ ... \\] is drawn like the dollar forms, in the thinking text and the answer.

### Added
- Web chat: LaTeX in replies (`$...$` inline, `$$...$$` as a block) is drawn as MathML with KaTeX, which needs no stylesheet or fonts so the CSP is unchanged; a price such as "$5 to $10" stays text. Adds `remark-math` and `rehype-katex` (MIT); the bundle grows by about 83 KB gzipped.

## [0.2.29] - 2026-11-22

### Changed
- Manifests (`web/package.json`, `web/package-lock.json`) and this file brought up to date with the versions v0.2.1 to v0.2.28, which were tagged afterwards: each tag points at the last commit of its pull request, whose manifests still read 0.2.0.

### Summary of 0.2.1 to 0.2.28
#### Added
- Tool service (`tools/chat_tools.py`): an MCP client (stdio and Streamable HTTP) on the model server with a pinned catalog (filesystem, memory, sequential thinking, time, git, fetch), lazy start, idle stop, a cap on running servers and a memory floor, and an optional web search through the user's own SearXNG. `tools/chat_server.py` forwards `/tools/*` to it.
- Web chat: MCP tools with a token budget and `find_tools`, approval cards (Allow once, Always allow, Deny), a Tools and internet panel in Settings (servers, per-tool switches, budget with the number of tools that fit, approval mode, SearXNG address with a test), and a header badge while the internet is on.
- Web chat: effort (thinking and answer length) and speed (image size, when the chat is compacted) per chat, next to the model picker; redirecting a reply while it streams.
- Web chat: the context bar shows the tool definitions sent and which ones; previews of attached files before and after sending; the model is told when to search and when to answer from its own knowledge.
- `scripts/install-tool-runtimes.sh`, `scripts/start-remote-chat-tools.sh` and `scripts/setup-remote-searxng.sh` (SearXNG installed from `scripts/searxng-requirements.txt`).
- `make test` runs the tool service's Python tests and the web chat's lint, typecheck, tests and build.

#### Changed
- Web chat defaults: effort High and speed Thorough (thinking on, answers up to 4,096 tokens, images at 1,024 px, compaction at 90% of the window); temperature 0.8, image size 1,024 px, tools on; at most 50 tool steps per message (default 20, was 8).
- The tool budget counts only the tools beyond the always-sent ones; the per-step cap is 30 tools.
- A read-only tool that reaches the network (fetch) now asks first; "always allow" is stored as `server/tool` (entries saved under the old names ask again).
- The tool service lists a stopped server's tools from the list kept at its last start, so opening the chat no longer starts every enabled server.

#### Fixed
- The reading countdown counts only what the server has not cached, and only for the same chat, model and tools.
- A cross-site GET can no longer start a tool server; tool requests work when the chat is served on a LAN address; web search no longer accepts the service itself or an unlisted private address as its target (`--search-allow`).
- The approval card takes focus itself, not the Allow once button, and cuts each argument on its own so no key is hidden.
- The Settings backdrop no longer turns opaque under the pointer.

## [0.2.28] - 2026-11-22

### Added
- Effort High and speed Thorough are the defaults ([#100](https://github.com/marcfs31/small-llm-in-c/pull/100)).

## [0.2.27] - 2026-11-22

### Fixed
- Review findings in approvals, the countdown, the budget and always-allow, plus changelog (F4, F5, F6, F7, F8, F12, F13, F15) ([#99](https://github.com/marcfs31/small-llm-in-c/pull/99)).

## [0.2.26] - 2026-11-22

### Fixed
- Review findings in the tool service, the proxy, CI and the SearXNG install (F1, F2, F3, F7, F9, F10, F11, F14) ([#98](https://github.com/marcfs31/small-llm-in-c/pull/98)).

## [0.2.25] - 2026-11-22

### Added
- Tell the model when to search and when to answer from knowledge ([#97](https://github.com/marcfs31/small-llm-in-c/pull/97)).

## [0.2.24] - 2026-11-22

### Added
- New defaults (0.8, 1,024 px, tools on) and preview of attached files ([#96](https://github.com/marcfs31/small-llm-in-c/pull/96)).

## [0.2.23] - 2026-11-22

### Fixed
- The modal backdrop no longer turns opaque when the pointer is over it ([#95](https://github.com/marcfs31/small-llm-in-c/pull/95)).

## [0.2.22] - 2026-11-22

### Added
- The tool size slider says how many tools fit; up to 30 per step ([#94](https://github.com/marcfs31/small-llm-in-c/pull/94)).

## [0.2.21] - 2026-11-22

### Added
- Allow up to 50 tool steps per message, default 20 ([#93](https://github.com/marcfs31/small-llm-in-c/pull/93)).

## [0.2.20] - 2026-11-22

### Added
- The context bar shows the tool definitions and which tools were sent ([#92](https://github.com/marcfs31/small-llm-in-c/pull/92)).

## [0.2.19] - 2026-11-22

### Fixed
- The reading countdown counts only what the server has not cached ([#91](https://github.com/marcfs31/small-llm-in-c/pull/91)).

## [0.2.18] - 2026-11-21

### Added
- Set up SearXNG on the model server for the chat's web search ([#90](https://github.com/marcfs31/small-llm-in-c/pull/90)).

## [0.2.17] - 2026-11-21

### Fixed
- Mark the catalog servers verified on the model server and drop sqlite ([#89](https://github.com/marcfs31/small-llm-in-c/pull/89)).

## [0.2.16] - 2026-11-21

### Added
- Install Node and uv for the catalog servers and start the tool service on the server ([#88](https://github.com/marcfs31/small-llm-in-c/pull/88)).

## [0.2.15] - 2026-11-21

### Added
- Settings panel for MCP tools and the internet ([#87](https://github.com/marcfs31/small-llm-in-c/pull/87)).

## [0.2.14] - 2026-11-21

### Added
- MCP tools with a token budget, find_tools, approvals and optional web search ([#84](https://github.com/marcfs31/small-llm-in-c/pull/84)).

## [0.2.13] - 2026-11-21

### Added
- A local tool service with an MCP client, a pinned catalog and web search ([#83](https://github.com/marcfs31/small-llm-in-c/pull/83)).

## [0.2.12] - 2026-11-21

### Added
- Effort and speed modes next to the model picker ([#82](https://github.com/marcfs31/small-llm-in-c/pull/82)).

## [0.2.11] - 2026-11-21

### Added
- Warn on Uncensored model choices ([#81](https://github.com/marcfs31/small-llm-in-c/pull/81)).

## [0.2.10] - 2026-11-21

### Added
- Open Settings as a visible modal ([#80](https://github.com/marcfs31/small-llm-in-c/pull/80)).

## [0.2.9] - 2026-11-21

### Added
- Show prompt-read ETA in minutes and seconds ([#79](https://github.com/marcfs31/small-llm-in-c/pull/79)).

## [0.2.8] - 2026-11-21

### Added
- Delete messages and rewind the chat ([#78](https://github.com/marcfs31/small-llm-in-c/pull/78)).

## [0.2.7] - 2026-11-21

### Changed
- How to run the Vite web chat ([#77](https://github.com/marcfs31/small-llm-in-c/pull/77)).

## [0.2.6] - 2026-11-21

### Added
- Redirect a reply while it streams ([#76](https://github.com/marcfs31/small-llm-in-c/pull/76)).

## [0.2.5] - 2026-11-21

### Fixed
- Update the context bar and chat totals while a reply streams ([#75](https://github.com/marcfs31/small-llm-in-c/pull/75)).

## [0.2.4] - 2026-11-21

### Added
- Add Gemma 4 E4B Uncensored with vision (Q6_K_P on llama-server) ([#74](https://github.com/marcfs31/small-llm-in-c/pull/74)).

## [0.2.3] - 2026-11-21

### Fixed
- Drop the "Images: available" line ([#73](https://github.com/marcfs31/small-llm-in-c/pull/73)).

## [0.2.2] - 2026-11-21

### Fixed
- Deleting a chat no longer looks like a dead server; start a model when it is down ([#72](https://github.com/marcfs31/small-llm-in-c/pull/72)).

## [0.2.1] - 2026-11-21

### Added
- Chat stats bar, auto and manual compact, open chat list during a reply, Retry ([#71](https://github.com/marcfs31/small-llm-in-c/pull/71)).

## [0.2.0] - 2026-11-21

### Added
- Web chat: live token counter (tokens read and written per reply and per chat, estimates marked until the server's usage arrives).
- Web chat: compacting of older turns into a model-written summary, automatically near the context limit (also between tool steps) or with the Compact button.
- Web chat: per-conversation model picker over `/v1/models`, showing each context length.
- Web chat: optional tool loop (calculator, current time) with a collapsible trace, a step cap (default 8) and Stop.

## [0.1.0] - 2026-11-21

### Added
- Web chat app for lm-serve (Vite, React, TypeScript) with streaming, attachments, English and Spanish, and the `tools/chat_server.py` proxy.
