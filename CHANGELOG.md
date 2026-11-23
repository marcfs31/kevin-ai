# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.25] - 2026-11-23

### Changed
- `AGENTS.md`: Kevin's SemVer. A minor holds 31 releases (patches 0 to 30): the release that would be the 31st patch (`0.3.31`) starts the next minor (`0.4.0`) instead. A major holds 13 minors (0 to 12): the release that would be the 13th minor (`0.13.0`) is the first major, `1.0.0`, which ships alone.

## [0.3.23] - 2026-11-23

### Added
- Edit on your own messages (those without a file or image): it rewinds the chat to that message, asking first when replies follow, and puts its text back in the message box to change and send again. Delete from here stays for when you want to drop the message.

## [0.3.22] - 2026-11-23

### Added
- The chat's stats are in a pane on the right of the chat, which can be hidden for a wider chat (Hide, and a "Show chat stats" button beside Compact chat) and is remembered (`statsPane`: `auto` shows it on wide screens only, `open`, `closed`). On a phone it opens in place of the chat.
- A tools viewer in that pane: the tools sent with the last request grouped by where they come from (built in, then one group per server) as chips, with the total and the token estimate.

### Changed
- The context bar is a slim strip with a legend under it (prompt, tools, last reply, free), which fits the narrow column.

## [0.3.21] - 2026-11-23

### Fixed
- The model picker button showed a light fill and a large triangle when the pointer was over it: the shared button hover sets the whole `background` shorthand, which reset the arrow's size and position, so its two gradients stretched over the button. The hover rule now states the image, position, size and repeat (shared as `--chevron-*` variables), and a test guards it. Found by the owner on the real page; the 0.3.20 check had looked only at the resting state.

## [0.3.20] - 2026-11-23

### Changed
- The model picker in the header looks like a button: the same bordered box and arrow as the Effort and Speed selects, with the status dot inside it, so the dot stays with its text when the row wraps on a phone. The panel it opens is unchanged.

## [0.3.19] - 2026-11-23

### Fixed
- Project git tools: the first commit of a new repository. `git_commit` and `git_push` asked git for the branch with a command that fails before the first commit, so a project made with `git_init` could never get one ("HEAD is not on a branch"). Found in the first live run on the model server. The branch is now read with `symbolic-ref`, so the default-branch rule applies as it should and a `kevin/` branch can make the first commit.

## [0.3.18] - 2026-11-23

### Added
- Projects on this Mac: Settings > Tools has a switch (`localHost`, off by default) to also use the tool service on this Mac. The Project picker then lists its projects as "(on this Mac)", a chat remembers the host of its project (`projectHost`, `server` for older chats), and that project's tools, git chip and workflows come from `/tools-local`; the approval card says "Mac". With the switch off the page never calls `/tools-local`.

## [0.3.17] - 2026-11-23

### Added
- Workflows: a project keeps step recipes in `.kevin/workflows.json`; a Workflow bar lists them when a project is chosen, a started workflow becomes a checklist saved with the chat (`workflow`, `null` for older chats), the model is told the steps and ticks them with `workflow_step` (done, skipped or blocked, with a note; never asks for approval), and the user can tick, untick or stop it. Limits: 20 workflows, 12 steps of 200 characters.

## [0.3.16] - 2026-11-23

### Changed
- `AGENTS.md`: where to edit the project tools and the project picker, and three rules for agents: never loosen what the agent can do in a project (jail, no force or delete, default branch, token only through askpass, fixed argv, audit, a negative test per new tool), never make the page ask for something that can fail on load, and never put the registry or token file into the repository, logs or fixtures.

## [0.3.15] - 2026-11-23

### Added
- `scripts/start-remote-chat-tools.sh` starts the tool service with `--projects-base` (default `/mnt/models/kevin-projects`, `LM_PROJECTS_BASE`, empty turns it off) and `--projects-min-free-gb` (default 20, `LM_PROJECTS_MIN_FREE_GB`); `scripts/start-local-chat-tools.sh` does the same on the Mac (`~/KevinProjects`, its own registry `~/.config/kevin-ai/projects-local.json`).
- Both scripts load `~/.config/kevin-ai/tool-service.env` (refused unless its mode is 600) so the GitHub token never sits in the repository, the command line or the browser.

### Changed
- README: "The five models" table (Qwen3.5-4B, Qwen3.5-4B Uncensored, Gemma 4 E4B, WhiteRabbitNeo 2.5 7B, Gemma 4 E4B Uncensored, each with its id, runtime, files and context), all five named in the intro and the web chat section, an owner guide for projects and the token, and a corrected note on vision.

## [0.3.14] - 2026-11-23

### Changed
- README: the chat, the tool service and projects are part of the overview, with hand-made diagrams (system overview, project approval flow, chat and tool service, one message with a tool call, repository layout) in place of the generated GitDiagram one, and updated Status, projects, build and test, web chat, releases and troubleshooting sections.

## [0.3.13] - 2026-11-23

### Added
- Tool service: `GET /tools/projects` lists the registered projects (ids, titles, branch rules, command names; no paths).
- A **Project** picker per chat, stored with the conversation (`projectId`, `null` for older chats). With a project chosen the model is told the project's rules and commands, every project tool call carries that project's id whatever the model wrote, and a chip shows the branch and the changed files.
- The approval card shows what the tool does; for an allowlisted command that is the exact command that will run.

## [0.3.12] - 2026-11-22

### Added
- Project server: remote git and pull requests (`git_clone`, `git_fetch`, `git_pull` fast-forward only, `git_push`, `pr_create` through the GitHub REST API). The registry takes `remote` (an https URL) and `tokenEnv`; the agent never names a URL, remote or token, and there is no force or delete option. The token goes to git only through a `GIT_ASKPASS` helper, is scrubbed from every result, and a repository that rewrites URLs is refused.

## [0.3.11] - 2026-11-22

### Security
- `katex` is pinned to 0.18.2 with an npm override (`rehype-katex` 7.0.1 still asks for 0.16): fixes the Dependabot advisory on trust-restriction bypass through prototype pollution.

## [0.3.10] - 2026-11-22

### Added
- Project server: allowlisted commands per project (`run_<project>_<name>`, argv fixed by the owner in the registry, no shell, minimal environment, private HOME, time limit killing the process group, output tail) and an audit log of every change (`--audit-log`, JSONL, private, no contents, rotates at 5 MB).

## [0.3.9] - 2026-11-22

### Added
- Project server: local git tools (`git_status`, `git_diff`, `git_log`, `git_show`, `git_branches`, `git_branch_create`, `git_switch`, `git_add`, `git_commit`, `git_restore`, `git_init`), hardened against the repository's own configuration (no hooks, fsmonitor, external diff, editor, pager or credential helper; no user or system git config). Commits on the default branch are refused unless the registry allows it, the committer is the registry identity, and every path goes through the jail. The registry takes `identity`, `defaultBranch` and `allowMain`.

### Changed
- The path jail and shared helpers moved to `tools/chat_jail.py`.

## [0.3.8] - 2026-11-22

### Changed
- The web package is named `kevin-ai` (was `lm-chat-web`) in `web/package.json` and `web/package-lock.json`.

## [0.3.7] - 2026-11-22

### Added
- A second tool-service host, the Mac: `/tools-local/*` in the Vite dev proxy (`LM_TOOLS_LOCAL_URL`, default `http://127.0.0.1:8082`) and in `tools/chat_server.py` (`--tools-local-upstream`; 404 `tools_local_not_configured` when unset; the same Origin check, Sec-Fetch-Site passing and no credentials as `/tools/*`; the prefix becomes `/tools/` upstream).
- Every `tools*` function in `web/src/lib/api.ts` takes an optional host (`'server' | 'local'`, default `'server'`).
- `scripts/start-local-chat-tools.sh` starts the tool service on this Mac (127.0.0.1:8082, own config and state, pid file, `--stop`).

## [0.3.6] - 2026-11-22

### Added
- Tool service: a built-in **project** server (`--projects-base`, `tools/chat_projects.py`) with file tools for the agent (`project_list`, `fs_read`, `fs_list`, `fs_search`, `fs_write`, `fs_edit`) on projects listed in an owner-edited registry. Paths are relative and jailed to the project (no `..`, no symlink out, nothing under `.git`), writes are capped and refused when the disk is short of free space, and the server runs in the service's process so it does not use up the MCP running limit. First step of projects, git and workflows for the agent.

## [0.3.5] - 2026-11-22

### Added
- `AGENTS.md` (and `CLAUDE.md` pointing to it): the release rule (every merged PR is a tagged version), the gate, the Lighthouse rule and where to edit ([#109](https://github.com/marcfs31/kevin-ai/pull/109)).

## [0.3.4] - 2026-11-22

### Changed
- The select arrows (effort, speed and every other select) are drawn inside the box with room around them instead of against its edge, and the items of the header row are separated by a thin rule instead of a dot; the dot that sat inside the "Internet on" pill is gone. On a phone the separators give way to spacing.
- The Settings, Attach files and Attach image buttons have a gear, a file and an image icon (drawn inline, nothing to load), and attached files that are not images show the file icon in their row.

## [0.3.3] - 2026-11-22

### Added
- `scripts/lighthouse.sh`: builds the web chat, serves it with `tools/chat_server.py` and runs Lighthouse (desktop and mobile, pinned version); it fails unless every category is 100.
- `tools/chat_server.py` gzips text files (HTML, JS, CSS, JSON, SVG) for browsers that accept it and keeps the compressed bytes until the file changes: the 791 kB script is 237 kB on the wire.

### Fixed
- On a phone the model row keeps its stacked height while it connects, so the page no longer jumps (Lighthouse mobile layout shift 0.17).
- The robots.txt that Lighthouse read as blocking indexing is removed (SEO 66 to 100).

## [0.3.2] - 2026-11-22

### Changed
- Type scale with real steps (13 px interface text, 16 px section titles, 20 px dialog titles, 25 px for the Kevin AI name), the Settings dialog is set apart by its border and the backdrop without a wide shadow, and the model button has 8 px of side padding.

### Fixed
- The page no longer jumps down when the model row loads (the header keeps the height it needs), which Lighthouse scored as a layout shift of 0.2 on desktop.
- The page has a description for search engines.

## [0.3.1] - 2026-11-22

### Changed
- The chat is named Kevin AI, with an Orbit logo (a coral disc with a K and an orbiting dot, `web/public/logo.svg`) in the header and as the tab icon; the page title and the app name are Kevin AI in English and Spanish. Every mention of the old project name is replaced: the README, NOTICE, scripts and service unit now say Kevin AI, and the checkout, deploy path and repository are `kevin-ai`. The README names Marc Fors i Soler as the creator.

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
- Effort High and speed Thorough are the defaults ([#100](https://github.com/marcfs31/kevin-ai/pull/100)).

## [0.2.27] - 2026-11-22

### Fixed
- Review findings in approvals, the countdown, the budget and always-allow, plus changelog (F4, F5, F6, F7, F8, F12, F13, F15) ([#99](https://github.com/marcfs31/kevin-ai/pull/99)).

## [0.2.26] - 2026-11-22

### Fixed
- Review findings in the tool service, the proxy, CI and the SearXNG install (F1, F2, F3, F7, F9, F10, F11, F14) ([#98](https://github.com/marcfs31/kevin-ai/pull/98)).

## [0.2.25] - 2026-11-22

### Added
- Tell the model when to search and when to answer from knowledge ([#97](https://github.com/marcfs31/kevin-ai/pull/97)).

## [0.2.24] - 2026-11-22

### Added
- New defaults (0.8, 1,024 px, tools on) and preview of attached files ([#96](https://github.com/marcfs31/kevin-ai/pull/96)).

## [0.2.23] - 2026-11-22

### Fixed
- The modal backdrop no longer turns opaque when the pointer is over it ([#95](https://github.com/marcfs31/kevin-ai/pull/95)).

## [0.2.22] - 2026-11-22

### Added
- The tool size slider says how many tools fit; up to 30 per step ([#94](https://github.com/marcfs31/kevin-ai/pull/94)).

## [0.2.21] - 2026-11-22

### Added
- Allow up to 50 tool steps per message, default 20 ([#93](https://github.com/marcfs31/kevin-ai/pull/93)).

## [0.2.20] - 2026-11-22

### Added
- The context bar shows the tool definitions and which tools were sent ([#92](https://github.com/marcfs31/kevin-ai/pull/92)).

## [0.2.19] - 2026-11-22

### Fixed
- The reading countdown counts only what the server has not cached ([#91](https://github.com/marcfs31/kevin-ai/pull/91)).

## [0.2.18] - 2026-11-21

### Added
- Set up SearXNG on the model server for the chat's web search ([#90](https://github.com/marcfs31/kevin-ai/pull/90)).

## [0.2.17] - 2026-11-21

### Fixed
- Mark the catalog servers verified on the model server and drop sqlite ([#89](https://github.com/marcfs31/kevin-ai/pull/89)).

## [0.2.16] - 2026-11-21

### Added
- Install Node and uv for the catalog servers and start the tool service on the server ([#88](https://github.com/marcfs31/kevin-ai/pull/88)).

## [0.2.15] - 2026-11-21

### Added
- Settings panel for MCP tools and the internet ([#87](https://github.com/marcfs31/kevin-ai/pull/87)).

## [0.2.14] - 2026-11-21

### Added
- MCP tools with a token budget, find_tools, approvals and optional web search ([#84](https://github.com/marcfs31/kevin-ai/pull/84)).

## [0.2.13] - 2026-11-21

### Added
- A local tool service with an MCP client, a pinned catalog and web search ([#83](https://github.com/marcfs31/kevin-ai/pull/83)).

## [0.2.12] - 2026-11-21

### Added
- Effort and speed modes next to the model picker ([#82](https://github.com/marcfs31/kevin-ai/pull/82)).

## [0.2.11] - 2026-11-21

### Added
- Warn on Uncensored model choices ([#81](https://github.com/marcfs31/kevin-ai/pull/81)).

## [0.2.10] - 2026-11-21

### Added
- Open Settings as a visible modal ([#80](https://github.com/marcfs31/kevin-ai/pull/80)).

## [0.2.9] - 2026-11-21

### Added
- Show prompt-read ETA in minutes and seconds ([#79](https://github.com/marcfs31/kevin-ai/pull/79)).

## [0.2.8] - 2026-11-21

### Added
- Delete messages and rewind the chat ([#78](https://github.com/marcfs31/kevin-ai/pull/78)).

## [0.2.7] - 2026-11-21

### Changed
- How to run the Vite web chat ([#77](https://github.com/marcfs31/kevin-ai/pull/77)).

## [0.2.6] - 2026-11-21

### Added
- Redirect a reply while it streams ([#76](https://github.com/marcfs31/kevin-ai/pull/76)).

## [0.2.5] - 2026-11-21

### Fixed
- Update the context bar and chat totals while a reply streams ([#75](https://github.com/marcfs31/kevin-ai/pull/75)).

## [0.2.4] - 2026-11-21

### Added
- Add Gemma 4 E4B Uncensored with vision (Q6_K_P on llama-server) ([#74](https://github.com/marcfs31/kevin-ai/pull/74)).

## [0.2.3] - 2026-11-21

### Fixed
- Drop the "Images: available" line ([#73](https://github.com/marcfs31/kevin-ai/pull/73)).

## [0.2.2] - 2026-11-21

### Fixed
- Deleting a chat no longer looks like a dead server; start a model when it is down ([#72](https://github.com/marcfs31/kevin-ai/pull/72)).

## [0.2.1] - 2026-11-21

### Added
- Chat stats bar, auto and manual compact, open chat list during a reply, Retry ([#71](https://github.com/marcfs31/kevin-ai/pull/71)).

## [0.2.0] - 2026-11-21

### Added
- Web chat: live token counter (tokens read and written per reply and per chat, estimates marked until the server's usage arrives).
- Web chat: compacting of older turns into a model-written summary, automatically near the context limit (also between tool steps) or with the Compact button.
- Web chat: per-conversation model picker over `/v1/models`, showing each context length.
- Web chat: optional tool loop (calculator, current time) with a collapsible trace, a step cap (default 8) and Stop.

## [0.1.0] - 2026-11-21

### Added
- Web chat app for lm-serve (Vite, React, TypeScript) with streaming, attachments, English and Spanish, and the `tools/chat_server.py` proxy.
