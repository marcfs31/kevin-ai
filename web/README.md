# Chat web app for lm-serve

The app is called **Kevin AI**; its logo (Orbit) is `public/logo.svg`, used in the header and as the tab icon.

A browser chat for `bin/lm-serve`: streaming replies with a Stop button, conversations kept
only in this browser, attachments (text and code files, PDFs, images when the server can read
them), English and Spanish. No accounts, no telemetry, no CDN: every byte the page loads
comes from the same server, and a strict Content-Security-Policy makes that checkable.

## Run it

Development (Node 24, see `.nvmrc`), with lm-serve on `127.0.0.1:8080`:

```bash
cd web && npm ci && npm run dev        # http://127.0.0.1:5173, /v1 and /health go to LM_SERVE_URL
```

`LM_SERVE_URL` (default `http://127.0.0.1:8080`) points the dev proxy elsewhere, for example at
an SSH tunnel: `ssh -N -L 8080:127.0.0.1:8080 toor@192.168.1.96`.

In Settings, **Start Qwen3.5-4B** / **Start Qwen3.5-4B Uncensored** / **Start Gemma 4 E4B** / **Start WhiteRabbitNeo 2.5 7B** / **Start Gemma 4 E4B Uncensored** call `POST /control/serve` on the Vite
dev server, which runs `scripts/start-remote-lm-serve.sh`. That script SSHs to the machine that
holds the weights (`LM_SERVE_SSH`, default `toor@192.168.1.96`), stops any existing `lm-serve`,
starts the chosen model on `127.0.0.1:8080`, and leaves the local tunnel alone. Optional fields
map to lm-serve flags (`--max-tokens`, `--seed`, `--ctx`, `--threads`, `--timeout`, `--cache-mb`,
`--name`, `--think`, `--fast`); leave them empty for lm-serve’s own defaults. Host and port are
not exposed — they stay `127.0.0.1:8080` for the tunnel.

WhiteRabbitNeo 2.5 7B (a Qwen2.5-Coder-7B fine-tune), Gemma 4 E4B Uncensored and the ten GGUFs in
`src/chat/lm_catalog.c` are files the C engine cannot run, so the script starts them on llama.cpp's
`llama-server` on the same port. Gemma 4 E2B, Nemotron 3.5 Bolt, Granite 4.2, the Qwen3.5-2B GGUF
and the older GGUFs use `~/opt/llama.cpp-v1.0.0`. Spark (spark2_5) uses `~/opt/llama.cpp-b10828`.
LFM3 (lfm3) uses `~/opt/llama.cpp-b11938`. A projector is passed as `--mmproj` only when that model
has a matching file (Gemma E2B, Gemma E2B Heretic, LFM3, LFM3 abliterated, Qwen3.5-2B; not Nemotron,
Granite or Spark). Weights live under `~/models/<alias>/` as named in the catalog. Only one model
runs at a time: starting any model stops the other server first. For llama-server, `--ctx` (default
8192; the picker caps WhiteRabbitNeo and Granite at 16k and the other GGUFs at 32k), `--threads`,
`--seed`, `--max-tokens`, `--timeout`, `--name` and `--think` / `--no-think` (`--reasoning on` /
`off`) are translated; `--cache-mb` and `--fast` are refused.
WhiteRabbitNeo has no vision; Gemma reads images. llama-server has no `capabilities` field, so the
app reads the window from `meta.n_ctx` and vision from `architecture.input_modalities` in
`/v1/models`. Gemma 4 thinks before it answers (shown under "Thinking"); `--no-think` turns that off.

Production: build once, then serve `web/dist` with the standard-library proxy, which needs only
Python 3 on the server:

```bash
cd web && npm ci && npm run build
python3 tools/chat_server.py --upstream http://127.0.0.1:8080        # http://127.0.0.1:8000
```

`scripts/deploy-web.sh` builds on this machine and copies `web/dist/` to the server
(`DEPLOY_HOST`, `DEPLOY_PATH` override `toor@192.168.1.96` and `~/kevin-ai/web/dist/`).

### tools/chat_server.py

- Serves `web/dist` with the right MIME types, a strict CSP, no directory listings, and a 404 for
  any path that resolves outside it.
- Forwards `/v1/*` and `/health` to `--upstream` and streams the answer back unbuffered. When the
  browser disconnects (Stop, closed tab), it closes the upstream connection, which stops
  lm-serve, even while lm-serve is still reading a long prompt.
- `LM_SERVE_API_KEY` is added upstream as `Authorization: Bearer …`; the browser never sees it.
- Listens on `127.0.0.1:8000`. Any other `--host` refuses to start unless `CHAT_PASSWORD` is set;
  then every request needs HTTP Basic auth with that password (any user name). Basic auth over
  plain HTTP is readable by anyone on the network path: use it on a trusted LAN only, or reach the
  machine through an SSH tunnel instead.
- Request bodies over 24 MiB get a 413.

Tests: `python3 -m unittest discover -s tools -p 'test_chat_server.py'`.

## What it does

- **Streaming and Stop.** While lm-serve reads the prompt the reply shows "Reading the prompt…
  N s · about M min S s left" (12 prompt tokens/s on the target machine), then tokens per second. Under it the page says
  what is being read: the model has not started thinking yet, it reads the whole prompt first (≈ tokens, the tool
  definitions counted apart, and what the server already holds from earlier turns). While tools are in use, the reply's
  Thinking and Tool calls are open, so what is asked, run and returned is in view; when the run ends they fold again.
  Stop (or Esc) aborts the
  request; the server stops when the client disconnects. A reply that ends without a finish
  reason is shown as cut short, never as complete.
- **Effort and speed.** Two selects beside the model picker, saved with each chat. Effort: Low
  answers without thinking and at most 512 tokens (never more than your Settings cap), Medium
  without thinking at your length, High thinks first (`chat_template_kwargs.enable_thinking`, sent
  with every request; llama-server honours it, lm-serve ignores it and thinks only if started with
  `--think`) and may write up to 4,096 tokens or your cap if larger. Speed: Fast sends images at
  512 px and compacts at 50% of the window, Balanced uses your Settings, Thorough sends 1,024 px
  images and compacts at 90%. Choosing a speed also sets the effort it starts from (Low, Medium,
  High); the effort can then be changed on its own. Before its first message a new chat starts
  from the last pick.
- **Time spent per mode.** Neither switch changes the engine's tokens/s. Wall time is mostly
  (thinking tokens + answer tokens) / generate rate, plus image encode and any compaction call.
  Measured 2026-11-25 on the target through the tunnel, Gemma 4 E4B Uncensored already loaded,
  same prompt ("In one short paragraph, explain what a mutex is and when to use one."),
  temperature 0.2:

  | Effort | Thinking | Cap | Wall | Prompt / completion tokens |
  |---|---|---:|---:|---|
  | Low | off | 512 | **26 s** | 25 / 124 |
  | Medium | off | 2048 | **25 s** | 25 / 123 |
  | High | on | 4096 | **54 s** | 31 / 258 |

  Low and Medium matched here because the model stopped around 120 tokens either way; High took
  about **2×** as long because thinking ran and the completion roughly doubled. Speed alone does
  not change that text-only timing: Fast / Balanced / Thorough only change image size and when
  auto-compact fires (and Fast/Balanced/Thorough also *start* Effort at Low/Medium/High).
- **Redirect.** While a reply streams in the open chat, type a new message and press Redirect (or
  Enter): the reply stops where it is, is kept and marked stopped, and the new message is sent
  right after it with everything already written, so the model carries on from there. Not offered
  while another chat is being written. It can be repeated.
- **Statistics per reply:** time to first token, tokens per second, prompt tokens and, when the
  server reports it, tokens taken from its cache.
- **Token counter.** While a reply streams: tokens read (prompt) and written, with `≈` and a note
  while they are estimates (prompt by characters / 3.5 plus images at the server's size, written
  by streamed chunks). Real `usage` from `stream_options.include_usage` replaces them when it
  arrives. The final figures stay under each reply.
- **This chat.** Above the composer: the context window as one bar of coloured boxes (prompt that
  is re-sent, last reply, free) with their numbers and a dashed mark where auto compact starts,
  then boxes with the chat's totals (tokens read and written, turns, mean tokens per second and,
  once compacted, how often). Figures are marked `≈` while any is estimated. The names and
  numbers are always written out, so colour is never the only cue.
- **The chat list stays open during a reply.** You can start or open another chat while one is
  being written (it is marked "writing…" and keeps streaming); sending waits until it ends.
  Deleting waits too, because it resets the server's cache.
- **Retry.** A reply that failed (server down, stream cut) shows Retry. It drops the failed
  attempt and asks again from the same point, tool steps already run included, so once the
  server is back one click carries on. A failed compaction and a failed cache reset have Retry too.
- **Delete from here / rewind.** Each saved message offers Delete from here: that message and
  everything after it are removed, and `POST /v1/chat/reset` clears the server’s reused state so
  the next reply matches the shorter chat. After deleting a reply, Regenerate asks again from the
  user message that remains. Not offered while a reply is running.
- **Compacting.** Auto compact (a checkbox above the composer, on by default, saved with the
  settings) summarises by itself when the next request would fill 80 % of the context (the
  server's own count of the last reply, else an estimate). With it off, only the Compact chat
  button does, and a message is sent as it is unless it overflows the window. Either way the older turns are
  summarised by the model into one summary message and the last two turns stay verbatim. The
  summary goes in the system message; the old messages stay saved and on screen, marked
  compacted. Images in the summarised turns become a text note, and a stopped or failed partial
  reply is passed to the summary marked as such. During a tool run the same check repeats before every step, since tool results grow the chat. If the summary fails, the chat is untouched and
  the message is sent as it was.
- **Model picker.** When `/v1/models` lists more than one model, each chat chooses its model
  (sent as `model`), with the context length the server reports for it. The context check,
  compaction and vision follow that choice.
- **Tools (off by default, Settings).** Offers a calculator and the current time through the
  OpenAI `tools` / `tool_calls` shape (a server with tool support, e.g. lm-serve on Qwen3.5 or
  Gemma 4). Calls and results are shown in a collapsible trace; the loop is capped at 1–20 steps
  (default 8), the last step sends `tool_choice: "none"`, and Stop ends it at once. The two
  built-in tools run in the page and touch nothing outside it.
- **MCP tools and web search (Settings > Tools and internet).** With `tools/chat_tools.py` running and reachable at `/tools/*`
  (`LM_TOOLS_URL`), the tools of the MCP servers turned on are fetched from it (one server at a
  time) and offered as `server__tool`. The service keeps each server's tool list from its last start, so
  opening the chat does not start the servers; they start when a tool is called. Only a **budget** of
  definitions is sent each step (default 1,500 tokens on top of the always-sent built-ins and `find_tools`,
  at most 30 tools): the tools that best match your message by keyword; `find_tools(query)` lets the model name what it needs and
  unlocks those tools for the rest of the message. A call to a tool that is not read-only, or that
  reaches the network (such as fetch: a URL can carry what the model read out of the machine), or to any
  tool in "always" mode, shows a card with the arguments (each long value cut on its own, so no key is
  hidden) and runs only after Allow once or Always allow this tool (stored as `server/tool`); the card
  takes focus itself, so a stray key press does not answer it. `web_search` is not asked about; Deny, or Stop, refuses it and the model is told. With
  nobody to ask, such a call is refused. Tool results are marked in the system message as data,
  never instructions. Servers that reach the network, and `web_search` (a SearXNG address), are
  offered only while the internet switch is on; it is off by default, a badge in the header shows
  it, and the SearXNG address can be tested from Settings. The panel lists the servers the service
  has (state, "needs the internet", their tools with read-only / asks first and their token
  cost, per-tool switches), offers the catalog's servers that are not added yet, sets the budget
  (with the cost to read it at the measured prompt speed) and the approval mode, and lists the
  tools you stopped being asked about. A service that is not running is shown quietly, with Retry.
  `LM_TOOLS_URL` (default `http://127.0.0.1:8081`) points the dev proxy at the service, as
  `--tools-upstream` does for `tools/chat_server.py`; an SSH tunnel to it works like the one to
  lm-serve. The dev proxy reads this at start, so a running dev server needs a restart for `/tools`.
  A second copy of the service on this Mac (`scripts/start-local-chat-tools.sh`, 127.0.0.1:8082) is reached at
  `/tools-local` with the same rules: `LM_TOOLS_LOCAL_URL` for the dev proxy, `--tools-local-upstream` for
  `chat_server.py` (404 `tools_local_not_configured` when unset). In `lib/api.ts` every `tools*` function takes
  an optional last argument `host: 'server' | 'local'` (default `'server'`).
- **Server state reuse.** Every earlier answer is re-sent exactly as received (no reasoning, no
  edits), because lm-serve reuses its state only then. Stopped replies are re-sent as far as
  they got; failed replies are left out, even when they streamed some text.
- **Thinking.** `reasoning_content` (lm-serve `--think`) goes in a collapsible block.
- **Lighthouse 100** in all categories, desktop and mobile, on the production build as `tools/chat_server.py` serves it (gzip, strict CSP): `scripts/lighthouse.sh` checks it. Measure that, not the Vite dev server, which sends about 78 unbundled files.
- **Markdown** with GitHub tables and code blocks with a copy button. Raw HTML in a reply is shown
  as text; images in a reply are never loaded (they could be remote trackers). LaTeX (`$...$`
  inline, `$$...$$` as a block) is drawn as MathML, which the browser renders itself, so there is no
  stylesheet or font to load and the CSP is unchanged; a price like "$5 to $10" stays text, and
  maths in code is left alone.
- **Conversations** are saved in `localStorage` under `lm-chat` (schema version 1): rename,
  delete, delete all. A saved document missing a newer field loads with its default; one that
  cannot be read is copied aside to `lm-chat.unreadable.<time>` rather than overwritten.
  A message is saved as soon as it is sent and its reply once it ends, never once per token.
- **Settings:** a modal (backdrop, Close, Escape) with the model starter and the chat settings:
  system prompt, temperature, longest reply, image size preset (512, 768 or 1024 px on the longer
  side), language.
- **Attachments** by button or drag and drop:
  - text and code (`.txt .md .json .csv .log` and common source files, up to 2 MiB) go in as a
    fenced block named after the file; binary content is refused;
  - PDFs (`pdfjs-dist`, its worker bundled, loaded only when a PDF is attached): text with
    `--- page N ---` markers and a page-range field; pages with no text layer are rendered to PNG
    and sent as images only when the server reports vision `ok`, otherwise they are listed as
    left out. pdf.js's JPEG 2000 and JBIG2 decoders for scanned pages are its JavaScript builds,
    copied to `dist/pdfjs/` with their BSD licences, so the CSP needs no `wasm-unsafe-eval`;
  - images (PNG, JPEG, WebP) are resized in the browser to the preset (and the server's
    `max_pixels`) and re-encoded as PNG data URLs.
- **Estimates before sending,** labelled as estimates: text ≈ characters / 3.5 tokens, images
  from the server's `tokens_per_image` or `pixels_per_token`, and the wait at about 12 prompt
  tokens a second (measured on the target i5-6500T). Above two minutes the wait is flagged.
  Sending is refused when the conversation would overflow the server's `context_length`, when a
  request would carry more than 8 images, or when its body would exceed 24 MiB.
- **Vision is tri-state** from `GET /v1/models` `data[0].capabilities.vision.status`: `ok`
  enables image upload, `unavailable` and a missing field disable it and say why. Today's
  lm-serve reports neither `capabilities` nor `context_length`, so images stay off and no context
  check is made until the server's image support lands.
- **Errors** are shown in the user's language by `code` (`image_unsupported`, `image_invalid`,
  `image_too_large`, `too_many_images`, `context_overflow`, and the proxy's
  `upstream_unreachable`, `upstream_failed`), else the server's own message.
- **Privacy test:** `src/privacy.test.ts` fails if the code names a remote URL, uses beacons,
  WebSockets or cookies, or calls `fetch` anywhere but `src/lib/api.ts`.
- **i18n:** `src/i18n/en.ts` and `src/i18n/es.ts`, written by hand; `Intl` formats numbers,
  dates, durations and plurals. The language follows the browser and can be switched in
  Settings. The model's reply language is not forced.
- **Accessibility:** keyboard only works (Enter sends, Shift+Enter adds a line, Esc stops), a skip
  link, visible focus, states never shown by colour alone, `prefers-reduced-motion` and
  `prefers-color-scheme` respected, logical CSS properties, usable at 320 px wide. axe-core runs
  on the main screens in the test suite.

## Gate

```bash
cd web && npm ci && npm run lint && npm run typecheck && npm test && npm run build
```

`.npmrc` sets `ignore-scripts=true` (no dependency lifecycle scripts run) and exact versions.

## Dependencies

Runtime:

| Package | Why |
| --- | --- |
| `react`, `react-dom` | The UI. |
| `react-markdown` | Renders replies as Markdown to React elements, without `dangerouslySetInnerHTML`. |
| `remark-gfm` | Tables, strikethrough and task lists, which models write often. |
| `remark-math`, `rehype-katex` | LaTeX in replies, drawn as MathML (KaTeX is pulled in by `rehype-katex`; all MIT). |
| `pdfjs-dist` | Reads PDF text and renders text-less pages; its worker is bundled locally. |

Development:

| Package | Why |
| --- | --- |
| `vite`, `@vitejs/plugin-react` | Dev server with the `/v1` proxy, and the production build. |
| `typescript`, `@types/react`, `@types/react-dom`, `@types/node` | Strict type checking (`@types/node` for `vite.config.ts`). |
| `vitest`, `jsdom` | Unit and component tests in a simulated DOM. |
| `@testing-library/react`, `@testing-library/dom` | Drive components the way a user does. |
| `axe-core` | Accessibility checks on the main screens. |
| `eslint`, `@eslint/js`, `typescript-eslint`, `eslint-plugin-react-hooks` | Lint, kept to the recommended rule sets plus React hook rules. |

TypeScript stays on 6.0 because `typescript-eslint` 8 does not support 7.x yet.
