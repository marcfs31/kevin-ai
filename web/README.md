# Chat web app for lm-serve

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

Production: build once, then serve `web/dist` with the standard-library proxy, which needs only
Python 3 on the server:

```bash
cd web && npm ci && npm run build
python3 tools/chat_server.py --upstream http://127.0.0.1:8080        # http://127.0.0.1:8000
```

`scripts/deploy-web.sh` builds on this machine and copies `web/dist/` to the server
(`DEPLOY_HOST`, `DEPLOY_PATH` override `toor@192.168.1.96` and `~/small-llm-in-c/web/dist/`).

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
  N s" (on this hardware that can take minutes), then tokens per second. Stop (or Esc) aborts the
  request; the server stops when the client disconnects. A reply that ends without a finish
  reason is shown as cut short, never as complete.
- **Statistics per reply:** time to first token, tokens per second, prompt tokens and, when the
  server reports it, tokens taken from its cache.
- **Server state reuse.** Every earlier answer is re-sent exactly as received (no reasoning, no
  edits), because lm-serve reuses its state only then. Failed replies are left out.
- **Thinking.** `reasoning_content` (lm-serve `--think`) goes in a collapsible block.
- **Markdown** with GitHub tables and code blocks with a copy button. Raw HTML in a reply is shown
  as text; images in a reply are never loaded (they could be remote trackers).
- **Conversations** are saved in `localStorage` under `lm-chat` (schema version 1): rename,
  delete, delete all. A saved document missing a newer field loads with its default; one that
  cannot be read is copied aside to `lm-chat.unreadable.<time>` rather than overwritten.
  Nothing is saved while a reply streams.
- **Settings:** system prompt, temperature, longest reply, image size preset (512, 768 or 1024 px
  on the longer side), language.
- **Attachments** by button or drag and drop:
  - text and code (`.txt .md .json .csv .log` and common source files, up to 2 MiB) go in as a
    fenced block named after the file; binary content is refused;
  - PDFs (`pdfjs-dist`, its worker bundled, loaded only when a PDF is attached): text with
    `--- page N ---` markers and a page-range field; pages with no text layer are rendered to PNG
    and sent as images only when the server reports vision `ok`, otherwise they are listed as
    left out;
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
  `image_too_large`, `too_many_images`, `context_overflow`), else the server's own message.
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
