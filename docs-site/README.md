# Kevin AI docs site

VitePress documentation for architecture (Vite vs `chat_server`, APIs, models) with Mermaid diagrams.

```bash
cd docs-site
npm ci
npm run docs:dev      # local preview
npm run docs:build    # static output under .vitepress/dist
```

If the chat Vite already uses port 5173:

```bash
npx vitepress dev --port 5174
```

Interactive GitDiagram: https://gitdiagram.com/marcfs31/kevin-ai

## GitHub Pages

Production site: <https://marcfs31.github.io/kevin-ai/>

`base` defaults to `/kevin-ai/` so the build matches project Pages. For a local preview at `/`, run `DOCS_BASE=/ npm run docs:dev`.

Rebuild and push the `gh-pages` branch from the repo root:

```bash
scripts/publish-docs-pages.sh
```

