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

Built with `npm run docs:build` (VitePress `base` is `/kevin-ai/`). The static files are published on the `gh-pages` branch.

