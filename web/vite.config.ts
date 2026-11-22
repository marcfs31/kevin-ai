/// <reference types="vitest/config" />
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { dirname, join } from 'node:path';
import { defineConfig, type Plugin } from 'vite';
import react from '@vitejs/plugin-react';
import { serveControl } from './serve-control-plugin';

const upstream = process.env.LM_SERVE_URL ?? 'http://127.0.0.1:8080';
// The tool service (tools/chat_tools.py): MCP servers and web search, behind /tools/*.
const toolsUpstream = process.env.LM_TOOLS_URL ?? 'http://127.0.0.1:8081';
// A second copy of it on this Mac (scripts/start-local-chat-tools.sh), behind /tools-local/*.
const toolsLocalUpstream = process.env.LM_TOOLS_LOCAL_URL ?? 'http://127.0.0.1:8082';

// pdf.js fetches its image decoders by fixed name from `wasmUrl` (see src/lib/pdf.ts): serve
// them at /pdfjs/ in dev and copy them to dist/pdfjs/ in the build, with their licences.
const PDFJS_FILES = [
  'openjpeg_nowasm_fallback.js', 'jbig2_nowasm_fallback.js',
  'LICENSE_OPENJPEG', 'LICENSE_PDFJS_OPENJPEG', 'LICENSE_JBIG2', 'LICENSE_PDFJS_JBIG2',
];
const pdfjsDir = join(dirname(createRequire(import.meta.url).resolve('pdfjs-dist/package.json')), 'wasm');

function pdfjsDecoders(): Plugin {
  return {
    name: 'pdfjs-decoders',
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        const name = req.url?.startsWith('/pdfjs/') ? req.url.slice('/pdfjs/'.length) : '';
        if (!PDFJS_FILES.includes(name)) return next();
        res.setHeader('Content-Type', name.endsWith('.js') ? 'text/javascript' : 'text/plain');
        res.end(readFileSync(join(pdfjsDir, name)));
      });
    },
    generateBundle() {
      for (const name of PDFJS_FILES)
        this.emitFile({ type: 'asset', fileName: `pdfjs/${name}`, source: readFileSync(join(pdfjsDir, name)) });
    },
  };
}

export default defineConfig({
  plugins: [react(), pdfjsDecoders(), serveControl()],
  server: {
    host: '127.0.0.1',
    port: 5173,
    strictPort: true,
    proxy: {
      '/v1': { target: upstream, changeOrigin: true },
      '/health': { target: upstream, changeOrigin: true },
      // Before '/tools': a string key matches by prefix, and '/tools-local' starts with '/tools'.
      '/tools-local': { target: toolsLocalUpstream, changeOrigin: true, rewrite: (p) => p.replace(/^\/tools-local/, '/tools') },
      '/tools': { target: toolsUpstream, changeOrigin: true },
    },
  },
  build: {
    // No inline scripts or data: URLs in the output, so a strict CSP holds.
    assetsInlineLimit: 0,
    sourcemap: false,
    target: 'es2022',
  },
  test: {
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
    include: ['src/**/*.test.{ts,tsx}'],
    restoreMocks: true,
  },
});
