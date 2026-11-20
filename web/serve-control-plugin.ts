// Dev-only Vite middleware: POST /control/serve runs scripts/start-remote-lm-serve.sh.
import { spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import type { Plugin } from 'vite';
import { parseServeOptions, serveScriptArgs, type ServeResult } from './src/lib/serveControl.ts';

const webDir = dirname(fileURLToPath(import.meta.url));
const scriptPath = resolve(webDir, '..', 'scripts', 'start-remote-lm-serve.sh');

function readBody(req: import('node:http').IncomingMessage): Promise<string> {
  return new Promise((resolveBody, reject) => {
    const chunks: Buffer[] = [];
    let n = 0;
    req.on('data', (c: Buffer) => {
      n += c.length;
      if (n > 16_384) {
        reject(new Error('body too large'));
        req.destroy();
        return;
      }
      chunks.push(c);
    });
    req.on('end', () => resolveBody(Buffer.concat(chunks).toString('utf8')));
    req.on('error', reject);
  });
}

function runScript(args: string[]): Promise<{ code: number; stdout: string; stderr: string }> {
  return new Promise((resolveRun) => {
    const child = spawn(scriptPath, args, {
      env: process.env,
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    let stdout = '';
    let stderr = '';
    child.stdout.on('data', (d: Buffer) => {
      stdout += d.toString('utf8');
    });
    child.stderr.on('data', (d: Buffer) => {
      stderr += d.toString('utf8');
    });
    child.on('error', (err) => resolveRun({ code: 1, stdout, stderr: err.message }));
    child.on('close', (code) => resolveRun({ code: code ?? 1, stdout, stderr }));
  });
}

export function serveControl(): Plugin {
  return {
    name: 'serve-control',
    configureServer(server) {
      server.middlewares.use(async (req, res, next) => {
        const url = req.url?.split('?')[0] ?? '';
        if (url !== '/control/serve') return next();
        if (req.method !== 'POST') {
          res.statusCode = 405;
          res.setHeader('Content-Type', 'application/json');
          res.end(JSON.stringify({ ok: false, error: 'POST only' } satisfies ServeResult));
          return;
        }
        if (!existsSync(scriptPath)) {
          res.statusCode = 500;
          res.setHeader('Content-Type', 'application/json');
          res.end(JSON.stringify({ ok: false, error: `missing ${scriptPath}` } satisfies ServeResult));
          return;
        }
        let raw: unknown;
        try {
          raw = JSON.parse(await readBody(req));
        } catch {
          res.statusCode = 400;
          res.setHeader('Content-Type', 'application/json');
          res.end(JSON.stringify({ ok: false, error: 'invalid JSON' } satisfies ServeResult));
          return;
        }
        const opts = parseServeOptions(raw);
        if (!opts) {
          res.statusCode = 400;
          res.setHeader('Content-Type', 'application/json');
          res.end(JSON.stringify({ ok: false, error: 'invalid options' } satisfies ServeResult));
          return;
        }
        const result = await runScript(serveScriptArgs(opts));
        const health = result.stdout.trim();
        if (result.code !== 0) {
          res.statusCode = 502;
          res.setHeader('Content-Type', 'application/json');
          const err = (result.stderr || result.stdout || `exit ${result.code}`).trim().slice(0, 2000);
          res.end(JSON.stringify({ ok: false, error: err } satisfies ServeResult));
          return;
        }
        res.statusCode = 200;
        res.setHeader('Content-Type', 'application/json');
        res.end(JSON.stringify({ ok: true, model: opts.model, health } satisfies ServeResult));
      });
    },
  };
}
