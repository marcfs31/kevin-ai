// Privacy: the app talks only to its own origin. The proxy's CSP enforces this in the browser;
// this test fails at review time if the code starts naming another host or a tracking API.
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join, relative } from 'node:path';
import { describe, expect, it } from 'vitest';

const SRC = join(__dirname);

function sources(dir: string): string[] {
  return readdirSync(dir).flatMap((name) => {
    const path = join(dir, name);
    if (statSync(path).isDirectory()) return name === 'test' ? [] : sources(path);
    return /\.(ts|tsx)$/.test(name) && !/\.test\.tsx?$/.test(name) ? [path] : [];
  });
}

describe('privacy', () => {
  const files = sources(SRC).map((path) => ({ path: relative(SRC, path), text: readFileSync(path, 'utf8') }));

  it('names no remote URL', () => {
    for (const f of files) expect(f.text.match(/\b(?:https?|wss?):\/\/[^\s'"`]*/g) ?? [], f.path).toEqual([]);
  });

  it('uses no beacon, WebSocket or third-party storage', () => {
    for (const f of files) expect(f.text, f.path).not.toMatch(/sendBeacon|new WebSocket|EventSource|indexedDB|document\.cookie/);
  });

  it('calls fetch only from lib/api.ts, and only with same-origin paths', () => {
    for (const f of files) {
      const calls = f.text.match(/\bfetch\(/g) ?? [];
      if (f.path === join('lib', 'api.ts')) {
        expect(calls).toHaveLength(1);
        expect(f.text.match(/send\('([^']+)'/g)).toEqual([
          "send('/v1/models'",
          "send('/control/serve'",
          "send('/v1/chat/reset'",
          "send('/v1/chat/completions'",
        ]);
        // The tool service's paths: same-origin /tools/* or /tools-local/* (toolsBase), nothing else.
        expect(f.text.match(/toolsJson\(\s*[`']([^`'?]*)/g)?.map((m) => m.replace(/^toolsJson\(\s*[`']/, ''))).toEqual([
          '${toolsBase(host)}/status',
          '${toolsBase(host)}/servers/${encodeURIComponent(server)}/tools',
          '${toolsBase(host)}/servers/${encodeURIComponent(server)}/call',
          '${toolsBase(host)}/search',
          '${toolsBase(host)}/servers/${encodeURIComponent(server)}/stop',
          '${toolsBase(host)}/catalog/${encodeURIComponent(id)}/add',
        ]);
      } else expect(calls, f.path).toEqual([]);
    }
  });
});
