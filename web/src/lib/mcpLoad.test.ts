import { afterEach, describe, expect, it, vi } from 'vitest';
import { loadMcpSpecs } from './mcpLoad';

const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });
const server = (id: string, network = false) => ({ id, title: id, kind: 'stdio', network, catalog: null, state: 'stopped', error: null, tools: null });
const tools = (...names: string[]) => ({ ok: true, tools: names.map((name) => ({ name, title: name, description: `does ${name}`, inputSchema: { type: 'object' }, readOnly: true, tokens: 10 })) });

function stub(routes: Record<string, () => Response>) {
  const calls: string[] = [];
  vi.stubGlobal('fetch', vi.fn(async (u: RequestInfo | URL) => {
    const url = String(u);
    calls.push(url);
    const key = Object.keys(routes).find((k) => url.startsWith(k));
    const route = key ? routes[key] : undefined;
    if (!route) throw new Error('no route ' + url);
    return route();
  }));
  return calls;
}
afterEach(() => vi.unstubAllGlobals());

describe('loadMcpSpecs', () => {
  it('asks each server that is on, in order, and names the tools server__tool', async () => {
    const calls = stub({
      '/tools/status': () => json({ ok: true, servers: [server('fs'), server('time')], catalog: [] }),
      '/tools/servers/fs/tools': () => json(tools('read', 'write')),
      '/tools/servers/time/tools': () => json(tools('now')),
    });
    const r = await loadMcpSpecs(['fs', 'time'], false);
    expect(r.specs.map((s) => s.name)).toEqual(['fs__read', 'fs__write', 'time__now']);
    expect(calls).toEqual(['/tools/status', '/tools/servers/fs/tools?internet=0', '/tools/servers/time/tools?internet=0']);
  });

  it('skips a server that needs the internet while it is off, and one the service does not have, without asking them', async () => {
    const calls = stub({
      '/tools/status': () => json({ ok: true, servers: [server('net', true)], catalog: [] }),
      '/tools/servers/net': () => json(tools('x')),
    });
    const r = await loadMcpSpecs(['net', 'gone'], false);
    expect(r.skipped).toEqual([{ id: 'net', reason: 'internet_off' }, { id: 'gone', reason: 'missing' }]);
    expect(r.specs).toEqual([]);
    expect(calls).toEqual(['/tools/status']);
    // with the internet on the same server is asked, with the flag
    const on = await loadMcpSpecs(['net'], true);
    expect(on.specs.map((s) => s.name)).toEqual(['net__x']);
    expect(calls.at(-1)).toBe('/tools/servers/net/tools?internet=1');
  });

  it('keeps the servers that answered when one fails, and names the failure', async () => {
    stub({
      '/tools/status': () => json({ ok: true, servers: [server('bad'), server('good')], catalog: [] }),
      '/tools/servers/bad': () => json({ ok: false, error: { code: 'start_failed', message: 'cannot run npx' } }, 502),
      '/tools/servers/good': () => json(tools('t')),
    });
    const r = await loadMcpSpecs(['bad', 'good'], false);
    expect(r.specs.map((s) => s.name)).toEqual(['good__t']);
    expect(r.failed).toEqual([{ id: 'bad', message: 'cannot run npx' }]);
  });

  it('never lets a server take a built-in name, and fails as a whole when the service is down', async () => {
    stub({
      '/tools/status': () => json({ ok: true, servers: [server('calculator')], catalog: [] }),
      '/tools/servers/calculator': () => json(tools('x')),
    });
    expect((await loadMcpSpecs(['calculator'], false)).specs[0]?.name).toBe('calculator__x');
    stub({ '/tools/status': () => json({ ok: false, error: { code: 'x', message: 'down' } }, 502) });
    await expect(loadMcpSpecs(['a'], false)).rejects.toMatchObject({ status: 502 });
  });
});
