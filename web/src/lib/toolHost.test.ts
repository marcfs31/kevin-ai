import { afterEach, describe, expect, it, vi } from 'vitest';
import { DEFAULT_SETTINGS, type ToolsConfig } from './storage';
import { MAX_TOOLS_PER_STEP, builtinHost, createToolHost, needsApproval, toolGuidance } from './toolHost';
import { mcpSpecs, type McpTool, type ToolSpec } from './toolRegistry';

const config = (over: Partial<ToolsConfig> = {}): ToolsConfig => ({ ...DEFAULT_SETTINGS.toolsConfig, servers: { fs: { on: true, disabledTools: [] } }, ...over });
const internet = (on = false, searchUrl = '') => ({ on, searchUrl });
const tool = (name: string, readOnly: boolean, description = `does ${name}`): McpTool => ({ name, title: name, description, inputSchema: { type: 'object', properties: {} }, readOnly, tokens: 0 });
const specs = (...tools: McpTool[]) => mcpSpecs('fs', tools, new Set());
const call = (name: string, args: unknown = {}, id = 'c1') => ({ id, name, arguments: typeof args === 'string' ? args : JSON.stringify(args) });
const noUnlock = () => undefined;
const signal = () => new AbortController().signal;

function stubFetch(handler: (url: string, init?: RequestInit) => Response | Promise<Response>) {
  const fn = vi.fn(async (u: RequestInfo | URL, init?: RequestInit) => handler(String(u), init));
  vi.stubGlobal('fetch', fn);
  return fn;
}
const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });
afterEach(() => vi.unstubAllGlobals());

describe('needsApproval', () => {
  const [ro, rw] = specs(tool('read', true), tool('write', false)) as [ToolSpec, ToolSpec];

  it('asks for what is not read-only by default, and never for a read-only tool', () => {
    const c = config();
    expect([needsApproval(ro, c), needsApproval(rw, c)]).toEqual([false, true]);
  });

  it('asks for everything in "always", except what the user stopped being asked about', () => {
    expect([needsApproval(ro, config({ approval: 'always' })), needsApproval(rw, config({ approval: 'always' }))]).toEqual([true, true]);
    expect(needsApproval(rw, config({ approval: 'always', alwaysAllow: [rw.name] }))).toBe(false);
  });

  it('never asks in "never", and an always-allow entry lifts only that tool', () => {
    expect(needsApproval(rw, config({ approval: 'never' }))).toBe(false);
    expect(needsApproval(rw, config({ alwaysAllow: [rw.name] }))).toBe(false);
    expect(needsApproval(rw, config({ alwaysAllow: ['other'] }))).toBe(true);
  });
});

describe('builtinHost', () => {
  it('offers the two built-ins, asks about nothing and runs them', async () => {
    const h = builtinHost();
    expect(h.external).toBe(false);
    expect(h.definitions('anything', new Set()).map((d) => d.function.name)).toEqual(['calculator', 'current_time']);
    expect(h.approval(call('calculator'))).toBeNull();
    expect(await h.run(call('calculator', { expression: '6*7' }), signal(), noUnlock)).toEqual({ content: '42', isError: false });
  });
});

describe('createToolHost: what is offered', () => {
  it('offers the built-ins and find_tools, and the matching tools within the budget', () => {
    const h = createToolHost({ mcp: specs(tool('read_file', true, 'read the contents of a file'), tool('git_status', true, 'repository status')), config: config(), internet: internet() });
    const names = h.definitions('please read my file', new Set()).map((d) => d.function.name);
    expect(names).toEqual(expect.arrayContaining(['calculator', 'current_time', 'find_tools', 'fs__read_file']));
    expect(names).not.toContain('fs__git_status');
    expect(h.external).toBe(true);
  });

  it('leaves out a server that is off, a tool the user disabled, and offers no find_tools without MCP tools', () => {
    const all = specs(tool('alpha', true), tool('beta', true));
    const off = createToolHost({ mcp: all, config: config({ servers: { fs: { on: false, disabledTools: [] } } }), internet: internet() });
    expect(off.external).toBe(false);
    expect(off.definitions('alpha beta', new Set()).map((d) => d.function.name)).toEqual(['calculator', 'current_time']);
    const some = createToolHost({ mcp: all, config: config({ servers: { fs: { on: true, disabledTools: ['beta'] } } }), internet: internet() });
    const names = some.definitions('alpha beta', new Set()).map((d) => d.function.name);
    expect(names).toContain('fs__alpha');
    expect(names).not.toContain('fs__beta');
  });

  it('offers web_search only with the internet on and an address to search', () => {
    const names = (net: ReturnType<typeof internet>) => createToolHost({ mcp: [], config: config(), internet: net }).definitions('x', new Set()).map((d) => d.function.name);
    expect(names(internet(true, 'http://127.0.0.1:8888'))).toContain('web_search');
    expect(names(internet(true, ''))).not.toContain('web_search');
    expect(names(internet(false, 'http://127.0.0.1:8888'))).not.toContain('web_search');
    expect(createToolHost({ mcp: [], config: config(), internet: internet(true, 'http://s') }).external).toBe(true);
  });

  it('never offers more than the cap, and unlocked tools join whatever the query', () => {
    const many = specs(...Array.from({ length: 30 }, (_, i) => tool(`file_tool_${i}`, true, 'handles a file')));
    const h = createToolHost({ mcp: many, config: config({ budgetTokens: 12000 }), internet: internet() });
    expect(h.definitions('file', new Set()).length).toBeLessThanOrEqual(MAX_TOOLS_PER_STEP);
    expect(h.definitions('hello', new Set(['fs__file_tool_29'])).map((d) => d.function.name)).toContain('fs__file_tool_29');
  });
});

describe('createToolHost: approval', () => {
  it('asks for a write and shows its arguments, and not for a read', () => {
    const h = createToolHost({ mcp: specs(tool('read', true), tool('write', false)), config: config(), internet: internet() });
    expect(h.approval(call('fs__read'))).toBeNull();
    expect(h.approval(call('fs__write', { path: '/x', text: 'hi' }, 'k'))).toEqual({
      callId: 'k', tool: 'fs__write', title: 'fs · write', server: 'fs', args: JSON.stringify({ path: '/x', text: 'hi' }, null, 2), readOnly: false,
    });
  });

  it('cuts each long value on its own so every key, the path included, stays in view', () => {
    const h = createToolHost({ mcp: specs(tool('write', false)), config: config(), internet: internet() });
    const shownArgs = h.approval(call('fs__write', { content: 'x'.repeat(5000), path: '/home/toor/chat-files/../../.ssh/authorized_keys', mode: 'a' }))?.args ?? '';
    expect(JSON.parse(shownArgs)).toEqual({ content: `${'x'.repeat(240)}… (4760 more characters)`, path: '/home/toor/chat-files/../../.ssh/authorized_keys', mode: 'a' });
    const many = Object.fromEntries(Array.from({ length: 45 }, (_, i) => [`k${i}`, i]));
    expect(JSON.parse(h.approval(call('fs__write', many))?.args ?? '{}')['…']).toBe('5 more arguments');
    expect(h.approval(call('fs__write', { nested: { a: 'y'.repeat(500) } }))?.args).toContain('more characters');
  });

  it('shows invalid JSON as written, both ends when it is long', () => {
    const h = createToolHost({ mcp: specs(tool('write', false)), config: config(), internet: internet() });
    expect(h.approval(call('fs__write', '{oops'))?.args).toBe('{oops');
    const long = h.approval(call('fs__write', '{' + 'z'.repeat(2000) + 'END'))?.args ?? '';
    expect(long.endsWith('END')).toBe(true);
    expect(long).toContain('…');
  });

  it('asks about nothing it does not know', () => {
    expect(createToolHost({ mcp: [], config: config(), internet: internet() }).approval(call('nope'))).toBeNull();
  });
});

describe('createToolHost: running', () => {
  const host = (over: Partial<Parameters<typeof createToolHost>[0]> = {}) =>
    createToolHost({ mcp: specs(tool('read', true), tool('write', false)), config: config(), internet: internet(true, 'http://127.0.0.1:8888'), ...over });

  it('calls the MCP tool through the service with the original tool name and the internet flag', async () => {
    const fetchMock = stubFetch(() => json({ ok: true, content: 'file text', isError: false }));
    const r = await host().run(call('fs__read', { path: '/a' }), signal(), noUnlock);
    expect(r).toEqual({ content: 'file text', isError: false });
    const [url, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toBe('/tools/servers/fs/call');
    expect(JSON.parse(String(init.body))).toEqual({ name: 'read', arguments: { path: '/a' }, internet: true });
  });

  it('hands the tool\'s own error flag on', async () => {
    stubFetch(() => json({ ok: true, content: 'denied', isError: true }));
    expect(await host().run(call('fs__read'), signal(), noUnlock)).toEqual({ content: 'denied', isError: true });
  });

  it('turns a service error, an unreachable service and a bad call into results the model can read', async () => {
    stubFetch(() => json({ ok: false, error: { code: 'internet_off', message: 'fs reaches the internet' } }, 403));
    expect(await host().run(call('fs__read'), signal(), noUnlock)).toEqual({ content: 'error: fs reaches the internet', isError: true });
    stubFetch(() => {
      throw new TypeError('down');
    });
    expect((await host().run(call('fs__read'), signal(), noUnlock)).content).toBe('error: the tool service could not be reached');
    expect((await host().run(call('nope'), signal(), noUnlock)).content).toBe('error: no tool named "nope"');
    expect((await host().run(call('fs__read', '[1]'), signal(), noUnlock)).content).toBe('error: the arguments must be a JSON object');
  });

  it('reports an aborted call as stopped, not as a failure of the tool', async () => {
    stubFetch(() => {
      throw new DOMException('aborted', 'AbortError');
    });
    expect((await host().run(call('fs__read'), signal(), noUnlock)).content).toBe('error: stopped');
  });

  it('find_tools returns matching names, unlocks them, and says when nothing matches', async () => {
    const h = host({ mcp: specs(tool('read_file', true, 'read the contents of a file'), tool('git_status', true, 'repository status')) });
    const unlocked: string[] = [];
    const r = await h.run(call('find_tools', { query: 'read a file' }), signal(), (n) => unlocked.push(...n));
    expect(r.content).toContain('fs__read_file: read the contents of a file');
    expect(r.content).toContain('now available');
    expect(unlocked).toEqual(['fs__read_file']);
    expect((await h.run(call('find_tools', { query: 'poetry' }), signal(), noUnlock)).content).toBe('no matching tool');
    expect((await h.run(call('find_tools', { query: '  ' }), signal(), noUnlock)).isError).toBe(true);
  });

  it('web_search posts the query and the address, with the internet flag, and needs a query', async () => {
    const fetchMock = stubFetch(() => json({ ok: true, content: '1. One', isError: false }));
    expect((await host().run(call('web_search', { query: 'llama' }), signal(), noUnlock)).content).toBe('1. One');
    const [url, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toBe('/tools/search');
    expect(JSON.parse(String(init.body))).toEqual({ query: 'llama', searchUrl: 'http://127.0.0.1:8888', internet: true });
    expect((await host().run(call('web_search', {}), signal(), noUnlock)).isError).toBe(true);
  });

  it('runs the built-in tools locally, without the service', async () => {
    const fetchMock = stubFetch(() => json({}));
    expect((await host().run(call('calculator', { expression: '2+2' }), signal(), noUnlock)).content).toBe('4');
    expect(fetchMock).not.toHaveBeenCalled();
  });
});

describe('toolGuidance', () => {
  it('tells the model to search when unsure, and not for what it knows, only when search is on, and says nothing for the built-ins alone', () => {
    expect(toolGuidance({ search: false, mcp: false })).toBe('');
    expect(builtinHost().guidance).toBe('');
    const withSearch = toolGuidance({ search: true, mcp: false });
    expect(withSearch).toContain('web_search');
    expect(withSearch).toContain('unsure');
    expect(withSearch).toContain('capital of France');
    expect(withSearch).not.toContain('find_tools');
    const mcpOnly = toolGuidance({ search: false, mcp: true });
    expect(mcpOnly).toContain('find_tools');
    expect(mcpOnly).not.toContain('web_search');
    // Always bounded: it is prompt the server reads on the first message of every chat.
    expect(toolGuidance({ search: true, mcp: true }).length).toBeLessThan(900);
  });

  it('follows what the host really offers', () => {
    expect(createToolHost({ mcp: [], config: config(), internet: internet(true, 'http://127.0.0.1:8888') }).guidance).toContain('web_search');
    expect(createToolHost({ mcp: [], config: config(), internet: internet(true, '') }).guidance).toBe('');
    expect(createToolHost({ mcp: [], config: config(), internet: internet(false, 'http://127.0.0.1:8888') }).guidance).toBe('');
  });
});
