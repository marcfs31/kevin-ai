import { afterEach, describe, expect, it, vi } from 'vitest';
import { DEFAULT_SETTINGS, type ToolsConfig } from './storage';
import { MAX_TOOLS_PER_STEP, builtinHost, createToolHost, needsApproval, toolGuidance } from './toolHost';
import { allowKey, mcpSpecs, type McpTool, type ToolSpec } from './toolRegistry';
import { startWorkflow } from './workflows';

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
    expect(needsApproval(rw, config({ approval: 'always', alwaysAllow: [allowKey(rw)] }))).toBe(false);
  });

  it('never asks in "never", and an always-allow entry lifts only that tool', () => {
    expect(needsApproval(rw, config({ approval: 'never' }))).toBe(false);
    expect(needsApproval(rw, config({ alwaysAllow: [allowKey(rw)] }))).toBe(false);
    // The model-facing name is not the key: it can change when names collide.
    expect(needsApproval(rw, config({ alwaysAllow: [rw.name] }))).toBe(true);
    expect(needsApproval(rw, config({ alwaysAllow: ['other'] }))).toBe(true);
  });
});

describe('needsApproval and the network', () => {
  const [get] = mcpSpecs('fetch', [tool('fetch', true)], new Set(), true) as [ToolSpec];
  const [read] = mcpSpecs('fs', [tool('read', true)], new Set(), false) as [ToolSpec];

  it('asks before a read-only tool that reaches the network, because a URL carries what the model read out of the machine', () => {
    expect([needsApproval(get, config()), needsApproval(read, config())]).toEqual([true, false]);
    expect(needsApproval(get, config({ alwaysAllow: [allowKey(get)] }))).toBe(false);
    expect(needsApproval(get, config({ approval: 'never' }))).toBe(false);
  });

  it('puts the network on the approval request, and leaves web search free', () => {
    const h = createToolHost({ mcp: [get, read], config: config({ servers: { fetch: { on: true, disabledTools: [] }, fs: { on: true, disabledTools: [] } } }), internet: internet(true, 'http://127.0.0.1:8888') });
    expect(h.approval(call('fetch__fetch', { url: 'https://example.org' }))).toMatchObject({ network: true, readOnly: true });
    expect(h.approval(call('fs__read'))).toBeNull();
    expect(h.approval(call('web_search', { query: 'x' }))).toBeNull();
  });
});

describe('allowKey', () => {
  it('is the server and the tool, so a collision suffix on the model-facing name cannot move a permission to another tool', () => {
    const taken = new Set<string>();
    const [a] = mcpSpecs('a', [tool('x__y', false)], taken) as [ToolSpec];
    const [b] = mcpSpecs('a__x', [tool('y', false)], taken) as [ToolSpec];
    expect([a.name, b.name]).toEqual(['a__x__y', 'a__x__y_2']);
    expect([allowKey(a), allowKey(b)]).toEqual(['a/x__y', 'a__x/y']);
    expect([needsApproval(a, config({ alwaysAllow: ['a__x/y'] })), needsApproval(b, config({ alwaysAllow: ['a__x/y'] }))]).toEqual([true, false]);
    // Loaded in the other order, the same entry still means the same tool.
    const again = new Set<string>();
    const [b2] = mcpSpecs('a__x', [tool('y', false)], again) as [ToolSpec];
    expect(b2.name).toBe('a__x__y');
    expect(needsApproval(b2, config({ alwaysAllow: ['a__x/y'] }))).toBe(false);
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
      callId: 'k', tool: 'fs__write', allowKey: 'fs/write', title: 'fs · write', server: 'fs', network: false, args: JSON.stringify({ path: '/x', text: 'hi' }, null, 2), readOnly: false, description: 'does write',
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

describe('createToolHost with a project', () => {
  const demo = { id: 'demo', title: 'Demo', defaultBranch: 'main', allowMain: false, hasRemote: false, commands: ['test'] };
  const projectSpecs = () => mcpSpecs('project', [tool('git_status', true, 'Show the branch.'), tool('git_commit', false, 'Commit what is staged.')], new Set());
  const host = (project: typeof demo | null) =>
    createToolHost({ mcp: projectSpecs(), config: config({ servers: { project: { on: true, disabledTools: [] } } }), internet: internet(), project });

  it('tells the model where it works, the branch rule and the commands it has; says nothing without a project', () => {
    const g = host(demo).guidance;
    expect(g).toContain('"Demo"');
    expect(g).toContain('main is the default branch and commits there are refused');
    expect(g).toContain('run_demo_test');
    expect(g).toContain('Do not push or open a pull request unless the user asks');
    expect(host(null).guidance).not.toContain('Demo');
  });

  it('always sends the chat\'s project, whatever the model wrote, and leaves other servers alone', async () => {
    const fetchMock = stubFetch(() => json({ ok: true, content: '## main', isError: false }));
    await host(demo).run(call('project__git_status', { project: 'someone-else', extra: 1 }), signal(), noUnlock);
    const [url, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toBe('/tools/servers/project/call');
    expect(JSON.parse(String(init.body)).arguments).toEqual({ project: 'demo', extra: 1 });
    await host(null).run(call('project__git_status', { project: 'x' }), signal(), noUnlock);
    expect(JSON.parse(String((fetchMock.mock.calls[1] as unknown as [string, RequestInit])[1].body)).arguments).toEqual({ project: 'x' });
  });

  it('sends the calls of a project on this Mac to the Mac\'s tool service, says so on the approval card, and keeps other servers on the model server', async () => {
    const fetchMock = stubFetch(() => json({ ok: true, content: 'ok', isError: false }));
    const mac = { ...demo, host: 'local' as const };
    const h = createToolHost({ mcp: [...projectSpecs(), ...mcpSpecs('fs', [tool('read', true)], new Set())], config: config({ servers: { project: { on: true, disabledTools: [] }, fs: { on: true, disabledTools: [] } } }), internet: internet(), project: mac });
    await h.run(call('project__git_status'), signal(), noUnlock);
    await h.run(call('fs__read'), signal(), noUnlock);
    expect(fetchMock.mock.calls.map((c) => (c as unknown as [string])[0])).toEqual(['/tools-local/servers/project/call', '/tools/servers/fs/call']);
    expect(h.approval(call('project__git_commit', { message: 'm' }))?.title).toBe('project · git_commit · Mac');
    expect(host(demo).approval(call('project__git_commit', { message: 'm' }))?.title).toBe('project · git_commit');
  });

  it('puts the tool\'s description on the approval request, so a command shows what will run', () => {
    const request = host(demo).approval(call('project__git_commit', { message: 'm' }));
    expect(request?.description).toBe('Commit what is staged.');
    expect(host(demo).approval(call('project__git_status'))).toBeNull();
  });
});

describe('createToolHost with a workflow', () => {
  const wf = startWorkflow({ id: 'ship', title: 'Ship it', steps: ['Run the tests', 'Commit', 'Push'] });
  const host = (workflow: typeof wf | null, onWorkflowChange = vi.fn()) => ({ onWorkflowChange, h: createToolHost({ mcp: [], config: config(), internet: internet(), workflow, onWorkflowChange }) });

  it('offers workflow_step and the checklist only while a workflow runs, and never asks before ticking', () => {
    const { h } = host(wf);
    expect(h.definitions('x', new Set()).map((d) => d.function.name)).toContain('workflow_step');
    expect(h.guidance).toContain('"Ship it"');
    expect(h.guidance).toContain('1. [todo] Run the tests');
    expect(h.approval(call('workflow_step', { step: 1, status: 'done' }))).toBeNull();
    const none = host(null).h;
    expect(none.definitions('x', new Set()).map((d) => d.function.name)).not.toContain('workflow_step');
    expect(none.guidance).not.toContain('workflow');
  });

  it('ticks the step, tells the page, and answers with the next step; several ticks in one run build on each other', async () => {
    const { h, onWorkflowChange } = host(wf);
    const first = await h.run(call('workflow_step', { step: 1, status: 'done', note: 'green' }), signal(), noUnlock);
    expect(first).toEqual({ content: 'step 1 is now done. Next: 2. [todo] Commit', isError: false });
    expect(onWorkflowChange).toHaveBeenCalledTimes(1);
    await h.run(call('workflow_step', { step: '2', status: 'skipped' }), signal(), noUnlock);
    const last = onWorkflowChange.mock.calls[1]?.[0] as typeof wf;
    expect(last.steps.map((s) => s.status)).toEqual(['done', 'skipped', 'todo']);
    expect(last.steps[0]?.note).toBe('green');
    expect(wf.steps[0]?.status).toBe('todo');          // the chat's own checklist is not mutated
  });

  it('refuses a bad step or status and changes nothing', async () => {
    const { h, onWorkflowChange } = host(wf);
    for (const args of [{ step: 0, status: 'done' }, { step: 4, status: 'done' }, { step: 1.5, status: 'done' }, { step: 1, status: 'finished' }, { step: 1 }, {}]) {
      const r = await h.run(call('workflow_step', args), signal(), noUnlock);
      expect(r.isError, JSON.stringify(args)).toBe(true);
    }
    expect(onWorkflowChange).not.toHaveBeenCalled();
  });
});
