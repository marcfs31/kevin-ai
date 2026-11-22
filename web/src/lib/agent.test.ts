import { describe, expect, it } from 'vitest';
import { TOOL_NOTE, runAgent, type AgentOptions, type StreamFn } from './agent';
import { ApiError, type ServerInfo } from './api';
import { applyCompaction } from './compact';
import { DEFAULT_SETTINGS, type AssistantMessage, type Message, type UserMessage } from './storage';
import type { ApprovalRequest, ToolHost } from './toolHost';

const user: UserMessage = { id: 'u', role: 'user', content: 'what is 6*7?', typed: 'what is 6*7?', attachmentNames: [], createdAt: 0 };
const settings = { ...DEFAULT_SETTINGS, tools: true, maxSteps: 3 };

type Step = (req: Parameters<StreamFn>[0], on: Parameters<StreamFn>[2]) => void | Promise<void>;

/** A fake upstream that answers each request with the next scripted step. */
function upstream(steps: Step[]) {
  const requests: Parameters<StreamFn>[0][] = [];
  const stream: StreamFn = async (req, _signal, on) => {
    requests.push(JSON.parse(JSON.stringify(req)));
    const step = steps[requests.length - 1];
    if (!step) throw new Error('unexpected request');
    await step(req, on);
    on({ kind: 'finish', reason: 'stop' });
  };
  return { stream, requests };
}
const call = (id: string, name: string, args: string): Step => (_r, on) => on({ kind: 'tool_call', index: 0, id, name, arguments: args });
const say = (text: string): Step => (_r, on) => on({ kind: 'content', text });

async function drive(steps: Step[], opts: { settings?: typeof settings; signal?: AbortSignal; history?: Message[]; info?: ServerInfo | null; compact?: AgentOptions['compact'] } = {}) {
  const up = upstream(steps);
  const committed: Message[] = [];
  await runAgent({
    history: opts.history ?? [user], settings: opts.settings ?? settings, model: null, info: opts.info ?? null, compact: opts.compact, signal: opts.signal ?? new AbortController().signal,
    stream: up.stream, onLive: () => undefined, onCommit: (m) => committed.push(m),
  });
  return { ...up, committed };
}

describe('runAgent', () => {
  it('runs a tool, sends its result back and ends on the text answer', async () => {
    const { committed, requests } = await drive([call('c1', 'calculator', '{"expression":"6*7"}'), say('It is 42.')]);
    expect(committed.map((m) => m.role)).toEqual(['assistant', 'tool', 'assistant']);
    expect(committed[0]).toMatchObject({ toolCalls: [{ id: 'c1', name: 'calculator', arguments: '{"expression":"6*7"}' }] });
    expect(committed[1]).toMatchObject({ toolCallId: 'c1', content: '42', isError: false });
    expect(committed[2]).toMatchObject({ content: 'It is 42.', status: 'done' });
    expect(requests[0]?.tools?.map((t) => t.function.name)).toEqual(['calculator', 'current_time']);
    expect(requests[0]?.toolChoice).toBe('auto');
    expect(requests[1]?.messages.slice(-2)).toEqual([
      { role: 'assistant', content: null, tool_calls: [{ id: 'c1', type: 'function', function: { name: 'calculator', arguments: '{"expression":"6*7"}' } }] },
      { role: 'tool', tool_call_id: 'c1', content: '42' },
    ]);
  });

  it('joins a call streamed in pieces and runs several calls of one reply in order', async () => {
    const pieces: Step = (_r, on) => {
      on({ kind: 'tool_call', index: 0, id: 'a', name: 'calculator', arguments: '{"expression"' });
      on({ kind: 'tool_call', index: 0, id: null, name: null, arguments: ':"1+1"}' });
      on({ kind: 'tool_call', index: 1, id: 'b', name: 'calculator', arguments: '{"expression":"2+2"}' });
    };
    const { committed } = await drive([pieces, say('done')]);
    expect(committed.filter((m) => m.role === 'tool').map((m) => (m.role === 'tool' ? [m.toolCallId, m.content] : []))).toEqual([['a', '2'], ['b', '4']]);
  });

  it('gives the model a failed call as an error result and goes on', async () => {
    const { committed } = await drive([call('c1', 'nope', '{}'), say('sorry')]);
    expect(committed[1]).toMatchObject({ role: 'tool', isError: true });
    expect(committed[2]).toMatchObject({ content: 'sorry' });
  });

  it('forces a text answer on the last step and records calls it still makes, unrun', async () => {
    const again = call('c', 'calculator', '{"expression":"1"}');
    const { committed, requests } = await drive([again, again, again]);
    expect(requests.map((r) => r.toolChoice)).toEqual(['auto', 'auto', 'none']);
    expect(requests).toHaveLength(3);
    expect(committed.map((m) => m.role)).toEqual(['assistant', 'tool', 'assistant', 'tool', 'assistant']);
    const lastReply = committed[4] as AssistantMessage;
    expect(lastReply.stepLimit).toBe(3);
    expect(lastReply.toolCalls).toHaveLength(1);
  });

  it('honours a cap of one: a single reply that may not call tools', async () => {
    const { requests, committed } = await drive([say('plain')], { settings: { ...settings, maxSteps: 1 } });
    expect(requests[0]?.toolChoice).toBe('none');
    expect(committed).toHaveLength(1);
  });

  it('sends no tools when they are off, and ignores calls', async () => {
    const { requests, committed } = await drive([call('c', 'calculator', '{}')], { settings: { ...settings, tools: false } });
    expect(requests[0]).not.toHaveProperty('tools', expect.anything());
    expect(requests[0]?.tools).toBeUndefined();
    expect(committed.map((m) => m.role)).toEqual(['assistant']);
  });

  it('stops with the user: the partial reply is kept and nothing more is asked', async () => {
    const controller = new AbortController();
    const stopMid: Step = async (_r, on) => {
      on({ kind: 'content', text: 'par' });
      controller.abort();
      throw new DOMException('aborted', 'AbortError');
    };
    const { committed, requests } = await drive([stopMid], { signal: controller.signal });
    expect(requests).toHaveLength(1);
    expect(committed).toHaveLength(1);
    expect(committed[0]).toMatchObject({ content: 'par', status: 'stopped' });
  });

  it('stopped while a tool ran: the results are kept and a stopped reply ends the chat', async () => {
    const controller = new AbortController();
    const asks: Step = (_r, on) => {
      on({ kind: 'tool_call', index: 0, id: 'c', name: 'calculator', arguments: '{"expression":"1+1"}' });
      controller.abort();
    };
    const { committed, requests } = await drive([asks], { signal: controller.signal });
    expect(requests).toHaveLength(1);
    expect(committed.map((m) => m.role)).toEqual(['assistant', 'tool', 'assistant']);
    expect(committed[2]).toMatchObject({ status: 'stopped', content: '' });
  });

  it('ends on an error without running the calls of a failed reply', async () => {
    const fail: Step = (_r, on) => {
      on({ kind: 'tool_call', index: 0, id: 'c', name: 'calculator', arguments: '{}' });
      throw new ApiError(500, { message: 'boom', code: 'upstream_failed', type: null });
    };
    const { committed } = await drive([fail]);
    expect(committed).toHaveLength(1);
    expect(committed[0]).toMatchObject({ status: 'error', error: { code: 'upstream_failed' } });
  });

  it('counts tokens per step from usage when sent, else keeps estimates', async () => {
    const withUsage: Step = (_r, on) => {
      on({ kind: 'content', text: 'x' });
      on({ kind: 'usage', usage: { promptTokens: 9, completionTokens: 2, cachedTokens: null } });
    };
    const { committed } = await drive([call('c', 'calculator', '{"expression":"1"}'), withUsage]);
    const [first, , second] = committed as AssistantMessage[];
    expect(first?.stats).toMatchObject({ promptTokens: null, completionTokens: null });
    expect(first?.stats?.promptEstimate).toBeGreaterThan(0);
    expect(second?.stats).toMatchObject({ promptTokens: 9, completionTokens: 2 });
  });

  it('records which tool definitions went with each reply, and none when tools are off', async () => {
    const withTools = await drive([say('a')]);
    expect(withTools.committed[0]).toMatchObject({ stats: { tools: { names: ['calculator', 'current_time'], tokens: expect.any(Number) } } });
    expect((withTools.committed[0] as AssistantMessage).stats?.tools?.tokens).toBeGreaterThan(0);
    const without = await drive([say('a')], { settings: { ...settings, tools: false } });
    expect((without.committed[0] as AssistantMessage).stats?.tools).toBeNull();
  });

  it('expects to read only what the server has not cached, and all of it when the tools changed', async () => {
    const prior: AssistantMessage = {
      id: 'a', role: 'assistant', content: 'earlier', reasoning: '', status: 'done', error: null, createdAt: 0,
      stats: { ttftMs: 1, tokensPerSecond: 1, promptTokens: 400, cachedTokens: null, completionTokens: 20, promptEstimate: 400, completionEstimate: 20, finish: 'stop' } as AssistantMessage['stats'],
    };
    const run = async (history: Message[], tools: boolean) => {
      const lives: number[][] = [];
      const up = upstream([say('ok')]);
      await runAgent({
        history, settings: { ...settings, tools }, model: null, info: null, signal: new AbortController().signal, stream: up.stream,
        onLive: (_m, info) => lives.push([info.promptEstimate, info.readEstimate]), onCommit: () => undefined,
      });
      return lives[0] as [number, number];
    };
    const long: UserMessage = { ...user, id: 'u2', content: 'x'.repeat(3500), typed: 'x'.repeat(3500) };
    await run([user, prior, long], false);
    // The same tool list as the request before: the earlier 420 tokens are cached.
    const [total, read] = await run([user, prior, long], false);
    expect(read).toBe(total - 420);
    // A different tool list moves everything after it, so all of it is read again.
    const [total2, read2] = await run([user, prior, long], true);
    expect(read2).toBe(total2);
  });

  describe('compacting between steps', () => {
    const old = (i: number): Message[] => [
      { ...user, id: `ou${i}`, typed: `old ${i}`, content: `old ${i} ${'x'.repeat(120)}` },
      { id: `oa${i}`, role: 'assistant', content: `old answer ${i} ${'y'.repeat(120)}`, reasoning: '', status: 'done', error: null, stats: null, createdAt: 0 },
    ];
    const info = (contextLength: number | null): ServerInfo => ({ model: 'm', contextLength, vision: 'unknown', maxPixels: null, pixelsPerToken: null, tokensPerImage: null, models: [] });
    const history = [...old(0), ...old(1), ...old(2), user];

    it('summarises older turns before the next step once tool results nearly fill the context', async () => {
      let asked = 0;
      const compact: AgentOptions['compact'] = async (h, plan) => {
        asked += 1;
        return applyCompaction(h, plan, 'SUMMARY', 's', 0, true);
      };
      const { requests } = await drive([call('c', 'calculator', '{"expression":"1"}'), say('done')], { history, info: info(200), compact });
      expect(asked).toBe(1);
      // Step 1 is sent whole (the caller compacts before a run); step 2 carries the summary, not the old turns.
      expect(requests[0]?.messages.length).toBeGreaterThan(requests[1]?.messages.length ?? 99);
      expect(JSON.stringify(requests[1]?.messages[0])).toContain('SUMMARY');
      expect(JSON.stringify(requests[1]?.messages)).not.toContain('old 0');
      expect(JSON.stringify(requests[1]?.messages)).toContain('what is 6*7?');
    });

    it('does nothing while the chat fits, without a context length, or when the summary fails', async () => {
      let asked = 0;
      const compact: AgentOptions['compact'] = async () => {
        asked += 1;
        return null;
      };
      const steps = [call('c', 'calculator', '{"expression":"1"}'), say('done')];
      await drive(steps, { history, info: info(100000), compact });
      await drive(steps, { history, info: info(null), compact });
      expect(asked).toBe(0);
      const failed = await drive(steps, { history, info: info(200), compact });
      expect(asked).toBe(1);
      expect(failed.requests[1]?.messages.length).toBeGreaterThan(5);
      expect(failed.committed[failed.committed.length - 1]).toMatchObject({ content: 'done' });
    });
  });
});

describe('runAgent with a tool host', () => {
  const request = (name: string): ApprovalRequest => ({ callId: 'c1', tool: name, title: name, server: 's', args: '{}', readOnly: false });
  /** A host whose "write" needs the user's OK and whose results are recorded. */
  function host(over: Partial<ToolHost> = {}) {
    const seen: { query: string; unlocked: string[] }[] = [];
    const ran: string[] = [];
    const h: ToolHost = {
      external: true,
      definitions: (query, unlocked) => {
        seen.push({ query, unlocked: [...unlocked] });
        return [{ type: 'function', function: { name: 'write', description: 'w', parameters: {} } }];
      },
      approval: (c) => (c.name === 'write' ? request('write') : null),
      run: async (c, _signal, unlock) => {
        ran.push(c.name);
        if (c.name === 'find') unlock(['unlocked_tool']);
        return { content: `ran ${c.name}`, isError: false };
      },
      ...over,
    };
    return { h, seen, ran };
  }
  async function go(h: ToolHost, steps: Step[], opts: { approve?: AgentOptions['approve']; settings?: typeof settings; signal?: AbortSignal } = {}) {
    const up = upstream(steps);
    const committed: Message[] = [];
    await runAgent({
      history: [user], settings: opts.settings ?? settings, model: null, info: null, signal: opts.signal ?? new AbortController().signal,
      stream: up.stream, onLive: () => undefined, onCommit: (m) => committed.push(m), tools: h, approve: opts.approve,
    });
    return { ...up, committed };
  }
  const toolText = (m: Message | undefined) => (m?.role === 'tool' ? m.content : null);

  it('asks before a call that needs it and runs it on allow', async () => {
    const { h, ran } = host();
    const asked: string[] = [];
    const { committed } = await go(h, [call('c1', 'write', '{}'), say('done')], { approve: async (r) => (asked.push(r.tool), 'allow') });
    expect(asked).toEqual(['write']);
    expect(ran).toEqual(['write']);
    expect(toolText(committed[1])).toBe('ran write');
  });

  it('hands the model a refusal on deny and does not run the tool, and the model carries on', async () => {
    const { h, ran } = host();
    const { committed, requests } = await go(h, [call('c1', 'write', '{}'), say('ok, not doing it')], { approve: async () => 'deny' });
    expect(ran).toEqual([]);
    expect(committed[1]).toMatchObject({ role: 'tool', isError: true, content: 'error: the user did not allow this tool call' });
    expect(committed.at(-1)).toMatchObject({ role: 'assistant', content: 'ok, not doing it' });
    expect(requests).toHaveLength(2);
  });

  it('refuses when nobody can be asked, and when asking fails', async () => {
    const a = host();
    const noAsker = await go(a.h, [call('c1', 'write', '{}'), say('x')]);
    expect(a.ran).toEqual([]);
    expect(toolText(noAsker.committed[1])).toContain('did not allow');
    const b = host();
    const failing = await go(b.h, [call('c1', 'write', '{}'), say('x')], { approve: async () => { throw new Error('ui gone'); } });
    expect(b.ran).toEqual([]);
    expect(toolText(failing.committed[1])).toContain('did not allow');
  });

  it('does not ask for a call the host lets through', async () => {
    const { h, ran } = host({ approval: () => null });
    let asked = 0;
    await go(h, [call('c1', 'write', '{}'), say('x')], { approve: async () => (asked++, 'allow') });
    expect(asked).toBe(0);
    expect(ran).toEqual(['write']);
  });

  it('stops where it is when Stop is pressed while a question is open, without running the tool', async () => {
    const { h, ran } = host();
    const controller = new AbortController();
    const { committed } = await go(h, [call('c1', 'write', '{}')], {
      signal: controller.signal,
      approve: (_r, signal) => new Promise((resolve) => {
        signal.addEventListener('abort', () => resolve('deny'));
        controller.abort();
      }),
    });
    expect(ran).toEqual([]);
    expect(committed.at(-1)).toMatchObject({ role: 'assistant', status: 'stopped' });
  });

  it('gives the host the user\'s last message, and what the model unlocked on the next step', async () => {
    const { h, seen } = host({ approval: () => null });
    await go(h, [call('c1', 'find', '{"query":"x"}'), say('done')]);
    expect(seen[0]).toEqual({ query: 'what is 6*7?', unlocked: [] });
    expect(seen[1]).toEqual({ query: 'what is 6*7?', unlocked: ['unlocked_tool'] });
  });

  it('tells the model to treat tool results as data, after the user\'s own system prompt, only when outside tools are in play', async () => {
    const withUser = await go(host().h, [say('x')], { settings: { ...settings, systemPrompt: 'Be brief.' } });
    expect(withUser.requests[0]?.messages[0]).toMatchObject({ role: 'system', content: `Be brief.\n\n${TOOL_NOTE}` });
    const alone = await go(host().h, [say('x')]);
    expect(alone.requests[0]?.messages[0]).toMatchObject({ role: 'system', content: TOOL_NOTE });
    const local = await go(host({ external: false }).h, [say('x')], { settings: { ...settings, systemPrompt: 'Be brief.' } });
    expect(local.requests[0]?.messages[0]).toMatchObject({ role: 'system', content: 'Be brief.' });
    // NEGATIVE: with tools off the prompt is untouched and no tools are sent
    const off = await go(host().h, [say('x')], { settings: { ...settings, tools: false, systemPrompt: 'Be brief.' } });
    expect(off.requests[0]?.messages[0]).toMatchObject({ content: 'Be brief.' });
    expect(off.requests[0]?.tools).toBeUndefined();
  });
});
