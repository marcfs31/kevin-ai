import { describe, expect, it } from 'vitest';
import { runAgent, type StreamFn } from './agent';
import { ApiError } from './api';
import { DEFAULT_SETTINGS, type AssistantMessage, type Message, type UserMessage } from './storage';

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

async function drive(steps: Step[], opts: { settings?: typeof settings; signal?: AbortSignal } = {}) {
  const up = upstream(steps);
  const committed: Message[] = [];
  await runAgent({
    history: [user], settings: opts.settings ?? settings, model: null, info: null, signal: opts.signal ?? new AbortController().signal,
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
});
