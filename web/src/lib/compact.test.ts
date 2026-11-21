import { describe, expect, it } from 'vitest';
import { AUTO_RATIO, applyCompaction, planCompaction, projectCompaction, shouldCompact, summaryRequest, transcript, usedTokens } from './compact';
import { toWire } from './request';
import type { AssistantMessage, Message, Stats, UserMessage } from './storage';

let n = 0;
const user = (text: string, content: UserMessage['content'] = text): UserMessage => ({
  id: `u${n++}`, role: 'user', content, typed: text, attachmentNames: [], createdAt: 0,
});
const stats = (o: Partial<Stats> = {}): Stats => ({
  ttftMs: null, tokensPerSecond: null, promptTokens: null, cachedTokens: null, completionTokens: null, promptEstimate: null, completionEstimate: null, ...o,
});
const bot = (text: string, o: Partial<AssistantMessage> = {}): AssistantMessage => ({
  id: `a${n++}`, role: 'assistant', content: text, reasoning: '', status: 'done', error: null, stats: null, createdAt: 0, ...o,
});
const turns = (count: number): Message[] => Array.from({ length: count }, (_, i) => [user(`q${i}`), bot(`a${i}`)]).flat();
const mustPlan = (m: Message[]) => {
  const p = planCompaction(m);
  if (!p) throw new Error('no plan');
  return p;
};
const info = (contextLength: number | null) => ({
  model: 'm', contextLength, vision: 'ok' as const, maxPixels: null, pixelsPerToken: null, tokensPerImage: 100, models: [],
});

describe('planCompaction', () => {
  it('keeps the last two turns and summarises the rest', () => {
    const plan = planCompaction(turns(5));
    expect(plan?.cut).toBe(6);
    expect(plan?.span.map((m) => (m.role === 'user' ? m.typed : m.role === 'assistant' ? m.content : ''))).toEqual(['q0', 'a0', 'q1', 'a1', 'q2', 'a2']);
  });

  it('has nothing to do for a short chat', () => {
    expect(planCompaction(turns(2))).toBeNull();
    expect(planCompaction(turns(3))?.span).toHaveLength(2);
    expect(planCompaction([])).toBeNull();
  });

  it('starts a cut on a user message so no turn is split', () => {
    const m = [...turns(2), user('q2'), bot('a2'), bot('a2b'), user('q3'), bot('a3')];
    const plan = planCompaction(m);
    expect(m[plan?.cut ?? 0]?.role).toBe('user');
  });

  it('builds on the previous summary and only counts what came after it', () => {
    const once = mustPlan(turns(5));
    const m = applyCompaction(turns(5), once, 'first', 's1', 0, false);
    expect(planCompaction(m)).toBeNull();
    const more = [...m, ...turns(3)];
    const plan = planCompaction(more);
    expect(plan?.previous?.content).toBe('first');
    expect(plan?.span.length).toBeGreaterThan(0);
    expect(plan?.span.every((x) => x.role !== 'summary')).toBe(true);
  });
});

describe('applyCompaction', () => {
  it('inserts the summary before the kept turns and drops nothing', () => {
    const m = turns(5);
    const out = applyCompaction(m, mustPlan(m), '  the summary \n', 's', 7, true);
    expect(out).toHaveLength(m.length + 1);
    expect(out[6]).toMatchObject({ role: 'summary', content: 'the summary', covered: 6, auto: true });
    expect(out.filter((x) => x.role !== 'summary')).toEqual(m);
  });
});

describe('toWire after a summary', () => {
  it('sends the summary in the system message and only the turns after it', () => {
    const m = turns(5);
    const out = applyCompaction(m, mustPlan(m), 'S', 's', 0, false);
    const wire = toWire('Be brief.', out);
    expect(wire[0]).toMatchObject({ role: 'system' });
    expect(String(wire[0]?.content)).toContain('Be brief.');
    expect(String(wire[0]?.content)).toContain('S');
    expect(wire.slice(1).map((w) => w.content)).toEqual(['q3', 'a3', 'q4', 'a4']);
    expect(toWire('', out)[0]?.content).toMatch(/^Summary of the earlier part/);
  });
});

describe('transcript', () => {
  it('turns images into a text note and keeps a stopped partial reply, marked', () => {
    const m: Message[] = [
      user('look', [{ type: 'image_url', image_url: { url: 'data:image/png;base64,AA' } }, { type: 'text', text: 'look' }]),
      bot('The picture shows a ca', { status: 'stopped' }),
      user('and'), bot('done'), ...turns(2),
    ];
    const text = transcript(mustPlan(m), null);
    expect(text).toContain('[1 image attached here; the picture itself is not available]');
    expect(text).not.toContain('base64');
    expect(text).toContain('The picture shows a ca');
    expect(text).toContain('[this reply was stopped by the user before it ended]');
  });

  it('puts tool calls and results into the transcript', () => {
    const asks = bot('', { toolCalls: [{ id: 'c', name: 'calculator', arguments: '{"expression":"6*7"}' }] });
    const res = { id: 't', role: 'tool' as const, toolCallId: 'c', name: 'calculator', content: '42', isError: false, createdAt: 0 };
    const text = transcript(mustPlan([user('q'), asks, res, bot('42'), ...turns(2)]), null);
    expect(text).toContain('[called tool calculator with {"expression":"6*7"}]');
    expect(text).toContain('Tool calculator returned:\n42');
  });

  it('carries the previous summary forward', () => {
    const m = applyCompaction(turns(5), mustPlan(turns(5)), 'old facts', 's', 0, false);
    const text = transcript(mustPlan([...m, ...turns(3)]), null);
    expect(text.startsWith('Earlier summary:\nold facts')).toBe(true);
  });

  it('cuts long messages evenly to fit the context, with a mark', () => {
    const m = [user('x'.repeat(20000)), bot('y'.repeat(20000)), ...turns(3)];
    const text = transcript(mustPlan(m), info(1000));
    expect(text.length).toBeLessThan(2 * 1000 * 3.5);
    expect(text).toContain('[… cut to fit]');
  });

  it('asks for a plain summary in the system message', () => {
    const req = summaryRequest(mustPlan(turns(5)), null);
    expect(req.map((r) => r.role)).toEqual(['system', 'user']);
  });
});

describe('shouldCompact', () => {
  const long = (promptTokens: number | null): Message[] => {
    const m = turns(5);
    const last = m[m.length - 1] as AssistantMessage;
    last.stats = promptTokens === null ? null : stats({ promptTokens, completionTokens: 10 });
    return m;
  };

  it('compacts at the share of the context, by the server’s own count', () => {
    const limit = 1000;
    const need = limit * AUTO_RATIO;
    expect(shouldCompact(long(Math.ceil(need) - 11 - 5), '', 5, info(limit))).toBe(false);
    expect(shouldCompact(long(Math.ceil(need) - 10 - 5), '', 5, info(limit))).toBe(true);
    expect(usedTokens(long(100), '', info(limit))).toBe(110);
  });

  it('falls back to an estimate when the server sent no usage', () => {
    const m = [user('x'.repeat(3500)), bot('y'.repeat(3500)), ...turns(3)];
    expect(shouldCompact(m, '', 0, info(1000))).toBe(true);
    expect(shouldCompact(m, '', 0, info(100000))).toBe(false);
  });

  it('never compacts without a known context, a short chat, or a reply that failed', () => {
    expect(shouldCompact(long(5000), '', 0, info(null))).toBe(false);
    expect(shouldCompact(turns(2), '', 99999, info(100))).toBe(false);
    const m = long(null);
    (m[m.length - 1] as AssistantMessage).stats = stats({ promptTokens: 900, completionTokens: 0 });
    (m[m.length - 1] as AssistantMessage).status = 'error';
    expect(usedTokens(m, '', info(1000))).toBeLessThan(900);
  });
});

describe('projectCompaction', () => {
  it('shows the history as it will be sent once compaction is due, and as it is otherwise', () => {
    const m = [user('x'.repeat(3500)), bot('y'.repeat(3500)), ...turns(3)];
    const out = projectCompaction(m, '', 0, info(1000));
    expect(out.find((x) => x.role === 'summary')).toBeTruthy();
    expect(toWire('', out)).toHaveLength(1 + 4);
    expect(projectCompaction(m, '', 0, info(100000))).toBe(m);
    expect(projectCompaction(m, '', 0, info(null))).toBe(m);
  });
});
