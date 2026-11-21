import { describe, expect, it } from 'vitest';
import type { AssistantMessage, Stats } from './storage';
import { estimateWireTokens, liveCount, replyCount, sessionTotals } from './tokens';

const stats = (o: Partial<Stats>): Stats => ({
  ttftMs: null, tokensPerSecond: null, promptTokens: null, cachedTokens: null, completionTokens: null, promptEstimate: null, completionEstimate: null, ...o,
});
const reply = (s: Stats | null): AssistantMessage => ({
  id: 'a', role: 'assistant', content: 'x', reasoning: '', status: 'done', error: null, stats: s, createdAt: 0,
});

describe('tokens', () => {
  it('estimates text, tool calls and images of known size', () => {
    const wire = [
      { role: 'user' as const, content: 'a'.repeat(35) },
      { role: 'user' as const, content: [{ type: 'image_url' as const, image_url: { url: 'data:image/png;base64,AA' } }, { type: 'text' as const, text: '' }] },
      { role: 'assistant' as const, content: null, tool_calls: [{ id: '1', type: 'function' as const, function: { name: 'f', arguments: '{}' } }] },
    ];
    const info = { model: 'm', contextLength: null, vision: 'ok' as const, maxPixels: null, pixelsPerToken: null, tokensPerImage: 100, models: [] };
    expect(estimateWireTokens(wire, info)).toBe(10 + 100 + 1);
    expect(estimateWireTokens(wire, null)).toBe(10 + 1);
  });

  it('prefers real usage over an estimate, per figure', () => {
    expect(replyCount(stats({ promptTokens: 12, completionTokens: 3, promptEstimate: 99, completionEstimate: 99 }))).toEqual({
      read: { n: 12, estimated: false }, written: { n: 3, estimated: false },
    });
    expect(replyCount(stats({ promptEstimate: 40, completionEstimate: 7 }))).toEqual({
      read: { n: 40, estimated: true }, written: { n: 7, estimated: true },
    });
    expect(replyCount(stats({ promptTokens: 40 }))).toEqual({ read: { n: 40, estimated: false } });
    expect(replyCount(stats({}))).toBeNull();
    expect(replyCount(null)).toBeNull();
  });

  it('counts a live reply from estimates until usage arrives', () => {
    expect(liveCount(null, 50, 4)).toEqual({ read: { n: 50, estimated: true }, written: { n: 4, estimated: true } });
    expect(liveCount({ promptTokens: 48, completionTokens: 5, cachedTokens: null }, 50, 4)).toEqual({
      read: { n: 48, estimated: false }, written: { n: 5, estimated: false },
    });
  });

  it('sums a session and marks it estimated when any reply was', () => {
    const real = reply(stats({ promptTokens: 10, completionTokens: 2 }));
    const est = reply(stats({ promptEstimate: 30, completionEstimate: 6 }));
    expect(sessionTotals([real, real])).toEqual({ read: { n: 20, estimated: false }, written: { n: 4, estimated: false } });
    expect(sessionTotals([real, est])).toEqual({ read: { n: 40, estimated: true }, written: { n: 8, estimated: true } });
    // A reply saved before completions were estimated still counts what it has.
    expect(sessionTotals([reply(stats({ promptTokens: 40 })), real])).toEqual({ read: { n: 50, estimated: false }, written: { n: 2, estimated: false } });
    expect(sessionTotals([reply(null)])).toBeNull();
    expect(sessionTotals([])).toBeNull();
  });
});
