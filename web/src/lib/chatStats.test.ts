import { describe, expect, it } from 'vitest';
import { chatFigures, contextBar } from './chatStats';
import type { AssistantMessage, Message, Stats, SummaryMessage, UserMessage } from './storage';
import type { Count } from './tokens';

let n = 0;
const user = (text: string): UserMessage => ({ id: `u${n++}`, role: 'user', content: text, typed: text, attachmentNames: [], createdAt: 0 });
const stats = (o: Partial<Stats> = {}): Stats => ({
  ttftMs: null, tokensPerSecond: null, promptTokens: null, cachedTokens: null, completionTokens: null, promptEstimate: null, completionEstimate: null, tools: null, ...o,
});
const bot = (o: Partial<Stats>, status: AssistantMessage['status'] = 'done'): AssistantMessage => ({
  id: `a${n++}`, role: 'assistant', content: 'a', reasoning: '', status, error: null, stats: stats(o), createdAt: 0,
});
const summary = (): SummaryMessage => ({ id: `s${n++}`, role: 'summary', content: 's', covered: 2, auto: true, createdAt: 0 });
const info = (contextLength: number | null) => ({
  model: 'm', contextLength, vision: 'ok' as const, maxPixels: null, pixelsPerToken: null, tokensPerImage: 100, models: [],
});

describe('chatFigures', () => {
  it('is null for an empty chat', () => {
    expect(chatFigures([])).toBeNull();
    expect(contextBar([], '', info(100))).toBeNull();
  });

  it('sums the tokens, counts the turns and compactions, and averages the speed of the replies that had one', () => {
    const m: Message[] = [
      user('q1'), bot({ promptTokens: 10, completionTokens: 4, tokensPerSecond: 4 }),
      summary(),
      user('q2'), bot({ promptTokens: 30, completionTokens: 6, tokensPerSecond: 6 }), bot({ promptTokens: 1, completionTokens: 1 }),
    ];
    expect(chatFigures(m)).toEqual({
      read: { n: 41, estimated: false }, written: { n: 11, estimated: false }, turns: 2, compactions: 1, rate: 5,
    });
  });

  it('marks a total estimated as soon as one reply is, and has no speed when none reported one', () => {
    const m: Message[] = [user('q'), bot({ promptTokens: 10, completionTokens: 4 }), user('q'), bot({ promptEstimate: 7, completionEstimate: 2 }, 'stopped')];
    expect(chatFigures(m)).toMatchObject({ read: { n: 17, estimated: true }, written: { n: 6, estimated: true }, rate: null });
  });

  it('has zero totals, not a crash, for a chat whose only reply failed before it counted anything', () => {
    const m: Message[] = [user('q'), { ...bot({}), status: 'error', stats: null }];
    expect(chatFigures(m)).toMatchObject({ read: { n: 0 }, written: { n: 0 }, turns: 1 });
  });
});

describe('contextBar', () => {
  const m: Message[] = [user('q'), bot({ promptTokens: 60, completionTokens: 20 })];

  it('takes the tool definitions of the last request out of the prompt and lists them', () => {
    const tools = { names: ['web_search', 'time__get_current_time'], tokens: 25 };
    const withTools: Message[] = [user('q'), bot({ promptTokens: 60, completionTokens: 20, tools })];
    expect(contextBar(withTools, '', info(200))).toMatchObject({ prompt: 35, tools, reply: 20, free: 120 });
    // The server's own count wins when it is smaller than the estimate.
    expect(contextBar([user('q'), bot({ promptTokens: 10, completionTokens: 1, tools })], '', info(200))).toMatchObject({ prompt: 0, tools: { tokens: 10 } });
  });

  it('shows no tools box when the request had none, after a summary, or for a live reply without tools', () => {
    const live = (read: number, written: number): Count => ({ read: { n: read, estimated: true }, written: { n: written, estimated: true } });
    expect(contextBar(m, '', info(200))?.tools).toBeNull();
    const old: Message[] = [user('q'), bot({ promptTokens: 60, completionTokens: 20, tools: { names: ['a'], tokens: 5 } }), summary(), user('q2')];
    expect(contextBar(old, '', info(200))?.tools).toBeNull();
    expect(contextBar(m, '', info(200), live(90, 7), undefined, null)?.tools).toBeNull();
    expect(contextBar(m, '', info(200), live(90, 7), undefined, { names: ['a'], tokens: 30 })).toMatchObject({ prompt: 60, tools: { names: ['a'], tokens: 30 } });
  });

  it('splits the window into prompt, last reply and what is free', () => {
    expect(contextBar(m, '', info(200))).toMatchObject({ prompt: 60, reply: 20, free: 120, limit: 200, share: 0.4, estimated: false });
  });

  it('never shows negative free space when the chat no longer fits', () => {
    expect(contextBar(m, '', info(50))).toMatchObject({ free: 0, share: 1.6 });
  });

  it('says nothing about free space or a share when the server gave no window size', () => {
    expect(contextBar(m, '', info(null))).toMatchObject({ prompt: 60, reply: 20, free: null, limit: null, share: null });
    expect(contextBar(m, '', null)?.free).toBeNull();
  });
});

describe('while a reply streams', () => {
  const live = (readN: number, writtenN: number, estimated = true): Count => ({ read: { n: readN, estimated }, written: { n: writtenN, estimated } });
  const m: Message[] = [user('q1'), bot({ promptTokens: 60, completionTokens: 20 }), user('q2')];

  it('adds the reply in progress to the totals and keeps the turn count', () => {
    expect(chatFigures(m, live(90, 7))).toMatchObject({ read: { n: 150, estimated: true }, written: { n: 27, estimated: true }, turns: 2 });
    // NEGATIVE: with nothing streaming the totals are the finished replies only
    expect(chatFigures(m, null)).toMatchObject({ read: { n: 60, estimated: false }, written: { n: 20, estimated: false } });
  });

  it('shows the window as the running request holds it, not the finished reply plus the new one', () => {
    expect(contextBar(m, '', info(200), live(90, 7))).toMatchObject({ prompt: 90, reply: 7, free: 103, share: 97 / 200, estimated: true });
    expect(contextBar(m, '', info(200), live(90, 7, false))?.estimated).toBe(false);
    expect(contextBar(m, '', info(200), null)).toMatchObject({ prompt: 61, reply: 20, estimated: true });
  });

  it('still says nothing for an empty chat', () => {
    expect(chatFigures([], live(1, 1))).toBeNull();
    expect(contextBar([], '', info(100), live(1, 1))).toBeNull();
  });
});
