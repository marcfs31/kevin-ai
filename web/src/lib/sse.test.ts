import { describe, expect, it } from 'vitest';
import { SseParser, parsePayload } from './sse';

describe('SseParser', () => {
  it('joins events split across chunks at any byte', () => {
    const text = 'data: {"a":1}\n\ndata: {"b":2}\n\n';
    for (let cut = 0; cut <= text.length; cut++) {
      const p = new SseParser();
      const out = [...p.push(text.slice(0, cut)), ...p.push(text.slice(cut)), ...p.end()];
      expect(out).toEqual(['{"a":1}', '{"b":2}']);
    }
  });

  it('handles CRLF split between chunks, comments and multi-line data', () => {
    const p = new SseParser();
    expect(p.push(': keep-alive\r')).toEqual([]);
    expect(p.push('\ndata: one\r')).toEqual([]);
    expect(p.push('\ndata: two\r\n\r\n')).toEqual(['one\ntwo']);
  });

  it('flushes a last event without a blank line', () => {
    const p = new SseParser();
    expect(p.push('data: [DONE]')).toEqual([]);
    expect(p.end()).toEqual(['[DONE]']);
  });

  it('ignores fields other than data', () => {
    const p = new SseParser();
    expect(p.push('event: x\nid: 3\nretry: 10\ndata:no-space\n\n')).toEqual(['no-space']);
  });
});

describe('parsePayload', () => {
  it('reads content and reasoning deltas', () => {
    expect(parsePayload('{"choices":[{"index":0,"delta":{"reasoning_content":"hm"},"finish_reason":null}]}')).toEqual([
      { kind: 'reasoning', text: 'hm' },
    ]);
    expect(parsePayload('{"choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}')).toEqual([]);
    expect(parsePayload('{"choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}]}')).toEqual([{ kind: 'content', text: 'Hi' }]);
  });

  it('reads finish reasons and usage, with and without cached tokens', () => {
    expect(parsePayload('{"choices":[{"index":0,"delta":{},"finish_reason":"length"}]}')).toEqual([{ kind: 'finish', reason: 'length' }]);
    expect(parsePayload('{"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}')).toEqual([
      { kind: 'usage', usage: { promptTokens: 5, completionTokens: 2, cachedTokens: null } },
    ]);
    expect(
      parsePayload('{"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2,"prompt_tokens_details":{"cached_tokens":4}}}'),
    ).toEqual([{ kind: 'usage', usage: { promptTokens: 5, completionTokens: 2, cachedTokens: 4 } }]);
  });

  it('reads errors in the stream and refuses malformed JSON', () => {
    expect(parsePayload('{"error":{"message":"boom","type":"server_error","code":"context_overflow"}}')).toEqual([
      { kind: 'error', error: { message: 'boom', type: 'server_error', code: 'context_overflow' } },
    ]);
    expect(parsePayload('{not json')).toEqual([{ kind: 'error', error: { message: '', code: 'bad_stream', type: null } }]);
  });

  it('marks the end of the stream', () => {
    expect(parsePayload('[DONE]')).toEqual([{ kind: 'done' }]);
  });

  it('ignores usage without token counts (negative case)', () => {
    expect(parsePayload('{"choices":[],"usage":{}}')).toEqual([]);
  });
});

describe('tool calls in a stream', () => {
  const payload = (calls: unknown[]) => JSON.stringify({ choices: [{ index: 0, delta: { tool_calls: calls }, finish_reason: null }] });

  it('reads a whole call and a call streamed in pieces', () => {
    expect(parsePayload(payload([{ index: 0, id: 'c1', type: 'function', function: { name: 'calculator', arguments: '{"a":1}' } }]))).toEqual([
      { kind: 'tool_call', index: 0, id: 'c1', name: 'calculator', arguments: '{"a":1}' },
    ]);
    expect(parsePayload(payload([{ index: 2, function: { arguments: '"x"}' } }]))).toEqual([
      { kind: 'tool_call', index: 2, id: null, name: null, arguments: '"x"}' },
    ]);
  });

  it('numbers calls by position when the server gives no index, and skips junk', () => {
    const events = parsePayload(payload([{ id: 'a', function: { name: 'f' } }, 'junk', { id: 'b', function: { name: 'g' } }]));
    expect(events.map((e) => (e.kind === 'tool_call' ? [e.index, e.id] : null))).toEqual([[0, 'a'], [2, 'b']]);
  });

  it('keeps the finish reason of a tool turn', () => {
    expect(parsePayload(JSON.stringify({ choices: [{ index: 0, delta: {}, finish_reason: 'tool_calls' }] }))).toEqual([{ kind: 'finish', reason: 'tool_calls' }]);
  });
});
