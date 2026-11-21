import { describe, expect, it, vi } from 'vitest';
import { DONE, FINISH, USAGE, delta, sseResponse } from '../test/helpers';
import { ApiError, parseModels, requestBody, streamChat } from './api';
import type { StreamEvent } from './sse';

describe('parseModels', () => {
  it('treats a server without capabilities as vision unknown', () => {
    expect(parseModels({ data: [{ id: 'm' }] })).toEqual({
      model: 'm', contextLength: null, vision: 'unknown', maxPixels: null, pixelsPerToken: null, tokensPerImage: null,
    });
  });

  it('reads context length and vision capabilities', () => {
    const qwen = parseModels({
      data: [{ id: 'm', context_length: 8192, capabilities: { vision: { status: 'ok', max_pixels: 1048576, pixels_per_token: 1024 } } }],
    });
    expect(qwen).toMatchObject({ contextLength: 8192, vision: 'ok', maxPixels: 1048576, pixelsPerToken: 1024, tokensPerImage: null });
    const gemma = parseModels({
      data: [{ id: 'g', capabilities: { vision: { status: 'ok', max_pixels: 1048576, tokens_per_image: 280 } } }],
    });
    expect(gemma).toMatchObject({ vision: 'ok', maxPixels: 1048576, pixelsPerToken: null, tokensPerImage: 280 });
  });

  it('keeps unavailable distinct from unknown, and refuses unknown statuses', () => {
    expect(parseModels({ data: [{ id: 'm', capabilities: { vision: { status: 'unavailable' } } }] })?.vision).toBe('unavailable');
    expect(parseModels({ data: [{ id: 'm', capabilities: { vision: { status: 'maybe' } } }] })?.vision).toBe('unknown');
    expect(parseModels({ data: [] })).toBeNull();
  });
});

describe('requestBody', () => {
  it('asks for a stream with usage and leaves max_tokens out when unset', () => {
    const body = JSON.parse(requestBody({ messages: [{ role: 'user', content: 'hi' }], temperature: 0.7, maxTokens: null }));
    expect(body).toEqual({ messages: [{ role: 'user', content: 'hi' }], stream: true, stream_options: { include_usage: true }, temperature: 0.7 });
    expect(JSON.parse(requestBody({ messages: [], temperature: 0, maxTokens: 64 })).max_tokens).toBe(64);
  });
});

describe('streamChat', () => {
  const req = { messages: [{ role: 'user' as const, content: 'hi' }], temperature: 0.7, maxTokens: null };

  it('delivers events in order across split chunks', async () => {
    const whole = delta('think', 'reasoning_content') + delta('Hel') + delta('lo') + FINISH + USAGE + DONE;
    const chunks = whole.match(/[\s\S]{1,7}/g) ?? [];
    vi.stubGlobal('fetch', vi.fn(async () => sseResponse(chunks)));
    const events: StreamEvent[] = [];
    await streamChat(req, new AbortController().signal, (e) => events.push(e));
    expect(events.map((e) => e.kind)).toEqual(['reasoning', 'content', 'content', 'finish', 'usage', 'done']);
  });

  it('throws the server error with its code on a 400', async () => {
    const body = { error: { message: 'too big', type: 'invalid_request_error', code: 'image_too_large' } };
    vi.stubGlobal('fetch', vi.fn(async () => new Response(JSON.stringify(body), { status: 400 })));
    const err = await streamChat(req, new AbortController().signal, () => {}).catch((e: unknown) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err).toMatchObject({ status: 400, code: 'image_too_large', serverMessage: 'too big' });
  });

  it('throws an error event sent mid-stream', async () => {
    const chunks = [delta('a'), 'data: {"error":{"message":"gone","type":"server_error","code":null}}\n\n', DONE];
    vi.stubGlobal('fetch', vi.fn(async () => sseResponse(chunks)));
    await expect(streamChat(req, new AbortController().signal, () => {})).rejects.toMatchObject({ serverMessage: 'gone' });
  });

  it('reports a stream that ends without a finish as cut short', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => sseResponse([delta('a')])));
    await expect(streamChat(req, new AbortController().signal, () => {})).rejects.toMatchObject({ code: 'stream_cut' });
  });

  it('reports an unreachable server as status 0', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => Promise.reject(new TypeError('Failed to fetch'))));
    await expect(streamChat(req, new AbortController().signal, () => {})).rejects.toMatchObject({ status: 0 });
  });
});
