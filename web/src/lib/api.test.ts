import { describe, expect, it, vi } from 'vitest';
import { DONE, FINISH, USAGE, delta, sseResponse } from '../test/helpers';
import {
  ApiError, infoForModel, parseModels, requestBody, resetServer, streamChat,
  toolsAddCatalog, toolsCall, toolsList, toolsSearch, toolsStatus, toolsStop, type ToolsHost,
} from './api';
import type { StreamEvent } from './sse';

describe('parseModels', () => {
  it('treats a server without capabilities as vision unknown', () => {
    expect(parseModels({ data: [{ id: 'm' }] })).toEqual({
      model: 'm', contextLength: null, vision: 'unknown', maxPixels: null, pixelsPerToken: null, tokensPerImage: null,
      models: [{ id: 'm', contextLength: null, vision: 'unknown', maxPixels: null, pixelsPerToken: null, tokensPerImage: null }],
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

  it('reads llama-server: the window from meta.n_ctx and vision from the input modalities', () => {
    const gemma = parseModels({ data: [{ id: 'g', architecture: { input_modalities: ['text', 'image', 'audio'] }, meta: { n_ctx: 32768 } }] });
    expect(gemma).toMatchObject({ contextLength: 32768, vision: 'ok' });
    // NEGATIVE: a text-only model says so; it is not left unknown
    expect(parseModels({ data: [{ id: 'w', architecture: { input_modalities: ['text'] }, meta: { n_ctx: 8192 } }] })).toMatchObject({ contextLength: 8192, vision: 'unavailable' });
    // lm-serve's own fields win when both are there, and a malformed list is unknown
    expect(parseModels({ data: [{ id: 'q', context_length: 4096, meta: { n_ctx: 99 }, capabilities: { vision: { status: 'unavailable' } }, architecture: { input_modalities: ['image'] } }] })).toMatchObject({ contextLength: 4096, vision: 'unavailable' });
    expect(parseModels({ data: [{ id: 'x', architecture: { input_modalities: 'image' }, meta: { n_ctx: 0 } }] })).toMatchObject({ contextLength: null, vision: 'unknown' });
  });

  it('keeps unavailable distinct from unknown, and refuses unknown statuses', () => {
    expect(parseModels({ data: [{ id: 'm', capabilities: { vision: { status: 'unavailable' } } }] })?.vision).toBe('unavailable');
    expect(parseModels({ data: [{ id: 'm', capabilities: { vision: { status: 'maybe' } } }] })?.vision).toBe('unknown');
    expect(parseModels({ data: [] })).toBeNull();
  });
});

describe('model list', () => {
  const two = parseModels({
    data: [
      { id: 'qwen', context_length: 8192, capabilities: { vision: { status: 'ok', tokens_per_image: 280 } } },
      { id: 'rabbit', context_length: 32768 },
      { id: '' },
      'junk',
    ],
  });

  it('keeps every named model with its own context length and vision', () => {
    expect(two?.models.map((m) => [m.id, m.contextLength, m.vision])).toEqual([['qwen', 8192, 'ok'], ['rabbit', 32768, 'unknown']]);
    expect(two).toMatchObject({ model: 'qwen', contextLength: 8192 });
  });

  it('answers with the chosen model’s figures, and the first model for an unknown choice', () => {
    expect(infoForModel(two, 'rabbit')).toMatchObject({ model: 'rabbit', contextLength: 32768, vision: 'unknown', tokensPerImage: null });
    expect(infoForModel(two, 'gone')).toMatchObject({ model: 'qwen', contextLength: 8192 });
    expect(infoForModel(two, null)?.model).toBe('qwen');
    expect(infoForModel(null, 'rabbit')).toBeNull();
  });
});

describe('resetServer', () => {
  const answer = (status: number) => vi.stubGlobal('fetch', vi.fn(async () => new Response('{}', { status })));

  it('is done when the server resets, and when it has nothing to reset (llama-server answers 404)', async () => {
    answer(200);
    await expect(resetServer()).resolves.toBeUndefined();
    answer(404);
    await expect(resetServer()).resolves.toBeUndefined();
  });

  it('still fails, with the server\'s status, on any other error, and as status 0 when unreachable', async () => {
    answer(500);
    await expect(resetServer()).rejects.toMatchObject({ status: 500 });
    vi.stubGlobal('fetch', vi.fn(async () => { throw new TypeError('Failed to fetch'); }));
    await expect(resetServer()).rejects.toMatchObject({ status: 0 });
  });
});

describe('requestBody thinking', () => {
  const base = { messages: [], temperature: 0.7, maxTokens: null };

  it('sends the template switch only when the effort chose one', () => {
    expect(JSON.parse(requestBody({ ...base, thinking: false })).chat_template_kwargs).toEqual({ enable_thinking: false });
    expect(JSON.parse(requestBody({ ...base, thinking: true })).chat_template_kwargs).toEqual({ enable_thinking: true });
    // NEGATIVE: no choice, no field
    expect('chat_template_kwargs' in JSON.parse(requestBody(base))).toBe(false);
  });
});

describe('requestBody', () => {
  it('asks for a stream with usage and leaves max_tokens out when unset', () => {
    const body = JSON.parse(requestBody({ messages: [{ role: 'user', content: 'hi' }], temperature: 0.7, maxTokens: null }));
    expect(body).toEqual({ messages: [{ role: 'user', content: 'hi' }], stream: true, stream_options: { include_usage: true }, temperature: 0.7 });
    expect(JSON.parse(requestBody({ messages: [], temperature: 0, maxTokens: 64 })).max_tokens).toBe(64);
  });
});

describe('requestBody with tools', () => {
  const tools = [{ type: 'function' as const, function: { name: 'f', description: 'd', parameters: { type: 'object' } } }];

  it('sends tools and the choice only when tools are given', () => {
    const base = { messages: [], temperature: 0, maxTokens: null };
    expect(JSON.parse(requestBody({ ...base, tools, toolChoice: 'none' }))).toMatchObject({ tools, tool_choice: 'none' });
    expect(JSON.parse(requestBody({ ...base, tools: [], toolChoice: 'auto' }))).not.toHaveProperty('tool_choice');
    expect(JSON.parse(requestBody(base))).not.toHaveProperty('tools');
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

describe('the tool service host', () => {
  const calls = () => {
    const fetchMock = vi.fn(async () => new Response('{"servers":[],"catalog":[],"tools":[],"content":"","added":true}', { status: 200 }));
    vi.stubGlobal('fetch', fetchMock);
    return fetchMock;
  };
  const run = async (host?: ToolsHost) => {
    const f = calls();
    await toolsStatus(undefined, host);
    await toolsList('s a', true, undefined, host);
    await toolsCall('s a', 't', {}, false, undefined, host);
    await toolsSearch('q', 'https://x/?q={q}', undefined, host);
    await toolsStop('s a', host);
    await toolsAddCatalog('c', true, host);
    return f.mock.calls.map((c) => String((c as unknown[])[0]));
  };

  it('goes to /tools by default and for the server', async () => {
    const want = ['/tools/status', '/tools/servers/s%20a/tools?internet=1', '/tools/servers/s%20a/call', '/tools/search', '/tools/servers/s%20a/stop', '/tools/catalog/c/add'];
    expect(await run()).toEqual(want);
    expect(await run('server')).toEqual(want);
  });

  it('goes to /tools-local for the local host, on every call', async () => {
    expect(await run('local')).toEqual([
      '/tools-local/status', '/tools-local/servers/s%20a/tools?internet=1', '/tools-local/servers/s%20a/call', '/tools-local/search',
      '/tools-local/servers/s%20a/stop', '/tools-local/catalog/c/add',
    ]);
  });

  it('never mixes the two prefixes', async () => {
    const local = await run('local');
    expect(local.every((u) => u.startsWith('/tools-local/'))).toBe(true);
    const server = await run('server');
    expect(server.some((u) => u.startsWith('/tools-local'))).toBe(false);
  });
});
