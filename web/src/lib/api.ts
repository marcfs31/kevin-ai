// The only module that calls fetch. Talks to lm-serve (directly in dev, through
// tools/chat_server.py in production) with same-origin relative URLs. Dev also
// exposes POST /control/serve on the Vite server to start a remote model.
import { SseParser, obj, parseErrorBody, parsePayload, type ApiErrorBody, type StreamEvent } from './sse';
import type { ServeOptions, ServeResult } from './serveControl';

export type VisionStatus = 'ok' | 'unavailable' | 'unknown';

export interface ServerInfo {
  model: string;
  contextLength: number | null;
  vision: VisionStatus;
  maxPixels: number | null;
  pixelsPerToken: number | null;
  tokensPerImage: number | null;
}

export type ContentPart = { type: 'text'; text: string } | { type: 'image_url'; image_url: { url: string } };

export interface WireMessage {
  role: 'system' | 'user' | 'assistant';
  content: string | ContentPart[];
}

export interface ChatRequest {
  messages: WireMessage[];
  temperature: number;
  maxTokens: number | null;
}

/** An HTTP or stream failure. status 0 means the server could not be reached. */
export class ApiError extends Error {
  readonly status: number;
  readonly code: string | null;
  readonly serverMessage: string;
  constructor(status: number, body: ApiErrorBody | null) {
    super(body?.message || `HTTP ${status}`);
    this.name = 'ApiError';
    this.status = status;
    this.code = body?.code ?? null;
    this.serverMessage = body?.message ?? '';
  }
}

function positive(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) && v > 0 ? v : null;
}

/** Reads /v1/models data[0]. Every extension field may be absent (today's lm-serve). */
export function parseModels(json: unknown): ServerInfo | null {
  const data = obj(json)?.data;
  const first = Array.isArray(data) ? obj(data[0]) : null;
  if (!first) return null;
  const vision = obj(obj(first.capabilities)?.vision);
  const status = vision?.status;
  return {
    model: typeof first.id === 'string' ? first.id : '',
    contextLength: positive(first.context_length),
    vision: status === 'ok' || status === 'unavailable' ? status : 'unknown',
    maxPixels: positive(vision?.max_pixels),
    pixelsPerToken: positive(vision?.pixels_per_token),
    tokensPerImage: positive(vision?.tokens_per_image),
  };
}

async function errorFrom(res: Response): Promise<ApiError> {
  const text = await res.text().catch(() => '');
  let json: unknown = null;
  try {
    json = JSON.parse(text);
  } catch {
    // Not JSON (a proxy's HTML error page, say): the status alone is reported.
  }
  return new ApiError(res.status, parseErrorBody(json));
}

async function send(url: string, init: RequestInit): Promise<Response> {
  try {
    return await fetch(url, init);
  } catch (e) {
    if (e instanceof DOMException && e.name === 'AbortError') throw e;
    throw new ApiError(0, null);
  }
}

export async function fetchServerInfo(signal?: AbortSignal): Promise<ServerInfo> {
  const res = await send('/v1/models', { signal, headers: { Accept: 'application/json' } });
  if (!res.ok) throw await errorFrom(res);
  const info = parseModels(await res.json().catch(() => null));
  if (!info) throw new ApiError(res.status, { message: '', code: 'bad_models', type: null });
  return info;
}

/** Starts Qwen, Qwen uncensored, Gemma or WhiteRabbitNeo on the remote host via the Vite /control/serve middleware. */
export async function startRemoteServe(opts: ServeOptions, signal?: AbortSignal): Promise<ServeResult> {
  const res = await send('/control/serve', {
    method: 'POST',
    signal,
    headers: { Accept: 'application/json', 'Content-Type': 'application/json' },
    body: JSON.stringify(opts),
  });
  const json = (await res.json().catch(() => null)) as ServeResult | null;
  if (!json || typeof json !== 'object' || typeof json.ok !== 'boolean') {
    throw new ApiError(res.status, { message: 'bad control response', code: null, type: null });
  }
  return json;
}

/** Drops lm-serve's conversation, system text and prefix cache. */
export async function resetServer(signal?: AbortSignal): Promise<void> {
  const res = await send('/v1/chat/reset', { method: 'POST', signal, headers: { Accept: 'application/json' } });
  if (!res.ok) throw await errorFrom(res);
}

export function requestBody(req: ChatRequest): string {
  const body: Record<string, unknown> = {
    messages: req.messages,
    stream: true,
    stream_options: { include_usage: true },
    temperature: req.temperature,
  };
  if (req.maxTokens !== null) body.max_tokens = req.maxTokens;
  return JSON.stringify(body);
}

/** Streams one reply; resolves when the stream ends, rejects with ApiError or AbortError. */
export async function streamChat(req: ChatRequest, signal: AbortSignal, onEvent: (e: StreamEvent) => void): Promise<void> {
  const res = await send('/v1/chat/completions', {
    method: 'POST',
    signal,
    headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
    body: requestBody(req),
  });
  if (!res.ok) throw await errorFrom(res);
  if (!res.body) throw new ApiError(res.status, { message: '', code: 'bad_stream', type: null });
  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
  const parser = new SseParser();
  let finished = false;
  const handle = (payloads: string[]) => {
    for (const p of payloads) {
      for (const e of parsePayload(p)) {
        if (e.kind === 'error') throw new ApiError(res.status, e.error);
        if (e.kind === 'finish' || e.kind === 'done') finished = true;
        onEvent(e);
      }
    }
  };
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      handle(parser.push(value));
    }
    handle(parser.end());
  } finally {
    reader.releaseLock();
  }
  // A stream that stops without a finish reason was cut short: never pass it off as complete.
  if (!finished) throw new ApiError(res.status, { message: '', code: 'stream_cut', type: null });
}
