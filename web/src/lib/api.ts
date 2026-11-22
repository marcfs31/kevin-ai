// The only module that calls fetch. Talks to lm-serve (directly in dev, through
// tools/chat_server.py in production) with same-origin relative URLs. Dev also
// exposes POST /control/serve on the Vite server to start a remote model.
import { SseParser, obj, parseErrorBody, parsePayload, type ApiErrorBody, type StreamEvent } from './sse';
import type { ServeOptions, ServeResult } from './serveControl';
import type { McpTool } from './toolRegistry';

export type VisionStatus = 'ok' | 'unavailable' | 'unknown';

export interface ServerInfo extends Omit<ModelEntry, 'id'> {
  model: string;
  /** Every model the server lists, in its order; the fields above describe the first. */
  models: ModelEntry[];
}

export interface ModelEntry {
  id: string;
  contextLength: number | null;
  vision: VisionStatus;
  maxPixels: number | null;
  pixelsPerToken: number | null;
  tokensPerImage: number | null;
}

export type ContentPart = { type: 'text'; text: string } | { type: 'image_url'; image_url: { url: string } };

export interface WireToolCall {
  id: string;
  type: 'function';
  function: { name: string; arguments: string };
}

export interface WireTool {
  type: 'function';
  function: { name: string; description: string; parameters: Record<string, unknown> };
}

export interface WireMessage {
  role: 'system' | 'user' | 'assistant' | 'tool';
  content: string | ContentPart[] | null;
  tool_calls?: WireToolCall[];
  tool_call_id?: string;
}

export interface ChatRequest {
  messages: WireMessage[];
  temperature: number;
  maxTokens: number | null;
  /** Sent as `model` when set; left out for a server that lists a single model. */
  model?: string | null;
  tools?: WireTool[];
  /** 'none' keeps the tools in the prompt (so the server's cache holds) but forbids calling them. */
  toolChoice?: 'auto' | 'none';
  /** The chat template's enable_thinking, when the effort says so; a server without it ignores the field. */
  thinking?: boolean;
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

function modelEntry(v: unknown): ModelEntry | null {
  const m = obj(v);
  if (!m) return null;
  const vision = obj(obj(m.capabilities)?.vision);
  const status = vision?.status;
  // llama-server (WhiteRabbitNeo, Gemma uncensored) has no capabilities.vision: it lists the kinds of
  // input the loaded model takes, and the window it was started with.
  const inputs = obj(m.architecture)?.input_modalities;
  const byModality = Array.isArray(inputs) ? (inputs.includes('image') ? 'ok' : 'unavailable') : 'unknown';
  return {
    id: typeof m.id === 'string' ? m.id : '',
    contextLength: positive(m.context_length) ?? positive(obj(m.meta)?.n_ctx),
    vision: status === 'ok' || status === 'unavailable' ? status : byModality,
    maxPixels: positive(vision?.max_pixels),
    pixelsPerToken: positive(vision?.pixels_per_token),
    tokensPerImage: positive(vision?.tokens_per_image),
  };
}

/** Reads /v1/models. Every extension field may be absent (today's lm-serve). */
export function parseModels(json: unknown): ServerInfo | null {
  const data = obj(json)?.data;
  const models = Array.isArray(data) ? data.map(modelEntry).filter((e): e is ModelEntry => e !== null && e.id !== '') : [];
  const first = Array.isArray(data) ? modelEntry(data[0]) : null;
  if (!first) return null;
  const { id, ...rest } = first;
  return { ...rest, model: id, models: models.length ? models : [first] };
}

/** The server's figures for one model: the context window and vision follow the model chosen. */
export function infoForModel(info: ServerInfo | null, id: string | null): ServerInfo | null {
  if (!info) return null;
  const e = info.models.find((m) => m.id === id) ?? info.models[0] ?? null;
  return e ? { ...e, model: e.id, models: info.models } : info;
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

/** Starts Qwen, Qwen uncensored, Gemma, Gemma uncensored or WhiteRabbitNeo on the remote host via the Vite /control/serve middleware. */
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

/** Drops lm-serve's conversation, system text and prefix cache. A server without the endpoint
 * (llama-server, which runs WhiteRabbitNeo and Gemma uncensored) keeps no conversation for us to drop: 404 is done, not a failure. */
export async function resetServer(signal?: AbortSignal): Promise<void> {
  const res = await send('/v1/chat/reset', { method: 'POST', signal, headers: { Accept: 'application/json' } });
  if (res.status === 404) return;
  if (!res.ok) throw await errorFrom(res);
}

// ---- the tool service (tools/chat_tools.py), reached through the same-origin /tools/* paths (and /tools-local/* for the Mac's copy) ----

/** Which tool service a call goes to: the model server's (`/tools`) or the one on this Mac (`/tools-local`). */
export type ToolsHost = 'server' | 'local';

const toolsBase = (host: ToolsHost): string => (host === 'local' ? '/tools-local' : '/tools');

export interface McpServerStatus {
  id: string;
  title: string;
  kind: 'stdio' | 'http';
  /** It reaches the internet, so it only runs while the internet switch is on. */
  network: boolean;
  catalog: string | null;
  state: 'ready' | 'stopped' | 'failed';
  error: string | null;
  tools: number | null;
}

export interface CatalogEntry {
  id: string;
  title: string;
  description: string;
  runtime: string;
  network: boolean;
  verified: boolean;
  added: boolean;
}

export interface ToolsStatus {
  servers: McpServerStatus[];
  catalog: CatalogEntry[];
  configError: string | null;
}

async function toolsJson(url: string, init: RequestInit): Promise<Record<string, unknown>> {
  const res = await send(url, { ...init, headers: { Accept: 'application/json', ...(init.method === 'POST' ? { 'Content-Type': 'application/json' } : {}) } });
  const text = await res.text().catch(() => '');
  let raw: unknown = null;
  try {
    raw = JSON.parse(text);
  } catch {
    // Not JSON (a proxy's error page): the status alone is reported.
  }
  if (!res.ok) throw new ApiError(res.status, parseErrorBody(raw));
  const json = obj(raw);
  if (!json) throw new ApiError(res.status, { message: '', code: 'bad_tools_reply', type: null });
  return json;
}

const str = (v: unknown, d = ''): string => (typeof v === 'string' ? v : d);

/** A project the owner registered on a host: what the page may know (no paths, no token). */
export interface ProjectInfo {
  id: string;
  title: string;
  defaultBranch: string;
  allowMain: boolean;
  hasRemote: boolean;
  commands: string[];
}

export async function toolsProjects(signal?: AbortSignal, host: ToolsHost = 'server'): Promise<ProjectInfo[]> {
  const j = await toolsJson(`${toolsBase(host)}/projects`, { signal });
  return (Array.isArray(j.projects) ? j.projects : []).flatMap((v): ProjectInfo[] => {
    const p = obj(v);
    if (!p || typeof p.id !== 'string') return [];
    return [{
      id: p.id, title: str(p.title, p.id), defaultBranch: str(p.defaultBranch, 'main'), allowMain: p.allowMain === true, hasRemote: p.hasRemote === true,
      commands: Array.isArray(p.commands) ? p.commands.filter((c): c is string => typeof c === 'string') : [],
    }];
  });
}

export async function toolsStatus(signal?: AbortSignal, host: ToolsHost = 'server'): Promise<ToolsStatus> {
  const j = await toolsJson(`${toolsBase(host)}/status`, { signal });
  const servers = (Array.isArray(j.servers) ? j.servers : []).flatMap((v): McpServerStatus[] => {
    const s = obj(v);
    if (!s || typeof s.id !== 'string') return [];
    return [{
      id: s.id, title: str(s.title, s.id), kind: s.kind === 'http' ? 'http' : 'stdio', network: s.network === true, catalog: typeof s.catalog === 'string' ? s.catalog : null,
      state: s.state === 'ready' || s.state === 'failed' ? s.state : 'stopped', error: typeof s.error === 'string' ? s.error : null, tools: typeof s.tools === 'number' ? s.tools : null,
    }];
  });
  const catalog = (Array.isArray(j.catalog) ? j.catalog : []).flatMap((v): CatalogEntry[] => {
    const c = obj(v);
    if (!c || typeof c.id !== 'string') return [];
    return [{ id: c.id, title: str(c.title, c.id), description: str(c.description), runtime: str(c.runtime), network: c.network === true, verified: c.verified === true, added: c.added === true }];
  });
  return { servers, catalog, configError: typeof j.configError === 'string' ? j.configError : null };
}

export async function toolsList(server: string, internet: boolean, signal?: AbortSignal, host: ToolsHost = 'server'): Promise<McpTool[]> {
  const j = await toolsJson(`${toolsBase(host)}/servers/${encodeURIComponent(server)}/tools?internet=${internet ? '1' : '0'}`, { signal });
  return (Array.isArray(j.tools) ? j.tools : []).flatMap((v): McpTool[] => {
    const t = obj(v);
    if (!t || typeof t.name !== 'string') return [];
    return [{
      name: t.name, title: str(t.title, t.name), description: str(t.description), inputSchema: obj(t.inputSchema) ?? { type: 'object', properties: {} },
      readOnly: t.readOnly === true, tokens: typeof t.tokens === 'number' ? t.tokens : 0,
    }];
  });
}

export interface ToolCallResult {
  content: string;
  isError: boolean;
}

function toolResult(j: Record<string, unknown>): ToolCallResult {
  return { content: str(j.content), isError: j.isError === true };
}

export async function toolsCall(server: string, name: string, args: Record<string, unknown>, internet: boolean, signal?: AbortSignal, host: ToolsHost = 'server'): Promise<ToolCallResult> {
  return toolResult(await toolsJson(`${toolsBase(host)}/servers/${encodeURIComponent(server)}/call`, { method: 'POST', signal, body: JSON.stringify({ name, arguments: args, internet }) }));
}

export async function toolsSearch(query: string, searchUrl: string, signal?: AbortSignal, host: ToolsHost = 'server'): Promise<ToolCallResult> {
  return toolResult(await toolsJson(`${toolsBase(host)}/search`, { method: 'POST', signal, body: JSON.stringify({ query, searchUrl, internet: true }) }));
}

export async function toolsStop(server: string, host: ToolsHost = 'server'): Promise<void> {
  await toolsJson(`${toolsBase(host)}/servers/${encodeURIComponent(server)}/stop`, { method: 'POST', body: '{}' });
}

export async function toolsAddCatalog(id: string, internet: boolean, host: ToolsHost = 'server'): Promise<boolean> {
  return (await toolsJson(`${toolsBase(host)}/catalog/${encodeURIComponent(id)}/add`, { method: 'POST', body: JSON.stringify({ internet }) })).added === true;
}

export function requestBody(req: ChatRequest): string {
  const body: Record<string, unknown> = {
    messages: req.messages,
    stream: true,
    stream_options: { include_usage: true },
    temperature: req.temperature,
  };
  if (req.maxTokens !== null) body.max_tokens = req.maxTokens;
  if (req.model) body.model = req.model;
  if (req.thinking !== undefined) body.chat_template_kwargs = { enable_thinking: req.thinking };
  if (req.tools?.length) {
    body.tools = req.tools;
    if (req.toolChoice) body.tool_choice = req.toolChoice;
  }
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
