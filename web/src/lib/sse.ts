// Server-sent events for OpenAI-style chat streams: a pure parser with no I/O.

export interface Usage {
  promptTokens: number;
  completionTokens: number;
  cachedTokens: number | null;
}

export interface ApiErrorBody {
  message: string;
  code: string | null;
  type: string | null;
}

export type StreamEvent =
  | { kind: 'content'; text: string }
  | { kind: 'reasoning'; text: string }
  /** One streamed piece of a tool call; id and name come once, arguments may arrive in pieces. */
  | { kind: 'tool_call'; index: number; id: string | null; name: string | null; arguments: string }
  | { kind: 'finish'; reason: string }
  | { kind: 'usage'; usage: Usage }
  | { kind: 'error'; error: ApiErrorBody }
  | { kind: 'done' };

/** Splits a decoded text stream into the data payloads of complete events. */
export class SseParser {
  private buffer = '';
  private data: string[] = [];

  push(chunk: string): string[] {
    this.buffer += chunk;
    const out: string[] = [];
    for (;;) {
      const nl = this.buffer.search(/\r\n|\r|\n/);
      if (nl < 0) break;
      // A lone \r at the very end may be the first half of \r\n: wait for more.
      if (this.buffer[nl] === '\r' && nl === this.buffer.length - 1) break;
      const width = this.buffer.startsWith('\r\n', nl) ? 2 : 1;
      const line = this.buffer.slice(0, nl);
      this.buffer = this.buffer.slice(nl + width);
      this.line(line, out);
    }
    return out;
  }

  /** Flushes a final event that was not followed by a blank line. */
  end(): string[] {
    const out: string[] = [];
    if (this.buffer) this.line(this.buffer, out);
    this.buffer = '';
    this.line('', out);
    return out;
  }

  private line(line: string, out: string[]): void {
    if (line === '') {
      if (this.data.length) out.push(this.data.join('\n'));
      this.data = [];
      return;
    }
    if (line.startsWith(':')) return;
    const colon = line.indexOf(':');
    const field = colon < 0 ? line : line.slice(0, colon);
    let value = colon < 0 ? '' : line.slice(colon + 1);
    if (value.startsWith(' ')) value = value.slice(1);
    if (field === 'data') this.data.push(value);
  }
}

function num(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

export function obj(v: unknown): Record<string, unknown> | null {
  return v !== null && typeof v === 'object' && !Array.isArray(v) ? (v as Record<string, unknown>) : null;
}

export function parseErrorBody(v: unknown): ApiErrorBody | null {
  const e = obj(obj(v)?.error);
  if (!e) return null;
  return {
    message: typeof e.message === 'string' ? e.message : '',
    code: typeof e.code === 'string' ? e.code : null,
    type: typeof e.type === 'string' ? e.type : null,
  };
}

/** Turns one data payload into the events it carries. Unknown fields are ignored. */
export function parsePayload(data: string): StreamEvent[] {
  if (data.trim() === '[DONE]') return [{ kind: 'done' }];
  let json: unknown;
  try {
    json = JSON.parse(data);
  } catch {
    return [{ kind: 'error', error: { message: '', code: 'bad_stream', type: null } }];
  }
  const err = parseErrorBody(json);
  if (err) return [{ kind: 'error', error: err }];
  const root = obj(json) ?? {};
  const events: StreamEvent[] = [];
  const choices = Array.isArray(root.choices) ? root.choices : [];
  for (const c of choices) {
    const choice = obj(c);
    const delta = obj(choice?.delta);
    if (delta) {
      if (typeof delta.reasoning_content === 'string' && delta.reasoning_content)
        events.push({ kind: 'reasoning', text: delta.reasoning_content });
      if (typeof delta.content === 'string' && delta.content) events.push({ kind: 'content', text: delta.content });
      if (Array.isArray(delta.tool_calls)) {
        for (const [i, raw] of delta.tool_calls.entries()) {
          const call = obj(raw);
          const fn = obj(call?.function);
          if (!call) continue;
          events.push({
            kind: 'tool_call',
            index: typeof call.index === 'number' && Number.isInteger(call.index) && call.index >= 0 ? call.index : i,
            id: typeof call.id === 'string' && call.id ? call.id : null,
            name: typeof fn?.name === 'string' && fn.name ? fn.name : null,
            arguments: typeof fn?.arguments === 'string' ? fn.arguments : '',
          });
        }
      }
    }
    if (typeof choice?.finish_reason === 'string') events.push({ kind: 'finish', reason: choice.finish_reason });
  }
  const usage = obj(root.usage);
  if (usage) {
    const p = num(usage.prompt_tokens);
    const c = num(usage.completion_tokens);
    if (p !== null && c !== null) {
      events.push({
        kind: 'usage',
        usage: { promptTokens: p, completionTokens: c, cachedTokens: num(obj(usage.prompt_tokens_details)?.cached_tokens) },
      });
    }
  }
  return events;
}
