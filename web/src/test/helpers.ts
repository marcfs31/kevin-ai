import axe from 'axe-core';
import { vi } from 'vitest';

/** A fetch Response whose body streams the given chunks, one per read. */
export function sseResponse(chunks: string[], init: { status?: number; hold?: boolean } = {}): Response {
  const encoder = new TextEncoder();
  let i = 0;
  const stream = new ReadableStream<Uint8Array>({
    pull(controller) {
      const chunk = chunks[i];
      if (chunk !== undefined) {
        i += 1;
        controller.enqueue(encoder.encode(chunk));
      } else if (!init.hold) controller.close();
    },
  });
  return new Response(stream, { status: init.status ?? 200, headers: { 'Content-Type': 'text/event-stream' } });
}

export function delta(content: string, kind: 'content' | 'reasoning_content' = 'content'): string {
  return `data: ${JSON.stringify({ choices: [{ index: 0, delta: { [kind]: content }, finish_reason: null }] })}\n\n`;
}

export const FINISH = `data: ${JSON.stringify({ choices: [{ index: 0, delta: {}, finish_reason: 'stop' }] })}\n\n`;
export const USAGE = `data: ${JSON.stringify({ choices: [], usage: { prompt_tokens: 12, completion_tokens: 3, total_tokens: 15, prompt_tokens_details: { cached_tokens: 8 } } })}\n\n`;
export const DONE = 'data: [DONE]\n\n';

export function modelsResponse(extra: Record<string, unknown> = {}): Response {
  return new Response(JSON.stringify({ object: 'list', data: [{ id: 'qwen3.5-4b', object: 'model', owned_by: 'local', ...extra }] }), {
    status: 200,
    headers: { 'Content-Type': 'application/json' },
  });
}

type Handler = (url: string, init: RequestInit | undefined) => Response | Promise<Response>;

/** Routes fetch by path; /v1/models answers with today's lm-serve unless overridden. */
export function mockFetch(chat: Handler, models: () => Response = () => modelsResponse()) {
  const fn = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    if (url.endsWith('/v1/models')) return models();
    return chat(url, init);
  });
  vi.stubGlobal('fetch', fn);
  return fn;
}

export async function axeViolations(node: Element): Promise<string[]> {
  const result = await axe.run(node, { rules: { 'color-contrast': { enabled: false } } });
  return result.violations.map((v) => `${v.id}: ${v.nodes.map((n) => n.target.join(' ')).join(', ')}`);
}
