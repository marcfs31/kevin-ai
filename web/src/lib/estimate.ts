// Token and wait estimates shown before sending. All of them are estimates, labelled so.
import type { ServerInfo } from './api';

/** Prompt reading speed measured on the target server (Qwen3.5-4B q8, i5-6500T). */
export const PROMPT_TOKENS_PER_SECOND = 12;
export const WARN_WAIT_SECONDS = 120;
export const CHARS_PER_TOKEN = 3.5;

export function estimateTextTokens(text: string): number {
  return Math.ceil(text.length / CHARS_PER_TOKEN);
}

/** Tokens for one image, or null when the server does not say. */
export function estimateImageTokens(width: number, height: number, info: ServerInfo | null): number | null {
  if (!info) return null;
  if (info.tokensPerImage) return info.tokensPerImage;
  if (info.pixelsPerToken) return Math.ceil((width * height) / info.pixelsPerToken);
  return null;
}

export function estimateWaitSeconds(tokens: number): number {
  return tokens / PROMPT_TOKENS_PER_SECOND;
}

/** Whole minutes and leftover seconds for an ETA shown as "M min S s". */
export function etaParts(seconds: number): { m: number; s: number } {
  const n = Math.max(0, Math.round(seconds));
  return { m: Math.floor(n / 60), s: n % 60 };
}

export type Overflow = { kind: 'ok' } | { kind: 'overflow'; tokens: number; limit: number } | { kind: 'unknown' };

/** Whether a prompt of `tokens` fits the server's context; unknown when it does not say. */
export function checkContext(tokens: number, info: ServerInfo | null): Overflow {
  const limit = info?.contextLength ?? null;
  if (limit === null) return { kind: 'unknown' };
  return tokens > limit ? { kind: 'overflow', tokens, limit } : { kind: 'ok' };
}
