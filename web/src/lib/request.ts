// Builds the wire conversation and checks it against the server's limits before sending.
import { requestBody, type ContentPart, type ServerInfo, type WireMessage } from './api';
import { MAX_BODY_BYTES, MAX_IMAGES, contentText, countImages, type Attachment, estimateAttachment } from './attachments';
import { checkContext, estimateTextTokens, estimateWaitSeconds } from './estimate';
import type { Message, Settings } from './storage';

/** The messages to send: each earlier answer exactly as received; failed replies are left out,
 * even when they streamed some text, so the model never sees a truncated answer as its own. */
export function toWire(systemPrompt: string, history: Message[]): WireMessage[] {
  const out: WireMessage[] = [];
  if (systemPrompt.trim()) out.push({ role: 'system', content: systemPrompt });
  for (const m of history) {
    if (m.role === 'user') out.push({ role: 'user', content: m.content });
    else if (m.content && m.status !== 'error') out.push({ role: 'assistant', content: m.content });
  }
  return out;
}

const encoder = new TextEncoder();
const jsonBytes = (v: unknown): number => encoder.encode(JSON.stringify(v)).length;
const IMAGE_PART_BYTES = jsonBytes({ type: 'image_url', image_url: { url: '' } });
// Image parts hold megabytes of base64: measure each content array once, not per keystroke.
const partsCache = new WeakMap<ContentPart[], number>();

function contentBytes(content: string | ContentPart[] | null): number {
  if (content === null) return 4;
  if (typeof content === 'string') return jsonBytes(content);
  const cached = partsCache.get(content);
  if (cached !== undefined) return cached;
  let n = 2 + Math.max(0, content.length - 1);
  for (const p of content) n += p.type === 'text' ? jsonBytes(p) : IMAGE_PART_BYTES + p.image_url.url.length;
  partsCache.set(content, n);
  return n;
}

/** The exact UTF-8 size of requestBody() for these messages, without serialising the images. */
export function requestBytes(messages: WireMessage[], settings: Pick<Settings, 'temperature' | 'maxTokens'>): number {
  let n = encoder.encode(requestBody({ messages: [], temperature: settings.temperature, maxTokens: settings.maxTokens })).length;
  n += Math.max(0, messages.length - 1);
  for (const m of messages) n += jsonBytes({ role: m.role, content: '' }) - 2 + contentBytes(m.content);
  return n;
}

export type Problem =
  | { kind: 'too_many_images'; count: number; max: number }
  | { kind: 'too_large'; bytes: number }
  | { kind: 'overflow'; tokens: number; limit: number };

export interface Assessment {
  /** Estimated tokens of the new message (text plus images with a known size). */
  newTokens: number;
  /** True when some images could not be counted. */
  imagesUnknown: boolean;
  waitSeconds: number;
  problems: Problem[];
}

function imageTokensOf(content: string | ContentPart[] | null, perImage: number | null): number {
  return perImage === null || content === null ? 0 : countImages(content) * perImage;
}

/** Estimates and limit checks for sending `draft` after `history`. */
export function assess(
  settings: Settings,
  history: Message[],
  draftText: string,
  attachments: Attachment[],
  draftContent: string | ContentPart[],
  info: ServerInfo | null,
): Assessment {
  let newTokens = estimateTextTokens(draftText);
  let imagesUnknown = false;
  for (const a of attachments) {
    const e = estimateAttachment(a, info);
    if (e.tokens === null) imagesUnknown = true;
    else newTokens += e.tokens;
  }
  const wire = [...toWire(settings.systemPrompt, history), { role: 'user' as const, content: draftContent }];
  // Earlier images are re-sent too; count them at the server's fixed size when it gives one.
  const perImage = info?.tokensPerImage ?? null;
  let promptTokens = newTokens;
  for (const m of wire.slice(0, -1)) promptTokens += estimateTextTokens(contentText(m.content ?? '')) + imageTokensOf(m.content, perImage);
  const problems: Problem[] = [];
  const images = wire.reduce((n, m) => n + countImages(m.content ?? ''), 0);
  if (images > MAX_IMAGES) problems.push({ kind: 'too_many_images', count: images, max: MAX_IMAGES });
  const bytes = requestBytes(wire, settings);
  if (bytes > MAX_BODY_BYTES) problems.push({ kind: 'too_large', bytes });
  const ctx = checkContext(promptTokens, info);
  if (ctx.kind === 'overflow') problems.push({ kind: 'overflow', tokens: ctx.tokens, limit: ctx.limit });
  // A continued conversation reuses the server's state, so only the new message is read.
  const readTokens = history.length ? newTokens : newTokens + estimateTextTokens(settings.systemPrompt);
  return { newTokens, imagesUnknown, waitSeconds: estimateWaitSeconds(readTokens), problems };
}
