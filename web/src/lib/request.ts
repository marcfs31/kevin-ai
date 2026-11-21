// Builds the wire conversation and checks it against the server's limits before sending.
import { requestBody, type ContentPart, type ServerInfo, type WireMessage } from './api';
import { MAX_BODY_BYTES, MAX_IMAGES, contentText, countImages, type Attachment, estimateAttachment } from './attachments';
import { checkContext, estimateTextTokens, estimateWaitSeconds } from './estimate';
import { lastSummaryIndex, toolResults, type Message, type Settings } from './storage';

/** Heads the summary inside the system message. */
export const SUMMARY_PREFIX = 'Summary of the earlier part of this conversation (it was compacted to save space):';

/** The messages to send: each earlier answer exactly as received; failed replies are left out,
 * even when they streamed some text, so the model never sees a truncated answer as its own. */
/** The server's rule for a tool name; a call the model named otherwise is sent under a safe one. */
const TOOL_NAME = /^[A-Za-z0-9_.-]{1,64}$/;

/** Arguments must be a JSON object string on the wire; whatever else the model wrote is sent as {}. */
function wireArguments(text: string): string {
  try {
    const v: unknown = JSON.parse(text);
    return v !== null && typeof v === 'object' && !Array.isArray(v) ? text : '{}';
  } catch {
    return '{}';
  }
}

function withoutImages(content: string | ContentPart[]): string | ContentPart[] {
  if (typeof content === 'string' || countImages(content) === 0) return content;
  return content.map((p): ContentPart => (p.type === 'image_url' ? { type: 'text', text: NO_IMAGE_NOTE } : p));
}

const NO_IMAGE_NOTE = '[an image was attached here; the model now chosen cannot read images]';

/** `keepImages` false (a model without vision) swaps every image for a text note, so a chat that
 * had images can still be continued on another model. */
export function toWire(systemPrompt: string, history: Message[], keepImages = true): WireMessage[] {
  const out: WireMessage[] = [];
  // Only what follows the last summary is sent; the summary rides in the system message.
  const s = lastSummaryIndex(history);
  const summary = s >= 0 ? history[s] : null;
  const system = [systemPrompt.trim() ? systemPrompt : '', summary ? `${SUMMARY_PREFIX}\n${summary.content}` : ''].filter(Boolean).join('\n\n');
  if (system) out.push({ role: 'system', content: system });
  const kept = history.slice(s + 1);
  for (const [i, m] of kept.entries()) {
    if (m.role === 'user') out.push({ role: 'user', content: keepImages ? m.content : withoutImages(m.content) });
    else if (m.role === 'assistant') {
      // Calls go back only with every result: the server refuses a call that nothing answered.
      const results = m.status === 'done' && m.toolCalls?.length ? toolResults(kept, i) : [];
      const answered = m.toolCalls?.every((c) => results.some((r) => r.toolCallId === c.id)) === true;
      if (m.toolCalls?.length && answered) {
        out.push({
          role: 'assistant',
          content: m.content || null,
          tool_calls: m.toolCalls.map((c) => ({ id: c.id, type: 'function', function: { name: TOOL_NAME.test(c.name) ? c.name : 'unknown_tool', arguments: wireArguments(c.arguments) } })),
        });
        for (const c of m.toolCalls) {
          const r = results.find((x) => x.toolCallId === c.id);
          if (r) out.push({ role: 'tool', tool_call_id: r.toolCallId, content: r.content });
        }
      } else if (m.content && m.status !== 'error') out.push({ role: 'assistant', content: m.content });
    }
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
  const wire = [...toWire(settings.systemPrompt, history, info?.vision === 'ok'), { role: 'user' as const, content: draftContent }];
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
