// Session compacting: older turns become one model-written summary, recent turns stay verbatim.
// Pure functions; the streaming call lives in App. Nothing is ever deleted from the saved chat:
// the summary is inserted after the turns it replaces and only what follows it is re-sent.
import type { ServerInfo, WireMessage } from './api';
import { contentText, countImages } from './attachments';
import { CHARS_PER_TOKEN } from './estimate';
import { toWire } from './request';
import { lastSummaryIndex, type Message, type SummaryMessage } from './storage';
import { estimateWireTokens } from './tokens';

/** Messages kept word for word at the end: the last two turns. */
export const KEEP_RECENT = 4;
/** Compact when the next request would fill this share of the context window. */
export const AUTO_RATIO = 0.8;
/** The summary request may use this share of the context window for the transcript. */
const TRANSCRIPT_RATIO = 0.6;

export interface Plan {
  /** Index of the first message that stays verbatim; the summary goes right before it. */
  cut: number;
  /** The summary this one builds on, if any. */
  previous: SummaryMessage | null;
  /** The messages to be summarised, after the previous summary. */
  span: Message[];
}

/** What to summarise, or null when too little is older than the kept turns. */
export function planCompaction(messages: Message[]): Plan | null {
  const start = lastSummaryIndex(messages) + 1;
  let cut = messages.length - KEEP_RECENT;
  // Keep whole turns: the first kept message is a user message.
  while (cut > start && messages[cut]?.role !== 'user') cut -= 1;
  if (cut - start < 2) return null;
  const previous = start > 0 ? (messages[start - 1] as SummaryMessage) : null;
  return { cut, previous, span: messages.slice(start, cut) };
}

/** Tokens the next request will hold: the server's own count for the last finished reply when it
 * sent one (that count includes everything re-sent), plus an estimate of what came after it;
 * else an estimate of the whole request. */
export function usedTokens(messages: Message[], systemPrompt: string, info: ServerInfo | null): number {
  const start = lastSummaryIndex(messages) + 1;
  for (let i = messages.length - 1; i >= start; i--) {
    const m = messages[i];
    if (m?.role === 'assistant' && m.status === 'done' && m.stats?.promptTokens != null)
      return m.stats.promptTokens + (m.stats.completionTokens ?? 0) + estimateWireTokens(toWire('', messages.slice(i + 1)), info);
  }
  return estimateWireTokens(toWire(systemPrompt, messages), info);
}

/** True when sending `newTokens` more would pass AUTO_RATIO of the context and there is something to compact. */
export function shouldCompact(messages: Message[], systemPrompt: string, newTokens: number, info: ServerInfo | null): boolean {
  const limit = info?.contextLength ?? null;
  if (limit === null || planCompaction(messages) === null) return false;
  return usedTokens(messages, systemPrompt, info) + newTokens >= limit * AUTO_RATIO;
}

/** A summary is assumed to take at most this share of the context when judging whether a draft fits. */
const SUMMARY_SHARE = 0.15;

/** The history as it will be sent once the automatic compaction has run, for the checks made
 * before sending: without it a long chat that compaction would rescue would be refused as too long. */
export function projectCompaction(messages: Message[], systemPrompt: string, newTokens: number, info: ServerInfo | null): Message[] {
  const plan = shouldCompact(messages, systemPrompt, newTokens, info) ? planCompaction(messages) : null;
  if (!plan || !info?.contextLength) return messages;
  const text = 'x'.repeat(Math.floor(info.contextLength * SUMMARY_SHARE * CHARS_PER_TOKEN));
  return applyCompaction(messages, plan, text, 'projected', 0, true);
}

const SUMMARY_SYSTEM =
  'You compress a conversation so the assistant can continue it later. Write a faithful summary of everything below: ' +
  'what the user asked, facts, names, numbers, code identifiers, decisions, results of tool calls and anything still open. ' +
  'Keep the conversation’s language. Plain text, no preamble, no new advice.';

function messageLine(m: Message): string {
  if (m.role === 'summary') return `Earlier summary:\n${m.content}`;
  if (m.role === 'user') {
    const images = countImages(m.content);
    const note = images > 0 ? `[${images} image${images === 1 ? '' : 's'} attached here; the picture itself is not available]\n` : '';
    return `User:\n${note}${contentText(m.content)}`;
  }
  // A reply that was stopped or failed part way is still part of what happened: say so, never drop it.
  const state = m.status === 'stopped' ? '[this reply was stopped by the user before it ended]\n' : m.status === 'error' ? '[this reply failed part way]\n' : '';
  return `Assistant:\n${state}${m.content}`;
}

/** The text handed to the model, cut evenly to fit the share of the context it may use. */
export function transcript(plan: Plan, info: ServerInfo | null): string {
  const lines: string[] = [];
  if (plan.previous) lines.push(messageLine(plan.previous));
  for (const m of plan.span) lines.push(messageLine(m));
  const limit = info?.contextLength ?? null;
  if (limit === null) return lines.join('\n\n');
  const budgetChars = Math.floor(limit * TRANSCRIPT_RATIO * 3.5);
  const total = lines.reduce((n, l) => n + l.length, 0);
  if (total <= budgetChars) return lines.join('\n\n');
  const each = Math.max(200, Math.floor(budgetChars / lines.length));
  return lines.map((l) => (l.length <= each ? l : `${l.slice(0, each)}\n[… cut to fit]`)).join('\n\n');
}

export function summaryRequest(plan: Plan, info: ServerInfo | null): WireMessage[] {
  return [
    { role: 'system', content: SUMMARY_SYSTEM },
    { role: 'user', content: transcript(plan, info) },
  ];
}

/** The history with the summary inserted before the kept turns. */
export function applyCompaction(messages: Message[], plan: Plan, text: string, id: string, now: number, auto: boolean): Message[] {
  const summary: SummaryMessage = { id, role: 'summary', content: text.trim(), covered: plan.span.length, auto, createdAt: now };
  return [...messages.slice(0, plan.cut), summary, ...messages.slice(plan.cut)];
}
