// Token figures shown to the user: real API usage when the server sends it, otherwise an
// estimate that is always marked as one. No I/O.
import type { ServerInfo, WireMessage } from './api';
import { contentText, countImages } from './attachments';
import { estimateTextTokens } from './estimate';
import type { Message, Stats } from './storage';
import type { Usage } from './sse';

export interface Figure {
  n: number;
  estimated: boolean;
}

/** Tokens the server reads (prompt) and writes (completion). */
export interface Count {
  read: Figure;
  written: Figure;
}

/** An estimate of what the server will read for this wire conversation. */
export function estimateWireTokens(wire: WireMessage[], info: ServerInfo | null): number {
  const perImage = info?.tokensPerImage ?? 0;
  let n = 0;
  for (const m of wire) {
    if (m.content !== null) n += estimateTextTokens(contentText(m.content)) + countImages(m.content) * perImage;
    for (const call of m.tool_calls ?? []) n += estimateTextTokens(call.function.name + call.function.arguments);
  }
  return n;
}

const figure = (real: number | null, estimate: number | null): Figure | null =>
  real !== null ? { n: real, estimated: false } : estimate !== null ? { n: estimate, estimated: true } : null;

/** One reply's figures, or null when neither a usage nor an estimate was kept. */
export function replyCount(s: Stats | null): Count | null {
  if (!s) return null;
  const read = figure(s.promptTokens, s.promptEstimate);
  const written = figure(s.completionTokens, s.completionEstimate);
  return read && written ? { read, written } : null;
}

/** What a reply that is still streaming has used so far: usage as soon as it arrives. */
export function liveCount(usage: Usage | null, promptEstimate: number, chunks: number): Count {
  return {
    read: usage ? { n: usage.promptTokens, estimated: false } : { n: promptEstimate, estimated: true },
    written: usage ? { n: usage.completionTokens, estimated: false } : { n: chunks, estimated: true },
  };
}

/** The sum over every reply of a conversation; estimated as soon as one figure is. */
export function sessionTotals(messages: Message[]): Count | null {
  let read = 0;
  let written = 0;
  let readEst = false;
  let writtenEst = false;
  let any = false;
  for (const m of messages) {
    if (m.role !== 'assistant') continue;
    const c = replyCount(m.stats);
    if (!c) continue;
    any = true;
    read += c.read.n;
    written += c.written.n;
    readEst ||= c.read.estimated;
    writtenEst ||= c.written.estimated;
  }
  return any ? { read: { n: read, estimated: readEst }, written: { n: written, estimated: writtenEst } } : null;
}
