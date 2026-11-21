// The figures of "This chat": what was read and written over the whole chat, how many turns,
// the mean speed, how often it was compacted, and how full the context window is now. No I/O.
import type { ServerInfo } from './api';
import { AUTO_RATIO, contextUse } from './compact';
import type { Message } from './storage';
import { sessionTotals, type Count, type Figure } from './tokens';

export interface ChatFigures {
  read: Figure;
  written: Figure;
  turns: number;
  compactions: number;
  /** Mean tokens per second over the replies that reported one. */
  rate: number | null;
}

const zero: Figure = { n: 0, estimated: false };

const add = (a: Figure, b: Figure): Figure => ({ n: a.n + b.n, estimated: a.estimated || b.estimated });

/** Null for an empty chat. `live` is the reply still streaming (not in `messages` yet): its
 * figures count as they arrive, so the totals move while it is written. */
export function chatFigures(messages: Message[], live: Count | null = null): ChatFigures | null {
  if (messages.length === 0) return null;
  const totals = sessionTotals(messages);
  const rates = messages.flatMap((m) => (m.role === 'assistant' && m.stats?.tokensPerSecond != null ? [m.stats.tokensPerSecond] : []));
  return {
    read: live ? add(totals?.read ?? zero, live.read) : (totals?.read ?? zero),
    written: live ? add(totals?.written ?? zero, live.written) : (totals?.written ?? zero),
    turns: messages.filter((m) => m.role === 'user').length,
    compactions: messages.filter((m) => m.role === 'summary').length,
    rate: rates.length ? rates.reduce((a, b) => a + b, 0) / rates.length : null,
  };
}

/** The context window as three boxes: what is re-sent, the last reply, and what is left. */
export interface ContextBar {
  prompt: number;
  reply: number;
  /** Tokens left; null when the server did not say how large the window is. */
  free: number | null;
  limit: number | null;
  /** Share of the window in use, 0..1 (can pass 1 when the chat no longer fits); null without a limit. */
  share: number | null;
  /** Where the automatic compaction starts, as a share of the window. */
  autoAt: number;
  estimated: boolean;
}

/** Null for an empty chat. While a reply streams (`live`) the window is what that request holds:
 * its prompt already contains the whole chat, so it replaces the figure taken from finished replies. */
export function contextBar(messages: Message[], systemPrompt: string, info: ServerInfo | null, live: Count | null = null, autoAt: number = AUTO_RATIO): ContextBar | null {
  if (messages.length === 0) return null;
  const use = live
    ? { prompt: live.read.n, reply: live.written.n, estimated: live.read.estimated || live.written.estimated }
    : contextUse(messages, systemPrompt, info);
  const limit = info?.contextLength ?? null;
  const used = use.prompt + use.reply;
  return {
    prompt: use.prompt,
    reply: use.reply,
    free: limit === null ? null : Math.max(0, limit - used),
    limit,
    share: limit === null ? null : used / limit,
    autoAt,
    estimated: use.estimated,
  };
}
