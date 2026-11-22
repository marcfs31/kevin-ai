// Which tool definitions to send this step. A small model reads its prompt slowly, so every
// definition costs seconds: send the built-ins and a budget of the tools that best match what was
// asked, plus the ones the model unlocked with find_tools. Plain keyword scoring, no model call.
import type { ToolSpec } from './toolRegistry';

const STOP = new Set(['the', 'an', 'of', 'to', 'in', 'on', 'and', 'or', 'for', 'with', 'is', 'are', 'this', 'that', 'it', 'me', 'my', 'please', 'can', 'you', 'what', 'how', 'do', 'does', 'be', 'at', 'by', 'from', 'as', 'we']);

export function terms(text: string): string[] {
  return text
    .replace(/([a-z])([A-Z])/g, '$1 $2')
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter((w) => w.length > 1 && !STOP.has(w));
}

/** Terms of a tool: its name (split at _ and -) counts three times a word of its description. */
function profile(spec: ToolSpec): Map<string, number> {
  const p = new Map<string, number>();
  const add = (words: string[], w: number) => {
    for (const t of words) p.set(t, Math.max(p.get(t) ?? 0, w));
  };
  add(terms(spec.description), 1);
  add(terms(spec.name), 3);
  return p;
}

/** The specs that match the query, best first; none for a query that shares no term with any. */
export function searchSpecs(specs: ToolSpec[], query: string, limit: number): ToolSpec[] {
  const q = [...new Set(terms(query))];
  if (!q.length || !specs.length) return [];
  const profiles = specs.map(profile);
  // A term found in few tools says more than one found in most.
  const idf = new Map(q.map((t) => [t, Math.log(1 + specs.length / (1 + profiles.filter((p) => p.has(t)).length))]));
  return specs
    .map((spec, i) => ({ spec, score: q.reduce((s, t) => s + (profiles[i]?.get(t) ?? 0) * (idf.get(t) ?? 0), 0) }))
    .filter((x) => x.score > 0)
    .sort((a, b) => b.score - a.score || a.spec.name.localeCompare(b.spec.name))
    .slice(0, limit)
    .map((x) => x.spec);
}

export interface SelectInput {
  /** Always offered, whatever the budget. */
  pinned: ToolSpec[];
  /** Offered when they match, within the budget. */
  candidates: ToolSpec[];
  query: string;
  budgetTokens: number;
  maxTools: number;
  /** Names the model asked for with find_tools: offered whatever the budget, up to maxTools. */
  unlocked: ReadonlySet<string>;
}

export function selectTools(i: SelectInput): ToolSpec[] {
  const out = [...i.pinned];
  const have = new Set(out.map((s) => s.name));
  // The budget is for the tools beyond the always-sent ones, so the slider starts at none and only ever adds.
  let used = 0;
  for (const s of i.candidates) {
    if (out.length >= i.maxTools) break;
    if (i.unlocked.has(s.name) && !have.has(s.name)) {
      out.push(s);
      have.add(s.name);
      used += s.tokens;
    }
  }
  for (const s of searchSpecs(i.candidates.filter((c) => !have.has(c.name)), i.query, i.candidates.length)) {
    if (out.length >= i.maxTools) break;
    if (used + s.tokens > i.budgetTokens) continue;
    out.push(s);
    used += s.tokens;
  }
  return out;
}

/** About how many of `enabled` fit in `budgetTokens` (the always-sent tools are not part of it), judged by their average size.
 * An estimate for the settings screen: the tools actually sent are the ones that match the message. */
export function fitCount(enabled: ToolSpec[], budgetTokens: number, maxTools: number, pinnedCount: number): number {
  if (!enabled.length) return 0;
  const average = enabled.reduce((n, s) => n + s.tokens, 0) / enabled.length;
  const byBudget = Math.floor(budgetTokens / average);
  return Math.max(0, Math.min(enabled.length, byBudget, maxTools - pinnedCount));
}
