// Turns the tools the user turned on into what one agent run needs: the definitions to offer at a
// step, whether a call needs the user's OK, and running it. Nothing here throws at the agent: a
// failure is a result the model can read.
import { ApiError, toolsCall, toolsSearch, type WireTool } from './api';
import type { ToolCall, InternetConfig, ToolsConfig } from './storage';
import { builtinSpecs, findToolsSpec, toWire, webSearchSpec, type ToolSpec } from './toolRegistry';
import { searchSpecs, selectTools } from './toolSelect';
import { parseArguments, runTool, toolDefinitions, type ToolResult } from './tools';

/** Most definitions in one request, however large the budget. */
export const MAX_TOOLS_PER_STEP = 30;

export interface ApprovalRequest {
  callId: string;
  /** The name the model called. */
  tool: string;
  title: string;
  server: string | null;
  /** The arguments as the model wrote them, pretty-printed and cut. */
  args: string;
  readOnly: boolean;
}

export interface ToolHost {
  /** Tools from outside this page are in play: the model is told to treat their results as data. */
  external: boolean;
  /** Said to the model once, in the system message: when to reach for the tools. Empty for none. */
  guidance: string;
  definitions(query: string, unlocked: ReadonlySet<string>): WireTool[];
  /** A request when the user must say yes first; null when the call may run. */
  approval(call: ToolCall): ApprovalRequest | null;
  run(call: ToolCall, signal: AbortSignal, unlock: (names: string[]) => void): Promise<ToolResult>;
}

const fail = (text: string): ToolResult => ({ content: `error: ${text}`, isError: true });

export function needsApproval(spec: ToolSpec, config: Pick<ToolsConfig, 'approval' | 'alwaysAllow'>): boolean {
  if (config.approval === 'never' || config.alwaysAllow.includes(spec.name)) return false;
  if (config.approval === 'always') return true;
  return !spec.readOnly;
}

/** The two built-in tools and nothing else: what the chat offered before MCP. */
export function builtinHost(): ToolHost {
  return {
    external: false,
    guidance: '',
    definitions: () => toolDefinitions(),
    approval: () => null,
    run: async (call) => runTool(call.name, call.arguments),
  };
}

/** The longest value shown in full on the approval card; a longer one is cut, and says by how much. */
const SHOWN_VALUE_CHARS = 240;
const SHOWN_KEYS = 40;

function shown(v: unknown): unknown {
  const text = typeof v === 'string' ? v : JSON.stringify(v);
  if (text === undefined || text.length <= SHOWN_VALUE_CHARS) return v;
  return `${text.slice(0, SHOWN_VALUE_CHARS)}… (${text.length - SHOWN_VALUE_CHARS} more characters)`;
}

/** The arguments for the approval card. Every key is shown whatever the order the model wrote them in and
 * each value is cut on its own, so a long `content` can never push the `path` out of sight. */
function prettyArguments(raw: string): string {
  try {
    const parsed: unknown = JSON.parse(raw || '{}');
    if (parsed !== null && typeof parsed === 'object' && !Array.isArray(parsed)) {
      const entries = Object.entries(parsed);
      const out = Object.fromEntries(entries.slice(0, SHOWN_KEYS).map(([k, v]) => [k, shown(v)]));
      const more = entries.length - SHOWN_KEYS;
      return JSON.stringify(more > 0 ? { ...out, '…': `${more} more arguments` } : out, null, 2);
    }
    return JSON.stringify(shown(parsed), null, 2);
  } catch {
    // Not JSON: show what the model wrote, its start and its end.
    return raw.length > 800 ? `${raw.slice(0, 600)}\n…\n${raw.slice(-200)}` : raw;
  }
}

function describe(e: unknown): string {
  if (e instanceof DOMException && e.name === 'AbortError') return 'stopped';
  if (e instanceof ApiError) return e.status === 0 ? 'the tool service could not be reached' : e.serverMessage || e.message;
  return 'the tool failed';
}

export interface HostInput {
  /** Every MCP tool the service listed for the servers that are on and allowed to run. */
  mcp: ToolSpec[];
  config: ToolsConfig;
  internet: InternetConfig;
}

/** When to use the tools, for a small model that otherwise answers from memory. Short: every word is prompt the server must read once. */
export function toolGuidance(o: { search: boolean; mcp: boolean }): string {
  // The built-in tools alone describe themselves well enough; no extra words in every request.
  if (!o.search && !o.mcp) return '';
  const lines = [
    'You can call tools. Answer from your own knowledge when you are confident: well-known facts, definitions, explanations, writing and code need no tool (for example the capital of France, who wrote Hamlet, what a loop is).',
  ];
  if (o.search)
    lines.push(
      'Call web_search only when the answer depends on recent or changing information (news, latest versions, prices, scores, current events) or on something specific you do not know or are unsure of.',
    );
  lines.push('Use calculator for arithmetic and current_time for the date or time.');
  if (o.search) lines.push('After a search, answer from the results and name the source address.');
  if (o.mcp) lines.push('For anything else a tool could do, such as files, notes or git, call find_tools to see what is available.');
  return lines.join(' ');
}

export function createToolHost(i: HostInput): ToolHost {
  const enabled = i.mcp.filter((s) => {
    if (s.source.kind !== 'mcp') return false;
    const server = i.config.servers[s.source.server];
    return !!server?.on && !server.disabledTools.includes(s.source.tool);
  });
  const searchOn = i.internet.on && i.internet.searchUrl !== '';
  const pinned = [...builtinSpecs(), ...(searchOn ? [webSearchSpec()] : []), ...(enabled.length ? [findToolsSpec()] : [])];
  const byName = new Map([...pinned, ...enabled].map((s) => [s.name, s]));
  return {
    external: enabled.length > 0 || searchOn,
    guidance: toolGuidance({ search: searchOn, mcp: enabled.length > 0 }),
    definitions: (query, unlocked) =>
      selectTools({ pinned, candidates: enabled, query, budgetTokens: i.config.budgetTokens, maxTools: MAX_TOOLS_PER_STEP, unlocked }).map(toWire),
    approval(call) {
      const spec = byName.get(call.name);
      if (!spec || !needsApproval(spec, i.config)) return null;
      const server = spec.source.kind === 'mcp' ? spec.source.server : null;
      return {
        callId: call.id, tool: spec.name, title: spec.source.kind === 'mcp' ? `${spec.source.server} · ${spec.source.tool}` : spec.name,
        server, args: prettyArguments(call.arguments), readOnly: spec.readOnly,
      };
    },
    async run(call, signal, unlock) {
      const spec = byName.get(call.name);
      if (!spec) return fail(`no tool named "${call.name}"`);
      const parsed = parseArguments(call.arguments);
      if ('error' in parsed) return fail(parsed.error);
      const query = typeof parsed.args.query === 'string' ? parsed.args.query.trim() : '';
      try {
        switch (spec.source.kind) {
          case 'builtin':
            return runTool(call.name, call.arguments);
          case 'find': {
            if (!query) return fail('query must be a non-empty string');
            const hits = searchSpecs(enabled, query, 8);
            unlock(hits.map((h) => h.name));
            const lines = hits.map((h) => `${h.name}: ${h.description.slice(0, 160)}`);
            return { content: hits.length ? `${lines.join('\n')}\nThese are now available to call.` : 'no matching tool', isError: false };
          }
          case 'search': {
            if (!query) return fail('query must be a non-empty string');
            return await toolsSearch(query, i.internet.searchUrl, signal);
          }
          case 'mcp':
            return await toolsCall(spec.source.server, spec.source.tool, parsed.args, i.internet.on, signal);
        }
      } catch (e) {
        return fail(describe(e));
      }
    },
  };
}
