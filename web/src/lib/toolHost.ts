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
    definitions: () => toolDefinitions(),
    approval: () => null,
    run: async (call) => runTool(call.name, call.arguments),
  };
}

function prettyArguments(raw: string): string {
  let text = raw;
  try {
    text = JSON.stringify(JSON.parse(raw || '{}'), null, 2);
  } catch {
    // Not JSON: show what the model wrote.
  }
  return text.length > 800 ? `${text.slice(0, 800)}\n…` : text;
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
