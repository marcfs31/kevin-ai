// What the model can be offered: the built-in tools, the tools of the MCP servers the user turned
// on, web search, and find_tools. Pure data and naming; running them is lib/toolHost.ts.
import type { WireTool } from './api';
import { toolDefinitions } from './tools';

export const FIND_TOOLS = 'find_tools';
export const WEB_SEARCH = 'web_search';
/** The longest function name the OpenAI-style tools field takes. */
const NAME_MAX = 64;

export type ToolSource = { kind: 'builtin' } | { kind: 'search' } | { kind: 'find' } | { kind: 'mcp'; server: string; tool: string };

export interface ToolSpec {
  /** The name the model sees and calls. */
  name: string;
  description: string;
  parameters: Record<string, unknown>;
  /** The server said the tool changes nothing (MCP readOnlyHint), or it is ours and does not. */
  readOnly: boolean;
  /** Estimated tokens of this definition in the request. */
  tokens: number;
  source: ToolSource;
}

/** What /tools/servers/<id>/tools returns for one tool. */
export interface McpTool {
  name: string;
  title: string;
  description: string;
  inputSchema: Record<string, unknown>;
  readOnly: boolean;
  tokens: number;
}

export const estimateTokens = (def: unknown): number => Math.ceil(JSON.stringify(def).length / 3.5);

export function toWire(spec: ToolSpec): WireTool {
  return { type: 'function', function: { name: spec.name, description: spec.description, parameters: spec.parameters } };
}

function make(name: string, description: string, parameters: Record<string, unknown>, readOnly: boolean, source: ToolSource): ToolSpec {
  return { name, description, parameters, readOnly, tokens: estimateTokens({ name, description, parameters }), source };
}

export function builtinSpecs(): ToolSpec[] {
  return toolDefinitions().map((d) => make(d.function.name, d.function.description, d.function.parameters, true, { kind: 'builtin' }));
}

export function findToolsSpec(): ToolSpec {
  return make(
    FIND_TOOLS,
    'Search the other available tools by what you need (for example "read a file" or "convert time zones"). Returns their names and what they do, and makes them available to call next.',
    { type: 'object', properties: { query: { type: 'string', description: 'What you want to do.' } }, required: ['query'] },
    true,
    { kind: 'find' },
  );
}

export function webSearchSpec(): ToolSpec {
  return make(
    WEB_SEARCH,
    'Search the web and return the top results with their titles, addresses and snippets.',
    { type: 'object', properties: { query: { type: 'string', description: 'The search terms.' } }, required: ['query'] },
    true,
    { kind: 'search' },
  );
}

const safe = (s: string) => s.replace(/[^a-zA-Z0-9_-]/g, '_');

/** server__tool, kept to the name rules and made unique among the names already `taken` (which it grows). */
export function wireName(server: string, tool: string, taken: Set<string>): string {
  const base = `${safe(server)}__${safe(tool)}`.slice(0, NAME_MAX);
  let name = base;
  for (let i = 2; taken.has(name); i++) name = `${base.slice(0, NAME_MAX - String(i).length - 1)}_${i}`;
  taken.add(name);
  return name;
}

export function mcpSpecs(server: string, tools: McpTool[], taken: Set<string>): ToolSpec[] {
  return tools.map((t) => {
    const name = wireName(server, t.name, taken);
    return { name, description: t.description, parameters: t.inputSchema, readOnly: t.readOnly, tokens: estimateTokens({ name, description: t.description, parameters: t.inputSchema }), source: { kind: 'mcp', server, tool: t.name } };
  });
}
