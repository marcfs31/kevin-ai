// What the model can be offered: the built-in tools, the tools of the MCP servers the user turned
// on, web search, and find_tools. Pure data and naming; running them is lib/toolHost.ts.
import type { WireTool } from './api';
import { toolDefinitions } from './tools';

export const FIND_TOOLS = 'find_tools';
export const WEB_SEARCH = 'web_search';
/** The longest function name the OpenAI-style tools field takes. */
const NAME_MAX = 64;

export const WORKFLOW_STEP = 'workflow_step';

export type ToolSource = { kind: 'builtin' } | { kind: 'search' } | { kind: 'find' } | { kind: 'workflow' } | { kind: 'mcp'; server: string; tool: string };

export interface ToolSpec {
  /** The name the model sees and calls. */
  name: string;
  description: string;
  parameters: Record<string, unknown>;
  /** The server said the tool changes nothing (MCP readOnlyHint), or it is ours and does not. */
  readOnly: boolean;
  /** Estimated tokens of this definition in the request. */
  tokens: number;
  /** The tool reaches the network, so what it is given can leave this machine (a server marked network, or web search). */
  network: boolean;
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

/** What "always allow" is stored under: the server and the tool's own name, which do not change when two tools'
 * model-facing names collide and one gets a suffix. Built-ins are known by their name. */
export function allowKey(spec: ToolSpec): string {
  return spec.source.kind === 'mcp' ? `${spec.source.server}/${spec.source.tool}` : spec.name;
}

export const estimateTokens = (def: unknown): number => Math.ceil(JSON.stringify(def).length / 3.5);

export function toWire(spec: ToolSpec): WireTool {
  return { type: 'function', function: { name: spec.name, description: spec.description, parameters: spec.parameters } };
}

function make(name: string, description: string, parameters: Record<string, unknown>, readOnly: boolean, source: ToolSource, network = false): ToolSpec {
  return { name, description, parameters, readOnly, tokens: estimateTokens({ name, description, parameters }), network, source };
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

/** Ticks a step of the chat's workflow. It only changes the checklist on this page, so it never asks. */
export function workflowStepSpec(): ToolSpec {
  return make(
    WORKFLOW_STEP,
    'Mark a step of the current workflow as done, skipped or blocked, with a short note. Answers with the next step.',
    {
      type: 'object',
      properties: {
        step: { type: 'integer', description: 'The step number, from 1.' },
        status: { type: 'string', enum: ['done', 'skipped', 'blocked'] },
        note: { type: 'string', description: 'One short sentence about what was done or what blocks it.' },
      },
      required: ['step', 'status'],
    },
    true,
    { kind: 'workflow' },
  );
}

export function webSearchSpec(): ToolSpec {
  return make(
    WEB_SEARCH,
    'Search the web and return the top results with their titles, addresses and snippets. Use it for recent or changing facts and whenever you are unsure of the answer.',
    { type: 'object', properties: { query: { type: 'string', description: 'The search terms.' } }, required: ['query'] },
    true,
    { kind: 'search' },
    true,
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

export function mcpSpecs(server: string, tools: McpTool[], taken: Set<string>, network = false): ToolSpec[] {
  return tools.map((t) => {
    const name = wireName(server, t.name, taken);
    return { name, description: t.description, parameters: t.inputSchema, readOnly: t.readOnly, tokens: estimateTokens({ name, description: t.description, parameters: t.inputSchema }), network, source: { kind: 'mcp', server, tool: t.name } };
  });
}
