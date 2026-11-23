// Groups the names of the tools sent with a request by where they come from, for the tools viewer: the page's own tools
// (built-in, web search, find_tools, workflow_step) and one group per MCP server (names are `server__tool`).

export const BUILTIN_TOOLS: readonly string[] = ['calculator', 'current_time', 'web_search', 'find_tools', 'workflow_step'];

export interface ToolGroup {
  /** The server's id, or `builtin`. */
  id: string;
  builtin: boolean;
  tools: { name: string; label: string }[];
}

export function groupTools(names: readonly string[]): ToolGroup[] {
  const groups = new Map<string, ToolGroup>();
  for (const name of names) {
    const cut = name.indexOf('__');
    const builtin = cut <= 0 || BUILTIN_TOOLS.includes(name);
    const id = builtin ? 'builtin' : name.slice(0, cut);
    const label = builtin ? name : name.slice(cut + 2);
    let g = groups.get(id);
    if (!g) groups.set(id, (g = { id, builtin, tools: [] }));
    g.tools.push({ name, label });
  }
  // The page's own tools first, then the servers in the order they were first used.
  return [...groups.values()].sort((a, b) => Number(b.builtin) - Number(a.builtin));
}
