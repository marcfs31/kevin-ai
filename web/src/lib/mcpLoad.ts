// Asks the tool service for the tools of the servers the user turned on. One server at a time, so
// a few starting together do not take the memory the model needs.
import { toolsList, toolsStatus } from './api';
import { FIND_TOOLS, WEB_SEARCH, mcpSpecs, type ToolSpec } from './toolRegistry';

export interface McpLoad {
  specs: ToolSpec[];
  /** Servers not asked: not in the service's config, or they need the internet and it is off. */
  skipped: { id: string; reason: 'missing' | 'internet_off' }[];
  /** Servers that were asked and did not answer. */
  failed: { id: string; message: string }[];
}

export async function loadMcpSpecs(ids: string[], internet: boolean, signal?: AbortSignal): Promise<McpLoad> {
  const status = await toolsStatus(signal);
  // Names the built-ins already use, so no server can shadow one.
  const taken = new Set(['calculator', 'current_time', FIND_TOOLS, WEB_SEARCH]);
  const out: McpLoad = { specs: [], skipped: [], failed: [] };
  for (const id of ids) {
    const server = status.servers.find((s) => s.id === id);
    if (!server) out.skipped.push({ id, reason: 'missing' });
    else if (server.network && !internet) out.skipped.push({ id, reason: 'internet_off' });
    else {
      try {
        out.specs.push(...mcpSpecs(id, await toolsList(id, internet, signal), taken));
      } catch (e) {
        if (e instanceof DOMException && e.name === 'AbortError') throw e;
        out.failed.push({ id, message: e instanceof Error ? e.message : 'failed' });
      }
    }
  }
  return out;
}
