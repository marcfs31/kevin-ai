import { describe, expect, it } from 'vitest';
import { FIND_TOOLS, WEB_SEARCH, builtinSpecs, estimateTokens, findToolsSpec, mcpSpecs, toWire, webSearchSpec, wireName, type McpTool } from './toolRegistry';

const tool = (name: string, extra: Partial<McpTool> = {}): McpTool => ({ name, title: name, description: `does ${name}`, inputSchema: { type: 'object', properties: {} }, readOnly: false, tokens: 0, ...extra });

describe('wireName', () => {
  it('joins server and tool and keeps to the characters a function name may have', () => {
    expect(wireName('fs', 'read.file', new Set())).toBe('fs__read_file');
    expect(wireName('my-server', 'a b/c', new Set())).toBe('my-server__a_b_c');
  });

  it('keeps to 64 characters and stays unique among the names taken', () => {
    const taken = new Set<string>();
    const long = 'x'.repeat(100);
    const a = wireName('s', long, taken);
    expect(a).toHaveLength(64);
    const b = wireName('s', long, taken);
    expect(b).toHaveLength(64);
    expect(b).not.toBe(a);
    expect(b.endsWith('_2')).toBe(true);
    expect(wireName('s', long, taken).endsWith('_3')).toBe(true);
    expect(taken.size).toBe(3);
  });

  it('turns a clash between two different tools into two names', () => {
    const taken = new Set<string>();
    expect(wireName('a', 'b_c', taken)).toBe('a__b_c');
    expect(wireName('a', 'b.c', taken)).toBe('a__b_c_2');
  });
});

describe('specs', () => {
  it('builds MCP specs with the server and tool kept apart from the wire name', () => {
    const [spec] = mcpSpecs('fs', [tool('read.file', { readOnly: true })], new Set());
    expect(spec).toMatchObject({ name: 'fs__read_file', readOnly: true, source: { kind: 'mcp', server: 'fs', tool: 'read.file' } });
    expect(spec?.tokens).toBeGreaterThan(5);
  });

  it('never lets a server take a name already in use', () => {
    const taken = new Set(['calculator', FIND_TOOLS, WEB_SEARCH]);
    const specs = mcpSpecs('x', [tool('a'), tool('a')], taken);
    expect(new Set(specs.map((s) => s.name)).size).toBe(2);
  });

  it('has the two built-ins read-only, find_tools and web_search with a required query', () => {
    expect(builtinSpecs().map((s) => s.name)).toEqual(['calculator', 'current_time']);
    expect(builtinSpecs().every((s) => s.readOnly)).toBe(true);
    for (const s of [findToolsSpec(), webSearchSpec()]) expect(s.parameters).toMatchObject({ required: ['query'] });
  });

  it('wires a spec as a function tool and estimates tokens from its size', () => {
    const spec = findToolsSpec();
    expect(toWire(spec)).toEqual({ type: 'function', function: { name: FIND_TOOLS, description: spec.description, parameters: spec.parameters } });
    expect(estimateTokens('x'.repeat(350))).toBe(Math.ceil(352 / 3.5));
  });
});
