import { describe, expect, it } from 'vitest';
import type { ToolSpec } from './toolRegistry';
import { fitCount, searchSpecs, selectTools, terms } from './toolSelect';

const spec = (name: string, description: string, tokens = 100): ToolSpec => ({ name, description, parameters: {}, readOnly: true, tokens, network: false, source: { kind: 'mcp', server: 's', tool: name } });
const FILES = [
  spec('fs__read_file', 'Read the complete contents of a file as text'),
  spec('fs__write_file', 'Create a new file or overwrite an existing file'),
  spec('fs__list_directory', 'List the files and folders in a directory'),
  spec('time__get_current_time', 'Get the current time in a time zone'),
  spec('time__convert_time', 'Convert a time between time zones'),
  spec('git__git_status', 'Show the working tree status of a repository'),
];

describe('terms', () => {
  it('lowercases, splits on punctuation and camelCase, and drops small words', () => {
    expect(terms('Read the readFile tool, please!')).toEqual(['read', 'read', 'file', 'tool']);
    expect(terms('a I of')).toEqual([]);
  });
});

describe('searchSpecs', () => {
  it('ranks what matches the request first, name hits above description hits', () => {
    expect(searchSpecs(FILES, 'read this file', 3)[0]?.name).toBe('fs__read_file');
    expect(searchSpecs(FILES, 'what time is it in Madrid time zone', 2).map((s) => s.name)).toEqual(['time__get_current_time', 'time__convert_time']);
    expect(searchSpecs(FILES, 'git status', 1)[0]?.name).toBe('git__git_status');
  });

  it('says nothing when nothing matches, and for an empty query or list', () => {
    expect(searchSpecs(FILES, 'translate french poetry', 5)).toEqual([]);
    expect(searchSpecs(FILES, '', 5)).toEqual([]);
    expect(searchSpecs([], 'file', 5)).toEqual([]);
  });

  it('prefers a rare term to a common one and breaks ties by name', () => {
    const specs = [spec('b_tool', 'common word'), spec('a_tool', 'common word'), spec('c_tool', 'common rareword')];
    expect(searchSpecs(specs, 'common rareword', 3)[0]?.name).toBe('c_tool');
    expect(searchSpecs(specs, 'common', 3).map((s) => s.name)).toEqual(['a_tool', 'b_tool', 'c_tool']);
  });

  it('respects the limit', () => {
    expect(searchSpecs(FILES, 'file time', 2)).toHaveLength(2);
  });
});

describe('selectTools', () => {
  const base = { pinned: [spec('calculator', 'math', 50)], candidates: FILES, query: 'read a file', budgetTokens: 1000, maxTools: 12, unlocked: new Set<string>() };

  it('offers the pinned tools always, then the best matches within the budget', () => {
    const names = selectTools(base).map((s) => s.name);
    expect(names[0]).toBe('calculator');
    expect(names).toContain('fs__read_file');
    expect(names).not.toContain('git__git_status');
  });

  it('stops at the budget, which is for the tools beyond the pinned ones; a match that does not fit is skipped', () => {
    const tight = selectTools({ ...base, budgetTokens: 110 }).map((s) => s.name);
    expect(tight).toHaveLength(2);                      // the 50-token pinned tool is free; one 100-token match fits
    expect(selectTools({ ...base, budgetTokens: 40 }).map((s) => s.name)).toEqual(['calculator']);   // nothing fits: only the pinned tool
    expect(selectTools({ ...base, budgetTokens: 300 }).length).toBeGreaterThan(tight.length);        // a bigger budget only ever adds
  });

  it('never goes over maxTools', () => {
    expect(selectTools({ ...base, maxTools: 2, query: 'file time git directory' })).toHaveLength(2);
  });

  it('offers a tool the model unlocked even when the query does not match it and the budget is spent', () => {
    const out = selectTools({ ...base, query: 'hello', budgetTokens: 10, unlocked: new Set(['git__git_status']) }).map((s) => s.name);
    expect(out).toEqual(['calculator', 'git__git_status']);
  });

  it('with a query that matches nothing offers only the pinned tools', () => {
    expect(selectTools({ ...base, query: 'tell me a joke' }).map((s) => s.name)).toEqual(['calculator']);
  });

  it('sends again what the chat sent before, in the same order, whatever the query and budget, and only adds', () => {
    const before = selectTools(base).map((s) => s.name);
    const later = selectTools({ ...base, query: 'what is the git status', budgetTokens: 1000, keep: before }).map((s) => s.name);
    expect(later.slice(0, before.length)).toEqual(before);
    expect(later).toContain('git__git_status');
    expect(selectTools({ ...base, query: 'tell me a joke', budgetTokens: 0, keep: before }).map((s) => s.name)).toEqual(before);
  });

  it('keeps an unlocked tool on the next message, puts a new pinned tool after the kept ones, and drops what is no longer on offer', () => {
    const first = selectTools({ ...base, query: 'hello', unlocked: new Set(['git__git_status']) }).map((s) => s.name);
    expect(selectTools({ ...base, query: 'hello', keep: first }).map((s) => s.name)).toEqual(first);
    const pinned = [...base.pinned, spec('workflow_step', 'tick', 30)];
    expect(selectTools({ ...base, pinned, query: 'hello', keep: first }).map((s) => s.name)).toEqual([...first, 'workflow_step']);
    const fewer = FILES.filter((s) => s.name !== 'git__git_status');
    expect(selectTools({ ...base, candidates: fewer, query: 'hello', keep: first }).map((s) => s.name)).toEqual(['calculator']);
  });

  it('never keeps more than maxTools, and always keeps the pinned tools', () => {
    const many = selectTools({ ...base, query: 'file time git directory', budgetTokens: 5000 }).map((s) => s.name);
    const out = selectTools({ ...base, maxTools: 3, keep: many }).map((s) => s.name);
    expect(out).toHaveLength(3);
    expect(out[0]).toBe('calculator');
    expect(out.slice(1)).toEqual(many.slice(1, 3));
  });

  it('does not offer a tool twice', () => {
    const out = selectTools({ ...base, unlocked: new Set(['fs__read_file']), keep: ['calculator', 'fs__read_file', 'fs__read_file'] }).map((s) => s.name);
    expect(out.filter((n) => n === 'fs__read_file')).toHaveLength(1);
    expect(out.filter((n) => n === 'calculator')).toHaveLength(1);
  });
});

describe('fitCount', () => {
  const spec = (name: string, tokens: number) => ({ name, description: '', parameters: {}, readOnly: true, tokens, network: false, source: { kind: 'mcp' as const, server: 's', tool: name } });
  const five = ['a', 'b', 'c', 'd', 'e'].map((n) => spec(n, 100));

  it('grows with the budget, stops at the number of tools and at the per-step cap', () => {
    expect(fitCount(five, 300, 30, 3)).toBe(3);
    expect(fitCount(five, 500, 30, 3)).toBe(5);
    expect(fitCount(five, 9000, 30, 3)).toBe(5);
    expect(fitCount(five, 9000, 6, 3)).toBe(3);
  });

  it('is zero without tools, with no budget, or when the per-step cap is already used by the always-sent ones', () => {
    expect(fitCount([], 5000, 30, 0)).toBe(0);
    expect(fitCount(five, 0, 30, 3)).toBe(0);
    expect(fitCount(five, 500, 3, 3)).toBe(0);
  });
});
