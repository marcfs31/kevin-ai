import { describe, expect, it } from 'vitest';
import { groupTools } from './toolGroups';

describe('groupTools', () => {
  it('puts the page\'s own tools first and one group per server after them, in the order first used', () => {
    const groups = groupTools(['project__fs_write', 'calculator', 'filesystem__write_file', 'web_search', 'project__pr_create', 'workflow_step']);
    expect(groups.map((g) => [g.id, g.builtin])).toEqual([['builtin', true], ['project', false], ['filesystem', false]]);
    expect(groups[0]?.tools.map((t) => t.label)).toEqual(['calculator', 'web_search', 'workflow_step']);
  });

  it('cuts the server off the label and keeps the full name for the tooltip', () => {
    const [group] = groupTools(['git__git_create_branch']);
    expect(group?.tools).toEqual([{ name: 'git__git_create_branch', label: 'git_create_branch' }]);
  });

  it('treats a name without a server prefix as the page\'s own, and keeps a collision suffix in the label', () => {
    expect(groupTools(['mystery'])[0]).toMatchObject({ id: 'builtin', builtin: true });
    expect(groupTools(['fs__read_2'])[0]?.tools[0]?.label).toBe('read_2');
    expect(groupTools(['__odd'])[0]?.id).toBe('builtin');      // an empty server name is not a server
  });

  it('is empty for no tools', () => {
    expect(groupTools([])).toEqual([]);
  });
});
