import { describe, expect, it } from 'vitest';
import { parseGitStatus } from './projectGit';

describe('parseGitStatus', () => {
  it('reads the branch and counts the changed files', () => {
    expect(parseGitStatus('## kevin/fix...origin/kevin/fix [ahead 1]\n M a.txt\n?? b.txt')).toEqual({ branch: 'kevin/fix', changes: 2 });
    expect(parseGitStatus('## main\nno changes')).toEqual({ branch: 'main', changes: 0 });
  });

  it('knows a repository with no commits yet, and the cut list of a big change', () => {
    expect(parseGitStatus('## No commits yet on main\nno changes')).toEqual({ branch: 'main', changes: 0 });
    expect(parseGitStatus('## main\n M a\n[150 more entries]')).toEqual({ branch: 'main', changes: 151 });
  });

  it('says unknown, never guesses, for anything else', () => {
    expect(parseGitStatus('error: not a git repository')).toBeNull();
    expect(parseGitStatus('')).toBeNull();
    expect(parseGitStatus('## ')).toBeNull();
  });
});
