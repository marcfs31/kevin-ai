// Reads the text git_status of the project server answers ("## branch...origin/branch [ahead 1]" and one line per
// changed file, or "no changes") into what the git chip shows. Anything else is "unknown", never a guess.
export interface GitChip {
  branch: string;
  /** Changed or untracked files. */
  changes: number;
}

export function parseGitStatus(text: string): GitChip | null {
  const lines = text.split('\n');
  const head = lines[0] ?? '';
  if (!head.startsWith('## ')) return null;
  let branch = head.slice(3).trim();
  const fresh = /^No commits yet on (\S+)/.exec(branch);
  if (fresh) branch = fresh[1] ?? '';
  else {
    const dots = branch.indexOf('...');
    if (dots > 0) branch = branch.slice(0, dots);
    branch = branch.replace(/ \[.*\]$/, '');
  }
  if (!branch) return null;
  const rest = lines.slice(1).filter((l) => l.trim() !== '' && l.trim() !== 'no changes' && !/^\[\d+ more entries\]$/.test(l.trim()));
  const more = /^\[(\d+) more entries\]$/m.exec(text);
  return { branch, changes: rest.length + (more ? Number(more[1]) : 0) };
}
