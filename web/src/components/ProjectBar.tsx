import { useEffect, useId, useState } from 'react';
import { useI18n } from '../i18n';
import { toolsCall, type ProjectInfo } from '../lib/api';
import { parseGitStatus, type GitChip } from '../lib/projectGit';
import { PROJECT_SERVER } from '../lib/toolHost';

/** The project this chat works in, and where its git stands (branch, changed files). Shown only when the tool service
 * lists projects; the chip is read with git_status, which changes nothing, and refreshed after each agent run. */
export function ProjectBar({ projects, projectId, disabled, refreshKey, onChange }: {
  projects: ProjectInfo[];
  projectId: string | null;
  disabled: boolean;
  /** Changes when a run ends, so the chip is read again. */
  refreshKey: number;
  onChange: (id: string | null) => void;
}) {
  const { t } = useI18n();
  const id = useId();
  const [chip, setChip] = useState<{ project: string; value: GitChip | null } | null>(null);
  const known = projectId !== null && projects.some((p) => p.id === projectId);
  useEffect(() => {
    if (!known || projectId === null) return;
    const controller = new AbortController();
    toolsCall(PROJECT_SERVER, 'git_status', { project: projectId }, false, controller.signal).then(
      (r) => setChip({ project: projectId, value: r.isError ? null : parseGitStatus(r.content) }),
      () => {
        if (!controller.signal.aborted) setChip({ project: projectId, value: null });
      },
    );
    return () => controller.abort();
  }, [known, projectId, refreshKey]);
  if (projects.length === 0) return null;
  const shown = known && chip?.project === projectId ? chip.value : null;
  return (
    <div className="project-bar">
      <span className="chat-model">
        <label htmlFor={`${id}-project`}>{t('project.label')}</label>
        <select id={`${id}-project`} value={known ? (projectId ?? '') : ''} disabled={disabled} onChange={(e) => onChange(e.target.value || null)}>
          <option value="">{t('project.none')}</option>
          {projects.map((p) => (
            <option key={p.id} value={p.id}>
              {p.title}
            </option>
          ))}
        </select>
      </span>
      {known && (
        <span className="git-chip" role="status">
          {shown ? t('project.git', { branch: shown.branch, n: shown.changes }) : t('project.gitUnknown')}
        </span>
      )}
    </div>
  );
}
