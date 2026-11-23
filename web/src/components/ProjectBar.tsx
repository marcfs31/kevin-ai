import { useEffect, useId, useState } from 'react';
import { useI18n } from '../i18n';
import { toolsCall, type HostedProject, type ToolsHost } from '../lib/api';
import { parseGitStatus, type GitChip } from '../lib/projectGit';
import { PROJECT_SERVER } from '../lib/toolHost';

const keyOf = (host: ToolsHost, id: string): string => `${host}:${id}`;

/** The project this chat works in (on the model server, or on this Mac when that host is on), and where its git stands
 * (branch, changed files). Shown only when a tool service lists projects; the chip is read with git_status, which
 * changes nothing, and refreshed after each agent run. */
export function ProjectBar({ projects, projectId, projectHost, disabled, refreshKey, onChange }: {
  projects: HostedProject[];
  projectId: string | null;
  projectHost: ToolsHost;
  disabled: boolean;
  /** Changes when a run ends, so the chip is read again; negative while a run is going (nothing is read then). */
  refreshKey: number;
  onChange: (id: string | null, host: ToolsHost) => void;
}) {
  const { t } = useI18n();
  const id = useId();
  const [chip, setChip] = useState<{ key: string; value: GitChip | null } | null>(null);
  const current = projectId === null ? null : keyOf(projectHost, projectId);
  const known = current !== null && projects.some((p) => keyOf(p.host, p.id) === current);
  useEffect(() => {
    if (!known || projectId === null || current === null || refreshKey < 0) return;
    const controller = new AbortController();
    toolsCall(PROJECT_SERVER, 'git_status', { project: projectId }, false, controller.signal, projectHost).then(
      (r) => setChip({ key: current, value: r.isError ? null : parseGitStatus(r.content) }),
      () => {
        if (!controller.signal.aborted) setChip({ key: current, value: null });
      },
    );
    return () => controller.abort();
  }, [known, projectId, projectHost, current, refreshKey]);
  if (projects.length === 0) return null;
  const shown = known && chip?.key === current ? chip.value : null;
  return (
    <div className="project-bar">
      <span className="chat-model">
        <label htmlFor={`${id}-project`}>{t('project.label')}</label>
        <select
          id={`${id}-project`}
          value={known ? (current ?? '') : ''}
          disabled={disabled}
          onChange={(e) => {
            const picked = projects.find((p) => keyOf(p.host, p.id) === e.target.value);
            onChange(picked ? picked.id : null, picked ? picked.host : 'server');
          }}
        >
          <option value="">{t('project.none')}</option>
          {projects.map((p) => (
            <option key={keyOf(p.host, p.id)} value={keyOf(p.host, p.id)}>
              {p.host === 'local' ? t('project.onMac', { title: p.title }) : p.title}
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
