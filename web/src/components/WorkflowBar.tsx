import { useEffect, useId, useState } from 'react';
import { useI18n } from '../i18n';
import { toolsCall, type ProjectInfo } from '../lib/api';
import { PROJECT_SERVER } from '../lib/toolHost';
import { WORKFLOWS_PATH, isFinished, parseWorkflows, type ParsedWorkflows, type StepStatus, type Workflow, type WorkflowDef } from '../lib/workflows';

/** Above the composer: choose a workflow of the project (from `.kevin/workflows.json`) and follow its checklist. The model
 * ticks the steps with workflow_step; the user can tick or untick them too, and stop the workflow. */
export function WorkflowBar({ project, workflow, disabled, onStart, onSet, onClear }: {
  project: ProjectInfo | null;
  workflow: Workflow | null;
  disabled: boolean;
  onStart: (def: WorkflowDef) => void;
  onSet: (step: number, status: StepStatus) => void;
  onClear: () => void;
}) {
  const { t } = useI18n();
  const id = useId();
  const [loaded, setLoaded] = useState<{ project: string; parsed: ParsedWorkflows } | null>(null);
  const [pick, setPick] = useState('');
  const projectId = project?.id ?? null;
  useEffect(() => {
    if (projectId === null || workflow) return;
    const controller = new AbortController();
    toolsCall(PROJECT_SERVER, 'fs_read', { project: projectId, path: WORKFLOWS_PATH }, false, controller.signal).then(
      (r) => setLoaded({ project: projectId, parsed: r.isError ? { defs: [], problem: null } : parseWorkflows(r.content) }),
      () => {
        if (!controller.signal.aborted) setLoaded({ project: projectId, parsed: { defs: [], problem: null } });
      },
    );
    return () => controller.abort();
  }, [projectId, workflow]);

  if (workflow) {
    const done = workflow.steps.filter((s) => s.status !== 'todo').length;
    return (
      <section className="workflow-bar" aria-label={t('workflow.label')}>
        <div className="workflow-head">
          <strong>{workflow.title}</strong>
          <span role="status">{isFinished(workflow) ? t('workflow.finished') : t('workflow.progress', { done, total: workflow.steps.length })}</span>
          <button type="button" onClick={onClear} disabled={disabled}>
            {t('workflow.stop')}
          </button>
        </div>
        <ol className="workflow-steps">
          {workflow.steps.map((s, i) => (
            <li key={i} className={`step-${s.status}`}>
              <label>
                <input type="checkbox" checked={s.status === 'done'} disabled={disabled} onChange={(e) => onSet(i + 1, e.target.checked ? 'done' : 'todo')} />
                <span>{s.text}</span>
              </label>
              {(s.status === 'skipped' || s.status === 'blocked') && <span className="step-state"> {t(`workflow.${s.status}`)}</span>}
              {s.note && <span className="step-note"> {s.note}</span>}
            </li>
          ))}
        </ol>
      </section>
    );
  }

  const mine = loaded && loaded.project === projectId ? loaded.parsed : null;
  if (!mine || (mine.defs.length === 0 && !mine.problem)) return null;
  if (mine.problem) {
    return (
      <p className="workflow-bar note" role="status">
        {t(mine.problem === 'too_long' ? 'workflow.tooLong' : 'workflow.invalid', { file: WORKFLOWS_PATH })}
      </p>
    );
  }
  const chosen = mine.defs.find((d) => d.id === pick) ?? mine.defs[0];
  return (
    <div className="workflow-bar">
      <span className="chat-model">
        <label htmlFor={`${id}-wf`}>{t('workflow.label')}</label>
        <select id={`${id}-wf`} value={chosen?.id ?? ''} disabled={disabled} onChange={(e) => setPick(e.target.value)}>
          {mine.defs.map((d) => (
            <option key={d.id} value={d.id}>
              {d.title}
            </option>
          ))}
        </select>
      </span>
      <button type="button" disabled={disabled || !chosen} onClick={() => chosen && onStart(chosen)}>
        {t('workflow.start')}
      </button>
    </div>
  );
}
