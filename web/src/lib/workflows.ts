// Workflows: saved step recipes for a project, kept in the repository as `.kevin/workflows.json` (they are prompts, not
// permissions: what the agent may do is still decided by the owner's registry). Choosing one in a chat copies its steps
// into the conversation as a checklist the model ticks off with workflow_step and the user can change.
//
//   {"workflows": [{"id": "fix-bug", "title": "Fix a bug", "steps": ["Reproduce it", "Write a failing test", ...]}]}

export const WORKFLOWS_PATH = '.kevin/workflows.json';
export const MAX_WORKFLOWS = 20;
export const MAX_STEPS = 12;
export const MAX_STEP_CHARS = 200;
export const MAX_NOTE_CHARS = 200;

export const STEP_STATUSES = ['done', 'skipped', 'blocked'] as const;
export type StepResolution = (typeof STEP_STATUSES)[number];
export type StepStatus = 'todo' | StepResolution;

export interface WorkflowDef {
  id: string;
  title: string;
  steps: string[];
}

export interface WorkflowStep {
  text: string;
  status: StepStatus;
  note: string;
}

/** A workflow started in a chat: the steps and where each one stands. */
export interface Workflow {
  id: string;
  title: string;
  steps: WorkflowStep[];
}

const ID = /^[a-z0-9][a-z0-9_-]{0,31}$/;
const isRecord = (v: unknown): v is Record<string, unknown> => typeof v === 'object' && v !== null && !Array.isArray(v);
const isStatus = (v: unknown): v is StepStatus => v === 'todo' || (STEP_STATUSES as readonly unknown[]).includes(v);

export interface ParsedWorkflows {
  defs: WorkflowDef[];
  /** The file exists but cannot be used; null when it is fine (or simply absent). */
  problem: 'invalid' | 'too_long' | null;
}

/** What the project server's fs_read returns: a header line, the lines, and a note when more follow. */
export function fileText(readResult: string): { text: string; complete: boolean } {
  const lines = readResult.split('\n');
  const body = /^\S+ lines \d+-\d+ of \d+$/.test(lines[0] ?? '') ? lines.slice(1) : lines;
  const complete = !/^\[more lines follow/.test(body[body.length - 1] ?? '');
  return { text: (complete ? body : body.slice(0, -1)).join('\n'), complete };
}

export function parseWorkflows(readResult: string): ParsedWorkflows {
  const { text, complete } = fileText(readResult);
  if (!complete) return { defs: [], problem: 'too_long' };
  let doc: unknown;
  try {
    doc = JSON.parse(text);
  } catch {
    return { defs: [], problem: 'invalid' };
  }
  const list = isRecord(doc) && Array.isArray(doc.workflows) ? doc.workflows : null;
  if (!list) return { defs: [], problem: 'invalid' };
  const defs: WorkflowDef[] = [];
  const seen = new Set<string>();
  for (const w of list.slice(0, MAX_WORKFLOWS)) {
    if (!isRecord(w) || typeof w.id !== 'string' || !ID.test(w.id) || seen.has(w.id) || !Array.isArray(w.steps)) continue;
    const steps = w.steps
      .filter((s): s is string => typeof s === 'string' && s.trim() !== '')
      .slice(0, MAX_STEPS)
      .map((s) => s.trim().slice(0, MAX_STEP_CHARS));
    if (steps.length === 0) continue;
    seen.add(w.id);
    defs.push({ id: w.id, title: typeof w.title === 'string' && w.title.trim() ? w.title.trim().slice(0, 80) : w.id, steps });
  }
  return { defs, problem: null };
}

export function startWorkflow(def: WorkflowDef): Workflow {
  return { id: def.id, title: def.title, steps: def.steps.map((text) => ({ text, status: 'todo', note: '' })) };
}

/** A saved workflow read back from storage; null when it is not one. */
export function normalizeWorkflow(v: unknown): Workflow | null {
  if (!isRecord(v) || typeof v.id !== 'string' || !ID.test(v.id) || !Array.isArray(v.steps)) return null;
  const steps = v.steps.flatMap((s): WorkflowStep[] => {
    if (!isRecord(s) || typeof s.text !== 'string' || s.text === '') return [];
    return [{ text: s.text.slice(0, MAX_STEP_CHARS), status: isStatus(s.status) ? s.status : 'todo', note: typeof s.note === 'string' ? s.note.slice(0, MAX_NOTE_CHARS) : '' }];
  });
  if (steps.length === 0 || steps.length > MAX_STEPS) return null;
  return { id: v.id, title: typeof v.title === 'string' && v.title ? v.title.slice(0, 80) : v.id, steps };
}

/** Marks step `step` (counting from 1). Null when there is no such step. */
export function setStep(w: Workflow, step: number, status: StepStatus, note = ''): Workflow | null {
  if (!Number.isInteger(step) || step < 1 || step > w.steps.length) return null;
  return { ...w, steps: w.steps.map((s, i) => (i === step - 1 ? { ...s, status, note: note.trim().slice(0, MAX_NOTE_CHARS) } : s)) };
}

export const isFinished = (w: Workflow): boolean => w.steps.every((s) => s.status !== 'todo');

const line = (s: WorkflowStep, i: number): string => `${i + 1}. [${s.status}] ${s.text}${s.note ? ` (${s.note})` : ''}`;

/** The checklist and how to use it, for the system message: short, because every word is prompt the server reads. */
export function workflowGuidance(w: Workflow): string {
  return [
    `You are following the workflow "${w.title}". Steps: ${w.steps.map(line).join(' ')}`,
    'Do the steps in order. After each one call workflow_step with its number, the status (done, skipped or blocked) and a short note. Do not mark a step done before it is. When a step is blocked, say why and stop.',
  ].join(' ');
}

/** What workflow_step answers: the next step to do, or that all are resolved. */
export function stepResult(w: Workflow, step: number): string {
  const next = w.steps.findIndex((s) => s.status === 'todo');
  const head = `step ${step} is now ${w.steps[step - 1]?.status}`;
  return next === -1 ? `${head}. All steps are resolved: summarise what was done.` : `${head}. Next: ${line(w.steps[next] as WorkflowStep, next)}`;
}
