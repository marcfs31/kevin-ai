import { describe, expect, it } from 'vitest';
import { MAX_STEPS, fileText, isFinished, normalizeWorkflow, parseWorkflows, setStep, startWorkflow, stepResult, workflowGuidance } from './workflows';

const read = (body: string) => `.kevin/workflows.json lines 1-9 of 9\n${body}`;
const doc = (workflows: unknown) => read(JSON.stringify({ workflows }));

describe('parseWorkflows', () => {
  it('reads the id, title and steps, trimming and capping them', () => {
    const { defs, problem } = parseWorkflows(doc([{ id: 'fix-bug', title: ' Fix a bug ', steps: [' Reproduce it ', 'Write a test', '', 7] }]));
    expect(problem).toBeNull();
    expect(defs).toEqual([{ id: 'fix-bug', title: 'Fix a bug', steps: ['Reproduce it', 'Write a test'] }]);
    const many = parseWorkflows(doc([{ id: 'big', steps: Array.from({ length: 40 }, (_, i) => `step ${i}`) }]));
    expect(many.defs[0]?.steps).toHaveLength(MAX_STEPS);
    expect(many.defs[0]?.title).toBe('big');
  });

  it('skips entries that are not usable, and duplicates, and keeps the rest', () => {
    const { defs } = parseWorkflows(doc([{ id: 'Bad Id', steps: ['a'] }, { id: 'no-steps', steps: [] }, { id: 'ok', steps: ['a'] }, { id: 'ok', steps: ['b'] }, 'x', null]));
    expect(defs.map((d) => d.id)).toEqual(['ok']);
    expect(defs[0]?.steps).toEqual(['a']);
  });

  it('says invalid for a file that is not workflows, and too long when fs_read cut it', () => {
    expect(parseWorkflows(read('not json')).problem).toBe('invalid');
    expect(parseWorkflows(read('{"workflows": 3}')).problem).toBe('invalid');
    expect(parseWorkflows(read('[]')).problem).toBe('invalid');
    expect(parseWorkflows(`${read('{"workflows": [')}\n[more lines follow: call again with offset 201]`).problem).toBe('too_long');
  });

  it('works on the bare text too (no header line)', () => {
    expect(fileText('{"a":1}')).toEqual({ text: '{"a":1}', complete: true });
    expect(parseWorkflows('{"workflows":[{"id":"x","steps":["a"]}]}').defs).toHaveLength(1);
  });
});

describe('a started workflow', () => {
  const w = startWorkflow({ id: 'x', title: 'X', steps: ['one', 'two', 'three'] });

  it('starts with every step to do, and ticks a step by its number from 1', () => {
    expect(w.steps.map((s) => s.status)).toEqual(['todo', 'todo', 'todo']);
    const next = setStep(w, 2, 'done', ' tests pass ');
    expect(next?.steps.map((s) => s.status)).toEqual(['todo', 'done', 'todo']);
    expect(next?.steps[1]?.note).toBe('tests pass');
    expect(w.steps[1]?.status).toBe('todo');          // the old checklist is untouched
  });

  it('refuses a step that does not exist', () => {
    for (const bad of [0, 4, -1, 1.5, Number.NaN]) expect(setStep(w, bad, 'done')).toBeNull();
  });

  it('is finished only when no step is left to do (skipped and blocked count as resolved)', () => {
    let cur = w;
    cur = setStep(cur, 1, 'done') ?? cur;
    cur = setStep(cur, 2, 'skipped') ?? cur;
    expect(isFinished(cur)).toBe(false);
    cur = setStep(cur, 3, 'blocked', 'needs a token') ?? cur;
    expect(isFinished(cur)).toBe(true);
  });

  it('tells the model the checklist and what to do, and answers with the next step', () => {
    const g = workflowGuidance(setStep(w, 1, 'done', 'found it') as typeof w);
    expect(g).toContain('"X"');
    expect(g).toContain('1. [done] one (found it)');
    expect(g).toContain('2. [todo] two');
    expect(g).toContain('workflow_step');
    const after = setStep(w, 1, 'done') as typeof w;
    expect(stepResult(after, 1)).toBe('step 1 is now done. Next: 2. [todo] two');
    let all = after;
    all = setStep(all, 2, 'done') ?? all;
    all = setStep(all, 3, 'done') ?? all;
    expect(stepResult(all, 3)).toContain('All steps are resolved');
  });
});

describe('normalizeWorkflow (what is read back from storage)', () => {
  it('keeps a valid one, fixes unknown statuses to todo, and drops anything else', () => {
    expect(normalizeWorkflow({ id: 'x', title: 'X', steps: [{ text: 'a', status: 'done', note: 'n' }, { text: 'b', status: 'weird' }] })).toEqual({
      id: 'x', title: 'X', steps: [{ text: 'a', status: 'done', note: 'n' }, { text: 'b', status: 'todo', note: '' }],
    });
    for (const bad of [null, 3, 'x', {}, { id: 'Bad Id', steps: [{ text: 'a' }] }, { id: 'x', steps: [] }, { id: 'x', steps: [{ text: '' }] }, { id: 'x', steps: Array.from({ length: 13 }, () => ({ text: 'a' })) }]) {
      expect(normalizeWorkflow(bad)).toBeNull();
    }
  });
});
