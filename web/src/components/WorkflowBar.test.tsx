import { fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { I18nProvider } from '../i18n';
import type { ProjectInfo } from '../lib/api';
import { startWorkflow, type Workflow } from '../lib/workflows';
import { WorkflowBar } from './WorkflowBar';

const demo: ProjectInfo = { id: 'demo', title: 'Demo', defaultBranch: 'main', allowMain: false, hasRemote: false, commands: [] };
const json = (body: unknown) => new Response(JSON.stringify(body), { status: 200, headers: { 'Content-Type': 'application/json' } });
const file = (workflows: unknown) => json({ ok: true, content: `.kevin/workflows.json lines 1-1 of 1\n${JSON.stringify({ workflows })}`, isError: false });
afterEach(() => vi.unstubAllGlobals());

function show(props: Partial<Parameters<typeof WorkflowBar>[0]> = {}) {
  const fns = { onStart: vi.fn(), onSet: vi.fn(), onClear: vi.fn() };
  render(
    <I18nProvider locale="en">
      <WorkflowBar project={demo} workflow={null} disabled={false} {...fns} {...props} />
    </I18nProvider>,
  );
  return fns;
}

describe('WorkflowBar', () => {
  it('reads the project\'s workflows with fs_read and starts the chosen one', async () => {
    const fetchMock = vi.fn(async () => file([{ id: 'fix', title: 'Fix a bug', steps: ['Reproduce'] }, { id: 'ship', title: 'Ship it', steps: ['Test', 'Commit'] }]));
    vi.stubGlobal('fetch', fetchMock);
    const { onStart } = show();
    const select = await screen.findByLabelText('Workflow');
    const [url, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toBe('/tools/servers/project/call');
    expect(JSON.parse(String(init.body))).toMatchObject({ name: 'fs_read', arguments: { project: 'demo', path: '.kevin/workflows.json' } });
    fireEvent.change(select, { target: { value: 'ship' } });
    fireEvent.click(screen.getByRole('button', { name: 'Start' }));
    expect(onStart).toHaveBeenCalledWith({ id: 'ship', title: 'Ship it', steps: ['Test', 'Commit'] });
  });

  it('shows nothing when the project has no workflows file, or no project is chosen', async () => {
    const fetchMock = vi.fn(async () => json({ ok: true, content: 'error: .kevin/workflows.json does not exist', isError: true }));
    vi.stubGlobal('fetch', fetchMock);
    const { container } = render(
      <I18nProvider locale="en">
        <WorkflowBar project={demo} workflow={null} disabled={false} onStart={() => {}} onSet={() => {}} onClear={() => {}} />
      </I18nProvider>,
    );
    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalled());
    expect(container.querySelector('.workflow-bar')).toBeNull();
    const none = render(
      <I18nProvider locale="en">
        <WorkflowBar project={null} workflow={null} disabled={false} onStart={() => {}} onSet={() => {}} onClear={() => {}} />
      </I18nProvider>,
    );
    expect(none.container.querySelector('.workflow-bar')).toBeNull();
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it('says so when the file cannot be read as workflows', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => json({ ok: true, content: '.kevin/workflows.json lines 1-1 of 1\nnot json', isError: false })));
    show();
    expect((await screen.findByRole('status')).textContent).toContain('.kevin/workflows.json could not be read as workflows.');
  });

  it('shows the checklist of a started workflow, lets the user tick a step, and stop it', () => {
    vi.stubGlobal('fetch', vi.fn());
    const w: Workflow = { ...startWorkflow({ id: 'x', title: 'Ship it', steps: ['Test', 'Commit', 'Push'] }) };
    w.steps[0] = { text: 'Test', status: 'done', note: 'all green' };
    w.steps[2] = { text: 'Push', status: 'blocked', note: 'no token' };
    const { onSet, onClear } = show({ workflow: w });
    expect(screen.getByText('Ship it')).toBeTruthy();
    expect(screen.getByRole('status').textContent).toBe('2 of 3 steps resolved');
    expect(screen.getByText('all green')).toBeTruthy();
    expect(screen.getByText('blocked')).toBeTruthy();
    expect((screen.getByLabelText('Test') as HTMLInputElement).checked).toBe(true);
    fireEvent.click(screen.getByLabelText('Commit'));
    expect(onSet).toHaveBeenCalledWith(2, 'done');
    fireEvent.click(screen.getByLabelText('Test'));
    expect(onSet).toHaveBeenCalledWith(1, 'todo');
    fireEvent.click(screen.getByRole('button', { name: 'Stop workflow' }));
    expect(onClear).toHaveBeenCalled();
  });

  it('locks the controls while a reply is being written', () => {
    vi.stubGlobal('fetch', vi.fn());
    show({ workflow: startWorkflow({ id: 'x', title: 'X', steps: ['one'] }), disabled: true });
    expect((screen.getByLabelText('one') as HTMLInputElement).disabled).toBe(true);
    expect((screen.getByRole('button', { name: 'Stop workflow' }) as HTMLButtonElement).disabled).toBe(true);
  });
});
