import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { I18nProvider } from '../i18n';
import type { HostedProject } from '../lib/api';
import { ProjectBar } from './ProjectBar';

const demo: HostedProject = { id: 'demo', title: 'Demo', defaultBranch: 'main', allowMain: false, hasRemote: false, commands: [], host: 'server' };
const mac: HostedProject = { ...demo, id: 'notes', title: 'Notes', host: 'local' };
const json = (body: unknown) => new Response(JSON.stringify(body), { status: 200, headers: { 'Content-Type': 'application/json' } });
afterEach(() => vi.unstubAllGlobals());

function show(props: Partial<Parameters<typeof ProjectBar>[0]> = {}) {
  const onChange = vi.fn();
  render(
    <I18nProvider locale="en">
      <ProjectBar projects={[demo]} projectId={null} projectHost="server" disabled={false} refreshKey={0} onChange={onChange} {...props} />
    </I18nProvider>,
  );
  return onChange;
}

describe('ProjectBar', () => {
  it('shows nothing when the service lists no projects', () => {
    const { container } = render(
      <I18nProvider locale="en">
        <ProjectBar projects={[]} projectId={null} projectHost="server" disabled={false} refreshKey={0} onChange={() => {}} />
      </I18nProvider>,
    );
    expect(container.querySelector('.project-bar')).toBeNull();
  });

  it('picks a project, or none, without asking git anything while none is picked', () => {
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock);
    const onChange = show();
    fireEvent.change(screen.getByLabelText('Project'), { target: { value: 'server:demo' } });
    expect(onChange).toHaveBeenCalledWith('demo', 'server');
    expect(fetchMock).not.toHaveBeenCalled();
    expect(screen.queryByRole('status')).toBeNull();
  });

  it('reads git_status of the chosen project, with the project named, and shows the branch and the changes', async () => {
    const fetchMock = vi.fn(async () => json({ ok: true, content: '## kevin/work\n M a.txt', isError: false }));
    vi.stubGlobal('fetch', fetchMock);
    show({ projectId: 'demo' });
    expect(await screen.findByText('kevin/work · 1 changed')).toBeTruthy();
    const [url, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toBe('/tools/servers/project/call');
    expect(JSON.parse(String(init.body))).toMatchObject({ name: 'git_status', arguments: { project: 'demo' } });
  });

  it('says the git state is unknown when it cannot be read, and for a project the service no longer lists', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => json({ ok: true, content: 'error: not a git repository', isError: true })));
    show({ projectId: 'demo' });
    await waitFor(() => expect(screen.getByText('git state unknown')).toBeTruthy());
  });

  it('lists the projects of both hosts, names the Mac ones, and says which host was picked', () => {
    vi.stubGlobal('fetch', vi.fn());
    const onChange = show({ projects: [demo, mac] });
    expect(screen.getByRole('option', { name: 'Notes (on this Mac)' })).toBeTruthy();
    expect(screen.getByRole('option', { name: 'Demo' })).toBeTruthy();
    fireEvent.change(screen.getByLabelText('Project'), { target: { value: 'local:notes' } });
    expect(onChange).toHaveBeenCalledWith('notes', 'local');
    fireEvent.change(screen.getByLabelText('Project'), { target: { value: '' } });
    expect(onChange).toHaveBeenLastCalledWith(null, 'server');
  });

  it('reads the git state of a Mac project from the Mac\'s tool service, and a same-named server project is not it', async () => {
    const fetchMock = vi.fn(async () => json({ ok: true, content: '## kevin/mac\n M a.txt\n M b.txt', isError: false }));
    vi.stubGlobal('fetch', fetchMock);
    show({ projects: [demo, { ...mac, id: 'demo' }], projectId: 'demo', projectHost: 'local' });
    expect(await screen.findByText('kevin/mac · 2 changed')).toBeTruthy();
    expect((fetchMock.mock.calls[0] as unknown as [string])[0]).toBe('/tools-local/servers/project/call');
    expect((screen.getByLabelText('Project') as HTMLSelectElement).value).toBe('local:demo');
  });

  it('treats a project that is gone as none', () => {
    vi.stubGlobal('fetch', vi.fn());
    show({ projectId: 'removed' });
    expect((screen.getByLabelText('Project') as HTMLSelectElement).value).toBe('');
    expect(screen.queryByRole('status')).toBeNull();
  });
});
