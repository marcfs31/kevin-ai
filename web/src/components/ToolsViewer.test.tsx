import { render, screen, within } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { I18nProvider } from '../i18n';
import { ToolsViewer } from './ToolsViewer';

const tools = { names: ['calculator', 'find_tools', 'filesystem__write_file', 'project__git_branch_create', 'project__pr_create'], tokens: 258 };

function show(locale: 'en' | 'es' = 'en') {
  return render(
    <I18nProvider locale={locale}>
      <ToolsViewer tools={tools} />
    </I18nProvider>,
  );
}

describe('ToolsViewer', () => {
  it('lists the tools sent by group, each as a chip, with the total and a tooltip carrying the full name', () => {
    show();
    const viewer = screen.getByLabelText('Tools sent with the last request');
    expect(within(viewer).getByText('5 tools, ≈258 tokens (estimated)')).toBeTruthy();
    expect(within(viewer).getByText('Built in')).toBeTruthy();
    expect(within(viewer).getByText('filesystem')).toBeTruthy();
    expect(within(viewer).getByText('project')).toBeTruthy();
    const chip = within(viewer).getByText('git_branch_create').closest('li');
    expect(chip?.getAttribute('title')).toBe('project__git_branch_create');
    expect(within(viewer).getAllByRole('listitem')).toHaveLength(5);
  });

  it('counts the tools of each group next to its name', () => {
    show();
    const project = screen.getByText('project').closest('h4');
    expect(project?.textContent).toBe('project2');
  });

  it('speaks Spanish', () => {
    show('es');
    expect(screen.getByText('Herramientas enviadas')).toBeTruthy();
    expect(screen.getByText('Integradas')).toBeTruthy();
  });
});
