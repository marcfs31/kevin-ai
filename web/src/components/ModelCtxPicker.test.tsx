import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { I18nProvider } from '../i18n';
import { ModelCtxPicker } from './ModelCtxPicker';

function renderPicker() {
  return render(
    <I18nProvider locale="en">
      <ModelCtxPicker modelName="qwen3.5-4b" contextLength={8192} busy={false} onStarted={() => {}} />
    </I18nProvider>,
  );
}

describe('ModelCtxPicker', () => {
  it('is the way to start a model when the server cannot be reached', () => {
    render(
      <I18nProvider locale="en">
        <ModelCtxPicker modelName={null} contextLength={null} busy={false} onStarted={() => {}} />
      </I18nProvider>,
    );
    fireEvent.click(screen.getByRole('button', { name: 'Start a model' }));
    expect(screen.getByRole('dialog', { name: 'Model and context' })).toBeTruthy();
    expect(screen.getByRole('button', { name: 'Start' })).toBeTruthy();
  });

  it('opens from the model header button', () => {
    renderPicker();
    fireEvent.click(screen.getByRole('button', { name: /Choose model and context window/ }));
    expect(screen.getByRole('dialog', { name: 'Model and context' })).toBeTruthy();
  });

  it('groups stock and uncensored models under section headings', () => {
    renderPicker();
    fireEvent.click(screen.getByRole('button', { name: /Choose model and context window/ }));
    expect(screen.getByText('Stock models')).toBeTruthy();
    expect(screen.getByText('Uncensored')).toBeTruthy();
    // One role=group for the whole model list (single selection); visual sections are headings only.
    expect(screen.getByRole('group', { name: 'Model' })).toBeTruthy();
    const groups = document.querySelectorAll('.model-picker-groups .model-picker-group');
    expect(groups).toHaveLength(2);
    expect(groups[0]?.querySelectorAll('button')).toHaveLength(8);
    expect(groups[1]?.querySelectorAll('button')).toHaveLength(7);
    expect(groups[0]?.querySelector('.model-warn')).toBeNull();
    expect(groups[1]?.querySelector('.model-warn')).toBeTruthy();
  });

  it('marks Uncensored models with a warning', () => {
    renderPicker();
    fireEvent.click(screen.getByRole('button', { name: /Choose model and context window/ }));
    const warns = screen.getAllByTitle('Uncensored: fewer safety filters than the stock model');
    expect(warns).toHaveLength(7);
    expect(screen.getByRole('button', { name: /Qwen3.5-4B Uncensored/ }).querySelector('.model-warn')).toBeTruthy();
    expect(screen.getByRole('button', { name: /Gemma 4 E4B Uncensored/ }).querySelector('.model-warn')).toBeTruthy();
    expect(screen.getByRole('button', { name: 'Gemma 4 E4B' }).querySelector('.model-warn')).toBeNull();
  });

  it('does not POST when confirm is cancelled', async () => {
    const fetchMock = vi.fn(async () => new Response(JSON.stringify({ ok: true }), { status: 200 }));
    vi.stubGlobal('fetch', fetchMock);
    renderPicker();
    fireEvent.click(screen.getByRole('button', { name: /Choose model and context window/ }));
    fireEvent.click(screen.getByRole('button', { name: 'Gemma 4 E4B' }));
    fireEvent.click(screen.getByRole('radio', { name: '8k' }));
    vi.spyOn(window, 'confirm').mockReturnValue(false);
    fireEvent.click(screen.getByRole('button', { name: 'Start' }));
    await waitFor(() => expect(fetchMock).not.toHaveBeenCalled());
    vi.unstubAllGlobals();
  });

  it('POSTs gemma with ctx 8192 when confirmed', async () => {
    const fetchMock = vi.fn(async () =>
      new Response(JSON.stringify({ ok: true, model: 'gemma', health: 'ok' }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }),
    );
    vi.stubGlobal('fetch', fetchMock);
    renderPicker();
    fireEvent.click(screen.getByRole('button', { name: /Choose model and context window/ }));
    fireEvent.click(screen.getByRole('button', { name: 'Gemma 4 E4B' }));
    fireEvent.click(screen.getByRole('radio', { name: '8k' }));
    vi.spyOn(window, 'confirm').mockReturnValue(true);
    fireEvent.click(screen.getByRole('button', { name: 'Start' }));
    await waitFor(() => expect(fetchMock).toHaveBeenCalled());
    const calls = fetchMock.mock.calls as unknown as [RequestInfo | URL, RequestInit | undefined][];
    const call = calls.find(([url]) => String(url).endsWith('/control/serve'));
    expect(JSON.parse(String(call?.[1]?.body))).toMatchObject({
      model: 'gemma',
      ctx: 8192,
      maxTokens: null,
      seed: null,
    });
    vi.unstubAllGlobals();
  });
});