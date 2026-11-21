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