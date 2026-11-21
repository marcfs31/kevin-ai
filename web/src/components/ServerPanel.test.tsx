import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { I18nProvider } from '../i18n';
import { ServerPanel } from './ServerPanel';

const tail = `lm-serve exited before becoming healthy:\n${'token dump\n'.repeat(30)}`;

describe('ServerPanel', () => {
  it('announces a short failure and keeps the log tail out of the alert', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => new Response(JSON.stringify({ ok: false, error: tail }), { status: 502, headers: { 'Content-Type': 'application/json' } })),
    );
    render(
      <I18nProvider locale="en">
        <ServerPanel busy={false} onStarted={() => {}} />
      </I18nProvider>,
    );
    vi.spyOn(window, 'confirm').mockReturnValue(true);
    fireEvent.click(screen.getByRole('button', { name: 'Start Qwen3.5-4B' }));
    const alert = await screen.findByRole('alert');
    expect(alert.textContent).toBe('Could not start the model on the server.');
    expect(alert.textContent).not.toContain('token dump');
    expect(screen.getByText(/token dump/)).toBeTruthy();
  });

  it('offers the uncensored Qwen and marks only the matching button as running', () => {
    render(
      <I18nProvider locale="en">
        <ServerPanel busy={false} runningModel="qwen3.5-4b-uncensored" onStarted={() => {}} />
      </I18nProvider>,
    );
    expect(screen.getByRole('button', { name: 'Start Qwen3.5-4B Uncensored' }).getAttribute('aria-pressed')).toBe('true');
    expect(screen.getByRole('button', { name: 'Start Qwen3.5-4B' }).getAttribute('aria-pressed')).toBe('false');
    expect(screen.getByRole('button', { name: 'Start Gemma 4 E4B' }).getAttribute('aria-pressed')).toBe('false');
  });

  it('keeps plain Qwen pressed for the plain model name', () => {
    render(
      <I18nProvider locale="en">
        <ServerPanel busy={false} runningModel="qwen3.5-4b" onStarted={() => {}} />
      </I18nProvider>,
    );
    expect(screen.getByRole('button', { name: 'Start Qwen3.5-4B' }).getAttribute('aria-pressed')).toBe('true');
    expect(screen.getByRole('button', { name: 'Start Qwen3.5-4B Uncensored' }).getAttribute('aria-pressed')).toBe('false');
  });

  it('sends --id qwen-uncensored when its button is confirmed', async () => {
    const fetchMock = vi.fn(async () => new Response(JSON.stringify({ ok: true }), { status: 200, headers: { 'Content-Type': 'application/json' } }));
    vi.stubGlobal('fetch', fetchMock);
    render(
      <I18nProvider locale="en">
        <ServerPanel busy={false} onStarted={() => {}} />
      </I18nProvider>,
    );
    const confirm = vi.spyOn(window, 'confirm').mockReturnValue(true);
    fireEvent.click(screen.getByRole('button', { name: 'Start Qwen3.5-4B Uncensored' }));
    expect(confirm).toHaveBeenCalledWith('Stop the current model and start Qwen3.5-4B Uncensored?');
    await screen.findByRole('status');
    const body = JSON.parse(String((fetchMock.mock.calls[0] as unknown as [string, RequestInit])[1].body));
    expect(body.model).toBe('qwen-uncensored');
  });

  it('sends --id whiterabbitneo and marks it running by its llama-server alias', async () => {
    const fetchMock = vi.fn(async () => new Response(JSON.stringify({ ok: true }), { status: 200, headers: { 'Content-Type': 'application/json' } }));
    vi.stubGlobal('fetch', fetchMock);
    render(
      <I18nProvider locale="en">
        <ServerPanel busy={false} runningModel="whiterabbitneo-2.5-7b" onStarted={() => {}} />
      </I18nProvider>,
    );
    expect(screen.getByRole('button', { name: 'Start WhiteRabbitNeo 2.5 7B' }).getAttribute('aria-pressed')).toBe('true');
    expect(screen.getByRole('button', { name: 'Start Qwen3.5-4B' }).getAttribute('aria-pressed')).toBe('false');
    vi.spyOn(window, 'confirm').mockReturnValue(true);
    fireEvent.click(screen.getByRole('button', { name: 'Start WhiteRabbitNeo 2.5 7B' }));
    await screen.findByRole('status');
    const body = JSON.parse(String((fetchMock.mock.calls[0] as unknown as [string, RequestInit])[1].body));
    expect(body.model).toBe('whiterabbitneo');
  });
});
