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
});
