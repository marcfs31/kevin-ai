import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { I18nProvider } from '../i18n';
import { SettingsModal } from './SettingsModal';

describe('SettingsModal', () => {
  it('shows a dialog over a backdrop and closes from Close or the backdrop', () => {
    const onClose = vi.fn();
    render(
      <I18nProvider locale="en">
        <SettingsModal open onClose={onClose}>
          <p>panel body</p>
        </SettingsModal>
      </I18nProvider>,
    );
    expect(screen.getByRole('dialog', { name: 'Settings' })).toBeTruthy();
    expect(screen.getByText('panel body')).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: 'Close' }));
    expect(onClose).toHaveBeenCalledTimes(1);

    onClose.mockClear();
    fireEvent.click(screen.getByRole('button', { name: 'Close settings' }));
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it('closes on Escape and hides when open is false', () => {
    const onClose = vi.fn();
    const { rerender } = render(
      <I18nProvider locale="en">
        <SettingsModal open onClose={onClose}>
          <p>inside</p>
        </SettingsModal>
      </I18nProvider>,
    );
    fireEvent.keyDown(window, { key: 'Escape' });
    expect(onClose).toHaveBeenCalledTimes(1);
    rerender(
      <I18nProvider locale="en">
        <SettingsModal open={false} onClose={onClose}>
          <p>inside</p>
        </SettingsModal>
      </I18nProvider>,
    );
    expect(screen.queryByRole('dialog')).toBeNull();
  });
});
