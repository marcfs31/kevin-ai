import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { I18nProvider } from '../i18n';
import type { ApprovalRequest } from '../lib/toolHost';
import { ApprovalCard } from './ApprovalCard';

const request: ApprovalRequest = { callId: 'c', tool: 'fs__write', title: 'fs · write', allowKey: 'fs/write', server: 'fs', network: false, args: '{}', readOnly: false };

describe('ApprovalCard', () => {
  it('puts focus on the card and not on a button, so a stray Enter or Space answers nothing', () => {
    const onAnswer = vi.fn();
    render(
      <I18nProvider locale="en">
        <ApprovalCard request={request} onAnswer={onAnswer} />
      </I18nProvider>,
    );
    const card = screen.getByRole('alertdialog');
    expect(document.activeElement).toBe(card);
    expect(document.activeElement).not.toBe(screen.getByRole('button', { name: 'Allow once' }));
    fireEvent.keyDown(card, { key: 'Enter' });
    expect(onAnswer).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole('button', { name: 'Deny' }));
    expect(onAnswer).toHaveBeenCalledWith('deny');
  });
});
