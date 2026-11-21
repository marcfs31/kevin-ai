import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { I18nProvider } from '../i18n';
import type { UserMessage } from '../lib/storage';
import { MessageView } from './MessageView';

function user(over: Partial<UserMessage> = {}): UserMessage {
  return {
    id: 'u',
    role: 'user',
    content: 'hi',
    typed: 'hi',
    attachmentNames: [],
    createdAt: 0,
    ...over,
  };
}

describe('MessageView user images', () => {
  it('shows data-URL image previews with an accessible label', () => {
    const content = [
      { type: 'image_url' as const, image_url: { url: 'data:image/png;base64,AA' } },
      { type: 'text' as const, text: 'what is this' },
    ];
    render(
      <I18nProvider locale="en">
        <MessageView m={user({ content, typed: 'what is this', attachmentNames: ['cat.png'] })} />
      </I18nProvider>,
    );
    const list = screen.getByRole('list', { name: 'Attached images' });
    expect(list.querySelectorAll('img.msg-thumb')).toHaveLength(1);
    expect(screen.getByText('what is this')).toBeTruthy();
    expect(screen.getByText('Attached: cat.png')).toBeTruthy();
  });

  it('hides remote image URLs', () => {
    const content = [
      { type: 'image_url' as const, image_url: { url: 'https://example.com/x.png' } },
      { type: 'text' as const, text: 'x' },
    ];
    render(
      <I18nProvider locale="en">
        <MessageView m={user({ content, typed: 'x' })} />
      </I18nProvider>,
    );
    expect(screen.queryByRole('list', { name: 'Attached images' })).toBeNull();
  });
});
