import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { I18nProvider } from '../i18n';
import type { AssistantMessage, UserMessage } from '../lib/storage';
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

function assistant(over: Partial<AssistantMessage> = {}): AssistantMessage {
  return {
    id: 'a',
    role: 'assistant',
    content: 'hello',
    reasoning: '',
    status: 'done',
    error: null,
    stats: null,
    createdAt: 1,
    ...over,
  };
}

describe('MessageView delete and regenerate', () => {
  it('calls onDelete from a user or assistant message', () => {
    const onDelete = vi.fn();
    const { rerender } = render(
      <I18nProvider locale="en">
        <MessageView m={user()} onDelete={onDelete} />
      </I18nProvider>,
    );
    fireEvent.click(screen.getByRole('button', { name: 'Delete from here' }));
    expect(onDelete).toHaveBeenCalledTimes(1);
    rerender(
      <I18nProvider locale="en">
        <MessageView m={assistant()} onDelete={onDelete} />
      </I18nProvider>,
    );
    fireEvent.click(screen.getByRole('button', { name: 'Delete from here' }));
    expect(onDelete).toHaveBeenCalledTimes(2);
  });

  it('offers regenerate on a dangling user message', () => {
    const onRegenerate = vi.fn();
    render(
      <I18nProvider locale="en">
        <MessageView m={user()} onRegenerate={onRegenerate} />
      </I18nProvider>,
    );
    fireEvent.click(screen.getByRole('button', { name: 'Regenerate' }));
    expect(onRegenerate).toHaveBeenCalledTimes(1);
  });
});


describe('MessageView reading ETA', () => {
  it('shows minutes and seconds left while reading the prompt', () => {
    render(
      <I18nProvider locale="en">
        <MessageView
          m={assistant({ content: '' })}
          live={{ startedAt: performance.now(), firstAt: null, chunks: 0, promptEstimate: 1440, readEstimate: 1440, usage: null }}
        />
      </I18nProvider>,
    );
    expect(screen.getByText(/Reading the prompt… .+ · about 2 min 0 s left/)).toBeTruthy();
  });

  it('drops the ETA once the estimate has run out', () => {
    render(
      <I18nProvider locale="en">
        <MessageView
          m={assistant({ content: '' })}
          live={{ startedAt: performance.now() - 130_000, firstAt: null, chunks: 0, promptEstimate: 1440, readEstimate: 1440, usage: null }}
        />
      </I18nProvider>,
    );
    expect(screen.getByText(/^Reading the prompt…/)).toBeTruthy();
    expect(screen.queryByText(/left/)).toBeNull();
  });
});

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
