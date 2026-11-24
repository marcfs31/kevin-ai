import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { I18nProvider, makeI18n } from '../i18n';
import type { AssistantMessage, UserMessage } from '../lib/storage';
import { MessageView, errorText } from './MessageView';

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
  it('offers Edit on a user message only when the page gives it, and calls it', () => {
    const onEdit = vi.fn();
    const { rerender } = render(
      <I18nProvider locale="en">
        <MessageView m={user()} onDelete={() => {}} />
      </I18nProvider>,
    );
    expect(screen.queryByRole('button', { name: 'Edit' })).toBeNull();
    rerender(
      <I18nProvider locale="en">
        <MessageView m={user()} onDelete={() => {}} onEdit={onEdit} />
      </I18nProvider>,
    );
    fireEvent.click(screen.getByRole('button', { name: 'Edit' }));
    expect(onEdit).toHaveBeenCalledTimes(1);
    // The assistant's reply has no Edit, even if one were passed.
    rerender(
      <I18nProvider locale="en">
        <MessageView m={assistant()} onDelete={() => {}} onEdit={onEdit} />
      </I18nProvider>,
    );
    expect(screen.queryByRole('button', { name: 'Edit' })).toBeNull();
  });

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


describe('MessageView headings', () => {
  it('labels each message with an h2, so the h3 headings inside a reply follow a level and none is skipped', () => {
    const { container, rerender } = render(
      <I18nProvider locale="en">
        <MessageView m={user()} />
      </I18nProvider>,
    );
    expect(container.querySelector('h2.role')?.textContent).toBe('You');
    rerender(
      <I18nProvider locale="en">
        <MessageView m={assistant({ content: '### Steps\n\nDo it.' })} />
      </I18nProvider>,
    );
    const levels = [...container.querySelectorAll('h1, h2, h3, h4, h5, h6')].map((h) => h.tagName);
    expect(levels).toEqual(['H2', 'H3']);
  });
});

describe('MessageView while a tool call waits for the user\'s answer', () => {
  const withCall = assistant({ reasoning: 'I should write the file.', toolCalls: [{ id: 'c1', name: 'fs__write_file', arguments: '{"path":"/x"}' }] });

  it('shows its thinking and its tool calls open, so the user sees what is being asked', () => {
    const { container } = render(
      <I18nProvider locale="en">
        <MessageView m={withCall} usingTools />
      </I18nProvider>,
    );
    expect((container.querySelector('details.think') as HTMLDetailsElement).open).toBe(true);
    expect((container.querySelector('details.trace') as HTMLDetailsElement).open).toBe(true);
    expect(screen.getByText('{"path":"/x"}')).toBeTruthy();
  });

  it('leaves them folded otherwise', () => {
    const { container } = render(
      <I18nProvider locale="en">
        <MessageView m={withCall} />
      </I18nProvider>,
    );
    expect((container.querySelector('details.think') as HTMLDetailsElement).open).toBe(false);
    expect((container.querySelector('details.trace') as HTMLDetailsElement).open).toBe(false);
  });
});

describe('MessageView reading ETA', () => {
  it('shows minutes and seconds left while reading the prompt', () => {
    render(
      <I18nProvider locale="en">
        <MessageView
          m={assistant({ content: '' })}
          live={{ startedAt: performance.now(), firstAt: null, chunks: 0, promptEstimate: 1440, readEstimate: 1440, tools: null, usage: null }}
        />
      </I18nProvider>,
    );
    expect(screen.getByText(/Reading the prompt… .+ · about 2 min 0 s left/)).toBeTruthy();
  });

  it('says the model has not started thinking, and what the prompt is made of, with the tools counted apart', () => {
    render(
      <I18nProvider locale="en">
        <MessageView
          m={assistant({ content: '' })}
          live={{ startedAt: performance.now(), firstAt: null, chunks: 0, promptEstimate: 1738, readEstimate: 1738, tools: { names: ['calculator'], tokens: 258 }, usage: null }}
        />
      </I18nProvider>,
    );
    expect(screen.getByText(/has not started thinking: it reads the whole prompt first, ≈1,738 tokens \(≈258 of tool definitions and ≈1,480 of instructions and conversation\)/)).toBeTruthy();
    expect(screen.queryByText(/already holds/)).toBeNull();
  });

  it('leaves the tools out when there are none, and says what the server already holds when it keeps part of the prompt', () => {
    render(
      <I18nProvider locale="en">
        <MessageView
          m={assistant({ content: '' })}
          live={{ startedAt: performance.now(), firstAt: null, chunks: 0, promptEstimate: 1000, readEstimate: 300, tools: null, usage: null }}
        />
      </I18nProvider>,
    );
    expect(screen.getByText(/reads the whole prompt first, ≈1,000 tokens of instructions and conversation/)).toBeTruthy();
    expect(screen.queryByText(/of tool definitions/)).toBeNull();
    expect(screen.getByText(/The server already holds ≈700 of them from earlier, so it reads ≈300\./)).toBeTruthy();
  });

  it('shows nothing about reading once the first token has arrived', () => {
    render(
      <I18nProvider locale="en">
        <MessageView
          m={assistant({ content: 'Hi' })}
          live={{ startedAt: performance.now() - 5000, firstAt: performance.now() - 1000, chunks: 3, promptEstimate: 1000, readEstimate: 1000, tools: null, usage: null }}
        />
      </I18nProvider>,
    );
    expect(screen.queryByText(/has not started thinking/)).toBeNull();
  });

  it('drops the ETA once the estimate has run out', () => {
    render(
      <I18nProvider locale="en">
        <MessageView
          m={assistant({ content: '' })}
          live={{ startedAt: performance.now() - 130_000, firstAt: null, chunks: 0, promptEstimate: 1440, readEstimate: 1440, tools: null, usage: null }}
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

describe('thinking text', () => {
  it('is drawn as Markdown like the answer: bold and italic marks become formatting, not asterisks', () => {
    const m = assistant({ content: 'Done.', reasoning: '**Total weight:** 12 g\n\n* first\n* second\n\n*note*' });
    const { container } = render(
      <I18nProvider locale="en">
        <MessageView m={m} />
      </I18nProvider>,
    );
    const body = container.querySelector('.think-body');
    expect([...(body?.querySelectorAll('strong') ?? [])].map((n) => n.textContent)).toEqual(['Total weight:', 'note']);
    expect(body?.querySelector('em')).toBeNull();                 // single asterisks mean bold in thinking
    expect(body?.querySelectorAll('li')).toHaveLength(2);
    expect(body?.textContent).not.toContain('*');
  });

  it('keeps the standard meaning in the answer: single asterisks are italics there', () => {
    const { container } = render(
      <I18nProvider locale="en">
        <MessageView m={assistant({ content: '*Total weight:* **12 g**', reasoning: '' })} />
      </I18nProvider>,
    );
    expect(container.querySelector('.md em')?.textContent).toBe('Total weight:');
    expect(container.querySelector('.md strong')?.textContent).toBe('12 g');
  });

  it('shows the screenshot line in thinking as bold with the text after it', () => {
    const { container } = render(
      <I18nProvider locale="en">
        <MessageView m={assistant({ content: 'x', reasoning: '*Total Weight:* 1.0 g to 1.2 g' })} />
      </I18nProvider>,
    );
    expect(container.querySelector('.think-body strong')?.textContent).toBe('Total Weight:');
    expect(container.querySelector('.think-body')?.textContent).toBe('Total Weight: 1.0 g to 1.2 g');
  });

  it('shows raw HTML in thinking as text, as in the answer', () => {
    const m = assistant({ content: 'x', reasoning: 'a <b>raw</b> tag' });
    const { container } = render(
      <I18nProvider locale="en">
        <MessageView m={m} />
      </I18nProvider>,
    );
    expect(container.querySelector('.think-body b')).toBeNull();
    expect(container.querySelector('.think-body')?.textContent).toContain('<b>raw</b>');
  });
});

describe('errorText', () => {
  const t = makeI18n('en').t;

  it('names a bare 502 as unreachable upstream, not no details', () => {
    expect(errorText(t, { status: 502, code: null, serverMessage: '' })).toBe(
      'The proxy cannot reach lm-serve. Is it running?',
    );
  });

  it('still shows a 502 body when the server sent one', () => {
    expect(errorText(t, { status: 502, code: null, serverMessage: 'bad gateway body' })).toBe(
      'The server answered 502: bad gateway body',
    );
  });
});
