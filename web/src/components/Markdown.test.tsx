import { render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { I18nProvider } from '../i18n';
import { DUAL_COPY_MIN_HEIGHT_PX, Markdown, needsDualCopy } from './Markdown';

function mockPreHeight(height: number) {
  vi.spyOn(Element.prototype, 'getBoundingClientRect').mockImplementation(function (this: Element) {
    const h = this.tagName === 'PRE' ? height : 0;
    return { height: h, width: 100, top: 0, left: 0, bottom: h, right: 100, x: 0, y: 0, toJSON() {} };
  });
}

describe('needsDualCopy', () => {
  it('is false below the threshold', () => {
    expect(needsDualCopy(DUAL_COPY_MIN_HEIGHT_PX - 1)).toBe(false);
  });

  it('is true at and above the threshold', () => {
    expect(needsDualCopy(DUAL_COPY_MIN_HEIGHT_PX)).toBe(true);
    expect(needsDualCopy(DUAL_COPY_MIN_HEIGHT_PX + 40)).toBe(true);
  });
});

describe('Markdown code copy', () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('shows one Copy code button above a short block', () => {
    mockPreHeight(80);
    const { container } = render(
      <I18nProvider locale="en">
        <Markdown text={'```js\nconst x = 1;\n```'} />
      </I18nProvider>,
    );
    const buttons = screen.getAllByRole('button', { name: 'Copy code' });
    expect(buttons).toHaveLength(1);
    const pre = container.querySelector('pre');
    const first = buttons[0];
    expect(pre).toBeTruthy();
    expect(first).toBeTruthy();
    expect(first.compareDocumentPosition(pre) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  });

  it('shows a second Copy code below a tall block, with the first still above', () => {
    mockPreHeight(DUAL_COPY_MIN_HEIGHT_PX);
    const { container } = render(
      <I18nProvider locale="en">
        <Markdown text={'```js\n' + 'line\n'.repeat(40) + '```'} />
      </I18nProvider>,
    );
    const buttons = screen.getAllByRole('button', { name: 'Copy code' });
    expect(buttons).toHaveLength(2);
    const pre = container.querySelector('pre');
    const first = buttons[0];
    const second = buttons[1];
    expect(pre).toBeTruthy();
    expect(first).toBeTruthy();
    expect(second).toBeTruthy();
    expect(first.compareDocumentPosition(pre) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(pre.compareDocumentPosition(second) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  });
});
