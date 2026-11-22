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
    if (!pre || !first) throw new Error('the code block or its first Copy code button is missing');
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
    if (!pre || !first || !second) throw new Error('the code block or one of its Copy code buttons is missing');
    expect(first.compareDocumentPosition(pre) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(pre.compareDocumentPosition(second) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  });
});

describe('Markdown maths', () => {
  const show = (text: string) =>
    render(
      <I18nProvider locale="en">
        <Markdown text={text} />
      </I18nProvider>,
    ).container;

  it('draws inline LaTeX as MathML instead of printing the dollar signs and commands', () => {
    const c = show('**Sodium bicarbonate ($\\text{NaHCO}_3$):** $1.0 \\text{ g}$ to $1.2 \\text{ g}$');
    expect(c.querySelectorAll('math')).toHaveLength(3);
    expect(c.querySelector('msub')).toBeTruthy();                 // the 3 under NaHCO
    // The MathML keeps the TeX source in an annotation the browser does not draw; what is drawn has neither.
    const drawn = c.cloneNode(true) as HTMLElement;
    drawn.querySelectorAll('annotation').forEach((n) => n.remove());
    expect((drawn.textContent ?? '').replace(/\s+/g, ' ')).toBe('Sodium bicarbonate (NaHCO3): 1.0 g to 1.2 g');
  });

  it('draws a $$ block as display maths', () => {
    const c = show('Area:\n\n$$\nA = \\pi r^2\n$$');
    expect(c.querySelector('math[display="block"]')).toBeTruthy();
  });

  it('leaves prices as text, and a lone dollar sign', () => {
    const c = show('It costs $5 to $10 each, and $3 more.');
    expect(c.querySelector('math')).toBeNull();
    expect(c.textContent).toBe('It costs $5 to $10 each, and $3 more.');
    expect(show('Pay $20').querySelector('math')).toBeNull();
  });

  it('leaves code alone and shows broken LaTeX without crashing', () => {
    const code = show('Use `$x$` in a shell: `echo $HOME`');
    expect(code.querySelector('math')).toBeNull();
    expect(code.textContent).toContain('$x$');
    const broken = show('Oops $\\frac{1$ here');
    expect(broken.textContent).toContain('here');
  });

  it('needs no inline style, so the strict style-src CSP holds', () => {
    const c = show('$x^2 + \\frac{a}{b}$ and $$\\sum_{i=1}^n i$$');
    expect(c.querySelector('math')).toBeTruthy();
    expect(c.querySelectorAll('[style]')).toHaveLength(0);
  });

  it('does not let LaTeX turn into a link or raw HTML', () => {
    const c = show('$\\href{javascript:alert(1)}{click}$ <b>raw</b>');
    expect(c.querySelector('a')).toBeNull();
    expect(c.querySelector('b')).toBeNull();
  });
});
