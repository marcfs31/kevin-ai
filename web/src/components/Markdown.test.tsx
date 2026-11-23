import { render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { I18nProvider } from '../i18n';
import { DUAL_COPY_MIN_HEIGHT_PX, Markdown, needsDualCopy, normalizeMathDelimiters } from './Markdown';

function mockPreHeight(height: number) {
  vi.spyOn(Element.prototype, 'getBoundingClientRect').mockImplementation(function (this: Element) {
    const h = this.tagName === 'PRE' ? height : 0;
    return { height: h, width: 100, top: 0, left: 0, bottom: h, right: 100, x: 0, y: 0, toJSON() {} };
  });
}

describe('Markdown headings', () => {
  it('renders every heading of a reply as an h3, so the page never gets a second h1 or skips a level, keeping the level the model wrote as a class', () => {
    const { container } = render(
      <I18nProvider locale="en">
        <Markdown text={'# one\n\n## two\n\n### three\n\n###### six'} />
      </I18nProvider>,
    );
    expect(container.querySelectorAll('h1, h2, h4, h5, h6')).toHaveLength(0);
    const headings = [...container.querySelectorAll('h3')];
    expect(headings.map((h) => [h.textContent, h.className])).toEqual([['one', 'md-h1'], ['two', 'md-h2'], ['three', 'md-h3'], ['six', 'md-h6']]);
  });
});

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

describe('Markdown maths with backslash delimiters', () => {
  const show = (text: string, emphasisAsBold = false) =>
    render(
      <I18nProvider locale="en">
        <Markdown text={text} emphasisAsBold={emphasisAsBold} />
      </I18nProvider>,
    ).container;

  it('draws \\( ... \\) inline and \\[ ... \\] as a block, like the dollar forms', () => {
    const c = show('Mass \\(m = \\rho V\\) and then\n\n\\[\nE = mc^2\n\\]');
    expect(c.querySelectorAll('math')).toHaveLength(2);
    expect(c.querySelector('math[display="block"]')).toBeTruthy();
    const drawn = c.cloneNode(true) as HTMLElement;
    drawn.querySelectorAll('annotation').forEach((n) => n.remove());
    expect(drawn.textContent).not.toContain('\\');
  });

  it('leaves them alone inside code, and in text that has no closing delimiter', () => {
    expect(normalizeMathDelimiters('Use `\\(x\\)` and\n```\n\\[y\\]\n```\n')).toBe('Use `\\(x\\)` and\n```\n\\[y\\]\n```\n');
    expect(normalizeMathDelimiters('a \\( open but never closed')).toBe('a \\( open but never closed');
    expect(normalizeMathDelimiters('plain text')).toBe('plain text');
  });

  it('works in the thinking text too, together with bold', () => {
    const c = show('*Total Weight:* \\(1.0 \\text{ g}\\) to $1.2 \\text{ g}$', true);
    expect(c.querySelector('strong')?.textContent).toBe('Total Weight:');
    expect(c.querySelectorAll('math')).toHaveLength(2);
  });
});
