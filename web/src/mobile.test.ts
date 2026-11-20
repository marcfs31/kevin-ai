import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const webRoot = join(dirname(fileURLToPath(import.meta.url)), '..');
const css = readFileSync(join(webRoot, 'src/styles.css'), 'utf8');
const html = readFileSync(join(webRoot, 'index.html'), 'utf8');

describe('phone layout', () => {
  it('covers the screen edge and resizes when the keyboard opens', () => {
    expect(html).toContain('viewport-fit=cover');
    expect(html).toContain('interactive-widget=resizes-content');
    expect(html).not.toContain('maximum-scale');
    expect(html).not.toContain('user-scalable=no');
  });

  it('tracks the dynamic viewport and scrolls settings inside the column', () => {
    expect(css).toContain('100dvh');
    expect(css).toMatch(/\.panels\s*\{[^}]*min-block-size:\s*0/);
    expect(css).toMatch(/\.panels\s*\{[^}]*overflow:\s*auto/);
    expect(css).toMatch(/\.log\s*\{[^}]*min-block-size:\s*0/);
    expect(css).toMatch(/\.main\s*\{[^}]*overflow:\s*hidden/);
    expect(css).toMatch(/\.composer\s*\{[^}]*flex-shrink:\s*0/);
  });

  it('pads the header, the composer and the conversation list for the safe area', () => {
    expect(css).toContain('safe-area-inset-top');
    expect(css).toContain('safe-area-inset-bottom');
    expect(css).toContain('safe-area-inset-left');
    expect(css).toContain('safe-area-inset-right');
  });

  it('stacks the model buttons and gives a coarse pointer a 44px target', () => {
    expect(css).toContain('pointer: coarse');
    expect(css).toContain('2.75rem');
    expect(css).toMatch(/\.serve-actions button\s*\{[^}]*inline-size:\s*100%/);
    expect(css).toMatch(/\.serve-actions button\s*\{[^}]*min-block-size:\s*2\.75rem/);
  });

  it('places copy above the code and lets a PDF name wrap', () => {
    expect(css).not.toMatch(/\.code-block \.copy\s*\{[^}]*position:\s*absolute/);
    expect(css).toMatch(/\.pdf-range label\s*\{[^}]*overflow-wrap:\s*anywhere/);
    expect(css).toMatch(/\.pdf-range label\s*\{[^}]*min-inline-size:\s*0/);
  });
});
