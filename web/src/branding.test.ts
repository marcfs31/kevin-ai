import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { en } from './i18n/en';
import { es } from './i18n/es';

const root = join(__dirname, '..');
const read = (p: string) => readFileSync(join(root, p), 'utf8');

describe('Kevin AI branding', () => {
  it('names the page and the app Kevin AI in both languages, with the logo as its tab icon', () => {
    const html = read('index.html');
    expect(html).toContain('<title>Kevin AI</title>');
    expect(html).toContain('<link rel="icon" type="image/svg+xml" href="/logo.svg" />');
    expect([en['app.title'], es['app.title']]).toEqual(['Kevin AI', 'Kevin AI']);
  });

  it('describes itself for search, and ships no robots.txt that would mark the page as blocked from indexing', () => {
    expect(read('index.html')).toMatch(/<meta name="description" content="[^"]{40,}" \/>/);
    expect(existsSync(join(root, 'public', 'robots.txt'))).toBe(false);
    expect(read('index.html')).not.toMatch(/name="robots"/i);
  });

  it('reserves the header height the loaded model row needs, so the page does not jump when the model answers', () => {
    const css = read('src/styles.css');
    expect(css).toMatch(/\.top \{[^}]*min-block-size: 4\.75rem;/);
    // The settings dialog is set apart by its border and the backdrop, not by a border plus a wide shadow too.
    expect(css.match(/\.settings-modal \{[^}]*\}/)?.[0]).not.toContain('box-shadow');
  });

  it('keeps the phone header at the height the loaded model row needs, so mobile does not shift either', () => {
    expect(read('src/styles.css')).toMatch(/@media \(max-width: 30rem\) \{\s*\.server \{ min-block-size: 6\.8rem; \}/);
  });

  it('ships the Orbit logo as a self-contained SVG: a disc, a ring, a K and a dot, with no script, link or remote reference', () => {
    const svg = read('public/logo.svg');
    expect(svg).toContain('viewBox="0 0 96 96"');
    expect(svg).toContain('<ellipse');                    // the orbit
    expect(svg).toContain('M38 28V68');                   // the K's stem
    expect((svg.match(/<circle/g) ?? []).length).toBe(2); // the disc and the dot
    expect(svg).not.toMatch(/<script|<a[ >]|href=|xlink|https?:\/\/(?!www\.w3\.org)/i);   // any case: <SCRIPT> counts too
    expect('<SCRIPT>').toMatch(/<script/i);
  });
});
