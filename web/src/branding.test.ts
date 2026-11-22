import { readFileSync } from 'node:fs';
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
