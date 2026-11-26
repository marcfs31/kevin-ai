import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

/** Regression: muted/warn marks on primary Start buttons vanished on the dark accent fill. */
describe('vision mark contrast on primary buttons', () => {
  it('forces accent-text colour for eye and warn marks inside button.primary', () => {
    const css = readFileSync(join(dirname(fileURLToPath(import.meta.url)), 'styles.css'), 'utf8');
    expect(css).toMatch(
      /button\.primary \.model-vision,\s*\nbutton\.primary \.model-warn \{\s*\n\s*color: var\(--accent-text\);/,
    );
    expect(css).toMatch(/button\.primary \.model-warn-icon path:last-of-type \{\s*\n\s*fill: var\(--accent\);/);
  });
});
