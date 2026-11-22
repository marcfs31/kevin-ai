import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { I18nProvider } from '../i18n';
import { Composer } from './Composer';
import { FileIcon, ImageIcon } from './Icons';

describe('icons', () => {
  it('are drawn inline, decorative, and in the text colour', () => {
    const { container } = render(
      <>
        <FileIcon />
        <ImageIcon />
      </>,
    );
    const svgs = [...container.querySelectorAll('svg')];
    expect(svgs).toHaveLength(2);
    for (const s of svgs) {
      expect(s.getAttribute('aria-hidden')).toBe('true');
      expect(s.getAttribute('stroke')).toBe('currentColor');
      expect(s.querySelector('image, use, script')).toBeNull();      // nothing to load: the CSP stays as it is
      expect(s.textContent).toBe('');
    }
  });

  it('sit in the two attach buttons without changing their names', () => {
    render(
      <I18nProvider locale="en">
        <Composer busy={false} info={null} imagePreset={1024} assessDraft={() => ({ newTokens: 0, imagesUnknown: false, waitSeconds: 0, problems: [] })} onSend={() => undefined} canRedirect={false} onRedirect={() => undefined} onStop={() => undefined} inputRef={{ current: null }} />
      </I18nProvider>,
    );
    for (const name of ['Attach files', 'Attach image']) {
      const button = screen.getByRole('button', { name });
      expect(button.querySelector('svg.icon')).toBeTruthy();
      expect(button.textContent).toBe(name);
    }
  });
});
