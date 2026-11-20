import { describe, expect, it } from 'vitest';
import { en } from './en';
import { CATALOGS, LOCALES, detectLocale, makeI18n } from './index';

const PLURAL = /\.(zero|one|two|few|many|other)$/;

function expectedKeys(locale: string): string[] {
  const categories = new Intl.PluralRules(locale).resolvedOptions().pluralCategories;
  const keys = new Set<string>();
  for (const key of Object.keys(en)) {
    if (PLURAL.test(key)) {
      const base = key.replace(PLURAL, '');
      for (const c of categories) keys.add(`${base}.${c}`);
    } else keys.add(key);
  }
  return [...keys].sort();
}

describe('i18n drift', () => {
  for (const locale of LOCALES) {
    it(`${locale} has exactly the keys it needs, every one non-empty`, () => {
      const catalog = CATALOGS[locale];
      expect(Object.keys(catalog).sort()).toEqual(expectedKeys(locale));
      for (const [key, value] of Object.entries(catalog)) expect(value.trim(), key).not.toBe('');
    });

    it(`${locale} keeps every placeholder of the English text`, () => {
      const catalog = CATALOGS[locale];
      for (const [key, value] of Object.entries(catalog)) {
        const base = key.replace(PLURAL, '');
        const source = (en as Record<string, string>)[key] ?? (en as Record<string, string>)[`${base}.other`] ?? '';
        const names = (s: string) => [...s.matchAll(/\{(\w+)\}/g)].map((m) => m[1]).sort();
        expect(names(value), key).toEqual(names(source));
      }
    });
  }
});

describe('i18n helpers', () => {
  it('picks the browser language, falling back to English', () => {
    expect(detectLocale(['es-ES', 'en'])).toBe('es');
    expect(detectLocale(['fr-FR', 'en-GB'])).toBe('en');
    expect(detectLocale(['de'])).toBe('en');
  });

  it('formats numbers and plurals per locale', () => {
    const es = makeI18n('es');
    expect(es.tp('sidebar.confirmDeleteAll', 1)).toContain('1 conversación');
    expect(es.tp('sidebar.confirmDeleteAll', 3)).toContain('las 3 conversaciones');
    expect(makeI18n('en').t('server.context', { n: 262144 })).toBe('Context: 262,144 tokens');
    expect(es.t('server.context', { n: 262144 })).toBe('Contexto: 262.144 tokens');
  });
});
