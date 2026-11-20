// A tiny i18n layer: one hand-written message file per locale, Intl for numbers,
// dates and plurals. No library.
import { createContext, useContext, useMemo, type ReactNode } from 'react';
import { en, type MessageKey } from './en';
import { es } from './es';

export type Locale = 'en' | 'es';
export const LOCALES: readonly Locale[] = ['en', 'es'];
export const CATALOGS: Record<Locale, Record<string, string>> = { en, es };
export const LANGUAGE_NAMES: Record<Locale, string> = { en: 'English', es: 'Español' };

type PluralBase = { [K in MessageKey]: K extends `${infer B}.other` ? B : never }[MessageKey];
export type Vars = Record<string, string | number>;

export function detectLocale(languages: readonly string[]): Locale {
  for (const lang of languages) {
    const base = lang.toLowerCase().split('-')[0];
    if (base === 'es' || base === 'en') return base;
  }
  return 'en';
}

export function interpolate(template: string, vars: Vars | undefined, locale: Locale): string {
  return template.replace(/\{(\w+)\}/g, (whole, name: string) => {
    const v = vars?.[name];
    if (v === undefined) return whole;
    return typeof v === 'number' ? new Intl.NumberFormat(locale).format(v) : v;
  });
}

export interface I18n {
  locale: Locale;
  t: (key: MessageKey, vars?: Vars) => string;
  tp: (base: PluralBase, n: number, vars?: Vars) => string;
  num: (n: number, digits?: number) => string;
  date: (ms: number) => string;
  /** A duration in seconds, or whole minutes from two minutes on. */
  duration: (seconds: number) => string;
  megabytes: (bytes: number) => string;
}

export function makeI18n(locale: Locale): I18n {
  const catalog = CATALOGS[locale];
  const rules = new Intl.PluralRules(locale);
  const dateFmt = new Intl.DateTimeFormat(locale, { dateStyle: 'medium', timeStyle: 'short' });
  const t = (key: string, vars?: Vars) => interpolate(catalog[key] ?? en[key as MessageKey] ?? key, vars, locale);
  return {
    locale,
    t,
    tp: (base, n, vars) => {
      const key = `${base}.${rules.select(n)}`;
      return t(key in catalog ? key : `${base}.other`, { n, count: n, ...vars });
    },
    num: (n, digits = 0) => new Intl.NumberFormat(locale, { maximumFractionDigits: digits, minimumFractionDigits: digits }).format(n),
    date: (ms) => dateFmt.format(new Date(ms)),
    duration: (seconds) => {
      const minutes = seconds >= 120;
      return new Intl.NumberFormat(locale, { style: 'unit', unit: minutes ? 'minute' : 'second', unitDisplay: 'long', maximumFractionDigits: 0 })
        .format(Math.max(1, Math.round(minutes ? seconds / 60 : seconds)));
    },
    megabytes: (bytes) =>
      new Intl.NumberFormat(locale, { style: 'unit', unit: 'megabyte', maximumFractionDigits: 1 }).format(bytes / (1024 * 1024)),
  };
}

const I18nContext = createContext<I18n>(makeI18n('en'));

export function I18nProvider({ locale, children }: { locale: Locale; children: ReactNode }) {
  const value = useMemo(() => makeI18n(locale), [locale]);
  return <I18nContext.Provider value={value}>{children}</I18nContext.Provider>;
}

export function useI18n(): I18n {
  return useContext(I18nContext);
}
