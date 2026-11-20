import { useId } from 'react';
import { LANGUAGE_NAMES, LOCALES, useI18n } from '../i18n';
import { IMAGE_PRESETS, type ImagePreset } from '../lib/attachments';
import type { LocaleSetting, Settings } from '../lib/storage';

export function SettingsPanel({ settings, onChange }: { settings: Settings; onChange: (s: Settings) => void }) {
  const { t, num } = useI18n();
  const id = useId();
  const set = <K extends keyof Settings>(k: K, v: Settings[K]) => onChange({ ...settings, [k]: v });
  return (
    <section className="settings" aria-labelledby={`${id}-h`}>
      <h2 id={`${id}-h`}>{t('settings.title')}</h2>
      <label htmlFor={`${id}-sys`}>{t('settings.systemPrompt')}</label>
      <textarea
        id={`${id}-sys`}
        rows={4}
        value={settings.systemPrompt}
        aria-describedby={`${id}-sys-hint`}
        onChange={(e) => set('systemPrompt', e.target.value)}
      />
      <p id={`${id}-sys-hint`} className="hint">
        {t('settings.systemPromptHint')}
      </p>
      <label htmlFor={`${id}-temp`}>
        {t('settings.temperature')}: <output htmlFor={`${id}-temp`}>{num(settings.temperature, 1)}</output>
      </label>
      <input
        id={`${id}-temp`}
        type="range"
        min={0}
        max={2}
        step={0.1}
        value={settings.temperature}
        onChange={(e) => set('temperature', Number(e.target.value))}
      />
      <label htmlFor={`${id}-max`}>{t('settings.maxTokens')}</label>
      <input
        id={`${id}-max`}
        type="number"
        inputMode="numeric"
        min={1}
        step={1}
        value={settings.maxTokens ?? ''}
        aria-describedby={`${id}-max-hint`}
        onChange={(e) => {
          const n = Number(e.target.value);
          set('maxTokens', e.target.value && Number.isInteger(n) && n > 0 ? n : null);
        }}
      />
      <p id={`${id}-max-hint`} className="hint">
        {t('settings.maxTokensHint')}
      </p>
      <label htmlFor={`${id}-img`}>{t('settings.imageSize')}</label>
      <select id={`${id}-img`} value={settings.imagePreset} onChange={(e) => set('imagePreset', Number(e.target.value) as ImagePreset)}>
        {IMAGE_PRESETS.map((p) => (
          <option key={p} value={p}>
            {t('settings.imageSizeOption', { n: p })}
          </option>
        ))}
      </select>
      <label htmlFor={`${id}-lang`}>{t('settings.language')}</label>
      <select id={`${id}-lang`} value={settings.locale} onChange={(e) => set('locale', e.target.value as LocaleSetting)}>
        <option value="auto">{t('settings.languageAuto')}</option>
        {LOCALES.map((l) => (
          <option key={l} value={l} lang={l}>
            {LANGUAGE_NAMES[l]}
          </option>
        ))}
      </select>
    </section>
  );
}
