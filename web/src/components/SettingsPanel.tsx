import { useId } from 'react';
import { LANGUAGE_NAMES, LOCALES, useI18n } from '../i18n';
import { IMAGE_PRESETS, type ImagePreset } from '../lib/attachments';
import { MAX_STEPS_LIMIT, type LocaleSetting, type Settings } from '../lib/storage';

export function SettingsPanel({ settings, onChange }: { settings: Settings; onChange: (s: Settings) => void }) {
  const { t, num } = useI18n();
  const id = useId();
  const set = <K extends keyof Settings>(k: K, v: Settings[K]) => onChange({ ...settings, [k]: v });
  return (
    <section className="settings" aria-labelledby={`${id}-h`}>
      <h2 id={`${id}-h`}>{t('settings.title')}</h2>
      <div className="fields">
        <div className="field field-wide">
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
        </div>
        <div className="field">
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
        </div>
        <div className="field">
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
        </div>
        <div className="field">
          <label htmlFor={`${id}-img`}>{t('settings.imageSize')}</label>
          <select id={`${id}-img`} value={settings.imagePreset} onChange={(e) => set('imagePreset', Number(e.target.value) as ImagePreset)}>
            {IMAGE_PRESETS.map((p) => (
              <option key={p} value={p}>
                {t('settings.imageSizeOption', { n: p })}
              </option>
            ))}
          </select>
        </div>
        <div className="field field-wide">
          <label className="check">
            <input type="checkbox" checked={settings.tools} aria-describedby={`${id}-tools-hint`} onChange={(e) => set('tools', e.target.checked)} />
            {t('settings.tools')}
          </label>
          <p id={`${id}-tools-hint`} className="hint">
            {t('settings.toolsHint')}
          </p>
        </div>
        <div className="field">
          <label htmlFor={`${id}-steps`}>{t('settings.maxSteps')}</label>
          <input
            id={`${id}-steps`}
            type="number"
            inputMode="numeric"
            min={1}
            max={MAX_STEPS_LIMIT}
            step={1}
            disabled={!settings.tools}
            value={settings.maxSteps}
            aria-describedby={`${id}-steps-hint`}
            onChange={(e) => {
              const n = Number(e.target.value);
              if (Number.isInteger(n) && n >= 1 && n <= MAX_STEPS_LIMIT) set('maxSteps', n);
            }}
          />
          <p id={`${id}-steps-hint`} className="hint">
            {t('settings.maxStepsHint', { max: MAX_STEPS_LIMIT })}
          </p>
        </div>
        <div className="field">
          <label htmlFor={`${id}-lang`}>{t('settings.language')}</label>
          <select id={`${id}-lang`} value={settings.locale} onChange={(e) => set('locale', e.target.value as LocaleSetting)}>
            <option value="auto">{t('settings.languageAuto')}</option>
            {LOCALES.map((l) => (
              <option key={l} value={l} lang={l}>
                {LANGUAGE_NAMES[l]}
              </option>
            ))}
          </select>
        </div>
      </div>
    </section>
  );
}
