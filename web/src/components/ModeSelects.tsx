import { useId } from 'react';
import { useI18n } from '../i18n';
import { EFFORTS, SPEEDS, type Effort, type Speed } from '../lib/modes';

/** Effort and speed for this chat, beside the model picker. Choosing a speed also sets the effort it
 * starts from; the effort can then be changed on its own. */
export function ModeSelects({ effort, speed, disabled, onEffort, onSpeed }: {
  effort: Effort;
  speed: Speed;
  disabled: boolean;
  onEffort: (e: Effort) => void;
  onSpeed: (s: Speed) => void;
}) {
  const { t } = useI18n();
  const id = useId();
  return (
    <>
      <span className="chat-model">
        <label htmlFor={`${id}-effort`}>{t('mode.effort')}</label>
        <select id={`${id}-effort`} value={effort} disabled={disabled} aria-describedby={`${id}-hint`} onChange={(e) => onEffort(e.target.value as Effort)}>
          {EFFORTS.map((e) => (
            <option key={e} value={e}>
              {t(`mode.effort.${e}`)}
            </option>
          ))}
        </select>
      </span>
      <span className="chat-model">
        <label htmlFor={`${id}-speed`}>{t('mode.speed')}</label>
        <select id={`${id}-speed`} value={speed} disabled={disabled} aria-describedby={`${id}-hint`} onChange={(e) => onSpeed(e.target.value as Speed)}>
          {SPEEDS.map((s) => (
            <option key={s} value={s}>
              {t(`mode.speed.${s}`)}
            </option>
          ))}
        </select>
      </span>
      <span id={`${id}-hint`} className="sr-only">
        {t('mode.hint')}
      </span>
    </>
  );
}
