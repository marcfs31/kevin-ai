import { useI18n } from '../i18n';

/** Warning mark for uncensored start targets; colour + title, not colour alone. */
export function UncensoredMark({ className = '' }: { className?: string }) {
  const { t } = useI18n();
  return (
    <span className={`model-warn ${className}`.trim()} title={t('picker.uncensoredWarn')}>
      <svg className="model-warn-icon" viewBox="0 0 16 16" width="14" height="14" aria-hidden="true" focusable="false">
        <path fill="currentColor" d="M8 1.2 15 14.2H1L8 1.2z" />
        <path fill="var(--paper)" d="M7.25 5.5h1.5v4.2H7.25zm0 5.2h1.5V12H7.25z" />
      </svg>
      <span className="sr-only">{t('picker.uncensoredWarn')}</span>
    </span>
  );
}
