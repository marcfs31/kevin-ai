import { useI18n } from '../i18n';

/** Small eye mark for start targets that can read images; colour + title, not colour alone. */
export function VisionMark({ className = '' }: { className?: string }) {
  const { t } = useI18n();
  return (
    <span className={`model-vision ${className}`.trim()} title={t('picker.visionMark')}>
      <svg className="model-vision-icon" viewBox="0 0 16 16" width="14" height="14" aria-hidden="true" focusable="false">
        <path
          fill="none"
          stroke="currentColor"
          strokeWidth="1.35"
          strokeLinecap="round"
          strokeLinejoin="round"
          d="M1.25 8C2.9 4.75 5.15 3.25 8 3.25S13.1 4.75 14.75 8C13.1 11.25 10.85 12.75 8 12.75S2.9 11.25 1.25 8z"
        />
        <circle cx="8" cy="8" r="2.2" fill="none" stroke="currentColor" strokeWidth="1.35" />
        <circle cx="8" cy="8" r="1.05" fill="currentColor" />
      </svg>
      <span className="sr-only">{t('picker.visionMark')}</span>
    </span>
  );
}
