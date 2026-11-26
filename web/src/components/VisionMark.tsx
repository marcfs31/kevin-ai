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
          strokeLinejoin="round"
          d="M1.5 8s2.4-4.5 6.5-4.5S14.5 8 14.5 8 12.1 12.5 8 12.5 1.5 8 1.5 8z"
        />
        <circle cx="8" cy="8" r="2.15" fill="currentColor" />
      </svg>
      <span className="sr-only">{t('picker.visionMark')}</span>
    </span>
  );
}
