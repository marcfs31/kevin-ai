import { useEffect, useId, useRef, type ReactNode } from 'react';
import { useI18n } from '../i18n';

/** Full-screen overlay for Settings: dimmed backdrop, centred panel, Escape / backdrop to close. */
export function SettingsModal({
  open,
  onClose,
  children,
}: {
  open: boolean;
  onClose: () => void;
  children: ReactNode;
}) {
  const { t } = useI18n();
  const titleId = useId();
  const closeRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (!open) return;
    const prev = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    closeRef.current?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        onClose();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => {
      document.body.style.overflow = prev;
      window.removeEventListener('keydown', onKey);
    };
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div className="modal-root">
      <button type="button" className="modal-backdrop" tabIndex={-1} aria-label={t('settings.closeBackdrop')} onClick={onClose} />
      <div id="settings" className="settings-modal" role="dialog" aria-modal="true" aria-labelledby={titleId}>
        <div className="settings-modal-head">
          <h2 id={titleId}>{t('settings.open')}</h2>
          <button type="button" ref={closeRef} className="settings-modal-close" onClick={onClose}>
            {t('settings.close')}
          </button>
        </div>
        <div className="settings-modal-body panels">{children}</div>
      </div>
    </div>
  );
}
