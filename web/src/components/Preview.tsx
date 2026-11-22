import { useI18n } from '../i18n';
import { PREVIEW_MAX_CHARS } from '../lib/attachments';

/** A file's text in a scrollable block, cut at PREVIEW_MAX_CHARS with a note. */
export function PreviewText({ text }: { text: string }) {
  const { t, num } = useI18n();
  const cut = text.length > PREVIEW_MAX_CHARS;
  return (
    <>
      <pre className="preview-text" tabIndex={0}>
        {cut ? text.slice(0, PREVIEW_MAX_CHARS) : text}
      </pre>
      {cut && <p className="hint">{t('preview.truncated', { n: num(PREVIEW_MAX_CHARS) })}</p>}
    </>
  );
}

/** An image at full size inside the preview. */
export function PreviewImage({ src, label }: { src: string; label: string }) {
  return <img src={src} alt={label} className="preview-img" />;
}

/** A disclosure that shows its content only when opened. */
export function Preview({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <details className="preview">
      <summary>{label}</summary>
      {children}
    </details>
  );
}
