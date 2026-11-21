import { useI18n } from '../i18n';
import type { Count } from '../lib/tokens';

type Key = 'tokens.reply' | 'tokens.session';

/** "Read 1,200 · written ≈40 tokens": a ≈ and a note mark every estimated figure. */
export function TokenLine({ className, labelKey, count }: { className: string; labelKey: Key; count: Count }) {
  const { t, num } = useI18n();
  const f = (x: Count['read']) => `${x.estimated ? '≈' : ''}${num(x.n)}`;
  const estimated = count.read.estimated || count.written.estimated;
  return (
    <p className={className} aria-label={t(labelKey === 'tokens.reply' ? 'tokens.replyLabel' : 'tokens.sessionLabel')}>
      {t(labelKey, { read: f(count.read), written: f(count.written) })}
      {estimated && ` · ${t('tokens.estimated')}`}
    </p>
  );
}
