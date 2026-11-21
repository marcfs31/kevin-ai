import { useI18n } from '../i18n';
import type { Count } from '../lib/tokens';

/** "Read 1,200 · written ≈40 tokens": a ≈ and a note mark every estimated figure. */
export function TokenLine({ className, count }: { className: string; count: Count }) {
  const { t, num } = useI18n();
  const f = (x: Count['read']) => `${x.estimated ? '≈' : ''}${num(x.n)}`;
  const estimated = count.read.estimated || count.written.estimated;
  return (
    <p className={className} aria-label={t('tokens.replyLabel')}>
      {t('tokens.reply', { read: f(count.read), written: f(count.written) })}
      {estimated && ` · ${t('tokens.estimated')}`}
    </p>
  );
}
