import { useI18n } from '../i18n';
import type { ServerInfo } from '../lib/api';
import { chatFigures, contextBar } from '../lib/chatStats';
import type { Message } from '../lib/storage';
import type { Figure } from '../lib/tokens';

function Box({ kind, label, value }: { kind: 'prompt' | 'reply' | 'free' | 'plain'; label: string; value: string }) {
  return (
    <li className={`box box-${kind}`}>
      <span className="box-label">{label}</span>
      <span className="box-value">{value}</span>
    </li>
  );
}

/** "This chat": the context window as one bar of coloured boxes with their numbers, and the
 * chat's totals as boxes under it. Every figure is also written out, so colour is never the only cue. */
export function ChatStats({ messages, systemPrompt, autoCompact, info }: { messages: Message[]; systemPrompt: string; autoCompact: boolean; info: ServerInfo | null }) {
  const { t, num } = useI18n();
  const figures = chatFigures(messages);
  const bar = contextBar(messages, systemPrompt, info);
  if (!figures || !bar) return null;
  const f = (x: Figure) => `${x.estimated ? '≈' : ''}${num(x.n)}`;
  const approx = bar.estimated ? '≈' : '';
  // A fresh chat is under 1%: say so rather than showing 0% next to tokens already in use.
  const rounded = bar.share === null ? null : Math.round(bar.share * 100);
  const percent = rounded === null ? null : rounded === 0 && bar.share !== null && bar.share > 0 ? '<1' : String(rounded);
  const near = bar.share !== null && bar.share >= bar.autoAt;
  const summary = bar.limit === null
    ? t('chatStats.barUnknown', { prompt: `${approx}${num(bar.prompt)}`, reply: num(bar.reply) })
    : t('chatStats.bar', { prompt: `${approx}${num(bar.prompt)}`, reply: num(bar.reply), free: `${approx}${num(bar.free ?? 0)}`, limit: num(bar.limit), percent: percent ?? '0' });
  return (
    <section className="chat-stats" aria-label={t('chatStats.label')}>
      <div className="ctx-wrap">
        <div className={`ctx-bar${near ? ' near' : ''}`} role="img" aria-label={summary}>
          <span className="seg seg-prompt" style={{ flexGrow: Math.max(bar.prompt, 1) }}>
            <span className="box-label">{t('chatStats.prompt')}</span>
            <span className="box-value">{approx}{num(bar.prompt)}</span>
          </span>
          <span className="seg seg-reply" style={{ flexGrow: Math.max(bar.reply, 1) }}>
            <span className="box-label">{t('chatStats.reply')}</span>
            <span className="box-value">{num(bar.reply)}</span>
          </span>
          {bar.free !== null && (
            <span className="seg seg-free" style={{ flexGrow: Math.max(bar.free, 1) }}>
              <span className="box-label">{t('chatStats.free')}</span>
              <span className="box-value">{approx}{num(bar.free)}</span>
            </span>
          )}
        </div>
        {autoCompact && bar.limit !== null && <span className="ctx-mark" style={{ insetInlineStart: `${bar.autoAt * 100}%` }} aria-hidden="true" />}
      </div>
      <p className="hint ctx-caption">
        {bar.limit === null
          ? t('chatStats.noLimit')
          : t(autoCompact ? 'chatStats.captionAuto' : 'chatStats.captionManual', { percent: percent ?? '0', limit: num(bar.limit), at: Math.round(bar.autoAt * 100) })}
        {bar.estimated && ` · ${t('tokens.estimated')}`}
      </p>
      <ul className="chat-boxes" aria-label={t('chatStats.totals')}>
        <Box kind="prompt" label={t('chatStats.read')} value={f(figures.read)} />
        <Box kind="reply" label={t('chatStats.written')} value={f(figures.written)} />
        <Box kind="plain" label={t('chatStats.turns')} value={num(figures.turns)} />
        {figures.rate !== null && <Box kind="plain" label={t('chatStats.rate')} value={num(figures.rate, 1)} />}
        {figures.compactions > 0 && <Box kind="plain" label={t('chatStats.compactions')} value={num(figures.compactions)} />}
      </ul>
    </section>
  );
}
