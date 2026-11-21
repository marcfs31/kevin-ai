import { useEffect, useState } from 'react';
import { useI18n, type I18n } from '../i18n';
import type { ApiError, ContentPart } from '../lib/api';
import { isAllowedImageDataUrl } from '../lib/attachments';
import type { AssistantMessage, Message, Stats, UserMessage } from '../lib/storage';
import { liveCount, replyCount } from '../lib/tokens';
import type { Usage } from '../lib/sse';
import { CopyButton, Markdown } from './Markdown';
import { TokenLine } from './TokenLine';

const KNOWN_CODES = new Set([
  'image_unsupported', 'image_invalid', 'image_too_large', 'too_many_images', 'context_overflow',
  'stream_cut', 'bad_stream', 'bad_models', 'upstream_unreachable', 'upstream_failed',
]);

/** A reply error in the user's language, by code where known. */
export function errorText(t: I18n['t'], e: Pick<ApiError, 'status' | 'code' | 'serverMessage'>): string {
  // lm-serve names this context_length_exceeded; the catalogs call it context_overflow.
  if (e.code === 'context_length_exceeded') return t('error.context_overflow');
  if (e.code && KNOWN_CODES.has(e.code)) return t(`error.${e.code}` as Parameters<I18n['t']>[0]);
  if (e.status === 0) return t('error.network');
  if (e.status === 401 || e.status === 403) return t('error.unauthorized');
  if (e.status === 413) return t('error.tooLarge');
  // An error event inside a stream that had started with 200.
  if (e.status >= 200 && e.status < 300) return t('error.stream', { message: e.serverMessage || '—' });
  return t('error.http', { status: String(e.status), message: e.serverMessage || '—' });
}

export interface LiveInfo {
  startedAt: number;
  firstAt: number | null;
  chunks: number;
  /** Estimated prompt tokens, used until the server sends its usage. */
  promptEstimate: number;
  usage: Usage | null;
}

function LiveStatus({ live }: { live: LiveInfo }) {
  const { t, num } = useI18n();
  const [now, setNow] = useState(() => performance.now());
  useEffect(() => {
    const id = setInterval(() => setNow(performance.now()), 250);
    return () => clearInterval(id);
  }, []);
  const counter = <TokenLine className="live" labelKey="tokens.reply" count={liveCount(live.usage, live.promptEstimate, live.chunks)} />;
  if (live.firstAt === null)
    return (
      <>
        <p className="live" aria-live="off">{t('message.readingPrompt', { s: num(Math.max(0, (now - live.startedAt) / 1000)) })}</p>
        {counter}
      </>
    );
  const secs = (now - live.firstAt) / 1000;
  const rate = live.chunks > 1 && secs > 0 ? (live.chunks - 1) / secs : null;
  return (
    <>
      <p className="live" aria-live="off">{rate === null ? t('message.waiting') : t('message.live', { rate: num(rate, 1) })}</p>
      {counter}
    </>
  );
}

function StatsLine({ stats }: { stats: Stats }) {
  const { t, num } = useI18n();
  const parts: string[] = [];
  if (stats.ttftMs !== null) parts.push(t('stats.ttft', { s: num(stats.ttftMs / 1000, 1) }));
  if (stats.tokensPerSecond !== null) parts.push(t('stats.rate', { rate: num(stats.tokensPerSecond, 1) }));
  if (stats.promptTokens !== null) parts.push(t('stats.prompt', { n: stats.promptTokens }));
  if (stats.cachedTokens !== null) parts.push(t('stats.cached', { n: stats.cachedTokens }));
  const count = replyCount(stats);
  if (!parts.length && !count) return null;
  return (
    <>
      {parts.length > 0 && (
        <p className="stats" aria-label={t('stats.label')}>
          {parts.join(' · ')}
        </p>
      )}
      {count && <TokenLine className="stats" labelKey="tokens.reply" count={count} />}
    </>
  );
}

function Assistant({ m, live }: { m: AssistantMessage; live: LiveInfo | null }) {
  const { t } = useI18n();
  return (
    <article className="msg assistant" aria-label={t('message.assistant')}>
      <p className="role">{t('message.assistant')}</p>
      {m.reasoning && (
        <details className="think" open={live !== null && !m.content}>
          <summary>{t('message.thinking')}</summary>
          <div className="think-body">{m.reasoning}</div>
        </details>
      )}
      {m.content && (
        <div className="md">
          <Markdown text={m.content} />
        </div>
      )}
      {live && <LiveStatus live={live} />}
      {m.status === 'stopped' && <p className="note">{t('message.stopped')}</p>}
      {m.error && (
        <p className="error" role="alert">
          {errorText(t, { status: m.error.status, code: m.error.code, serverMessage: m.error.message })}
        </p>
      )}
      {!live && (
        <div className="msg-foot">
          {m.stats && <StatsLine stats={m.stats} />}
          {m.content && <CopyButton text={m.content} label={t('message.copy')} />}
        </div>
      )}
    </article>
  );
}

function imageParts(content: string | ContentPart[]): Extract<ContentPart, { type: 'image_url' }>[] {
  if (typeof content === 'string') return [];
  return content.filter(
    (p): p is Extract<ContentPart, { type: 'image_url' }> =>
      p.type === 'image_url' && isAllowedImageDataUrl(p.image_url.url),
  );
}

function User({ m }: { m: UserMessage }) {
  const { t } = useI18n();
  const images = imageParts(m.content);
  return (
    <article className="msg user" aria-label={t('message.you')}>
      <p className="role">{t('message.you')}</p>
      {images.length > 0 && (
        <ul className="msg-images" aria-label={t('message.images')}>
          {images.map((p, i) => (
            <li key={i}>
              <img src={p.image_url.url} alt="" className="msg-thumb" />
            </li>
          ))}
        </ul>
      )}
      {m.typed && <p className="typed">{m.typed}</p>}
      {m.attachmentNames.length > 0 && <p className="attached">{t('message.attached', { names: m.attachmentNames.join(', ') })}</p>}
    </article>
  );
}

export function MessageView({ m, live = null }: { m: Message; live?: LiveInfo | null }) {
  if (m.role === 'assistant') return <Assistant m={m} live={live} />;
  return <User m={m} />;
}
