import { useEffect, useState } from 'react';
import { useI18n, type I18n } from '../i18n';
import type { ApiError, ContentPart } from '../lib/api';
import { isAllowedImageDataUrl } from '../lib/attachments';
import type { AssistantMessage, Message, Stats, SummaryMessage, ToolMessage, UserMessage } from '../lib/storage';
import { liveCount, replyCount } from '../lib/tokens';
import type { LiveInfo } from '../lib/agent';
import { CopyButton, Markdown } from './Markdown';
import { TokenLine } from './TokenLine';

const KNOWN_CODES = new Set([
  'image_unsupported', 'image_invalid', 'image_too_large', 'too_many_images', 'context_overflow',
  'stream_cut', 'bad_stream', 'summary_empty', 'bad_models', 'upstream_unreachable', 'upstream_failed',
]);

/** A reply error in the user's language, by code where known. */
export function errorText(t: I18n['t'], e: Pick<ApiError, 'status' | 'code' | 'serverMessage'>): string {
  // lm-serve names this context_length_exceeded; the catalogs call it context_overflow.
  if (e.code === 'context_length_exceeded') return t('error.context_overflow');
  if (e.code === 'invalid_tools' || e.code === 'invalid_tool_call' || e.code === 'invalid_tool_message')
    return t('error.toolRequest', { message: e.serverMessage || '—' });
  if (e.code && KNOWN_CODES.has(e.code)) return t(`error.${e.code}` as Parameters<I18n['t']>[0]);
  if (e.status === 0) return t('error.network');
  if (e.status === 401 || e.status === 403) return t('error.unauthorized');
  if (e.status === 413) return t('error.tooLarge');
  // An error event inside a stream that had started with 200.
  if (e.status >= 200 && e.status < 300) return t('error.stream', { message: e.serverMessage || '—' });
  return t('error.http', { status: String(e.status), message: e.serverMessage || '—' });
}

function LiveStatus({ live }: { live: LiveInfo }) {
  const { t, num } = useI18n();
  const [now, setNow] = useState(() => performance.now());
  useEffect(() => {
    const id = setInterval(() => setNow(performance.now()), 250);
    return () => clearInterval(id);
  }, []);
  const counter = <TokenLine className="live" count={liveCount(live.usage, live.promptEstimate, live.chunks)} />;
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
  const figures = replyCount(stats);
  const count = figures?.read && figures.written ? { read: figures.read, written: figures.written } : null;
  if (!parts.length && !count) return null;
  return (
    <>
      {parts.length > 0 && (
        <p className="stats" aria-label={t('stats.label')}>
          {parts.join(' · ')}
        </p>
      )}
      {count && <TokenLine className="stats" count={count} />}
    </>
  );
}

function Assistant({ m, live, results, onRetry, onDelete }: { m: AssistantMessage; live: LiveInfo | null; results: ToolMessage[]; onRetry?: () => void; onDelete?: () => void }) {
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
      {m.toolCalls && m.toolCalls.length > 0 && <ToolTrace m={m} results={results} />}
      {live && <LiveStatus live={live} />}
      {m.status === 'stopped' && <p className="note">{t('message.stopped')}</p>}
      {m.error && (
        <p className="error" role="alert">
          {errorText(t, { status: m.error.status, code: m.error.code, serverMessage: m.error.message })}
        </p>
      )}
      {m.error && onRetry && (
        <p className="retry">
          <button type="button" onClick={onRetry}>
            {t('message.retry')}
          </button>
          <span className="hint">{t('message.retryHint')}</span>
        </p>
      )}
      {!live && (
        <div className="msg-foot">
          {m.stats && <StatsLine stats={m.stats} />}
          {m.content && <CopyButton text={m.content} label={t('message.copy')} />}
          {onDelete && (
            <button type="button" className="danger" onClick={onDelete} aria-label={t('message.delete')}>
              {t('message.delete')}
            </button>
          )}
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

function User({ m, onDelete, onRegenerate }: { m: UserMessage; onDelete?: () => void; onRegenerate?: () => void }) {
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
      {(onDelete || onRegenerate) && (
        <div className="msg-foot">
          {onRegenerate && (
            <p className="retry">
              <button type="button" onClick={onRegenerate}>
                {t('message.regenerate')}
              </button>
              <span className="hint">{t('message.regenerateHint')}</span>
            </p>
          )}
          {onDelete && (
            <button type="button" className="danger" onClick={onDelete} aria-label={t('message.delete')}>
              {t('message.delete')}
            </button>
          )}
        </div>
      )}
    </article>
  );
}

/** What the model asked for and what came back, folded away until the user wants to see it. */
function ToolTrace({ m, results }: { m: AssistantMessage; results: ToolMessage[] }) {
  const { t } = useI18n();
  const calls = m.toolCalls ?? [];
  return (
    <>
      <details className="trace">
        <summary>{t('tools.trace', { names: calls.map((c) => c.name || '?').join(', ') })}</summary>
        <ol className="trace-calls">
          {calls.map((c) => {
            const r = results.find((x) => x.toolCallId === c.id);
            return (
              <li key={c.id}>
                <p className="trace-name">{c.name || '?'}</p>
                <p className="hint">{t('tools.arguments')}</p>
                <pre>{c.arguments || '{}'}</pre>
                <p className="hint">{r ? (r.isError ? t('tools.resultError') : t('tools.result')) : t('tools.notRun')}</p>
                {r && <pre>{r.content}</pre>}
              </li>
            );
          })}
        </ol>
      </details>
      {m.stepLimit !== undefined && <p className="note">{t('tools.stepLimit', { n: m.stepLimit })}</p>}
    </>
  );
}

/** The marker where older turns were replaced by a summary; the summary itself folds away. */
function Summary({ m, onDelete }: { m: SummaryMessage; onDelete?: () => void }) {
  const { t, tp } = useI18n();
  return (
    <aside className="msg summary" aria-label={t('compact.label')}>
      <details>
        <summary>{tp('compact.marker', m.covered)}</summary>
        <p className="hint">{m.auto ? t('compact.byAuto') : t('compact.byHand')}</p>
        <div className="md">
          <Markdown text={m.content} />
        </div>
      </details>
      {onDelete && (
        <div className="msg-foot">
          <button type="button" className="danger" onClick={onDelete} aria-label={t('message.delete')}>
            {t('message.delete')}
          </button>
        </div>
      )}
    </aside>
  );
}

/** onRetry / onRegenerate only for the last message when nothing is running; onDelete rewinds from here. */
export function MessageView({ m, live = null, results = [], onRetry, onDelete, onRegenerate }: {
  m: Message;
  live?: LiveInfo | null;
  results?: ToolMessage[];
  onRetry?: () => void;
  onDelete?: () => void;
  onRegenerate?: () => void;
}) {
  if (m.role === 'assistant') return <Assistant m={m} live={live} results={results} onRetry={onRetry} onDelete={onDelete} />;
  if (m.role === 'summary') return <Summary m={m} onDelete={onDelete} />;
  // A tool result is shown inside the reply that asked for it.
  if (m.role === 'tool') return null;
  return <User m={m} onDelete={onDelete} onRegenerate={onRegenerate} />;
}
