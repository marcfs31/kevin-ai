import { useEffect, useState } from 'react';
import { useI18n, type I18n } from '../i18n';
import type { ApiError, ContentPart } from '../lib/api';
import { isAllowedImageDataUrl, sentFileText } from '../lib/attachments';
import type { AssistantMessage, Message, Stats, SummaryMessage, ToolMessage, UserMessage } from '../lib/storage';
import { liveCount, replyCount } from '../lib/tokens';
import type { LiveInfo } from '../lib/agent';
import { estimateWaitSeconds, etaParts } from '../lib/estimate';
import { CopyButton, Markdown } from './Markdown';
import { Preview, PreviewImage, PreviewText } from './Preview';
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
    return t('error.toolRequest', { message: e.serverMessage || t('error.noDetail') });
  if (e.code && KNOWN_CODES.has(e.code)) return t(`error.${e.code}` as Parameters<I18n['t']>[0]);
  if (e.status === 0) return t('error.network');
  if (e.status === 401 || e.status === 403) return t('error.unauthorized');
  if (e.status === 413) return t('error.tooLarge');
  // Bare 502 from a proxy HTML page has no body: say the upstream is down, not "no details".
  if (e.status === 502 && !e.serverMessage) return t('error.upstream_unreachable');
  // An error event inside a stream that had started with 200.
  if (e.status >= 200 && e.status < 300) return t('error.stream', { message: e.serverMessage || t('error.noDetail') });
  return t('error.http', { status: String(e.status), message: e.serverMessage || t('error.noDetail') });
}

function LiveStatus({ live }: { live: LiveInfo }) {
  const { t, num } = useI18n();
  const [now, setNow] = useState(() => performance.now());
  useEffect(() => {
    const id = setInterval(() => setNow(performance.now()), 250);
    return () => clearInterval(id);
  }, []);
  const counter = <TokenLine className="live" count={liveCount(live.usage, live.promptEstimate, live.chunks)} />;
  if (live.firstAt === null) {
    const elapsed = Math.max(0, (now - live.startedAt) / 1000);
    const left = Math.max(0, estimateWaitSeconds(live.readEstimate) - elapsed);
    const eta = etaParts(left);
    // While the estimate still has time left, show minutes and seconds remaining.
    const line =
      left >= 1
        ? t('message.readingPromptEta', { elapsed: num(elapsed), m: num(eta.m), s: num(eta.s) })
        : t('message.readingPrompt', { s: num(elapsed) });
    // What the server is busy with, so a long wait is not a mystery: the model has not started thinking, it reads first.
    const tools = live.tools?.tokens ?? 0;
    const cached = Math.max(0, live.promptEstimate - live.readEstimate);
    const what = [
      live.tools
        ? t('message.readingWhat', { prompt: num(live.promptEstimate), tools: num(tools), rest: num(Math.max(0, live.promptEstimate - tools)) })
        : t('message.readingWhatNoTools', { prompt: num(live.promptEstimate) }),
      cached > 0 ? t('message.readingCached', { cached: num(cached), read: num(live.readEstimate) }) : '',
    ]
      .filter(Boolean)
      .join(' ');
    return (
      <>
        <p className="live" aria-live="off">{line}</p>
        <p className="hint live-what">{what}</p>
        {counter}
      </>
    );
  }
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

function Assistant({ m, live, results, usingTools, onRetry, onDelete, onEdit }: { m: AssistantMessage; live: LiveInfo | null; results: ToolMessage[]; usingTools: boolean; onRetry?: () => void; onDelete?: () => void; onEdit?: () => void }) {
  const { t } = useI18n();
  return (
    <article className="msg assistant" aria-label={t('message.assistant')}>
      <h2 className="role">{t('message.assistant')}</h2>
      {m.reasoning && (
        <details className="think" open={(live !== null && !m.content) || usingTools}>
          <summary>{t('message.thinking')}</summary>
          <div className="think-body md">
            <Markdown text={m.reasoning} emphasisAsBold />
          </div>
        </details>
      )}
      {m.content && (
        <div className="md">
          <Markdown text={m.content} />
        </div>
      )}
      {m.toolCalls && m.toolCalls.length > 0 && <ToolTrace m={m} results={results} open={usingTools} />}
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
          {onEdit && (
            <button type="button" onClick={onEdit} aria-describedby={`${m.id}-edit-hint`}>
              {t('message.edit')}
            </button>
          )}
          {onEdit && (
            <span id={`${m.id}-edit-hint`} className="sr-only">
              {t('message.editHint')}
            </span>
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

function imageParts(content: string | ContentPart[]): Extract<ContentPart, { type: 'image_url' }>[] {
  if (typeof content === 'string') return [];
  return content.filter(
    (p): p is Extract<ContentPart, { type: 'image_url' }> =>
      p.type === 'image_url' && isAllowedImageDataUrl(p.image_url.url),
  );
}

function User({ m, onDelete, onRegenerate, onEdit }: { m: UserMessage; onDelete?: () => void; onRegenerate?: () => void; onEdit?: () => void }) {
  const { t } = useI18n();
  const images = imageParts(m.content);
  const files = sentFileText(m.content, m.typed);
  return (
    <article className="msg user" aria-label={t('message.you')}>
      <h2 className="role">{t('message.you')}</h2>
      {images.length > 0 && (
        <ul className="msg-images" aria-label={t('message.images')}>
          {images.map((p, i) => (
            <li key={i}>
              <details className="preview preview-image">
                <summary aria-label={t('preview.image', { n: i + 1 })}>
                  <img src={p.image_url.url} alt="" className="msg-thumb" />
                </summary>
                <PreviewImage src={p.image_url.url} label={t('preview.image', { n: i + 1 })} />
              </details>
            </li>
          ))}
        </ul>
      )}
      {m.typed && <p className="typed">{m.typed}</p>}
      {m.attachmentNames.length > 0 && <p className="attached">{t('message.attached', { names: m.attachmentNames.join(', ') })}</p>}
      {m.attachmentNames.length > 0 && files !== '' && (
        <Preview label={t('preview.sent')}>
          <PreviewText text={files} />
        </Preview>
      )}
      {(onDelete || onRegenerate || onEdit) && (
        <div className="msg-foot">
          {onRegenerate && (
            <p className="retry">
              <button type="button" onClick={onRegenerate}>
                {t('message.regenerate')}
              </button>
              <span className="hint">{t('message.regenerateHint')}</span>
            </p>
          )}
          {onEdit && (
            <button type="button" onClick={onEdit} aria-describedby={`${m.id}-edit-hint`}>
              {t('message.edit')}
            </button>
          )}
          {onEdit && (
            <span id={`${m.id}-edit-hint`} className="sr-only">
              {t('message.editHint')}
            </span>
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
function ToolTrace({ m, results, open }: { m: AssistantMessage; results: ToolMessage[]; open: boolean }) {
  const { t } = useI18n();
  const calls = m.toolCalls ?? [];
  return (
    <>
      <details className="trace" open={open}>
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

/** onRetry / onRegenerate only for the last message when nothing is running; onDelete rewinds from here; onEdit rewinds
 * to the prompt that produced this turn (your own message, or the user turn before an assistant reply) and puts that
 * text back in the message box to change and send again. */
export function MessageView({ m, live = null, results = [], usingTools = false, onRetry, onDelete, onRegenerate, onEdit }: {
  m: Message;
  live?: LiveInfo | null;
  results?: ToolMessage[];
  /** This reply's tools are in use, in the run that is going now: its thinking and tool calls are shown while it lasts
   * (so the user sees what is asked, run and returned) and fold again when the run ends, to keep the chat short. */
  usingTools?: boolean;
  onRetry?: () => void;
  onDelete?: () => void;
  onRegenerate?: () => void;
  onEdit?: () => void;
}) {
  if (m.role === 'assistant') return <Assistant m={m} live={live} results={results} usingTools={usingTools} onRetry={onRetry} onDelete={onDelete} onEdit={onEdit} />;
  if (m.role === 'summary') return <Summary m={m} onDelete={onDelete} />;
  // A tool result is shown inside the reply that asked for it.
  if (m.role === 'tool') return null;
  return <User m={m} onDelete={onDelete} onRegenerate={onRegenerate} onEdit={onEdit} />;
}
