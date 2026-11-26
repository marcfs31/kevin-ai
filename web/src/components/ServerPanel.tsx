import { useId, useState } from 'react';
import { useI18n } from '../i18n';
import { ApiError } from '../lib/api';
import { SERVE_DEFAULTS, startRemoteServe, type ServeModelId, type ServeOptions } from '../lib/serve';
import { stockOptions, isServeModelRunning, serveModelLabel, uncensoredOptions } from '../lib/serveModels';
import { UncensoredMark } from './UncensoredMark';
import { VisionMark } from './VisionMark';

export interface ServeDraft {
  maxTokens: string;
  seed: string;
  ctx: string;
  threads: string;
  timeout: string;
  cacheMb: string;
  name: string;
  think: boolean;
  fast: boolean;
}

export const EMPTY_SERVE_DRAFT: ServeDraft = {
  maxTokens: '',
  seed: '',
  ctx: '',
  threads: '',
  timeout: '',
  cacheMb: '',
  name: '',
  think: false,
  fast: false,
};

function optionalInt(s: string): number | null | undefined {
  const t = s.trim();
  if (!t) return null;
  if (!/^-?\d+$/.test(t)) return undefined;
  const n = Number(t);
  return Number.isInteger(n) ? n : undefined;
}

export function draftToOptions(model: ServeModelId, draft: ServeDraft): ServeOptions | null {
  const maxTokens = optionalInt(draft.maxTokens);
  const seed = optionalInt(draft.seed);
  const ctx = optionalInt(draft.ctx);
  const threads = optionalInt(draft.threads);
  const timeout = optionalInt(draft.timeout);
  const cacheMb = optionalInt(draft.cacheMb);
  if ([maxTokens, seed, ctx, threads, timeout, cacheMb].includes(undefined)) return null;
  return {
    model,
    maxTokens: maxTokens ?? null,
    seed: seed ?? null,
    ctx: ctx ?? null,
    threads: threads ?? null,
    timeout: timeout ?? null,
    cacheMb: cacheMb ?? null,
    name: draft.name.trim() || null,
    think: draft.think ? true : null,
    fast: draft.fast ? true : null,
  };
}

type Status =
  | { kind: 'idle' }
  | { kind: 'starting'; model: ServeModelId }
  | { kind: 'ok'; model: ServeModelId }
  | { kind: 'error'; message: string; detail?: string };

interface Props {
  busy: boolean;
  /** Model id reported by the server, when one is running. */
  runningModel?: string | null;
  onStarted: () => void;
}

export function ServerPanel({ busy, runningModel = null, onStarted }: Props) {
  const { t, locale } = useI18n();
  const id = useId();
  const [draft, setDraft] = useState<ServeDraft>(EMPTY_SERVE_DRAFT);
  const [status, setStatus] = useState<Status>({ kind: 'idle' });
  const starting = status.kind === 'starting';

  const set = <K extends keyof ServeDraft>(k: K, v: ServeDraft[K]) => setDraft((d) => ({ ...d, [k]: v }));

  const renderServeButton = (opt: (typeof stockOptions)[number]) => {
    const name = serveModelLabel(opt.id, locale);
    return (
      <button
        key={opt.id}
        type="button"
        className={[
          'primary',
          opt.uncensored || opt.vision ? 'has-model-mark' : '',
          opt.uncensored ? 'has-uncensored-warn' : '',
        ].filter(Boolean).join(' ')}
        aria-pressed={isServeModelRunning(runningModel, opt.id)}
        disabled={busy || starting}
        onClick={() => void start(opt.id)}
      >
        <span className="model-choice-label">
          {starting && status.model === opt.id ? t('serve.starting', { model: name }) : t('serve.startNamed', { model: name })}
        </span>
        {(opt.vision || opt.uncensored) && (
          <span className="model-marks">
            {opt.vision && <VisionMark />}
            {opt.uncensored && <UncensoredMark />}
          </span>
        )}
      </button>
    );
  };

  const start = async (model: ServeModelId) => {
    const opts = draftToOptions(model, draft);
    if (!opts) {
      setStatus({ kind: 'error', message: t('serve.badParams') });
      return;
    }
    if (!window.confirm(t('serve.confirm', { model: serveModelLabel(model, locale) }))) return;
    setStatus({ kind: 'starting', model });
    try {
      const result = await startRemoteServe(opts);
      if (!result.ok) {
        setStatus({ kind: 'error', message: t('serve.failed'), detail: result.error || undefined });
        return;
      }
      setStatus({ kind: 'ok', model });
      onStarted();
    } catch (e) {
      if (e instanceof ApiError && e.status === 0) {
        setStatus({ kind: 'error', message: t('error.network') });
        return;
      }
      const detail = e instanceof Error ? e.message : '';
      setStatus({ kind: 'error', message: t('serve.failed'), detail: detail || undefined });
    }
  };

  return (
    <section className="settings serve-panel" aria-labelledby={`${id}-h`}>
      <h2 id={`${id}-h`}>{t('serve.title')}</h2>
      <p className="hint" id={`${id}-hint`}>
        {t('serve.hint')}
      </p>
      <div className="serve-model-groups" role="group" aria-describedby={`${id}-hint`}>
        <div className="model-picker-group">
          <h3 id={`${id}-stock`} className="model-picker-group-title">{t('picker.modelGroupStock')}</h3>
          <div className="serve-actions" aria-labelledby={`${id}-stock`}>{stockOptions.map(renderServeButton)}</div>
        </div>
        <div className="model-picker-group">
          <h3 id={`${id}-uncensored`} className="model-picker-group-title">{t('picker.modelGroupUncensored')}</h3>
          <div className="serve-actions" aria-labelledby={`${id}-uncensored`}>{uncensoredOptions.map(renderServeButton)}</div>
        </div>
      </div>
      {status.kind === 'ok' && (
        <p className="hint" role="status">
          {t('serve.started', { model: serveModelLabel(status.model, locale) })}
        </p>
      )}
      {status.kind === 'error' && (
        <>
          <p className="error" role="alert">
            {status.message}
          </p>
          {status.detail && <pre className="serve-log">{status.detail}</pre>}
        </>
      )}
      <details className="serve-advanced">
        <summary>{t('serve.advanced')}</summary>
        <p className="hint">{t('serve.advancedHint')}</p>
        <div className="fields">
          <div className="field">
            <label htmlFor={`${id}-max`}>{t('serve.maxTokens')}</label>
        <input
          id={`${id}-max`}
          type="number"
          inputMode="numeric"
          min={0}
          step={1}
          placeholder={String(SERVE_DEFAULTS.maxTokens)}
          value={draft.maxTokens}
          onChange={(e) => set('maxTokens', e.target.value)}
            />
          </div>
          <div className="field">
            <label htmlFor={`${id}-seed`}>{t('serve.seed')}</label>
        <input
          id={`${id}-seed`}
          type="number"
          inputMode="numeric"
          min={0}
          step={1}
          placeholder={String(SERVE_DEFAULTS.seed)}
          value={draft.seed}
          onChange={(e) => set('seed', e.target.value)}
            />
          </div>
          <div className="field">
            <label htmlFor={`${id}-ctx`}>{t('serve.ctx')}</label>
        <input
          id={`${id}-ctx`}
          type="number"
          inputMode="numeric"
          min={1}
          step={1}
          placeholder={t('serve.auto')}
          value={draft.ctx}
          onChange={(e) => set('ctx', e.target.value)}
            />
          </div>
          <div className="field">
            <label htmlFor={`${id}-threads`}>{t('serve.threads')}</label>
        <input
          id={`${id}-threads`}
          type="number"
          inputMode="numeric"
          min={1}
          step={1}
          placeholder={t('serve.auto')}
          value={draft.threads}
          onChange={(e) => set('threads', e.target.value)}
            />
          </div>
          <div className="field">
            <label htmlFor={`${id}-timeout`}>{t('serve.timeout')}</label>
        <input
          id={`${id}-timeout`}
          type="number"
          inputMode="numeric"
          min={1}
          step={1}
          placeholder={String(SERVE_DEFAULTS.timeout)}
          value={draft.timeout}
          onChange={(e) => set('timeout', e.target.value)}
            />
          </div>
          <div className="field">
            <label htmlFor={`${id}-cache`}>{t('serve.cacheMb')}</label>
        <input
          id={`${id}-cache`}
          type="number"
          inputMode="numeric"
          min={0}
          step={1}
          placeholder={String(SERVE_DEFAULTS.cacheMb)}
          value={draft.cacheMb}
          onChange={(e) => set('cacheMb', e.target.value)}
            />
          </div>
          <div className="field field-wide">
            <label htmlFor={`${id}-name`}>{t('serve.name')}</label>
            <input id={`${id}-name`} type="text" maxLength={200} placeholder={t('serve.nameDefault')} value={draft.name} onChange={(e) => set('name', e.target.value)} />
          </div>
          <div className="field field-wide checks">
            <label className="serve-check">
              <input type="checkbox" checked={draft.think} onChange={(e) => set('think', e.target.checked)} />
              {t('serve.think')}
            </label>
            <label className="serve-check">
              <input type="checkbox" checked={draft.fast} onChange={(e) => set('fast', e.target.checked)} />
              {t('serve.fast')}
            </label>
          </div>
        </div>
      </details>
    </section>
  );
}
