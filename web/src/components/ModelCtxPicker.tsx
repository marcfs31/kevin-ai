import { useCallback, useEffect, useId, useRef, useState } from 'react';
import { useI18n } from '../i18n';
import { ApiError } from '../lib/api';
import { startRemoteServe, type ServeModelId } from '../lib/serve';
import {
  SERVE_MODEL_LABEL,
  SERVE_MODEL_OPTIONS,
  ctxPresetsFor,
  formatCtxPreset,
  matchServeModel,
} from '../lib/serveModels';
import { draftToOptions, EMPTY_SERVE_DRAFT } from './ServerPanel';

type Status =
  | { kind: 'idle' }
  | { kind: 'starting'; model: ServeModelId }
  | { kind: 'error'; message: string; detail?: string };

interface Props {
  /** The model the server reports, or null when it cannot be reached: the picker is then how to start one. */
  modelName: string | null;
  contextLength: number | null;
  busy: boolean;
  onStarted: () => void;
}

export function ModelCtxPicker({ modelName, contextLength, busy, onStarted }: Props) {
  const { t } = useI18n();
  const uid = useId();
  const panelId = `${uid}-panel`;
  const titleId = `${uid}-title`;
  const rootRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const [open, setOpen] = useState(false);
  const [model, setModel] = useState<ServeModelId>(() => matchServeModel(modelName) ?? 'qwen');
  const [ctxPreset, setCtxPreset] = useState<number | null>(null);
  const [status, setStatus] = useState<Status>({ kind: 'idle' });
  const starting = status.kind === 'starting';
  const disabled = busy || starting;

  const presets = ctxPresetsFor(model, modelName, contextLength);
  const selectedCtx = ctxPreset !== null && presets.includes(ctxPreset) ? ctxPreset : null;

  const close = useCallback(() => {
    setOpen(false);
    if (status.kind === 'error') setStatus({ kind: 'idle' });
  }, [status.kind]);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        e.stopPropagation();
        close();
        triggerRef.current?.focus();
      }
    };
    const onPointer = (e: MouseEvent) => {
      const root = rootRef.current;
      if (root && e.target instanceof Node && !root.contains(e.target)) {
        close();
        triggerRef.current?.focus();
      }
    };
    document.addEventListener('keydown', onKey);
    document.addEventListener('mousedown', onPointer);
    return () => {
      document.removeEventListener('keydown', onKey);
      document.removeEventListener('mousedown', onPointer);
    };
  }, [open, close]);

  const start = async () => {
    const draft = {
      ...EMPTY_SERVE_DRAFT,
      ctx: selectedCtx === null ? '' : String(selectedCtx),
    };
    const opts = draftToOptions(model, draft);
    if (!opts) {
      setStatus({ kind: 'error', message: t('serve.badParams') });
      return;
    }
    if (!window.confirm(t('serve.confirm', { model: SERVE_MODEL_LABEL[model] }))) return;
    setStatus({ kind: 'starting', model });
    try {
      const result = await startRemoteServe(opts);
      if (!result.ok) {
        setStatus({ kind: 'error', message: t('serve.failed'), detail: result.error || undefined });
        return;
      }
      setStatus({ kind: 'idle' });
      onStarted();
      setOpen(false);
      triggerRef.current?.focus();
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
    <div className="model-picker" ref={rootRef}>
      <button
        type="button"
        className="model-picker-trigger"
        ref={triggerRef}
        aria-haspopup="dialog"
        aria-expanded={open}
        aria-controls={open ? panelId : undefined}
        aria-label={modelName === null ? t('picker.startNone') : `${t('server.ok', { model: modelName })}. ${t('picker.open')}`}
        disabled={disabled}
        onClick={() => setOpen((o) => !o)}
      >
        {modelName === null ? t('picker.startNone') : t('server.ok', { model: modelName })}
      </button>
      {open && (
        <div
          id={panelId}
          role="dialog"
          aria-modal="false"
          aria-labelledby={titleId}
          className="model-picker-panel"
        >
          <h2 id={titleId} className="model-picker-title">
            {t('picker.title')}
          </h2>
          <div className="model-picker-section">
            <h3>{t('picker.model')}</h3>
            <div className="model-picker-choices" role="group">
              {SERVE_MODEL_OPTIONS.map(({ id, label }) => (
                <button
                  key={id}
                  type="button"
                  aria-pressed={model === id}
                  disabled={disabled}
                  onClick={() => {
                    setModel(id);
                    const next = ctxPresetsFor(id, modelName, contextLength);
                    if (ctxPreset !== null && !next.includes(ctxPreset)) setCtxPreset(null);
                  }}
                >
                  {label}
                </button>
              ))}
            </div>
          </div>
          <div className="model-picker-section">
            <h3>{t('picker.ctx')}</h3>
            <div className="model-picker-choices" role="radiogroup" aria-label={t('picker.ctx')}>
              {presets.map((size) => {
                const auto = size === null;
                const label = auto
                  ? t('picker.ctxAuto')
                  : t('picker.ctxSize', { label: formatCtxPreset(size) });
                const selected = selectedCtx === size;
                return (
                  <button
                    key={auto ? 'auto' : size}
                    type="button"
                    role="radio"
                    aria-checked={selected}
                    disabled={disabled}
                    onClick={() => setCtxPreset(size)}
                  >
                    {label}
                  </button>
                );
              })}
            </div>
          </div>
          <button type="button" className="primary" disabled={disabled} onClick={() => void start()}>
            {starting ? t('serve.starting', { model: SERVE_MODEL_LABEL[status.model] }) : t('picker.start')}
          </button>
          {status.kind === 'error' && (
            <>
              <p className="error" role="alert">
                {status.message}
              </p>
              {status.detail && <pre className="serve-log">{status.detail}</pre>}
            </>
          )}
        </div>
      )}
    </div>
  );
}
