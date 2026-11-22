import { useEffect, useMemo, useRef, useState, type DragEvent, type KeyboardEvent, type RefObject } from 'react';
import { useI18n } from '../i18n';
import type { ServerInfo } from '../lib/api';
import {
  ACCEPT_ATTRIBUTE, AttachError, DEFAULT_MAX_PIXELS, checkText, classifyFile, estimateAttachment, maxBytes, newId, parsePageRange,
  selectPdf, type Attachment,
} from '../lib/attachments';
import { Preview, PreviewImage, PreviewText } from './Preview';
import { WARN_WAIT_SECONDS } from '../lib/estimate';
import { imageToPng } from '../lib/image';
import { finePointer } from '../lib/pointer';
import type { Assessment } from '../lib/request';

interface Props {
  busy: boolean;
  info: ServerInfo | null;
  imagePreset: number;
  assessDraft: (text: string, attachments: Attachment[]) => Assessment;
  onSend: (text: string, attachments: Attachment[]) => void;
  /** True while the reply streaming in this chat can be steered: a message typed now stops it and goes next. */
  canRedirect: boolean;
  onRedirect: (text: string, attachments: Attachment[]) => void;
  onStop: () => void;
  inputRef: RefObject<HTMLTextAreaElement | null>;
}

async function readAttachment(file: File, info: ServerInfo | null, preset: number): Promise<Attachment> {
  const kind = classifyFile(file.name, file.type);
  if (!kind) throw new AttachError('unsupported_type', file.name);
  if (file.size > maxBytes(kind)) throw new AttachError('too_large', file.name);
  const visionOk = info?.vision === 'ok';
  const maxPixels = info?.maxPixels ?? DEFAULT_MAX_PIXELS;
  if (kind === 'image') {
    if (!visionOk) throw new AttachError('vision_off', file.name);
    return { id: newId(), kind: 'image', name: file.name, ...(await imageToPng(file, file.name, preset, maxPixels)) };
  }
  if (kind === 'pdf') {
    // pdf.js is large: load it only when a PDF is attached.
    const { readPdf } = await import('../lib/pdf');
    const pages = await readPdf(await file.arrayBuffer(), file.name, visionOk, preset, maxPixels);
    return { id: newId(), kind: 'pdf', name: file.name, pages, range: `1-${pages.length}` };
  }
  let text: string;
  try {
    text = await file.text();
  } catch {
    throw new AttachError('read_failed', file.name);
  }
  return { id: newId(), kind: 'text', name: file.name, text: checkText(file.name, text) };
}

export function Composer({ busy, info, imagePreset, assessDraft, onSend, canRedirect, onRedirect, onStop, inputRef }: Props) {
  const { t, duration, megabytes } = useI18n();
  const [text, setText] = useState('');
  const [attachments, setAttachments] = useState<Attachment[]>([]);
  const [errors, setErrors] = useState<string[]>([]);
  const [reading, setReading] = useState<Array<{ id: string; name: string }>>([]);
  const [dragging, setDragging] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);
  const imageRef = useRef<HTMLInputElement>(null);
  const vision = info?.vision ?? 'unknown';
  const visionOk = vision === 'ok';

  const assessment = useMemo(() => assessDraft(text, attachments), [assessDraft, text, attachments]);
  const rangeInvalid = attachments.some((a) => a.kind === 'pdf' && !parsePageRange(a.range, a.pages.length));
  const empty = !text.trim() && attachments.length === 0;
  const blocked = assessment.problems.length > 0 || rangeInvalid || reading.length > 0;

  const addFiles = async (files: File[]) => {
    setErrors([]);
    for (const file of files) {
      const pending = { id: newId(), name: file.name };
      setReading((r) => [...r, pending]);
      try {
        const att = await readAttachment(file, info, imagePreset);
        setAttachments((list) => [...list, att]);
      } catch (e) {
        const code = e instanceof AttachError ? e.code : 'read_failed';
        setErrors((list) => [...list, t(`attach.${code}`, { name: file.name })]);
      } finally {
        setReading((r) => r.filter((p) => p.id !== pending.id));
      }
    }
  };

  const send = () => {
    if (empty || blocked || (busy && !canRedirect)) return;
    if (busy) onRedirect(text, attachments);
    else onSend(text, attachments);
    setText('');
    setAttachments([]);
    setErrors([]);
  };

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    // A phone keyboard has no Shift+Enter. Return inserts a line; Send is the button.
    if (!finePointer()) return;
    if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      send();
    }
  };

  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setDragging(false);
    void addFiles(Array.from(e.dataTransfer.files));
  };

  const wasBusy = useRef(false);
  useEffect(() => {
    const ended = wasBusy.current && !busy;
    wasBusy.current = busy;
    if (!ended || !finePointer()) return;
    const el = document.activeElement;
    if (el instanceof HTMLElement && el !== document.body && !el.closest('.composer')) return;
    inputRef.current?.focus();
  }, [busy, inputRef]);

  const visionReason = vision === 'unavailable' ? t('vision.unavailable') : t('vision.unknown');
  const problemText = assessment.problems.map((p) =>
    p.kind === 'too_many_images'
      ? t('composer.tooManyImages', { max: p.max, count: p.count })
      : p.kind === 'too_large'
        ? t('composer.tooLarge', { size: megabytes(p.bytes) })
        : t('composer.overflow', { tokens: p.tokens, limit: p.limit }),
  );

  return (
    <form
      className={`composer${dragging ? ' dragging' : ''}`}
      onSubmit={(e) => {
        e.preventDefault();
        send();
      }}
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={onDrop}
    >
      {dragging && <p className="drop-hint">{t('composer.drop')}</p>}
      {errors.length > 0 && (
        <ul className="errors" role="alert">
          {errors.map((e, i) => (
            <li key={i}>{e}</li>
          ))}
        </ul>
      )}
      {(attachments.length > 0 || reading.length > 0) && (
        <ul className="attachments" aria-label={t('composer.attachments')}>
          {attachments.map((a) => (
            <AttachmentItem
              key={a.id}
              a={a}
              info={info}
              onRemove={() => setAttachments((list) => list.filter((x) => x.id !== a.id))}
              onRange={(range) => setAttachments((list) => list.map((x) => (x.id === a.id && x.kind === 'pdf' ? { ...x, range } : x)))}
            />
          ))}
          {reading.map((p) => (
            <li key={p.id} className="attachment muted">
              {t('composer.reading', { name: p.name })}
            </li>
          ))}
        </ul>
      )}
      <label htmlFor="composer-input" className="sr-only">
        {t('composer.label')}
      </label>
      <textarea
        id="composer-input"
        ref={inputRef}
        rows={3}
        value={text}
        placeholder={finePointer() ? t('composer.placeholder') : t('composer.placeholderTouch')}
        enterKeyHint={finePointer() ? 'send' : 'enter'}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={onKeyDown}
      />
      {!empty && (
        <div className="estimate" aria-live="polite">
          <p>
            {assessment.imagesUnknown
              ? t('composer.totalUnknown', { n: assessment.newTokens })
              : t('composer.total', { n: assessment.newTokens, wait: duration(assessment.waitSeconds) })}
          </p>
          {assessment.waitSeconds > WARN_WAIT_SECONDS && <p className="warn">{t('composer.waitWarn', { wait: duration(assessment.waitSeconds) })}</p>}
          {problemText.map((p) => (
            <p key={p} className="error" role="alert">
              {p}
            </p>
          ))}
        </div>
      )}
      <div className="composer-bar">
        <input
          ref={fileRef}
          type="file"
          multiple
          hidden
          accept={visionOk ? ACCEPT_ATTRIBUTE : ACCEPT_ATTRIBUTE.split(',').filter((e) => !['.png', '.jpg', '.jpeg', '.webp'].includes(e)).join(',')}
          onChange={(e) => {
            void addFiles(Array.from(e.target.files ?? []));
            e.target.value = '';
          }}
          data-testid="file-input"
        />
        <input
          ref={imageRef}
          type="file"
          multiple
          hidden
          accept="image/png,image/jpeg,image/webp"
          onChange={(e) => {
            void addFiles(Array.from(e.target.files ?? []));
            e.target.value = '';
          }}
        />
        <button type="button" onClick={() => fileRef.current?.click()}>
          {t('composer.attach')}
        </button>
        <button type="button" onClick={() => imageRef.current?.click()} disabled={!visionOk} aria-describedby={visionOk ? undefined : 'image-reason'}>
          {t('composer.attachImage')}
        </button>
        {!visionOk && (
          <span id="image-reason" className="sr-only">
            {visionReason}
          </span>
        )}
        <span className="spacer" />
        {busy ? (
          <>
            {canRedirect && (
              <button type="submit" className="primary" disabled={empty || blocked} aria-describedby="redirect-hint">
                {t('composer.redirect')}
              </button>
            )}
            {canRedirect && (
              <span id="redirect-hint" className="sr-only">
                {t('composer.redirectHint')}
              </span>
            )}
            <button type="button" className="danger" onClick={onStop}>
              {t('composer.stop')}
            </button>
          </>
        ) : (
          <button type="submit" className="primary" disabled={empty || blocked}>
            {t('composer.send')}
          </button>
        )}
      </div>
    </form>
  );
}

function AttachmentItem({ a, info, onRemove, onRange }: { a: Attachment; info: ServerInfo | null; onRemove: () => void; onRange: (r: string) => void }) {
  const { t, tp } = useI18n();
  const est = estimateAttachment(a, info);
  const visionOk = info?.vision === 'ok';
  const sel = a.kind === 'pdf' ? selectPdf(a, visionOk) : null;
  const rangeId = `range-${a.id}`;
  return (
    <li className="attachment">
      <div className="attachment-head">
        {a.kind === 'image' && <img src={a.dataUrl} alt="" width={40} height={40} className="thumb" />}
        <span className="attachment-name">{a.name}</span>
        <span className="muted">{est.tokens === null ? t('composer.tokensUnknown') : t('composer.tokens', { n: est.tokens })}</span>
        <button type="button" onClick={onRemove} aria-label={t('composer.remove', { name: a.name })}>
          ×
        </button>
      </div>
      <Preview label={t('preview.show', { name: a.name })}>
        {a.kind === 'image' && <PreviewImage src={a.dataUrl} label={a.name} />}
        {a.kind === 'text' && <PreviewText text={a.text} />}
        {a.kind === 'pdf' && sel && (
          <>
            {sel.text && <PreviewText text={sel.text} />}
            {sel.images.map((img, i) => (
              <PreviewImage key={i} src={img.dataUrl} label={t('preview.page', { name: a.name })} />
            ))}
          </>
        )}
        {a.kind === 'pdf' && !sel && <p className="error">{t('composer.pdfRangeInvalid')}</p>}
      </Preview>
      {a.kind === 'pdf' && (
        <div className="pdf-range">
          <label htmlFor={rangeId}>{t('composer.pdfPages', { name: a.name })}</label>
          <input
            id={rangeId}
            value={a.range}
            onChange={(e) => onRange(e.target.value)}
            aria-describedby={`${rangeId}-hint`}
            aria-invalid={sel === null}
          />
          <span id={`${rangeId}-hint`} className="hint">
            {tp('composer.pdfPagesHint', a.pages.length)}
          </span>
          {sel === null && <p className="error">{t('composer.pdfRangeInvalid')}</p>}
          {sel && sel.images.length > 0 && <p className="hint">{tp('composer.pdfImages', sel.images.length)}</p>}
          {sel && sel.skipped.length > 0 && <p className="warn">{t('composer.pdfSkipped', { pages: sel.skipped.join(', ') })}</p>}
        </div>
      )}
    </li>
  );
}
