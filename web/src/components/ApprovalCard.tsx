import { useId } from 'react';
import { useI18n } from '../i18n';
import type { ApprovalRequest } from '../lib/toolHost';

export type Answer = 'allow' | 'always' | 'deny';

/** The model asked to run a tool that is not read-only (or the user asked to be asked every time):
 * nothing runs until one of these buttons is pressed; Stop also refuses it. */
export function ApprovalCard({ request, onAnswer }: { request: ApprovalRequest; onAnswer: (a: Answer) => void }) {
  const { t } = useI18n();
  const id = useId();
  return (
    <section className="approval" role="alertdialog" aria-labelledby={`${id}-t`} aria-describedby={`${id}-d`}>
      <h2 id={`${id}-t`}>{t('approval.title', { tool: request.title })}</h2>
      <p id={`${id}-d`} className="hint">
        {request.readOnly ? t('approval.hintReadOnly') : t('approval.hint')}
      </p>
      <p className="hint">{t('approval.args')}</p>
      <pre>{request.args}</pre>
      <div className="approval-actions">
        <button type="button" className="primary" onClick={() => onAnswer('allow')} autoFocus>
          {t('approval.allow')}
        </button>
        <button type="button" onClick={() => onAnswer('always')}>
          {t('approval.always')}
        </button>
        <button type="button" className="danger" onClick={() => onAnswer('deny')}>
          {t('approval.deny')}
        </button>
      </div>
    </section>
  );
}
