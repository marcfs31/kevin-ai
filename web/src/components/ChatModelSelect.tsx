import { useId } from 'react';
import { useI18n } from '../i18n';
import type { ModelEntry } from '../lib/api';

/** The model one conversation is sent to, each with the context window the server reports for it. */
export function ChatModelSelect({ models, value, disabled, onChange }: {
  models: ModelEntry[];
  value: string;
  disabled: boolean;
  onChange: (id: string) => void;
}) {
  const { t } = useI18n();
  const id = useId();
  return (
    <span className="chat-model">
      <label htmlFor={id}>{t('chatModel.label')}</label>
      <select id={id} value={value} disabled={disabled} onChange={(e) => onChange(e.target.value)}>
        {models.map((m) => (
          <option key={m.id} value={m.id}>
            {m.contextLength === null ? t('chatModel.optionNoContext', { model: m.id }) : t('chatModel.option', { model: m.id, n: m.contextLength })}
          </option>
        ))}
      </select>
    </span>
  );
}
