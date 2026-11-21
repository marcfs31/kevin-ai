import { useEffect, useRef, useState, type FormEvent } from 'react';
import { useI18n } from '../i18n';
import type { Conversation } from '../lib/storage';

interface Props {
  conversations: Conversation[];
  activeId: string | null;
  /** A reply, tool step or summary is running. Only deleting waits for it: that resets the server's cache. */
  busy: boolean;
  /** The chat the running reply belongs to; it can be left and come back to. */
  generatingId: string | null;
  onSelect: (id: string) => void;
  onNew: () => void;
  onRename: (id: string, title: string) => void;
  onDelete: (id: string) => void;
  onDeleteAll: () => void;
}

export function Sidebar({ conversations, activeId, busy, generatingId, onSelect, onNew, onRename, onDelete, onDeleteAll }: Props) {
  const { t, tp, date } = useI18n();
  const [editing, setEditing] = useState<string | null>(null);
  const [draft, setDraft] = useState('');
  const restoreFocus = useRef<string | null>(null);
  const sorted = [...conversations].sort((a, b) => b.updatedAt - a.updatedAt);

  const finishEdit = (id: string) => {
    restoreFocus.current = id;
    setEditing(null);
  };
  useEffect(() => {
    const id = restoreFocus.current;
    if (!id || editing !== null) return;
    restoreFocus.current = null;
    document.getElementById(`rename-btn-${id}`)?.focus();
  }, [editing]);

  const submit = (e: FormEvent, id: string) => {
    e.preventDefault();
    if (draft.trim()) onRename(id, draft.trim());
    finishEdit(id);
  };

  return (
    <nav className="sidebar" aria-label={t('sidebar.label')}>
      <button type="button" className="primary" onClick={onNew}>
        {t('sidebar.new')}
      </button>
      {sorted.length === 0 ? (
        <p className="muted">{t('sidebar.none')}</p>
      ) : (
        <ul className="conv-list">
          {sorted.map((c) => {
            const title = c.title || t('sidebar.untitled');
            return (
              <li key={c.id} className={c.id === activeId ? 'active' : undefined}>
                {editing === c.id ? (
                  <form onSubmit={(e) => submit(e, c.id)} className="rename">
                    <label className="sr-only" htmlFor={`rename-${c.id}`}>
                      {t('sidebar.renameLabel', { title })}
                    </label>
                    <input id={`rename-${c.id}`} value={draft} onChange={(e) => setDraft(e.target.value)} autoFocus maxLength={200} />
                    <button type="submit">{t('sidebar.save')}</button>
                    <button type="button" onClick={() => finishEdit(c.id)}>
                      {t('sidebar.cancel')}
                    </button>
                  </form>
                ) : (
                  <>
                    <button
                      type="button"
                      className="conv"
                      aria-current={c.id === activeId ? 'page' : undefined}
                      onClick={() => onSelect(c.id)}
                    >
                      <span className="conv-title">{title}</span>
                      {c.id === generatingId && <span className="conv-live">{t('sidebar.generating')}</span>}
                      <span className="conv-date">{date(c.updatedAt)}</span>
                    </button>
                    <div className="conv-actions">
                      <button
                        type="button"
                        id={`rename-btn-${c.id}`}
                        aria-label={t('sidebar.renameNamed', { title })}
                        onClick={() => {
                          setDraft(c.title);
                          setEditing(c.id);
                        }}
                      >
                        {t('sidebar.rename')}
                      </button>
                      <button
                        type="button"
                        aria-label={t('sidebar.deleteNamed', { title })}
                        disabled={busy}
                        onClick={() => {
                          if (window.confirm(t('sidebar.confirmDelete', { title }))) onDelete(c.id);
                        }}
                      >
                        {t('sidebar.delete')}
                      </button>
                    </div>
                  </>
                )}
              </li>
            );
          })}
        </ul>
      )}
      {sorted.length > 0 && (
        <button
          type="button"
          className="danger"
          disabled={busy}
          onClick={() => {
            if (window.confirm(tp('sidebar.confirmDeleteAll', sorted.length))) onDeleteAll();
          }}
        >
          {t('sidebar.deleteAll')}
        </button>
      )}
    </nav>
  );
}
