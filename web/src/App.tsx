import { useCallback, useEffect, useRef, useState, useSyncExternalStore } from 'react';
import { Composer } from './components/Composer';
import { MessageView, errorText, type LiveInfo } from './components/MessageView';
import { TokenLine } from './components/TokenLine';
import { ModelCtxPicker } from './components/ModelCtxPicker';
import { ServerPanel } from './components/ServerPanel';
import { SettingsPanel } from './components/SettingsPanel';
import { Sidebar } from './components/Sidebar';
import { I18nProvider, detectLocale, useI18n, type Locale } from './i18n';
import { ApiError, fetchServerInfo, resetServer, streamChat, type ServerInfo } from './lib/api';
import { buildUserContent, newId, type Attachment } from './lib/attachments';
import { assess, toWire } from './lib/request';
import { estimateWireTokens, sessionTotals } from './lib/tokens';
import { isNearBottom } from './lib/scroll';
import type { Usage } from './lib/sse';
import {
  backupKeys, createSaver, deleteBackups, loadState, STORAGE_KEY, type AssistantMessage, type Conversation, type LoadResult, type Settings, type State,
  type UserMessage,
} from './lib/storage';

type ServerState = { kind: 'loading' } | { kind: 'ok'; info: ServerInfo } | { kind: 'error'; error: ApiError };

interface Live {
  convId: string;
  message: AssistantMessage;
  info: LiveInfo;
}

export interface AppProps {
  storage?: Storage;
  languages?: readonly string[];
}

export function App({ storage = window.localStorage, languages = navigator.languages }: AppProps) {
  const [loaded] = useState<LoadResult>(() => loadState(storage));
  const [state, setState] = useState<State>(loaded.state);
  const locale: Locale = state.settings.locale === 'auto' ? detectLocale(languages) : state.settings.locale;
  useEffect(() => {
    document.documentElement.lang = locale;
  }, [locale]);
  return (
    <I18nProvider locale={locale}>
      <Chat state={state} setState={setState} storage={storage} notice={loaded.notice} />
    </I18nProvider>
  );
}

function titleFrom(typed: string, attachments: Attachment[]): string {
  const line = typed.trim().split('\n')[0] ?? '';
  const base = line || attachments[0]?.name || '';
  return base.length > 60 ? `${base.slice(0, 57)}…` : base;
}

function Chat({ state, setState, storage, notice }: {
  state: State;
  setState: React.Dispatch<React.SetStateAction<State>>;
  storage: Storage;
  notice: LoadResult['notice'];
}) {
  const { t, tp } = useI18n();
  const [hold, setHold] = useState(notice);
  const [copies, setCopies] = useState(() => backupKeys(storage));
  const [forgetFailed, setForgetFailed] = useState(false);
  const [server, setServer] = useState<ServerState>({ kind: 'loading' });
  const [live, setLive] = useState<Live | null>(null);
  const [saver] = useState(() => createSaver(storage));
  const saveResult = useSyncExternalStore(saver.subscribe, saver.get);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const controllerRef = useRef<AbortController | null>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const logRef = useRef<HTMLDivElement>(null);
  const stickRef = useRef(true);
  const busy = live !== null;
  const info = server.kind === 'ok' ? server.info : null;
  const active = state.conversations.find((c) => c.id === state.activeId) ?? null;

  const fetchInfo = useCallback((signal?: AbortSignal) => {
    fetchServerInfo(signal).then(
      (i) => setServer({ kind: 'ok', info: i }),
      (e: unknown) => {
        if (e instanceof ApiError) setServer({ kind: 'error', error: e });
      },
    );
  }, []);
  useEffect(() => {
    const controller = new AbortController();
    fetchInfo(controller.signal);
    return () => controller.abort();
  }, [fetchInfo]);
  const retryServer = () => {
    setServer({ kind: 'loading' });
    fetchInfo();
  };

  // Streamed tokens go to `live`, not `state`, so this saves the user's message as soon as it
  // is sent and the reply once it ends, never once per token.
  useEffect(() => {
    if (hold !== 'locked') saver.save(state);
  }, [state, saver, hold]);

  useEffect(() => {
    stickRef.current = true;
  }, [state.activeId]);

  useEffect(() => {
    if (!stickRef.current) return;
    const log = logRef.current;
    if (!log) return;
    log.scrollTop = log.scrollHeight;
  }, [live?.message.content, live?.message.reasoning, active?.messages.length]);

  const stop = useCallback(() => controllerRef.current?.abort(), []);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && controllerRef.current) {
        e.preventDefault();
        stop();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [stop]);

  const updateConv = (id: string, f: (c: Conversation) => Conversation) =>
    setState((s) => ({ ...s, conversations: s.conversations.map((c) => (c.id === id ? f(c) : c)) }));

  const assessDraft = useCallback(
    (typed: string, attachments: Attachment[]) =>
      assess(state.settings, active?.messages ?? [], typed, attachments, buildUserContent(typed, attachments, info?.vision === 'ok'), info),
    [state.settings, active?.messages, info],
  );

  const send = async (typed: string, attachments: Attachment[]) => {
    if (controllerRef.current) return;
    stickRef.current = true;
    const now = Date.now();
    const user: UserMessage = {
      id: newId(),
      role: 'user',
      content: buildUserContent(typed, attachments, info?.vision === 'ok'),
      typed,
      attachmentNames: attachments.map((a) => a.name),
      createdAt: now,
    };
    let conv = active;
    if (!conv) {
      conv = { id: newId(), title: titleFrom(typed, attachments), createdAt: now, updatedAt: now, messages: [] };
      const created = conv;
      setState((s) => ({ ...s, conversations: [...s.conversations, created], activeId: created.id }));
    }
    const convId = conv.id;
    const history = [...conv.messages, user];
    updateConv(convId, (c) => ({ ...c, title: c.title || titleFrom(typed, attachments), updatedAt: now, messages: [...c.messages, user] }));

    const controller = new AbortController();
    controllerRef.current = controller;
    const message: AssistantMessage = {
      id: newId(), role: 'assistant', content: '', reasoning: '', status: 'done', error: null, stats: null, createdAt: now,
    };
    const wire = toWire(state.settings.systemPrompt, history);
    const promptEstimate = estimateWireTokens(wire, info);
    const liveInfo: LiveInfo = { startedAt: performance.now(), firstAt: null, chunks: 0, promptEstimate, usage: null };
    let usage: Usage | null = null;
    const publish = () => setLive({ convId, message: { ...message }, info: { ...liveInfo } });
    publish();
    try {
      await streamChat(
        { messages: wire, temperature: state.settings.temperature, maxTokens: state.settings.maxTokens },
        controller.signal,
        (e) => {
          if (e.kind === 'content' || e.kind === 'reasoning') {
            if (liveInfo.firstAt === null) liveInfo.firstAt = performance.now();
            liveInfo.chunks += 1;
            if (e.kind === 'content') message.content += e.text;
            else message.reasoning += e.text;
            publish();
          } else if (e.kind === 'usage') {
            usage = e.usage;
            liveInfo.usage = e.usage;
            publish();
          }
        },
      );
    } catch (e) {
      if (e instanceof DOMException && e.name === 'AbortError') message.status = 'stopped';
      else {
        const err = e instanceof ApiError ? e : new ApiError(0, null);
        message.status = 'error';
        message.error = { code: err.code, message: err.serverMessage, status: err.status };
      }
    } finally {
      const end = performance.now();
      const u = usage as Usage | null;
      const generated = u?.completionTokens ?? liveInfo.chunks;
      const decodeSecs = liveInfo.firstAt === null ? 0 : (end - liveInfo.firstAt) / 1000;
      message.stats = {
        ttftMs: liveInfo.firstAt === null ? null : liveInfo.firstAt - liveInfo.startedAt,
        tokensPerSecond: generated > 1 && decodeSecs > 0 ? (generated - 1) / decodeSecs : null,
        promptTokens: u?.promptTokens ?? null,
        cachedTokens: u?.cachedTokens ?? null,
        completionTokens: u?.completionTokens ?? null,
        promptEstimate,
        completionEstimate: liveInfo.chunks,
      };
      const done = { ...message };
      updateConv(convId, (c) => ({ ...c, updatedAt: Date.now(), messages: [...c.messages, done] }));
      controllerRef.current = null;
      setLive(null);
    }
  };

  const updateSettings = (settings: Settings) => setState((s) => ({ ...s, settings }));

  const forgetServer = () => {
    void resetServer().then(
      () => setForgetFailed(false),
      () => setForgetFailed(true),
    );
  };

  const discardCopies = () => {
    const n = Math.max(1, backupKeys(storage).length + (hold === 'locked' ? 1 : 0));
    if (!window.confirm(tp('sidebar.confirmDeleteAll', n))) return;
    deleteBackups(storage);
    if (hold === 'locked') storage.removeItem(STORAGE_KEY);
    setCopies([]);
    setHold('none');
  };

  const serverLine =
    server.kind === 'loading' ? t('server.checking') : server.kind === 'error' ? errorText(t, server.error) : null;
  const onServeStarted = () => {
    setServer({ kind: 'loading' });
    fetchInfo();
  };
  // When images are off, the composer says why next to the disabled button.
  const visionLine = info?.vision === 'ok' ? t('vision.ok') : null;
  const storageError =
    hold === 'locked'
      ? t('error.storageLocked')
      : saveResult === 'full'
        ? t('error.storageFull')
        : hold === 'unreadable' || copies.length > 0
          ? t('error.storageUnreadable')
          : null;
  const canDiscard = hold === 'locked' || hold === 'unreadable' || copies.length > 0;
  const messages = active?.messages ?? [];
  const totals = sessionTotals(messages);

  return (
    <div className={`app${sidebarOpen ? ' sidebar-open' : ''}`}>
      <a className="skip" href="#composer-input">
        {t('app.skip')}
      </a>
      <header className="top">
        <div className="brand">
          <button
            type="button"
            className="sidebar-toggle"
            aria-expanded={sidebarOpen}
            aria-controls="sidebar"
            aria-label={sidebarOpen ? t('sidebar.hide') : t('sidebar.show')}
            onClick={() => setSidebarOpen((o) => !o)}
          >
            {sidebarOpen ? t('sidebar.hideShort') : t('sidebar.showShort')}
          </button>
          <h1>{t('app.title')}</h1>
        </div>
        <div className={`server server-${server.kind}`} role="status">
          {server.kind === 'ok' ? (
            <ModelCtxPicker
              modelName={server.info.model}
              contextLength={server.info.contextLength}
              busy={busy}
              onStarted={onServeStarted}
            />
          ) : (
            serverLine && <span>{serverLine}</span>
          )}
          {info?.contextLength && <span>{t('server.context', { n: info.contextLength })}</span>}
          {server.kind === 'error' && (
            <button type="button" onClick={retryServer}>
              {t('server.retry')}
            </button>
          )}
        </div>
        <button type="button" className="header-action" aria-expanded={settingsOpen} aria-controls="settings" onClick={() => setSettingsOpen((o) => !o)}>
          {t('settings.open')}
        </button>
      </header>
      <aside id="sidebar" className="side">
        <Sidebar
          conversations={state.conversations}
          activeId={state.activeId}
          busy={busy}
          onSelect={(id) => {
            setState((s) => ({ ...s, activeId: id }));
            setSidebarOpen(false);
          }}
          onNew={() => {
            setState((s) => ({ ...s, activeId: null }));
            setSidebarOpen(false);
          }}
          onRename={(id, title) => updateConv(id, (c) => ({ ...c, title }))}
          onDelete={(id) => {
            setState((s) => ({ ...s, conversations: s.conversations.filter((c) => c.id !== id), activeId: s.activeId === id ? null : s.activeId }));
            forgetServer();
          }}
          onDeleteAll={() => {
            deleteBackups(storage);
            setCopies([]);
            setState((s) => ({ ...s, conversations: [], activeId: null }));
            forgetServer();
          }}
        />
      </aside>
      <main className="main">
        {settingsOpen && (
          <div id="settings" className="panels">
            <ServerPanel busy={busy} runningModel={info?.model ?? null} onStarted={onServeStarted} />
            <SettingsPanel settings={state.settings} onChange={updateSettings} />
          </div>
        )}
        {storageError && (
          <p className="error banner" role="alert">
            {storageError}
            {canDiscard && (
              <button type="button" className="danger" onClick={discardCopies}>
                {t('sidebar.deleteAll')}
              </button>
            )}
          </p>
        )}
        {forgetFailed && (
          <p className="error banner" role="alert">
            {t('error.network')}
          </p>
        )}
        <div
          className="log"
          ref={logRef}
          tabIndex={0}
          role="region"
          aria-label={t('app.transcript')}
          onScroll={(e) => {
            stickRef.current = isNearBottom(e.currentTarget);
          }}
        >
          <div className="log-entries" role="log" aria-busy={busy}>
            {messages.length === 0 && !live && <p className="empty">{t('app.empty')}</p>}
            {messages.map((m) => (
              <MessageView key={m.id} m={m} />
            ))}
          </div>
          {live && live.convId === active?.id && <MessageView m={live.message} live={live.info} />}
        </div>
        {totals && <TokenLine className="session hint" labelKey="tokens.session" count={totals} />}
        {visionLine && <p className="vision hint">{visionLine}</p>}
        <Composer busy={busy} info={info} imagePreset={state.settings.imagePreset} assessDraft={assessDraft} onSend={(tx, a) => void send(tx, a)} onStop={stop} inputRef={inputRef} />
      </main>
    </div>
  );
}
