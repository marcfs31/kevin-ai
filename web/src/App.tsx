import { useCallback, useEffect, useRef, useState, useSyncExternalStore } from 'react';
import { Composer } from './components/Composer';
import { MessageView, errorText } from './components/MessageView';
import { ChatStats } from './components/ChatStats';
import { ChatModelSelect } from './components/ChatModelSelect';
import { ModelCtxPicker } from './components/ModelCtxPicker';
import { ServerPanel } from './components/ServerPanel';
import { SettingsPanel } from './components/SettingsPanel';
import { Sidebar } from './components/Sidebar';
import { I18nProvider, detectLocale, useI18n, type Locale } from './i18n';
import { ApiError, fetchServerInfo, infoForModel, resetServer, streamChat, type ServerInfo } from './lib/api';
import { buildUserContent, newId, type Attachment } from './lib/attachments';
import { AUTO_RATIO, applyCompaction, planCompaction, projectCompaction, shouldCompact, summaryRequest, type Plan } from './lib/compact';
import { runAgent, type LiveInfo } from './lib/agent';
import { assess } from './lib/request';
import { estimateWireTokens, liveCount } from './lib/tokens';
import { isNearBottom } from './lib/scroll';
import {
  backupKeys, createSaver, deleteBackups, lastSummaryIndex, loadState, STORAGE_KEY, toolResults, truncateFrom, type AssistantMessage, type Conversation, type LoadResult, type Settings, type State,
  type Message, type UserMessage,
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
  const [forgetFailed, setForgetFailed] = useState<ApiError | null>(null);
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
  // Busy from send to the end of the reply (and any tool or compaction step in between).
  const [busy, setBusy] = useState(false);
  // The chat the running reply belongs to: it can be left and come back to while it runs.
  const [runConv, setRunConv] = useState<string | null>(null);
  const [compacting, setCompacting] = useState(false);
  // Belongs to one chat: its banner and Retry show only there, so Retry never compacts another chat.
  const [compactError, setCompactError] = useState<{ convId: string; error: ApiError } | null>(null);
  const serverInfo = server.kind === 'ok' ? server.info : null;
  const active = state.conversations.find((c) => c.id === state.activeId) ?? null;
  // The model is per conversation; before the first message it is the pick for the next new one.
  const [draftModel, setDraftModel] = useState<string | null>(null);
  const modelId = active ? active.model : draftModel;
  // Context window and vision follow the model chosen, so everything below reads `info`.
  const info = infoForModel(serverInfo, modelId);
  const requestModel = serverInfo && serverInfo.models.length > 1 ? (info?.model ?? null) : null;

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
    (typed: string, attachments: Attachment[]) => {
      const content = buildUserContent(typed, attachments, info?.vision === 'ok');
      const history = active?.messages ?? [];
      const sent = state.settings.autoCompact
        ? projectCompaction(history, state.settings.systemPrompt, estimateWireTokens([{ role: 'user', content }], info), info)
        : history;
      return assess(state.settings, sent, typed, attachments, content, info);
    },
    [state.settings, active?.messages, info],
  );

  // Asks the model for a summary of the plan's span and inserts it. Returns the new history, or
  // null when it failed or was stopped: then nothing changed and every message is still sent.
  const compact = async (convId: string, history: Message[], plan: Plan, auto: boolean, signal: AbortSignal): Promise<Message[] | null> => {
    setCompacting(true);
    let text = '';
    try {
      await streamChat({ messages: summaryRequest(plan, info), temperature: 0.2, maxTokens: null, model: requestModel }, signal, (e) => {
        if (e.kind === 'content') text += e.text;
      });
      if (!text.trim()) throw new ApiError(0, { message: '', code: 'summary_empty', type: null });
    } catch (e) {
      if (!(e instanceof DOMException && e.name === 'AbortError')) setCompactError({ convId, error: e instanceof ApiError ? e : new ApiError(0, null) });
      return null;
    } finally {
      setCompacting(false);
    }
    const id = newId();
    const now = Date.now();
    updateConv(convId, (c) => ({ ...c, messages: applyCompaction(c.messages, plan, text, id, now, auto) }));
    return applyCompaction(history, plan, text, id, now, auto);
  };

  const compactNow = async () => {
    if (controllerRef.current || !active) return;
    const plan = planCompaction(active.messages);
    if (!plan) return;
    const controller = new AbortController();
    controllerRef.current = controller;
    setBusy(true);
    setRunConv(active.id);
    setCompactError(null);
    try {
      await compact(active.id, active.messages, plan, false, controller.signal);
    } finally {
      controllerRef.current = null;
      setBusy(false);
      setRunConv(null);
    }
  };

  // Runs the model on `history` (ending with the user's message, or with the step to resume from)
  // and commits what comes back into the chat. `prior` is the saved chat before the new message.
  // With automatic compacting on and the chat near the limit, the older turns are summarised first.
  // Returns the chat as it stands when the run ends: the history it started from plus every
  // message it committed (a reply that was stopped included), which is what a redirect continues from.
  const run = async (convId: string, prior: Message[], history: Message[], newTokens: number): Promise<Message[]> => {
    const controller = new AbortController();
    controllerRef.current = controller;
    setBusy(true);
    setRunConv(convId);
    setCompactError(null);
    const auto = state.settings.autoCompact;
    try {
      if (auto && shouldCompact(prior, state.settings.systemPrompt, newTokens, info)) {
        const plan = planCompaction(prior);
        if (plan) history = (await compact(convId, history, plan, true, controller.signal)) ?? history;
      }
      let current = history;
      await runAgent({
        history,
        settings: state.settings,
        model: requestModel,
        info,
        signal: controller.signal,
        stream: streamChat,
        compact: auto
          ? async (h, plan) => {
              const next = await compact(convId, h, plan, true, controller.signal);
              if (next) current = next;
              return next;
            }
          : undefined,
        onLive: (message, liveInfo) => setLive({ convId, message, info: liveInfo }),
        onCommit: (m) => {
          current = [...current, m];
          updateConv(convId, (c) => ({ ...c, updatedAt: Date.now(), messages: [...c.messages, m] }));
          setLive(null);
        },
      });
      return current;
    } finally {
      controllerRef.current = null;
      setLive(null);
      setBusy(false);
      setRunConv(null);
    }
  };

  const runRef = useRef<Promise<Message[]> | null>(null);
  const startRun = (convId: string, prior: Message[], history: Message[], newTokens: number) => {
    const p = run(convId, prior, history, newTokens);
    runRef.current = p;
    return p;
  };

  // The last reply failed (server down, stream cut): drop it and ask again from the same point,
  // so once the server is back one click carries on, tool steps already run included.
  const failedLast = active?.messages[active.messages.length - 1];
  const canRetry = !busy && failedLast?.role === 'assistant' && failedLast.status === 'error';
  const retry = () => {
    if (controllerRef.current || !active || !canRetry || failedLast?.role !== 'assistant') return;
    const prior = active.messages.slice(0, -1);
    stickRef.current = true;
    updateConv(active.id, (c) => ({ ...c, updatedAt: Date.now(), messages: c.messages.filter((m) => m.id !== failedLast.id) }));
    void startRun(active.id, prior, prior, 0);
  };

  // After deleting a reply, the chat can end on a user message: regenerate asks again from there.
  const canRegenerate = !busy && failedLast?.role === 'user';
  const regenerate = () => {
    if (controllerRef.current || !active || !canRegenerate) return;
    stickRef.current = true;
    void startRun(active.id, active.messages, active.messages, 0);
  };

  // Rewind: drop this message and everything after, and clear the server's reused state.
  const deleteFrom = (index: number) => {
    if (busy || !active) return;
    const target = active.messages[index];
    if (!target || target.role === 'tool') return;
    if (!window.confirm(t('message.confirmDelete'))) return;
    updateConv(active.id, (c) => ({ ...c, updatedAt: Date.now(), messages: truncateFrom(c.messages, index) }));
    forgetServer();
  };

  const send = async (typed: string, attachments: Attachment[], target: Conversation | null = active) => {
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
    let conv = target;
    if (!conv) {
      conv = { id: newId(), title: titleFrom(typed, attachments), createdAt: now, updatedAt: now, model: requestModel ?? draftModel, messages: [] };
      const created = conv;
      setState((s) => ({ ...s, conversations: [...s.conversations, created], activeId: created.id }));
    }
    const convId = conv.id;
    updateConv(convId, (c) => ({ ...c, title: c.title || titleFrom(typed, attachments), updatedAt: now, messages: [...c.messages, user] }));

    await startRun(convId, conv.messages, [...conv.messages, user], estimateWireTokens([{ role: 'user', content: user.content }], info));
  };

  // Steering a reply: the message typed while it streams stops it, keeps what was already written
  // (saved as a stopped reply) and is sent next, so the model carries on from there.
  const [redirecting, setRedirecting] = useState(false);
  const redirectNow = async (typed: string, attachments: Attachment[]) => {
    const pending = runRef.current;
    if (!controllerRef.current || !active || runConv !== active.id || !pending || redirecting) return;
    const conv = active;
    setRedirecting(true);
    stop();
    let messages: Message[];
    try {
      messages = await pending;
    } catch {
      setRedirecting(false);
      return;
    }
    // Lifted before the new run starts, so that run can be redirected in turn.
    setRedirecting(false);
    await send(typed, attachments, { ...conv, messages });
  };

  const updateSettings = (settings: Settings) => setState((s) => ({ ...s, settings }));

  const forgetServer = () => {
    void resetServer().then(
      () => setForgetFailed(null),
      (e: unknown) => setForgetFailed(e instanceof ApiError ? e : new ApiError(0, null)),
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
  const lastSummary = lastSummaryIndex(messages);
  const canCompact = active !== null && planCompaction(active.messages) !== null;

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
          {server.kind === 'error' && <ModelCtxPicker modelName={null} contextLength={null} busy={busy} onStarted={onServeStarted} />}
          {serverInfo && serverInfo.models.length > 1 && (
            <ChatModelSelect
              models={serverInfo.models}
              value={info?.model ?? ''}
              disabled={busy}
              onChange={(id) => (active ? updateConv(active.id, (c) => ({ ...c, model: id })) : setDraftModel(id))}
            />
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
          generatingId={runConv}
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
            {errorText(t, forgetFailed)}
            <button type="button" onClick={forgetServer} disabled={busy}>
              {t('server.retry')}
            </button>
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
            {messages.map((m, i) => (
              <div key={m.id} className={i < lastSummary ? 'compacted' : undefined}>
                <MessageView
                  m={m}
                  results={m.role === 'assistant' ? toolResults(messages, i) : undefined}
                  onRetry={canRetry && i === messages.length - 1 ? retry : undefined}
                  onRegenerate={canRegenerate && i === messages.length - 1 ? regenerate : undefined}
                  onDelete={!busy && m.role !== 'tool' ? () => deleteFrom(i) : undefined}
                />
              </div>
            ))}
          </div>
          {live && live.convId === active?.id && <MessageView m={live.message} live={live.info} />}
        </div>
        {compacting && (
          <p className="note" role="status">
            {t('compact.running')}
          </p>
        )}
        {compactError && compactError.convId === active?.id && (
          <p className="error banner" role="alert">
            {t('compact.failed', { reason: errorText(t, compactError.error) })}
            {canCompact && (
              <button type="button" onClick={() => void compactNow()} disabled={busy}>
                {t('server.retry')}
              </button>
            )}
          </p>
        )}
        <ChatStats
          messages={messages}
          systemPrompt={state.settings.systemPrompt}
          autoCompact={state.settings.autoCompact}
          info={info}
          live={live && live.convId === active?.id ? liveCount(live.info.usage, live.info.promptEstimate, live.info.chunks) : null}
        />
        <div className="chat-tools" role="group" aria-label={t('compact.group')}>
          <label className="check">
            <input
              type="checkbox"
              checked={state.settings.autoCompact}
              aria-describedby="compact-hint"
              onChange={(e) => updateSettings({ ...state.settings, autoCompact: e.target.checked })}
            />
            {t('compact.auto')}
          </label>
          <span className="spacer" />
          <button type="button" onClick={() => void compactNow()} disabled={busy || !canCompact} aria-describedby="compact-hint">
            {t('compact.button')}
          </button>
          <span id="compact-hint" className="sr-only">
            {state.settings.autoCompact ? t('compact.hintAuto', { at: Math.round(AUTO_RATIO * 100) }) : t('compact.hintManual')}
          </span>
        </div>
        {busy && runConv !== null && runConv !== active?.id && (
          <p className="note" role="status">
            {t('app.busyElsewhere')}
          </p>
        )}
        <Composer busy={busy} info={info} imagePreset={state.settings.imagePreset} assessDraft={assessDraft} onSend={(tx, a) => void send(tx, a)} canRedirect={busy && runConv !== null && runConv === active?.id && !compacting && !redirecting} onRedirect={(tx, a) => void redirectNow(tx, a)} onStop={stop} inputRef={inputRef} />
      </main>
    </div>
  );
}
