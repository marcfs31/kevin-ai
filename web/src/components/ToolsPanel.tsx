import { useCallback, useEffect, useId, useState } from 'react';
import { useI18n } from '../i18n';
import { ApiError, toolsAddCatalog, toolsSearch, toolsStatus, toolsStop, type ToolsStatus } from '../lib/api';
import { estimateWaitSeconds } from '../lib/estimate';
import { builtinSpecs, findToolsSpec, webSearchSpec } from '../lib/toolRegistry';
import { MAX_TOOLS_PER_STEP } from '../lib/toolHost';
import { fitCount } from '../lib/toolSelect';
import type { McpLoad } from '../lib/mcpLoad';
import { APPROVAL_MODES, MAX_TOOL_BUDGET, MIN_TOOL_BUDGET, type ApprovalMode, type Settings } from '../lib/storage';
import { errorText } from './MessageView';

interface Props {
  settings: Settings;
  onChange: (s: Settings) => void;
  /** What the chat asked of the servers that are on (App owns it, so the tools and this list agree). */
  load: McpLoad | null;
  reload: () => void;
}

type Test = { kind: 'idle' } | { kind: 'running' } | { kind: 'ok'; n: number } | { kind: 'error'; reason: string };

/** Settings > Tools and internet: the MCP servers and their tools, the internet switch and the search
 * address, how much of each request the tool definitions may take, and when to ask first. */
export function ToolsPanel({ settings, onChange, load, reload }: Props) {
  const { t, tp, num, duration } = useI18n();
  const id = useId();
  const cfg = settings.toolsConfig;
  const net = settings.internet;
  const [state, setState] = useState<{ status: ToolsStatus | null; error: ApiError | null }>({ status: null, error: null });
  const [refresh, setRefresh] = useState(0);
  const [test, setTest] = useState<Test>({ kind: 'idle' });
  const [busyId, setBusyId] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    toolsStatus(controller.signal).then(
      (status) => setState({ status, error: null }),
      (e: unknown) => {
        if (!(e instanceof DOMException && e.name === 'AbortError')) setState({ status: null, error: e instanceof ApiError ? e : new ApiError(0, null) });
      },
    );
    return () => controller.abort();
  }, [refresh]);
  const again = useCallback(() => {
    setRefresh((n) => n + 1);
    reload();
  }, [reload]);

  const setTools = (patch: Partial<Settings['toolsConfig']>) => onChange({ ...settings, toolsConfig: { ...cfg, ...patch } });
  const setServer = (sid: string, patch: Partial<{ on: boolean; disabledTools: string[] }>) =>
    setTools({ servers: { ...cfg.servers, [sid]: { on: cfg.servers[sid]?.on ?? false, disabledTools: cfg.servers[sid]?.disabledTools ?? [], ...patch } } });
  const setNet = (patch: Partial<Settings['internet']>) => onChange({ ...settings, internet: { ...net, ...patch } });

  const specs = load?.specs ?? [];
  const enabledSpecs = specs.filter((s) => s.source.kind === 'mcp' && cfg.servers[s.source.server]?.on && !cfg.servers[s.source.server]?.disabledTools.includes(s.source.tool));
  const enabledTokens = enabledSpecs.reduce((n, s) => n + s.tokens, 0);
  const always = [...builtinSpecs(), ...(net.on && net.searchUrl !== '' ? [webSearchSpec()] : []), ...(enabledSpecs.length ? [findToolsSpec()] : [])];
  const fit = fitCount(enabledSpecs, cfg.budgetTokens, MAX_TOOLS_PER_STEP, always.length);

  const act = async (key: string, f: () => Promise<unknown>) => {
    setBusyId(key);
    setActionError(null);
    try {
      await f();
      again();
    } catch (e) {
      setActionError(e instanceof ApiError ? errorText(t, e) : t('error.network'));
    } finally {
      setBusyId(null);
    }
  };

  const runTest = async () => {
    setTest({ kind: 'running' });
    try {
      const r = await toolsSearch('test', net.searchUrl);
      setTest(r.isError ? { kind: 'error', reason: r.content } : { kind: 'ok', n: r.content === 'no results' ? 0 : r.content.split('\n').filter((l) => /^\d+\. /.test(l)).length });
    } catch (e) {
      setTest({ kind: 'error', reason: e instanceof ApiError ? errorText(t, e) : t('error.network') });
    }
  };

  const status = state.status;
  return (
    <section className="settings tools-panel" aria-labelledby={`${id}-h`}>
      <h2 id={`${id}-h`}>{t('tools.title')}</h2>
      {!settings.tools && <p className="hint">{t('tools.needTools')}</p>}

      <div className="field field-wide">
        <h3>{t('tools.service')}</h3>
        {state.error ? (
          <p className="hint" role="status">
            {t('tools.serviceDown', { reason: errorText(t, state.error) })}{' '}
            <button type="button" onClick={again}>
              {t('server.retry')}
            </button>
          </p>
        ) : status ? (
          <p className="hint" role="status">
            {status.configError ? t('tools.configError', { error: status.configError }) : t('tools.serviceOk')}
          </p>
        ) : (
          <p className="hint">{t('server.checking')}</p>
        )}
        {actionError && (
          <p className="error" role="alert">
            {actionError}
          </p>
        )}
      </div>

      <div className="field field-wide">
        <label className="check">
          <input type="checkbox" checked={net.on} aria-describedby={`${id}-net-hint`} onChange={(e) => setNet({ on: e.target.checked })} />
          {t('tools.internet')}
        </label>
        <p id={`${id}-net-hint`} className="hint">
          {t('tools.internetHint')}
        </p>
        <label htmlFor={`${id}-url`}>{t('tools.searchUrl')}</label>
        <input id={`${id}-url`} type="url" value={net.searchUrl} disabled={!net.on} aria-describedby={`${id}-url-hint`} onChange={(e) => setNet({ searchUrl: e.target.value.trim() })} />
        <p id={`${id}-url-hint`} className="hint">
          {t('tools.searchUrlHint')}
        </p>
        <div>
          <button type="button" disabled={!net.on || !net.searchUrl || test.kind === 'running'} onClick={() => void runTest()}>
            {t('tools.testSearch')}
          </button>{' '}
          <span role="status">
            {test.kind === 'ok' && t('tools.testOk', { n: test.n })}
            {test.kind === 'error' && t('tools.testFail', { reason: test.reason })}
          </span>
        </div>
      </div>

      <div className="field field-wide">
        <h3>{t('tools.servers')}</h3>
        {status && status.servers.length === 0 && <p className="hint">{t('tools.noServers')}</p>}
        <ul className="tool-servers">
          {(status?.servers ?? []).map((sv) => {
            const on = cfg.servers[sv.id]?.on ?? false;
            const disabled = cfg.servers[sv.id]?.disabledTools ?? [];
            const mine = specs.filter((s) => s.source.kind === 'mcp' && s.source.server === sv.id);
            const skipped = load?.skipped.find((s) => s.id === sv.id);
            const failed = load?.failed.find((f) => f.id === sv.id);
            return (
              <li key={sv.id}>
                <label className="check">
                  <input type="checkbox" checked={on} disabled={!settings.tools} onChange={(e) => setServer(sv.id, { on: e.target.checked })} />
                  {t('tools.serverOn', { title: sv.title })}
                </label>
                <span className="hint">
                  {' '}
                  {t(`tools.state.${sv.state}`)}
                  {sv.network && ` · ${t('tools.needsInternet')}`}
                </span>
                {sv.error && <p className="error">{sv.error}</p>}
                {on && skipped && <p className="hint">{t(`tools.skipped.${skipped.reason}`)}</p>}
                {on && failed && <p className="error">{t('tools.failed', { message: failed.message })}</p>}
                {sv.state === 'ready' && (
                  <button type="button" disabled={busyId === `stop:${sv.id}`} onClick={() => void act(`stop:${sv.id}`, () => toolsStop(sv.id))}>
                    {t('tools.stop')}
                  </button>
                )}
                {on && mine.length > 0 && (
                  <details>
                    <summary>{tp('tools.count', mine.length)}</summary>
                    <ul className="tool-list">
                      {mine.map((s) => {
                        const tool = s.source.kind === 'mcp' ? s.source.tool : s.name;
                        return (
                          <li key={s.name}>
                            <label className="check">
                              <input
                                type="checkbox"
                                checked={!disabled.includes(tool)}
                                onChange={(e) => setServer(sv.id, { disabledTools: e.target.checked ? disabled.filter((d) => d !== tool) : [...disabled, tool] })}
                              />
                              {t('tools.toolOn', { name: tool })}
                            </label>
                            <span className="hint">
                              {' '}
                              {s.readOnly ? t('tools.readOnly') : t('tools.asksFirst')} · {t('tools.tokens', { n: num(s.tokens) })}
                            </span>
                          </li>
                        );
                      })}
                    </ul>
                  </details>
                )}
              </li>
            );
          })}
        </ul>
      </div>

      {status && status.catalog.some((c) => !c.added) && (
        <div className="field field-wide">
          <h3>{t('tools.catalog')}</h3>
          <p className="hint">{t('tools.catalogHint')}</p>
          <ul className="tool-catalog">
            {status.catalog
              .filter((c) => !c.added)
              .map((c) => (
                <li key={c.id}>
                  <strong>{c.title}</strong> <span className="hint">{c.description}</span>
                  <span className="hint">
                    {c.network && ` · ${t('tools.needsInternet')}`}
                    {!c.verified && ` · ${t('tools.unverified')}`}
                  </span>{' '}
                  <button type="button" disabled={busyId === `add:${c.id}` || (c.network && !net.on)} onClick={() => void act(`add:${c.id}`, () => toolsAddCatalog(c.id, net.on))}>
                    {t('tools.add', { title: c.title })}
                  </button>
                  {c.network && !net.on && <span className="hint"> {t('tools.addNeedsInternet')}</span>}
                </li>
              ))}
          </ul>
        </div>
      )}

      <div className="field">
        <label htmlFor={`${id}-budget`}>{t('tools.budget', { tokens: num(cfg.budgetTokens) })}</label>
        <input
          id={`${id}-budget`}
          type="range"
          min={MIN_TOOL_BUDGET}
          max={MAX_TOOL_BUDGET}
          step={100}
          value={cfg.budgetTokens}
          aria-describedby={`${id}-budget-hint`}
          onChange={(e) => setTools({ budgetTokens: Number(e.target.value) })}
        />
        <p id={`${id}-budget-hint`} className="hint">
          {t('tools.budgetHint', { all: num(enabledTokens), wait: duration(estimateWaitSeconds(Math.min(enabledTokens, cfg.budgetTokens))) })}
        </p>
        {enabledSpecs.length > 0 && (
          <p className="hint" role="status">
            {t('tools.budgetFit', { n: num(fit), total: num(enabledSpecs.length), cap: num(MAX_TOOLS_PER_STEP) })}
          </p>
        )}
      </div>

      <div className="field">
        <label htmlFor={`${id}-approval`}>{t('tools.approval')}</label>
        <select id={`${id}-approval`} value={cfg.approval} onChange={(e) => setTools({ approval: e.target.value as ApprovalMode })}>
          {APPROVAL_MODES.map((m) => (
            <option key={m} value={m}>
              {t(`tools.approval.${m}`)}
            </option>
          ))}
        </select>
      </div>

      <div className="field field-wide">
        <label>
          <input type="checkbox" checked={cfg.localHost} onChange={(e) => setTools({ localHost: e.target.checked })} /> {t('tools.localHost')}
        </label>
        <p className="hint">{t('tools.localHostHint')}</p>
      </div>

      {cfg.alwaysAllow.length > 0 && (
        <div className="field field-wide">
          <h3>{t('tools.alwaysAllow')}</h3>
          <ul>
            {cfg.alwaysAllow.map((name) => (
              <li key={name}>
                <code>{name}</code>{' '}
                <button type="button" aria-label={t('tools.forget', { name })} onClick={() => setTools({ alwaysAllow: cfg.alwaysAllow.filter((n) => n !== name) })}>
                  {t('tools.forgetShort')}
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}
    </section>
  );
}
