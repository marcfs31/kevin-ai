import { describe, expect, it } from 'vitest';
import { BACKUP_KEY, DEFAULT_SETTINGS, STORAGE_KEY, deleteBackups, emptyState, loadState, migrate, saveState, truncateFrom, type Message } from './storage';

describe('tools and internet settings', () => {
  const load = (settings: unknown) => migrate({ version: 1, settings })?.settings;

  it('default to nothing on and the internet off, so a chat saved before them loads as it was', () => {
    expect(load({})?.toolsConfig).toEqual({ servers: {}, budgetTokens: 1500, approval: 'readonly-free', alwaysAllow: [], localHost: false });
    expect(load({})?.internet).toEqual({ on: false, searchUrl: '' });
  });

  it('keep a valid configuration', () => {
    const cfg = { servers: { fs: { on: true, disabledTools: ['write_file'] } }, budgetTokens: 3000, approval: 'always', alwaysAllow: ['fs/read_file'] };
    expect(load({ toolsConfig: cfg, internet: { on: true, searchUrl: ' http://127.0.0.1:8888 ' } })).toMatchObject({ toolsConfig: cfg, internet: { on: true, searchUrl: 'http://127.0.0.1:8888' } });
  });

  it('refuse what is not valid, one field at a time', () => {
    expect(load({ toolsConfig: { servers: { 'Bad Name': { on: true }, ok: { on: 'yes', disabledTools: [1, 'x', 'x'] }, list: [] }, budgetTokens: 5, approval: 'maybe', alwaysAllow: [3, '', 'a'] } })?.toolsConfig).toEqual({
      servers: { ok: { on: false, disabledTools: ['x'] } }, budgetTokens: 1500, approval: 'readonly-free', alwaysAllow: ['a'], localHost: false,
    });
    expect(load({ toolsConfig: { localHost: true } })?.toolsConfig.localHost).toBe(true);
    expect(load({ toolsConfig: { localHost: 'yes' } })?.toolsConfig.localHost).toBe(false);
    expect(load({ toolsConfig: { budgetTokens: 99999 } })?.toolsConfig.budgetTokens).toBe(1500);
    expect(load({ internet: { on: 'true', searchUrl: 'x'.repeat(400) } })?.internet).toEqual({ on: false, searchUrl: '' });
    expect(load({ toolsConfig: 'x', internet: 7 })).toMatchObject({ toolsConfig: { servers: {} }, internet: { on: false } });
  });
});

describe('effort and speed per conversation', () => {
  it('load with the defaults (high, thorough) when absent or not a known value, and keep a valid pair', () => {
    const state = migrate({
      version: 1,
      conversations: [
        { id: 'a', messages: [] },
        { id: 'b', effort: 'low', speed: 'fast', messages: [] },
        { id: 'c', effort: 'max', speed: 7, messages: [] },
      ],
    });
    expect(state?.conversations.map((c) => [c.effort, c.speed])).toEqual([['high', 'thorough'], ['low', 'fast'], ['high', 'thorough']]);
  });
});

describe('the stats pane setting', () => {
  it('is auto unless the user chose to keep it open or hidden', () => {
    expect(migrate({ version: 1, settings: {} })?.settings.statsPane).toBe('auto');
    expect(migrate({ version: 1, settings: { statsPane: 'open' } })?.settings.statsPane).toBe('open');
    expect(migrate({ version: 1, settings: { statsPane: 'closed' } })?.settings.statsPane).toBe('closed');
    expect(migrate({ version: 1, settings: { statsPane: 'wide' } })?.settings.statsPane).toBe('auto');
    expect(migrate({ version: 1, settings: { statsPane: true } })?.settings.statsPane).toBe('auto');
  });
});

describe('project host per conversation', () => {
  it('is the model server unless the chat says it is on this Mac', () => {
    const state = migrate({
      version: 1,
      conversations: [{ id: 'a', messages: [] }, { id: 'b', projectId: 'notes', projectHost: 'local', messages: [] }, { id: 'c', projectHost: 'elsewhere', messages: [] }],
    });
    expect(state?.conversations.map((c) => c.projectHost)).toEqual(['server', 'local', 'server']);
  });
});

describe('workflow per conversation', () => {
  it('is kept with its checklist, and is null when absent or not a workflow', () => {
    const saved = { id: 'ship', title: 'Ship it', steps: [{ text: 'Test', status: 'done', note: 'green' }, { text: 'Push', status: 'todo', note: '' }] };
    const state = migrate({
      version: 1,
      conversations: [{ id: 'a', messages: [] }, { id: 'b', workflow: saved, messages: [] }, { id: 'c', workflow: { id: 'x', steps: 'no' }, messages: [] }],
    });
    expect(state?.conversations.map((c) => c.workflow)).toEqual([null, saved, null]);
  });
});

describe('project per conversation', () => {
  it('loads null when absent or not a plain id, and keeps a valid id', () => {
    const state = migrate({
      version: 1,
      conversations: [
        { id: 'a', messages: [] },
        { id: 'b', projectId: 'demo-1', messages: [] },
        { id: 'c', projectId: '../etc', messages: [] },
        { id: 'd', projectId: 7, messages: [] },
      ],
    });
    expect(state?.conversations.map((c) => c.projectId)).toEqual([null, 'demo-1', null, null]);
  });
});

describe('migrate', () => {
  it('loads a document missing newer fields with their defaults', () => {
    const state = migrate({
      version: 1,
      conversations: [{ id: 'c', messages: [{ id: 'm', role: 'user', content: 'hi' }, { id: 'a', role: 'assistant', content: 'yo' }] }],
      activeId: 'c',
    });
    expect(state?.settings).toEqual(DEFAULT_SETTINGS);
    expect(state?.activeId).toBe('c');
    const conv = state?.conversations[0];
    expect(conv).toMatchObject({ id: 'c', title: '', createdAt: 0 });
    expect(conv?.messages[0]).toEqual({ id: 'm', role: 'user', content: 'hi', typed: 'hi', attachmentNames: [], createdAt: 0 });
    expect(conv?.messages[1]).toMatchObject({ role: 'assistant', content: 'yo', reasoning: '', status: 'done', error: null, stats: null });
  });

  it('loads saved statistics from before estimates were kept with no estimate', () => {
    const state = migrate({
      version: 1,
      conversations: [{ id: 'c', messages: [{ id: 'a', role: 'assistant', content: 'yo', stats: { promptTokens: 4, completionTokens: 2 } }] }],
    });
    expect(state?.conversations[0]?.messages[0]).toMatchObject({ stats: { promptTokens: 4, completionTokens: 2, promptEstimate: null, completionEstimate: null } });
  });

  it('loads a conversation saved before models were chosen on the default, and keeps a chosen one', () => {
    const state = migrate({ version: 1, conversations: [{ id: 'a', messages: [] }, { id: 'b', model: 'rabbit', messages: [] }, { id: 'c', model: 7, messages: [] }] });
    expect(state?.conversations.map((c) => c.model)).toEqual([null, 'rabbit', null]);
  });

  it('loads tool calls and results, and settings from before tools existed with the defaults (tools on)', () => {
    const state = migrate({
      version: 1,
      settings: { systemPrompt: 'x' },
      conversations: [{ id: 'c', messages: [
        { id: 'a', role: 'assistant', content: '', toolCalls: [{ id: 'k', name: 'f', arguments: '{}' }, { name: 'bad' }], stepLimit: 3 },
        { id: 't', role: 'tool', toolCallId: 'k', name: 'f', content: 'r', isError: true },
        { id: 'x', role: 'tool', content: 'no call id' },
      ] }],
    });
    expect(state?.settings).toMatchObject({ tools: true, maxSteps: 20 });
    expect(migrate({ version: 1, settings: { tools: false } })?.settings.tools).toBe(false);
    expect(state?.conversations[0]?.messages).toMatchObject([
      { toolCalls: [{ id: 'k', name: 'f', arguments: '{}' }], stepLimit: 3 },
      { role: 'tool', toolCallId: 'k', isError: true },
    ]);
    expect(migrate({ version: 1, settings: { tools: true, maxSteps: 99 } })?.settings).toMatchObject({ tools: true, maxSteps: 20 });
    expect(migrate({ version: 1, settings: { maxSteps: 50 } })?.settings.maxSteps).toBe(50);
    expect(migrate({ version: 1, settings: { maxSteps: 51 } })?.settings.maxSteps).toBe(20);
    expect(migrate({ version: 1, settings: { maxSteps: 5 } })?.settings.maxSteps).toBe(5);
    // A chat saved before Auto compact existed keeps compacting by itself; a saved choice is kept.
    expect(migrate({ version: 1, settings: {} })?.settings.autoCompact).toBe(true);
    expect(migrate({ version: 1, settings: { autoCompact: false } })?.settings.autoCompact).toBe(false);
    expect(migrate({ version: 1, settings: { autoCompact: 'no' } })?.settings.autoCompact).toBe(true);
  });

  it('drops broken entries and resets invalid settings', () => {
    const state = migrate({
      version: 1,
      conversations: [{ id: 'c', messages: [{ role: 'user' }, { id: 'x', role: 'robot' }, { id: 'y', role: 'user', content: [{ type: 'audio' }] }] }, 'junk'],
      activeId: 'missing',
      settings: { temperature: 9, maxTokens: -1, imagePreset: 300, locale: 'fr', systemPrompt: 'keep' },
    });
    expect(state?.conversations).toHaveLength(1);
    expect(state?.conversations[0]?.messages).toEqual([]);
    expect(state?.activeId).toBeNull();
    expect(state?.settings).toEqual({ ...DEFAULT_SETTINGS, systemPrompt: 'keep' });
  });

  it('keeps image parts exactly', () => {
    const content = [{ type: 'image_url', image_url: { url: 'data:image/png;base64,AA' } }, { type: 'text', text: 'q' }];
    const state = migrate({ version: 1, conversations: [{ id: 'c', messages: [{ id: 'm', role: 'user', content }] }] });
    expect(state?.conversations[0]?.messages[0]?.content).toEqual(content);
  });

  it('refuses a newer or unknown schema (negative case)', () => {
    expect(migrate({ version: 2, conversations: [] })).toBeNull();
    expect(migrate(null)).toBeNull();
    expect(migrate([])).toBeNull();
  });
});

describe('loadState and saveState', () => {
  it('round-trips', () => {
    const state = { ...emptyState(), settings: { ...DEFAULT_SETTINGS, systemPrompt: 's' } };
    expect(saveState(localStorage, state)).toBe('ok');
    expect(loadState(localStorage)).toEqual({ state, notice: 'none' });
  });

  it('sets an unreadable document aside instead of overwriting it', () => {
    localStorage.setItem(STORAGE_KEY, '{"version":99}');
    const r = loadState(localStorage);
    expect(r.notice).toBe('unreadable');
    expect(localStorage.getItem(STORAGE_KEY)).toBeNull();
    const backups = Object.keys(localStorage).filter((k) => k.startsWith(BACKUP_KEY));
    expect(backups).toHaveLength(1);
    expect(localStorage.getItem(backups[0] ?? '')).toBe('{"version":99}');
  });

  it('deletes unreadable copies and leaves the primary key', () => {
    localStorage.setItem(`${BACKUP_KEY}.1`, '{"version":99}');
    localStorage.setItem(STORAGE_KEY, '{"version":1}');
    deleteBackups(localStorage);
    expect(Object.keys(localStorage).filter((k) => k.startsWith(BACKUP_KEY))).toEqual([]);
    expect(localStorage.getItem(STORAGE_KEY)).toBe('{"version":1}');
  });

  it('reports a full storage', () => {
    const full = { setItem: () => { throw new DOMException('full', 'QuotaExceededError'); } } as unknown as Storage;
    expect(saveState(full, emptyState())).toBe('full');
  });
});

describe('truncateFrom', () => {
  const msgs = [
    { id: 'u1', role: 'user', content: 'a', typed: 'a', attachmentNames: [], createdAt: 0 },
    { id: 'a1', role: 'assistant', content: 'b', reasoning: '', status: 'done', error: null, stats: null, createdAt: 1 },
    { id: 't1', role: 'tool', toolCallId: 'c', name: 'f', content: 'r', isError: false, createdAt: 2 },
    { id: 'u2', role: 'user', content: 'c', typed: 'c', attachmentNames: [], createdAt: 3 },
  ] as Message[];

  it('drops the chosen message and everything after, including tool results', () => {
    expect(truncateFrom(msgs, 1).map((m) => m.id)).toEqual(['u1']);
    expect(truncateFrom(msgs, 0)).toEqual([]);
    expect(truncateFrom(msgs, 3).map((m) => m.id)).toEqual(['u1', 'a1', 't1']);
  });

  it('leaves the list alone when the index is out of range', () => {
    expect(truncateFrom(msgs, -1)).toBe(msgs);
    expect(truncateFrom(msgs, 99)).toBe(msgs);
  });
});
