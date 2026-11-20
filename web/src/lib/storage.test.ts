import { describe, expect, it } from 'vitest';
import { BACKUP_KEY, DEFAULT_SETTINGS, STORAGE_KEY, emptyState, loadState, migrate, saveState } from './storage';

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

  it('reports a full storage', () => {
    const full = { setItem: () => { throw new DOMException('full', 'QuotaExceededError'); } } as unknown as Storage;
    expect(saveState(full, emptyState())).toBe('full');
  });
});
