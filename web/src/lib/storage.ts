// Conversations and settings, kept only in this browser's localStorage.
// Versioned schema: a saved document missing a newer field loads with its default.
import type { ContentPart } from './api';
import { IMAGE_PRESETS, type ImagePreset } from './attachments';
import { obj } from './sse';

export const STORAGE_KEY = 'lm-chat';
export const BACKUP_KEY = 'lm-chat.unreadable';
export const SCHEMA_VERSION = 1;

export interface Stats {
  ttftMs: number | null;
  tokensPerSecond: number | null;
  promptTokens: number | null;
  cachedTokens: number | null;
  completionTokens: number | null;
  /** What was counted locally, kept when the server sent no usage. */
  promptEstimate: number | null;
  completionEstimate: number | null;
}

export interface ToolCall {
  id: string;
  name: string;
  /** The JSON text the model wrote, kept as received. */
  arguments: string;
}

export type ReplyStatus = 'done' | 'stopped' | 'error';

export interface UserMessage {
  id: string;
  role: 'user';
  /** Exactly what was sent. */
  content: string | ContentPart[];
  /** What the user typed, for display. */
  typed: string;
  attachmentNames: string[];
  createdAt: number;
}

export interface AssistantMessage {
  id: string;
  role: 'assistant';
  /** Exactly as received: re-sent unchanged so the server can reuse its state. */
  content: string;
  reasoning: string;
  status: ReplyStatus;
  error: { code: string | null; message: string; status: number } | null;
  stats: Stats | null;
  /** Tools the model asked for in this reply, if any. */
  toolCalls?: ToolCall[];
  /** Set to the step cap when the loop ended here with calls left unrun. */
  stepLimit?: number;
  createdAt: number;
}

/** The result of one tool call, sent back to the model right after the reply that asked for it. */
export interface ToolMessage {
  id: string;
  role: 'tool';
  toolCallId: string;
  name: string;
  content: string;
  isError: boolean;
  createdAt: number;
}

/** Stands in, for the model, for the `covered` messages just before it. Those stay in the chat, shown as compacted. */
export interface SummaryMessage {
  id: string;
  role: 'summary';
  content: string;
  covered: number;
  /** Made by the automatic trigger rather than the Compact button. */
  auto: boolean;
  createdAt: number;
}

export type Message = UserMessage | AssistantMessage | SummaryMessage | ToolMessage;

/** The tool messages that directly follow the assistant message at `index`. */
export function toolResults(messages: Message[], index: number): ToolMessage[] {
  const out: ToolMessage[] = [];
  for (let i = index + 1; i < messages.length; i++) {
    const m = messages[i];
    if (m?.role !== 'tool') break;
    out.push(m);
  }
  return out;
}

export function lastSummaryIndex(messages: Message[]): number {
  for (let i = messages.length - 1; i >= 0; i--) if (messages[i]?.role === 'summary') return i;
  return -1;
}

export interface Conversation {
  id: string;
  title: string;
  createdAt: number;
  updatedAt: number;
  /** The model this conversation is sent to; null = the server's first. */
  model: string | null;
  messages: Message[];
}

export type LocaleSetting = 'auto' | 'en' | 'es';

export interface Settings {
  systemPrompt: string;
  temperature: number;
  maxTokens: number | null;
  imagePreset: ImagePreset;
  locale: LocaleSetting;
  /** Offer the model its tools and run what it calls. Off until the user turns it on. */
  tools: boolean;
  /** Most model calls for one message when tools are on; the last one may not call tools. */
  maxSteps: number;
  /** Summarise older turns by itself near the context limit. Off: only the Compact button does. */
  autoCompact: boolean;
}

export interface State {
  version: typeof SCHEMA_VERSION;
  conversations: Conversation[];
  activeId: string | null;
  settings: Settings;
}

export const DEFAULT_SETTINGS: Settings = {
  systemPrompt: '',
  temperature: 0.7,
  maxTokens: null,
  imagePreset: 512,
  locale: 'auto',
  tools: false,
  maxSteps: 8,
  autoCompact: true,
};

export const MAX_STEPS_LIMIT = 20;

export function emptyState(): State {
  return { version: SCHEMA_VERSION, conversations: [], activeId: null, settings: { ...DEFAULT_SETTINGS } };
}

const str = (v: unknown, d = ''): string => (typeof v === 'string' ? v : d);
const time = (v: unknown): number => (typeof v === 'number' && Number.isFinite(v) ? v : 0);
const numOrNull = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null);

function content(v: unknown): string | ContentPart[] | null {
  if (typeof v === 'string') return v;
  if (!Array.isArray(v)) return null;
  const parts: ContentPart[] = [];
  for (const p of v) {
    const o = obj(p);
    if (o?.type === 'text' && typeof o.text === 'string') parts.push({ type: 'text', text: o.text });
    else if (o?.type === 'image_url' && typeof obj(o.image_url)?.url === 'string')
      parts.push({ type: 'image_url', image_url: { url: str(obj(o.image_url)?.url) } });
    else return null;
  }
  return parts;
}

function stats(v: unknown): Stats | null {
  const o = obj(v);
  if (!o) return null;
  return {
    ttftMs: numOrNull(o.ttftMs),
    tokensPerSecond: numOrNull(o.tokensPerSecond),
    promptTokens: numOrNull(o.promptTokens),
    cachedTokens: numOrNull(o.cachedTokens),
    completionTokens: numOrNull(o.completionTokens),
    promptEstimate: numOrNull(o.promptEstimate),
    completionEstimate: numOrNull(o.completionEstimate),
  };
}

function toolFields(o: Record<string, unknown>): Pick<AssistantMessage, 'toolCalls' | 'stepLimit'> {
  const out: Pick<AssistantMessage, 'toolCalls' | 'stepLimit'> = {};
  if (Array.isArray(o.toolCalls)) {
    const calls = o.toolCalls.flatMap((c): ToolCall[] => {
      const x = obj(c);
      return x && typeof x.id === 'string' && typeof x.name === 'string' ? [{ id: x.id, name: x.name, arguments: str(x.arguments) }] : [];
    });
    if (calls.length) out.toolCalls = calls;
  }
  if (typeof o.stepLimit === 'number' && o.stepLimit >= 1) out.stepLimit = Math.floor(o.stepLimit);
  return out;
}

function message(v: unknown): Message | null {
  const o = obj(v);
  if (!o || typeof o.id !== 'string') return null;
  if (o.role === 'user') {
    const c = content(o.content);
    if (c === null) return null;
    return {
      id: o.id,
      role: 'user',
      content: c,
      typed: str(o.typed, typeof c === 'string' ? c : ''),
      attachmentNames: Array.isArray(o.attachmentNames) ? o.attachmentNames.filter((n): n is string => typeof n === 'string') : [],
      createdAt: time(o.createdAt),
    };
  }
  if (o.role === 'assistant') {
    const e = obj(o.error);
    const status = o.status === 'stopped' || o.status === 'error' ? o.status : 'done';
    return {
      id: o.id,
      role: 'assistant',
      content: str(o.content),
      reasoning: str(o.reasoning),
      status,
      error: e ? { code: typeof e.code === 'string' ? e.code : null, message: str(e.message), status: time(e.status) } : null,
      stats: stats(o.stats),
      ...toolFields(o),
      createdAt: time(o.createdAt),
    };
  }
  if (o.role === 'tool' && typeof o.toolCallId === 'string') {
    return { id: o.id, role: 'tool', toolCallId: o.toolCallId, name: str(o.name), content: str(o.content), isError: o.isError === true, createdAt: time(o.createdAt) };
  }
  if (o.role === 'summary' && typeof o.content === 'string' && o.content.trim()) {
    const covered = numOrNull(o.covered);
    return { id: o.id, role: 'summary', content: o.content, covered: covered !== null && covered >= 0 ? Math.floor(covered) : 0, auto: o.auto === true, createdAt: time(o.createdAt) };
  }
  return null;
}

function conversation(v: unknown): Conversation | null {
  const o = obj(v);
  if (!o || typeof o.id !== 'string') return null;
  const messages = Array.isArray(o.messages) ? o.messages.map(message).filter((m): m is Message => m !== null) : [];
  const model = typeof o.model === 'string' && o.model ? o.model : null;
  return { id: o.id, title: str(o.title), createdAt: time(o.createdAt), updatedAt: time(o.updatedAt), model, messages };
}

function settings(v: unknown): Settings {
  const o = obj(v) ?? {};
  const t = o.temperature;
  const m = o.maxTokens;
  return {
    systemPrompt: str(o.systemPrompt, DEFAULT_SETTINGS.systemPrompt),
    temperature: typeof t === 'number' && t >= 0 && t <= 2 ? t : DEFAULT_SETTINGS.temperature,
    maxTokens: typeof m === 'number' && Number.isInteger(m) && m > 0 ? m : null,
    imagePreset: (IMAGE_PRESETS as readonly unknown[]).includes(o.imagePreset) ? (o.imagePreset as ImagePreset) : DEFAULT_SETTINGS.imagePreset,
    locale: o.locale === 'en' || o.locale === 'es' ? o.locale : 'auto',
    tools: o.tools === true,
    maxSteps: typeof o.maxSteps === 'number' && Number.isInteger(o.maxSteps) && o.maxSteps >= 1 && o.maxSteps <= MAX_STEPS_LIMIT ? o.maxSteps : DEFAULT_SETTINGS.maxSteps,
    autoCompact: typeof o.autoCompact === 'boolean' ? o.autoCompact : DEFAULT_SETTINGS.autoCompact,
  };
}

/** A stored document as the current schema, or null when it cannot be read (newer or broken). */
export function migrate(raw: unknown): State | null {
  const o = obj(raw);
  if (!o || o.version !== SCHEMA_VERSION) return null;
  const conversations = Array.isArray(o.conversations)
    ? o.conversations.map(conversation).filter((c): c is Conversation => c !== null)
    : [];
  const activeId = typeof o.activeId === 'string' && conversations.some((c) => c.id === o.activeId) ? o.activeId : null;
  return { version: SCHEMA_VERSION, conversations, activeId, settings: settings(o.settings) };
}

export type LoadResult = { state: State; notice: 'none' | 'unreadable' | 'locked' };

/** Loads the saved state. An unreadable document is moved aside, never overwritten. */
export function loadState(storage: Storage): LoadResult {
  const text = storage.getItem(STORAGE_KEY);
  if (text === null) return { state: emptyState(), notice: 'none' };
  let raw: unknown;
  try {
    raw = JSON.parse(text);
  } catch {
    raw = undefined;
  }
  const state = migrate(raw);
  if (state) return { state, notice: 'none' };
  try {
    storage.setItem(`${BACKUP_KEY}.${Date.now()}`, text);
  } catch {
    // No room for a copy: keep the original where it is and do not save over it.
    return { state: emptyState(), notice: 'locked' };
  }
  storage.removeItem(STORAGE_KEY);
  return { state: emptyState(), notice: 'unreadable' };
}

export type SaveResult = 'ok' | 'full';

/** Keys of unreadable documents moved aside (`lm-chat.unreadable.<time>`). */
export function backupKeys(storage: Storage): string[] {
  const keys: string[] = [];
  for (let i = 0; i < storage.length; i++) {
    const key = storage.key(i);
    if (key && key.startsWith(`${BACKUP_KEY}.`)) keys.push(key);
  }
  return keys;
}

export function deleteBackups(storage: Storage): void {
  for (const key of backupKeys(storage)) storage.removeItem(key);
}

export function saveState(storage: Storage, state: State): SaveResult {
  try {
    storage.setItem(STORAGE_KEY, JSON.stringify(state));
    return 'ok';
  } catch {
    return 'full';
  }
}

/** Saves state and lets the UI subscribe to the last result (useSyncExternalStore). */
export function createSaver(storage: Storage) {
  let last: SaveResult = 'ok';
  const listeners = new Set<() => void>();
  return {
    save(state: State) {
      const r = saveState(storage, state);
      if (r !== last) {
        last = r;
        listeners.forEach((l) => l());
      }
    },
    subscribe(l: () => void) {
      listeners.add(l);
      return () => listeners.delete(l);
    },
    get: (): SaveResult => last,
  };
}
