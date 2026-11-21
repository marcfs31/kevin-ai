// Effort and speed: two per-chat switches next to the model picker. Effort is how much the model
// thinks and how long it may write; speed trims the work around a message (image size, how early
// the chat is compacted). Speed also moves effort to its own level when chosen, and effort can
// then be changed on its own. Pure: the numbers live here so the screen and the request agree.
import type { ImagePreset } from './attachments';
import type { Settings } from './storage';

export const EFFORTS = ['low', 'medium', 'high'] as const;
export type Effort = (typeof EFFORTS)[number];
export const SPEEDS = ['fast', 'balanced', 'thorough'] as const;
export type Speed = (typeof SPEEDS)[number];

export const DEFAULT_EFFORT: Effort = 'medium';
export const DEFAULT_SPEED: Speed = 'balanced';

/** The effort each speed starts from. */
export const SPEED_EFFORT: Record<Speed, Effort> = { fast: 'low', balanced: 'medium', thorough: 'high' };

/** Longest answer at low effort, and the least a high-effort answer may run to. */
export const LOW_MAX_TOKENS = 512;
export const HIGH_MAX_TOKENS = 4096;

export interface ModeParams {
  /** Sent as the chat template's enable_thinking; a server that has no such switch ignores it. */
  thinking: boolean;
  /** Cap on the reply, or null for the server's own. */
  maxTokens: number | null;
  imagePreset: ImagePreset;
  /** Share of the context window at which a chat is compacted by itself. */
  compactAt: number;
}

/** Where the chat is compacted by itself at balanced speed (lib/compact.ts calls it AUTO_RATIO). */
export const AUTO_COMPACT_AT = 0.8;
const COMPACT_AT: Record<Speed, number> = { fast: 0.5, balanced: AUTO_COMPACT_AT, thorough: 0.9 };

export function resolveModes(effort: Effort, speed: Speed, settings: Pick<Settings, 'maxTokens' | 'imagePreset'>): ModeParams {
  const set = settings.maxTokens;
  return {
    thinking: effort === 'high',
    maxTokens: effort === 'low' ? Math.min(set ?? LOW_MAX_TOKENS, LOW_MAX_TOKENS) : effort === 'high' ? Math.max(set ?? 0, HIGH_MAX_TOKENS) : set,
    imagePreset: speed === 'fast' ? 512 : speed === 'thorough' ? 1024 : settings.imagePreset,
    compactAt: COMPACT_AT[speed],
  };
}

export const isEffort = (v: unknown): v is Effort => (EFFORTS as readonly unknown[]).includes(v);
export const isSpeed = (v: unknown): v is Speed => (SPEEDS as readonly unknown[]).includes(v);
