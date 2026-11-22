import { describe, expect, it } from 'vitest';
import { AUTO_COMPACT_AT, DEFAULT_EFFORT, DEFAULT_SPEED, HIGH_MAX_TOKENS, LOW_MAX_TOKENS, SPEED_EFFORT, isEffort, isSpeed, resolveModes } from './modes';

const settings = (maxTokens: number | null = null, imagePreset: 512 | 768 | 1024 = 768) => ({ maxTokens, imagePreset });

describe('resolveModes', () => {
  it('thinks only at high effort', () => {
    expect(resolveModes('low', 'balanced', settings()).thinking).toBe(false);
    expect(resolveModes('medium', 'balanced', settings()).thinking).toBe(false);
    expect(resolveModes('high', 'balanced', settings()).thinking).toBe(true);
  });

  it('caps the answer at low effort and lets it run long at high, around the length set in Settings', () => {
    expect(resolveModes('low', 'balanced', settings()).maxTokens).toBe(LOW_MAX_TOKENS);
    expect(resolveModes('low', 'balanced', settings(128)).maxTokens).toBe(128);          // never longer than the user allowed
    expect(resolveModes('low', 'balanced', settings(9000)).maxTokens).toBe(LOW_MAX_TOKENS);
    expect(resolveModes('medium', 'balanced', settings()).maxTokens).toBeNull();           // the server's own
    expect(resolveModes('medium', 'balanced', settings(300)).maxTokens).toBe(300);
    expect(resolveModes('high', 'balanced', settings()).maxTokens).toBe(HIGH_MAX_TOKENS);
    expect(resolveModes('high', 'balanced', settings(100)).maxTokens).toBe(HIGH_MAX_TOKENS);
    expect(resolveModes('high', 'balanced', settings(8000)).maxTokens).toBe(8000);
  });

  it('sizes images and the compaction point by speed, and leaves balanced to the settings', () => {
    expect(resolveModes('medium', 'fast', settings(null, 1024))).toMatchObject({ imagePreset: 512, compactAt: 0.5 });
    expect(resolveModes('medium', 'balanced', settings(null, 768))).toMatchObject({ imagePreset: 768, compactAt: AUTO_COMPACT_AT });
    expect(resolveModes('medium', 'thorough', settings(null, 512))).toMatchObject({ imagePreset: 1024, compactAt: 0.9 });
  });

  it('keeps effort and speed independent: a fast chat can still think', () => {
    expect(resolveModes('high', 'fast', settings())).toMatchObject({ thinking: true, imagePreset: 512, compactAt: 0.5 });
  });
});

describe('defaults and guards', () => {
  it('starts each speed from its own effort, and the defaults are the top ones', () => {
    expect(SPEED_EFFORT).toEqual({ fast: 'low', balanced: 'medium', thorough: 'high' });
    expect([DEFAULT_EFFORT, DEFAULT_SPEED]).toEqual(['high', 'thorough']);
  });

  it('accepts only the known values', () => {
    expect(isEffort('high')).toBe(true);
    expect(isEffort('max')).toBe(false);
    expect(isSpeed('thorough')).toBe(true);
    expect(isSpeed('medium')).toBe(false);
    expect(isSpeed(undefined)).toBe(false);
  });
});
