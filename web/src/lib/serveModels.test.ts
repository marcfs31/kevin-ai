import { describe, expect, it } from 'vitest';
import { ctxPresetsFor, formatCtxPreset, matchServeModel } from './serveModels';

describe('formatCtxPreset', () => {
  it('formats ladder sizes as k labels', () => {
    expect(formatCtxPreset(4096)).toBe('4k');
    expect(formatCtxPreset(131072)).toBe('128k');
  });

  it('keeps non-multiples of 1024 as plain numbers', () => {
    expect(formatCtxPreset(100)).toBe('100');
    expect(formatCtxPreset(10000)).toBe('10000');
  });
});

describe('matchServeModel', () => {
  it('maps /v1/models ids to serve targets', () => {
    expect(matchServeModel('qwen3.5-4b')).toBe('qwen');
    expect(matchServeModel('gemma-4-e4b')).toBe('gemma');
    expect(matchServeModel('qwen3.5-4b-uncensored')).toBe('qwen-uncensored');
  });

  it('returns null when unknown or missing', () => {
    expect(matchServeModel(null)).toBeNull();
    expect(matchServeModel('llama-7b')).toBeNull();
  });
});

describe('ctxPresetsFor', () => {
  it('starts with auto and caps presets at the model max', () => {
    expect(ctxPresetsFor('qwen', null, null)).toEqual([
      null,
      4096,
      8192,
      16384,
      32768,
      65536,
      131072,
    ]);
  });

  it('uses reported context_length as cap when that model is running', () => {
    expect(ctxPresetsFor('qwen', 'qwen3.5-4b', 8192)).toEqual([null, 4096, 8192]);
    expect(ctxPresetsFor('gemma', 'qwen3.5-4b', 8192)).toEqual([
      null,
      4096,
      8192,
      16384,
      32768,
      65536,
      131072,
    ]);
  });

  it('does not cap the plain Qwen target by a running uncensored model', () => {
    expect(ctxPresetsFor('qwen', 'qwen3.5-4b-uncensored', 8192)).toHaveLength(7);
    expect(ctxPresetsFor('qwen-uncensored', 'qwen3.5-4b-uncensored', 8192)).toEqual([null, 4096, 8192]);
  });

  it('includes a running ctx value that is not on the ladder', () => {
    const presets = ctxPresetsFor('qwen', 'qwen3.5-4b', 10000);
    expect(presets).toEqual([null, 4096, 8192, 10000]);
  });
});