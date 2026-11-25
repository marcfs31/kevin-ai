import { describe, expect, it } from 'vitest';
import { ctxPresetsFor, formatCtxPreset, isServeModelRunning, isUncensoredServeModel, matchServeModel, serveModelLabel } from './serveModels';

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

  it('keeps Gemma and Gemma uncensored apart, by their different aliases', () => {
    expect(matchServeModel('gemma-4-e4b')).toBe('gemma');
    expect(matchServeModel('gemma-4-e4b-uncensored')).toBe('gemma-uncensored');
    // NEGATIVE: each name turns the other target off
    expect(isServeModelRunning('gemma-4-e4b-uncensored', 'gemma')).toBe(false);
    expect(isServeModelRunning('gemma-4-e4b', 'gemma-uncensored')).toBe(false);
    expect(isServeModelRunning('qwen3.5-4b-uncensored', 'gemma-uncensored')).toBe(false);
    expect(isServeModelRunning('gemma-4-e4b-uncensored', 'qwen-uncensored')).toBe(false);
    expect(ctxPresetsFor('gemma-uncensored', null, null)).toEqual([null, 4096, 8192, 16384, 32768]);
  });

  it('maps the llama-server WhiteRabbitNeo model and caps its window at 16k', () => {
    expect(matchServeModel('whiterabbitneo-2.5-7b')).toBe('whiterabbitneo');
    // NEGATIVE: its name has no "qwen", so the Qwen targets stay off
    expect(isServeModelRunning('whiterabbitneo-2.5-7b', 'qwen')).toBe(false);
    expect(isServeModelRunning('qwen3.5-4b', 'whiterabbitneo')).toBe(false);
    expect(ctxPresetsFor('whiterabbitneo', null, null)).toEqual([null, 4096, 8192, 16384]);
  });

  it('includes a running ctx value that is not on the ladder', () => {
    const presets = ctxPresetsFor('qwen', 'qwen3.5-4b', 10000);
    expect(presets).toEqual([null, 4096, 8192, 10000]);
  });
});
describe('isUncensoredServeModel', () => {
  it('is true only for the uncensored start targets', () => {
    expect(isUncensoredServeModel('qwen-uncensored')).toBe(true);
    expect(isUncensoredServeModel('gemma-uncensored')).toBe(true);
    expect(isUncensoredServeModel('gemma-e2b-heretic')).toBe(true);
    expect(isUncensoredServeModel('spark-heretic')).toBe(true);
    expect(isUncensoredServeModel('granite-heretic')).toBe(true);
    expect(isUncensoredServeModel('qwen2b')).toBe(true);
    expect(isUncensoredServeModel('lfm3-abliterated')).toBe(true);
    expect(isUncensoredServeModel('qwen')).toBe(false);
    expect(isUncensoredServeModel('gemma')).toBe(false);
    expect(isUncensoredServeModel('gemma-e2b')).toBe(false);
    expect(isUncensoredServeModel('nemotron')).toBe(false);
    expect(isUncensoredServeModel('whiterabbitneo')).toBe(false);
  });
});

describe('the ten GGUF targets', () => {
  it('keeps Gemma E2B off the E4B button, and Nemotron text-only in the cap', () => {
    expect(matchServeModel('gemma-4-e2b')).toBe('gemma-e2b');
    expect(isServeModelRunning('gemma-4-e2b', 'gemma')).toBe(false);
    expect(isServeModelRunning('gemma-4-e4b', 'gemma-e2b')).toBe(false);
    expect(matchServeModel('nemotron-3.5-bolt-e4b')).toBe('nemotron');
    expect(ctxPresetsFor('nemotron', null, null)).toEqual([null, 4096, 8192, 16384, 32768]);
  });

  it('maps Spark and LFM3 by their own aliases and caps Granite at 16k', () => {
    expect(matchServeModel('spark-x2.6-4b')).toBe('spark');
    expect(matchServeModel('spark-x2.6-4b-heretic')).toBe('spark-heretic');
    expect(isServeModelRunning('spark-x2.6-4b-heretic', 'spark')).toBe(false);
    expect(matchServeModel('lfm3-vl-2b')).toBe('lfm3');
    expect(matchServeModel('lfm3-vl-2b-abliterated')).toBe('lfm3-abliterated');
    expect(isServeModelRunning('lfm3-vl-2b-abliterated', 'lfm3')).toBe(false);
    expect(ctxPresetsFor('granite', null, null)).toEqual([null, 4096, 8192, 16384]);
    expect(ctxPresetsFor('granite-heretic', null, null)).toEqual([null, 4096, 8192, 16384]);
  });

  it('maps the Qwen 2B abliterated alias and not the installed 4B', () => {
    expect(matchServeModel('qwen3.5-2b-abliterated')).toBe('qwen2b');
    expect(isServeModelRunning('qwen3.5-2b-abliterated', 'qwen')).toBe(false);
    expect(isServeModelRunning('qwen3.5-4b', 'qwen2b')).toBe(false);
    expect(isServeModelRunning('qwen3.5-4b-uncensored', 'qwen2b')).toBe(false);
    expect(serveModelLabel('qwen2b', 'es')).toBe('Qwen3.5-2B abliterado');
    expect(serveModelLabel('gemma-e2b', 'en')).toBe('Gemma 4 E2B');
  });
});
