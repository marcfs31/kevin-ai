import { describe, expect, it } from 'vitest';
import { parseServeOptions, serveScriptArgs } from './serveControl';
import { draftToOptions, EMPTY_SERVE_DRAFT } from '../components/ServerPanel';

describe('parseServeOptions', () => {
  it('accepts qwen with no optional flags', () => {
    expect(parseServeOptions({ model: 'qwen' })).toEqual({
      model: 'qwen',
      maxTokens: null,
      seed: null,
      ctx: null,
      threads: null,
      timeout: null,
      cacheMb: null,
      name: null,
      think: null,
      fast: null,
    });
  });

  it('rejects unknown models and bad integers', () => {
    expect(parseServeOptions({ model: 'llama' })).toBeNull();
    expect(parseServeOptions({ model: 'gemma', maxTokens: 1.5 })).toBeNull();
    expect(parseServeOptions({ model: 'gemma', ctx: 0 })).toBeNull();
  });

  it('builds only the flags that were set', () => {
    expect(serveScriptArgs({ model: 'gemma' })).toEqual(['--id', 'gemma']);
    expect(
      serveScriptArgs({
        model: 'qwen',
        maxTokens: 128,
        seed: null,
        ctx: 4096,
        threads: null,
        timeout: null,
        cacheMb: 256,
        name: 'local-qwen',
        think: true,
        fast: null,
      }),
    ).toEqual(['--id', 'qwen', '--max-tokens', '128', '--ctx', '4096', '--cache-mb', '256', '--name', 'local-qwen', '--think']);
  });
});

describe('draftToOptions', () => {
  it('treats empty fields as omitted defaults', () => {
    expect(draftToOptions('qwen', EMPTY_SERVE_DRAFT)).toMatchObject({ model: 'qwen', maxTokens: null, think: null });
  });

  it('rejects non-integer text', () => {
    expect(draftToOptions('gemma', { ...EMPTY_SERVE_DRAFT, ctx: 'abc' })).toBeNull();
  });
});
