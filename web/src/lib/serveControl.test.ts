import { describe, expect, it } from 'vitest';
import { controlRequestAllowed, parseServeOptions, serveScriptArgs } from './serveControl';
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

  it('accepts qwen-uncensored and passes it to the script', () => {
    expect(parseServeOptions({ model: 'qwen-uncensored' })).toMatchObject({ model: 'qwen-uncensored' });
    expect(serveScriptArgs({ model: 'qwen-uncensored' })).toEqual(['--id', 'qwen-uncensored']);
  });

  it('accepts whiterabbitneo and passes it to the script', () => {
    expect(parseServeOptions({ model: 'whiterabbitneo' })).toMatchObject({ model: 'whiterabbitneo' });
    expect(serveScriptArgs({ model: 'whiterabbitneo' })).toEqual(['--id', 'whiterabbitneo']);
  });

  it('rejects unknown models and bad integers', () => {
    expect(parseServeOptions({ model: 'llama' })).toBeNull();
    expect(parseServeOptions({ model: 'qwen-uncensored-x' })).toBeNull();
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

describe('controlRequestAllowed', () => {
  const page = {
    method: 'POST',
    contentType: 'application/json',
    origin: 'http://127.0.0.1:5173',
    secFetchSite: 'same-origin',
    host: '127.0.0.1:5173',
  };

  it('allows the Settings page on this dev server', () => {
    expect(controlRequestAllowed(page)).toEqual({ ok: true });
    expect(controlRequestAllowed({ ...page, contentType: 'application/json; charset=utf-8' })).toEqual({ ok: true });
    expect(controlRequestAllowed({ ...page, host: 'localhost:5173', origin: 'http://localhost:5173' })).toEqual({ ok: true });
  });

  it('refuses a request that is not the dev page', () => {
    expect(controlRequestAllowed({ ...page, method: 'GET' }).ok).toBe(false);
    expect(controlRequestAllowed({ ...page, contentType: 'text/plain' }).ok).toBe(false);
    expect(controlRequestAllowed({ ...page, contentType: null }).ok).toBe(false);
    expect(controlRequestAllowed({ ...page, secFetchSite: 'cross-site' }).ok).toBe(false);
    expect(controlRequestAllowed({ ...page, secFetchSite: null }).ok).toBe(false);
    expect(controlRequestAllowed({ ...page, origin: 'http://127.0.0.1:9999' }).ok).toBe(false);
    expect(controlRequestAllowed({ ...page, origin: null }).ok).toBe(false);
    expect(controlRequestAllowed({ ...page, host: '192.0.2.10:5173', origin: 'http://192.0.2.10:5173' }).ok).toBe(false);
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
