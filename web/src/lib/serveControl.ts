// Shared request shape for POST /control/serve (Vite middleware → start-remote-lm-serve.sh).
// Only lm-serve flags that already exist on the binary are accepted. Null/undefined omits
// the flag so lm-serve keeps its own default.

export type ServeModelId = 'qwen' | 'gemma';

export interface ServeOptions {
  model: ServeModelId;
  /** lm-serve --max-tokens; omit for 0 (run until stop / context full). */
  maxTokens?: number | null;
  /** lm-serve --seed; omit for 1. */
  seed?: number | null;
  /** lm-serve --ctx; omit for 0 (engine chooses from free memory). */
  ctx?: number | null;
  /** lm-serve --threads; omit for 0 (auto). */
  threads?: number | null;
  /** lm-serve --timeout seconds; omit for 30. */
  timeout?: number | null;
  /** lm-serve --cache-mb; omit for 512. */
  cacheMb?: number | null;
  /** lm-serve --name; omit to use the model directory basename. */
  name?: string | null;
  /** lm-serve --think when true; omit (default off) when false/null. */
  think?: boolean | null;
  /** lm-serve --fast when true; omit when false/null. */
  fast?: boolean | null;
}

export interface ServeOk {
  ok: true;
  model: ServeModelId;
  health: string;
}

export interface ServeErr {
  ok: false;
  error: string;
}

export type ServeResult = ServeOk | ServeErr;

const MODELS: ServeModelId[] = ['qwen', 'gemma'];

function intOrNull(v: unknown, lo: number, hi: number): number | null | undefined {
  if (v === undefined || v === null || v === '') return null;
  if (typeof v !== 'number' || !Number.isInteger(v) || v < lo || v > hi) return undefined;
  return v;
}

/** Parse and validate a JSON body. Returns null when the body is invalid. */
export function parseServeOptions(raw: unknown): ServeOptions | null {
  if (!raw || typeof raw !== 'object') return null;
  const o = raw as Record<string, unknown>;
  if (typeof o.model !== 'string' || !MODELS.includes(o.model as ServeModelId)) return null;
  const maxTokens = intOrNull(o.maxTokens, 0, 1 << 24);
  const seed = intOrNull(o.seed, 0, Number.MAX_SAFE_INTEGER);
  const ctx = intOrNull(o.ctx, 1, 1 << 24);
  const threads = intOrNull(o.threads, 1, 4096);
  const timeout = intOrNull(o.timeout, 1, 3600);
  const cacheMb = intOrNull(o.cacheMb, 0, 1 << 20);
  if ([maxTokens, seed, ctx, threads, timeout, cacheMb].includes(undefined)) return null;
  let name: string | null = null;
  if (o.name !== undefined && o.name !== null && o.name !== '') {
    if (typeof o.name !== 'string' || o.name.length > 200 || /[\0\r\n]/.test(o.name)) return null;
    name = o.name;
  }
  if (o.think !== undefined && o.think !== null && typeof o.think !== 'boolean') return null;
  if (o.fast !== undefined && o.fast !== null && typeof o.fast !== 'boolean') return null;
  return {
    model: o.model as ServeModelId,
    maxTokens: maxTokens ?? null,
    seed: seed ?? null,
    ctx: ctx ?? null,
    threads: threads ?? null,
    timeout: timeout ?? null,
    cacheMb: cacheMb ?? null,
    name,
    think: typeof o.think === 'boolean' ? o.think : null,
    fast: typeof o.fast === 'boolean' ? o.fast : null,
  };
}

/** argv after `--id <model>` for start-remote-lm-serve.sh (only set flags). */
export function serveScriptArgs(opts: ServeOptions): string[] {
  const args = ['--id', opts.model];
  if (opts.maxTokens !== null && opts.maxTokens !== undefined) args.push('--max-tokens', String(opts.maxTokens));
  if (opts.seed !== null && opts.seed !== undefined) args.push('--seed', String(opts.seed));
  if (opts.ctx !== null && opts.ctx !== undefined) args.push('--ctx', String(opts.ctx));
  if (opts.threads !== null && opts.threads !== undefined) args.push('--threads', String(opts.threads));
  if (opts.timeout !== null && opts.timeout !== undefined) args.push('--timeout', String(opts.timeout));
  if (opts.cacheMb !== null && opts.cacheMb !== undefined) args.push('--cache-mb', String(opts.cacheMb));
  if (opts.name) args.push('--name', opts.name);
  if (opts.think === true) args.push('--think');
  if (opts.fast === true) args.push('--fast');
  return args;
}

export const SERVE_DEFAULTS = {
  host: '127.0.0.1',
  port: 8080,
  think: false,
  fast: false,
  maxTokens: 0,
  seed: 1,
  ctx: 0,
  threads: 0,
  timeout: 30,
  cacheMb: 512,
  name: '(model directory basename)',
} as const;
