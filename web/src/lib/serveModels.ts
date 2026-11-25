import type { ServeModelId } from './serveControl';

export interface ServeModelOption {
  id: ServeModelId;
  /** English name, also the confirm dialog when the page is in English. */
  label: string;
  /** Spanish name. The Start / Arrancar prefix lives in i18n. */
  labelEs: string;
  /** Directory under ~/models and the id /v1/models reports. */
  alias: string;
  /** Picker cap. llama-server allocates the whole window up front. */
  maxCtx: number;
  uncensored: boolean;
}

/** Fixed start targets Settings / the header picker can launch. */
export const SERVE_MODEL_OPTIONS: ServeModelOption[] = [
  { id: 'qwen', label: 'Qwen3.5-4B', labelEs: 'Qwen3.5-4B', alias: 'qwen3.5-4b', maxCtx: 131072, uncensored: false },
  { id: 'qwen-uncensored', label: 'Qwen3.5-4B Uncensored', labelEs: 'Qwen3.5-4B sin censura', alias: 'qwen3.5-4b-uncensored', maxCtx: 131072, uncensored: true },
  { id: 'gemma', label: 'Gemma 4 E4B', labelEs: 'Gemma 4 E4B', alias: 'gemma-4-e4b', maxCtx: 131072, uncensored: false },
  { id: 'whiterabbitneo', label: 'WhiteRabbitNeo 2.5 7B', labelEs: 'WhiteRabbitNeo 2.5 7B', alias: 'whiterabbitneo-2.5-7b', maxCtx: 16384, uncensored: false },
  { id: 'gemma-uncensored', label: 'Gemma 4 E4B Uncensored', labelEs: 'Gemma 4 E4B Uncensored', alias: 'gemma-4-e4b-uncensored', maxCtx: 32768, uncensored: true },
  // Gemma 4 E2B and Nemotron follow the gemma4 path (llama-server v1.0.0). Nemotron has no mmproj.
  { id: 'gemma-e2b', label: 'Gemma 4 E2B', labelEs: 'Gemma 4 E2B', alias: 'gemma-4-e2b', maxCtx: 32768, uncensored: false },
  { id: 'nemotron', label: 'Nemotron 3.5 Bolt E4B', labelEs: 'Nemotron 3.5 Bolt E4B', alias: 'nemotron-3.5-bolt-e4b', maxCtx: 32768, uncensored: false },
  { id: 'granite', label: 'Granite 4.2 3B', labelEs: 'Granite 4.2 3B', alias: 'granite-4.2-3b', maxCtx: 16384, uncensored: false },
  { id: 'spark', label: 'Spark X2.6 4B', labelEs: 'Spark X2.6 4B', alias: 'spark-x2.6-4b', maxCtx: 32768, uncensored: false },
  { id: 'lfm3', label: 'LFM3-VL 2B', labelEs: 'LFM3-VL 2B', alias: 'lfm3-vl-2b', maxCtx: 32768, uncensored: false },
  { id: 'gemma-e2b-heretic', label: 'Gemma 4 E2B Heretic', labelEs: 'Gemma 4 E2B heretic', alias: 'gemma-4-e2b-heretic', maxCtx: 32768, uncensored: true },
  { id: 'spark-heretic', label: 'Spark X2.6 4B Heretic', labelEs: 'Spark X2.6 4B heretic', alias: 'spark-x2.6-4b-heretic', maxCtx: 32768, uncensored: true },
  { id: 'granite-heretic', label: 'Granite 4.2 3B Heretic', labelEs: 'Granite 4.2 3B heretic', alias: 'granite-4.2-3b-heretic', maxCtx: 16384, uncensored: true },
  // The 2B abliterated GGUF, not the installed Qwen3.5-4B.
  { id: 'qwen2b', label: 'Qwen3.5-2B Abliterated', labelEs: 'Qwen3.5-2B abliterado', alias: 'qwen3.5-2b-abliterated', maxCtx: 32768, uncensored: true },
  { id: 'lfm3-abliterated', label: 'LFM3-VL 2B Abliterated', labelEs: 'LFM3-VL 2B abliterado', alias: 'lfm3-vl-2b-abliterated', maxCtx: 32768, uncensored: true },
];

export const censoredOptions = SERVE_MODEL_OPTIONS.filter((o) => !o.uncensored);
export const uncensoredOptions = SERVE_MODEL_OPTIONS.filter((o) => o.uncensored);

export const SERVE_MODEL_LABEL: Record<ServeModelId, string> = Object.fromEntries(
  SERVE_MODEL_OPTIONS.map((o) => [o.id, o.label]),
) as Record<ServeModelId, string>;

/** Upper bound used when /v1/models has not reported context_length for this target. */
export const SERVE_MODEL_MAX_CTX: Record<ServeModelId, number> = Object.fromEntries(
  SERVE_MODEL_OPTIONS.map((o) => [o.id, o.maxCtx]),
) as Record<ServeModelId, number>;

function option(id: ServeModelId): ServeModelOption {
  const row = SERVE_MODEL_OPTIONS.find((o) => o.id === id);
  if (!row) throw new Error(id);
  return row;
}

/** Name in the page language. English is the default for a locale this page does not speak. */
export function serveModelLabel(id: ServeModelId, locale: string): string {
  const row = option(id);
  return locale === 'es' ? row.labelEs : row.label;
}

/** Uncensored start targets (fewer safety filters than the stock checkpoints). */
export function isUncensoredServeModel(id: ServeModelId): boolean {
  return option(id).uncensored;
}

const CTX_LADDER = [4096, 8192, 16384, 32768, 65536, 131072] as const;

export function isServeModelRunning(runningModel: string | null | undefined, id: ServeModelId): boolean {
  if (!runningModel) return false;
  return runningModel.toLowerCase() === option(id).alias.toLowerCase();
}

/** Map a /v1/models id (e.g. qwen3.5-4b) to a serve start target when possible. */
export function matchServeModel(runningModel: string | null | undefined): ServeModelId | null {
  if (!runningModel) return null;
  for (const { id } of SERVE_MODEL_OPTIONS) {
    if (isServeModelRunning(runningModel, id)) return id;
  }
  return null;
}

/**
 * Context presets for a start target: auto (null) plus the ladder capped by the
 * model's max. When that model is already running and reported context_length,
 * that value is the cap (and is included if not already on the ladder).
 */
export function ctxPresetsFor(
  model: ServeModelId,
  runningModel: string | null | undefined,
  runningCtx: number | null | undefined,
): (number | null)[] {
  let max = SERVE_MODEL_MAX_CTX[model];
  if (isServeModelRunning(runningModel, model) && runningCtx && runningCtx > 0) {
    max = runningCtx;
  }
  const sizes: number[] = CTX_LADDER.filter((n) => n <= max);
  if (max > 0 && !sizes.includes(max)) {
    sizes.push(max);
    sizes.sort((a, b) => a - b);
  }
  return [null, ...sizes];
}

/** Short label for a preset size (4096 → "4k"). */
export function formatCtxPreset(n: number): string {
  if (n >= 1024 && n % 1024 === 0) return `${n / 1024}k`;
  return String(n);
}
