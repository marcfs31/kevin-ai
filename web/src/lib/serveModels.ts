import type { ServeModelId } from './serveControl';

/** Fixed start targets Settings / the header picker can launch. */
export const SERVE_MODEL_OPTIONS: { id: ServeModelId; label: string }[] = [
  { id: 'qwen', label: 'Qwen3.5-4B' },
  { id: 'qwen-uncensored', label: 'Qwen3.5-4B Uncensored' },
  { id: 'gemma', label: 'Gemma 4 E4B' },
  { id: 'whiterabbitneo', label: 'WhiteRabbitNeo 2.5 7B' },
  { id: 'gemma-uncensored', label: 'Gemma 4 E4B Uncensored' },
];

export const SERVE_MODEL_LABEL: Record<ServeModelId, string> = {
  qwen: 'Qwen3.5-4B',
  'qwen-uncensored': 'Qwen3.5-4B Uncensored',
  gemma: 'Gemma 4 E4B',
  whiterabbitneo: 'WhiteRabbitNeo 2.5 7B',
  'gemma-uncensored': 'Gemma 4 E4B Uncensored',
};

/** Upper bound used when /v1/models has not reported context_length for this target. */
export const SERVE_MODEL_MAX_CTX: Record<ServeModelId, number> = {
  qwen: 131072,
  'qwen-uncensored': 131072,
  gemma: 131072,
  // llama-server allocates the whole window up front; 8 GB of Q8_0 weights leave room for 16k.
  whiterabbitneo: 16384,
  // llama-server too. Q6_K_P and the projector are 7.5 GiB resident at 32k (measured), and the
  // KV cache is small (sliding-window layers); the prompt reads at about 16 tokens/s, so more is of little use.
  'gemma-uncensored': 32768,
};

const CTX_LADDER = [4096, 8192, 16384, 32768, 65536, 131072] as const;

export function isServeModelRunning(runningModel: string | null | undefined, id: ServeModelId): boolean {
  if (!runningModel) return false;
  const name = runningModel.toLowerCase();
  // The uncensored model's name also contains "qwen"; keep the two targets apart.
  if (id === 'qwen') return name.includes('qwen') && !name.includes('uncensored');
  if (id === 'qwen-uncensored') return name.includes('qwen') && name.includes('uncensored');
  // Same for Gemma: the uncensored llama-server alias is gemma-4-e4b-uncensored, lm-serve's is gemma-4-e4b.
  if (id === 'gemma') return name.includes('gemma') && !name.includes('uncensored');
  if (id === 'gemma-uncensored') return name.includes('gemma') && name.includes('uncensored');
  return name.includes(id);
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
