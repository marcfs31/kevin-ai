// Re-exports for the ServerPanel. The actual fetch lives in api.ts (privacy gate).
export type { ServeModelId, ServeOptions, ServeResult } from './serveControl';
export { SERVE_DEFAULTS, parseServeOptions, serveScriptArgs } from './serveControl';
export { startRemoteServe } from './api';
