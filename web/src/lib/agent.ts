// One user message to its final answer: stream a reply, run the tools it asks for, send the
// results back, and repeat until it answers in text, is stopped, fails, or hits the step cap.
// No React and no fetch of its own: the caller passes the streaming function and gets every
// finished message through onCommit, so a test can drive it with a fake upstream.
import type { ApiError, ChatRequest, ServerInfo } from './api';
import { ApiError as ApiErrorClass } from './api';
import { newId } from './attachments';
import { planCompaction, shouldCompact, type Plan } from './compact';
import { toWire } from './request';
import type { StreamEvent, Usage } from './sse';
import type { AssistantMessage, Message, Settings, ToolCall, ToolMessage } from './storage';
import { estimateWireTokens } from './tokens';
import { runTool, toolDefinitions, type ToolResult } from './tools';

/** What the screen shows about the reply that is streaming. */
export interface LiveInfo {
  startedAt: number;
  firstAt: number | null;
  chunks: number;
  /** Estimated prompt tokens, used until the server sends its usage. */
  promptEstimate: number;
  usage: Usage | null;
}

export type StreamFn = (req: ChatRequest, signal: AbortSignal, onEvent: (e: StreamEvent) => void) => Promise<void>;

export interface AgentOptions {
  /** The chat so far, ending with the user's new message. */
  history: Message[];
  settings: Settings;
  /** Sent as `model` when the server lists several. */
  model: string | null;
  info: ServerInfo | null;
  signal: AbortSignal;
  stream: StreamFn;
  /** The reply as it streams; called on every change. */
  onLive: (message: AssistantMessage, info: LiveInfo) => void;
  /** A reply or tool result that is final. */
  onCommit: (message: Message) => void;
  /** Summarises older turns when the chat nears the context between steps (tool results grow it).
   * Returns the new history, or null when it failed or was stopped: then nothing changed. */
  compact?: (history: Message[], plan: Plan) => Promise<Message[] | null>;
  /** Runs a tool call; the real one unless a test replaces it. */
  run?: (call: ToolCall) => ToolResult;
}

const isAbort = (e: unknown): boolean => e instanceof DOMException && e.name === 'AbortError';

/** Streams one reply into `message`, adding the tool calls the model asked for. Never throws. */
async function streamReply(o: AgentOptions, wire: ReturnType<typeof toWire>, message: AssistantMessage, tools: boolean, last: boolean): Promise<void> {
  const promptEstimate = estimateWireTokens(wire, o.info);
  const live: LiveInfo = { startedAt: performance.now(), firstAt: null, chunks: 0, promptEstimate, usage: null };
  const calls = new Map<number, ToolCall>();
  const publish = () => o.onLive({ ...message, toolCalls: calls.size ? [...calls.values()] : undefined }, { ...live });
  publish();
  let usage: Usage | null = null;
  try {
    await o.stream(
      {
        messages: wire,
        temperature: o.settings.temperature,
        maxTokens: o.settings.maxTokens,
        model: o.model,
        tools: tools ? toolDefinitions() : undefined,
        // On the last step the tools stay in the prompt but the model must answer in text.
        toolChoice: tools ? (last ? 'none' : 'auto') : undefined,
      },
      o.signal,
      (e) => {
        if (e.kind === 'content' || e.kind === 'reasoning') {
          if (live.firstAt === null) live.firstAt = performance.now();
          live.chunks += 1;
          if (e.kind === 'content') message.content += e.text;
          else message.reasoning += e.text;
          publish();
        } else if (e.kind === 'tool_call') {
          if (live.firstAt === null) live.firstAt = performance.now();
          const prev = calls.get(e.index);
          calls.set(e.index, {
            id: e.id ?? prev?.id ?? `call_${e.index}`,
            name: e.name ?? prev?.name ?? '',
            arguments: (prev?.arguments ?? '') + e.arguments,
          });
          publish();
        } else if (e.kind === 'usage') {
          usage = e.usage;
          live.usage = e.usage;
          publish();
        }
      },
    );
  } catch (e) {
    if (isAbort(e)) message.status = 'stopped';
    else {
      const err: ApiError = e instanceof ApiErrorClass ? e : new ApiErrorClass(0, null);
      message.status = 'error';
      message.error = { code: err.code, message: err.serverMessage, status: err.status };
    }
  }
  const end = performance.now();
  const u = usage as Usage | null;
  const generated = u?.completionTokens ?? live.chunks;
  const decodeSecs = live.firstAt === null ? 0 : (end - live.firstAt) / 1000;
  message.stats = {
    ttftMs: live.firstAt === null ? null : live.firstAt - live.startedAt,
    tokensPerSecond: generated > 1 && decodeSecs > 0 ? (generated - 1) / decodeSecs : null,
    promptTokens: u?.promptTokens ?? null,
    cachedTokens: u?.cachedTokens ?? null,
    completionTokens: u?.completionTokens ?? null,
    promptEstimate,
    completionEstimate: live.chunks,
  };
  if (calls.size) message.toolCalls = [...calls.values()];
}

/** Runs the loop and returns once the last message is committed. */
export async function runAgent(o: AgentOptions): Promise<void> {
  const useTools = o.settings.tools;
  const cap = Math.max(1, o.settings.maxSteps);
  const run = o.run ?? ((c: ToolCall) => runTool(c.name, c.arguments));
  let history = [...o.history];
  const commit = (m: Message) => {
    history.push(m);
    o.onCommit(m);
  };
  for (let step = 1; ; step++) {
    const message: AssistantMessage = {
      id: newId(), role: 'assistant', content: '', reasoning: '', status: 'done', error: null, stats: null, createdAt: Date.now(),
    };
    // Stopped between two steps: say so instead of leaving a chat that just ends.
    if (o.signal.aborted) {
      commit({ ...message, status: 'stopped' });
      return;
    }
    if (step > 1 && o.compact && shouldCompact(history, o.settings.systemPrompt, 0, o.info)) {
      const plan = planCompaction(history);
      if (plan) history = (await o.compact(history, plan)) ?? history;
    }
    const last = useTools && step >= cap;
    await streamReply(o, toWire(o.settings.systemPrompt, history, o.info?.vision === 'ok'), message, useTools, last);
    const calls = useTools && message.status === 'done' ? (message.toolCalls ?? []) : [];
    if (!calls.length) {
      commit(message);
      return;
    }
    if (last) {
      // The model asked again on the last step: record the calls, run nothing.
      commit({ ...message, stepLimit: cap });
      return;
    }
    commit(message);
    for (const call of calls) {
      const r = run(call);
      const result: ToolMessage = {
        id: newId(), role: 'tool', toolCallId: call.id, name: call.name, content: r.content, isError: r.isError, createdAt: Date.now(),
      };
      commit(result);
    }
  }
}
