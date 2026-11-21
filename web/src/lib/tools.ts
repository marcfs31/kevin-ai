// The tools the chat offers the model. Each runs in this page, on the user's data only:
// no network, no files. Add one only if it is safe to run on whatever the model asks.
import type { WireTool } from './api';

export interface ToolResult {
  content: string;
  isError: boolean;
}

interface ToolDef {
  name: string;
  description: string;
  parameters: Record<string, unknown>;
  run: (args: Record<string, unknown>, now: Date) => string;
}

/** + - * / % ^ and parentheses over decimal numbers. No variables, no calls, no eval. */
export function calculate(expression: string): number {
  const src = expression.replace(/\s+/g, '');
  if (!src) throw new Error('empty expression');
  if (src.length > 200) throw new Error('expression too long');
  let i = 0;
  const peek = () => src[i] ?? '';
  const number = (): number => {
    const m = /^(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?/i.exec(src.slice(i));
    if (!m) throw new Error(`unexpected "${peek() || 'end'}" at position ${i + 1}`);
    i += m[0].length;
    return Number(m[0]);
  };
  const atom = (): number => {
    if (peek() === '(') {
      i += 1;
      const v = sum();
      if (peek() !== ')') throw new Error('missing )');
      i += 1;
      return v;
    }
    return number();
  };
  // Unary minus binds looser than ^, as in maths: -2 ^ 2 is -4.
  const unary = (): number => {
    if (peek() === '-') {
      i += 1;
      return -unary();
    }
    if (peek() === '+') {
      i += 1;
      return unary();
    }
    return power();
  };
  const power = (): number => {
    const base = atom();
    if (peek() !== '^') return base;
    i += 1;
    return base ** unary();
  };
  const product = (): number => {
    let v = unary();
    for (;;) {
      const op = peek();
      if (op !== '*' && op !== '/' && op !== '%') return v;
      i += 1;
      const r = unary();
      v = op === '*' ? v * r : op === '/' ? v / r : v % r;
    }
  };
  function sum(): number {
    let v = product();
    for (;;) {
      const op = peek();
      if (op !== '+' && op !== '-') return v;
      i += 1;
      const r = product();
      v = op === '+' ? v + r : v - r;
    }
  }
  const v = sum();
  if (i < src.length) throw new Error(`unexpected "${peek()}" at position ${i + 1}`);
  if (!Number.isFinite(v)) throw new Error('the result is not a finite number');
  return v;
}

const TOOLS: ToolDef[] = [
  {
    name: 'calculator',
    description: 'Evaluate an arithmetic expression with + - * / % ^ and parentheses, for example "(12.5 + 3) * 4".',
    parameters: {
      type: 'object',
      properties: { expression: { type: 'string', description: 'The expression to evaluate.' } },
      required: ['expression'],
    },
    run: (args) => {
      if (typeof args.expression !== 'string') throw new Error('expression must be a string');
      return String(calculate(args.expression));
    },
  },
  {
    name: 'current_time',
    description: 'The current date and time, optionally in an IANA time zone such as "Europe/Madrid".',
    parameters: {
      type: 'object',
      properties: { timezone: { type: 'string', description: 'IANA time zone name. Defaults to the browser’s.' } },
    },
    run: (args, now) => {
      const zone = typeof args.timezone === 'string' && args.timezone ? args.timezone : undefined;
      let text: string;
      try {
        text = new Intl.DateTimeFormat('en-GB', { dateStyle: 'full', timeStyle: 'long', timeZone: zone }).format(now);
      } catch {
        throw new Error(`unknown time zone "${String(args.timezone)}"`);
      }
      return `${text} (UTC ${now.toISOString()})`;
    },
  },
];

export function toolDefinitions(): WireTool[] {
  return TOOLS.map((t) => ({ type: 'function', function: { name: t.name, description: t.description, parameters: t.parameters } }));
}

/** Runs one call as the model wrote it. A failure is a result the model can read, never a throw. */
export function runTool(name: string, argumentsJson: string, now: Date = new Date()): ToolResult {
  const tool = TOOLS.find((t) => t.name === name);
  if (!tool) return { content: `error: no tool named "${name}"`, isError: true };
  let args: unknown;
  try {
    args = argumentsJson.trim() ? JSON.parse(argumentsJson) : {};
  } catch {
    return { content: 'error: the arguments are not valid JSON', isError: true };
  }
  if (args === null || typeof args !== 'object' || Array.isArray(args)) return { content: 'error: the arguments must be a JSON object', isError: true };
  try {
    return { content: tool.run(args as Record<string, unknown>, now), isError: false };
  } catch (e) {
    return { content: `error: ${e instanceof Error ? e.message : 'the tool failed'}`, isError: true };
  }
}
