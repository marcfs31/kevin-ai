import { describe, expect, it } from 'vitest';
import { calculate, runTool, toolDefinitions } from './tools';

describe('calculate', () => {
  it('follows precedence, powers, unary minus and parentheses', () => {
    expect(calculate('1 + 2 * 3')).toBe(7);
    expect(calculate('(1 + 2) * 3')).toBe(9);
    expect(calculate('2 ^ 3 ^ 2')).toBe(512);
    expect(calculate('-2 ^ 2')).toBe(-4);
    expect(calculate('10 % 4 + 7 / 2')).toBe(5.5);
    expect(calculate('1.5e2 + .5')).toBe(150.5);
  });

  it('refuses anything that is not arithmetic', () => {
    for (const bad of ['', '2 +', '(1', '1)', 'alert(1)', '1; 2', 'process', '1/0', '0/0', '2'.repeat(201)]) expect(() => calculate(bad), bad).toThrow();
  });
});

describe('runTool', () => {
  const now = new Date('2026-11-21T12:30:00Z');

  it('offers well-formed definitions', () => {
    const defs = toolDefinitions();
    expect(defs.map((d) => d.function.name)).toEqual(['calculator', 'current_time']);
    for (const d of defs) expect(d.function.parameters.type).toBe('object');
  });

  it('runs a call', () => {
    expect(runTool('calculator', '{"expression":"6*7"}')).toEqual({ content: '42', isError: false });
    const t = runTool('current_time', '{"timezone":"Europe/Madrid"}', now);
    expect(t.isError).toBe(false);
    expect(t.content).toContain('13:30');
    expect(runTool('current_time', '', now).isError).toBe(false);
  });

  it('turns every failure into a result the model can read', () => {
    expect(runTool('nope', '{}')).toMatchObject({ isError: true, content: 'error: no tool named "nope"' });
    expect(runTool('calculator', '{oops')).toMatchObject({ isError: true });
    expect(runTool('calculator', '[1]')).toMatchObject({ isError: true });
    expect(runTool('calculator', '{"expression":5}')).toMatchObject({ isError: true });
    expect(runTool('calculator', '{"expression":"1/0"}')).toMatchObject({ isError: true });
    expect(runTool('current_time', '{"timezone":"Mars/Base"}', now)).toMatchObject({ isError: true });
  });
});
