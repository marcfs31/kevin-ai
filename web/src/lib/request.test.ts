import { describe, expect, it } from 'vitest';
import { toWire } from './request';
import type { UserMessage } from './storage';

describe('toWire', () => {
  const user = (over: Partial<UserMessage> = {}): UserMessage => ({ id: 'u', role: 'user', content: 'hi', typed: 'hi', attachmentNames: [], createdAt: 1, ...over });

  it('sends a user message\'s note after its content, the same way every time it is sent', () => {
    expect(toWire('', [user({ note: 'Workflow "X" now stands: 1. [done] a' })])).toEqual([{ role: 'user', content: 'hi\n\nWorkflow "X" now stands: 1. [done] a' }]);
    expect(toWire('', [user()])).toEqual([{ role: 'user', content: 'hi' }]);
    const parts = toWire('', [user({ content: [{ type: 'text', text: 'look' }, { type: 'image_url', image_url: { url: 'data:x' } }], note: 'n' })]);
    expect(parts[0]?.content).toEqual([{ type: 'text', text: 'look' }, { type: 'image_url', image_url: { url: 'data:x' } }, { type: 'text', text: 'n' }]);
  });
});
