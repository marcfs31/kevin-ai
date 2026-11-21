import { describe, expect, it } from 'vitest';
import { NEAR_BOTTOM_PX, isNearBottom } from './scroll';

describe('isNearBottom', () => {
  it('is true when the viewport sits on the last pixels', () => {
    expect(isNearBottom({ scrollTop: 920, clientHeight: 400, scrollHeight: 1320 })).toBe(true);
  });

  it('is true within the stickiness threshold', () => {
    expect(isNearBottom({ scrollTop: 920 - NEAR_BOTTOM_PX, clientHeight: 400, scrollHeight: 1320 })).toBe(true);
  });

  it('is false once the user has scrolled further up', () => {
    expect(isNearBottom({ scrollTop: 920 - NEAR_BOTTOM_PX - 1, clientHeight: 400, scrollHeight: 1320 })).toBe(false);
  });

  it('is true for a short transcript that does not overflow', () => {
    expect(isNearBottom({ scrollTop: 0, clientHeight: 400, scrollHeight: 200 })).toBe(true);
  });
});
