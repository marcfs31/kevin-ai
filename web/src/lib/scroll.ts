/** How close to the end counts as “still following” the live reply. */
export const NEAR_BOTTOM_PX = 80;

export interface ScrollMetrics {
  scrollTop: number;
  clientHeight: number;
  scrollHeight: number;
}

/** True when the scrollport is already at (or within NEAR_BOTTOM_PX of) the bottom. */
export function isNearBottom(el: ScrollMetrics, threshold = NEAR_BOTTOM_PX): boolean {
  return el.scrollHeight - el.scrollTop - el.clientHeight <= threshold;
}
