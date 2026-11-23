import { useSyncExternalStore } from 'react';

/** The width at which the layout leaves its one-column phone form (the same 48rem as the stylesheet). */
const WIDE = '(min-width: 48.01rem)';

function subscribe(onChange: () => void): () => void {
  if (typeof window.matchMedia !== 'function') return () => undefined;
  const query = window.matchMedia(WIDE);
  query.addEventListener('change', onChange);
  return () => query.removeEventListener('change', onChange);
}

const snapshot = (): boolean => (typeof window.matchMedia === 'function' ? window.matchMedia(WIDE).matches : true);

/** True on a screen wide enough for the side panes. Without matchMedia (old test environments) it says yes. */
export function useWide(): boolean {
  return useSyncExternalStore(subscribe, snapshot, () => true);
}
