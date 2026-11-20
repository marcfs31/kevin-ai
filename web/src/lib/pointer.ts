/** A mouse or trackpad. A phone reports a coarse pointer, so Enter must insert a line. */
export function finePointer(): boolean {
  if (typeof window.matchMedia !== 'function') return true;
  return window.matchMedia('(hover: hover) and (pointer: fine)').matches;
}
