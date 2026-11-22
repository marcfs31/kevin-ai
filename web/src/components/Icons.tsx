// Small outline icons drawn inline (no icon font, no image request, so the CSP stays as it is).
// They only decorate a label that is always written next to them, so they are hidden from screen readers.
function Icon({ children }: { children: React.ReactNode }) {
  return (
    <svg className="icon" viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" focusable="false">
      {children}
    </svg>
  );
}

/** A document with a folded corner: files. */
export function FileIcon() {
  return (
    <Icon>
      <path d="M14 3v4a1 1 0 0 0 1 1h4" />
      <path d="M17 21h-10a2 2 0 0 1-2-2v-14a2 2 0 0 1 2-2h7l5 5v11a2 2 0 0 1-2 2z" />
      <path d="M9 13h6M9 17h4" />
    </Icon>
  );
}

/** A framed picture with a sun and hills: images. */
export function ImageIcon() {
  return (
    <Icon>
      <path d="M3 6a3 3 0 0 1 3-3h12a3 3 0 0 1 3 3v12a3 3 0 0 1-3 3h-12a3 3 0 0 1-3-3z" />
      <path d="M15 8h.01" />
      <path d="M3 16l5-5c.93-.89 2.07-.89 3 0l5 5" />
      <path d="M14 14l1-1c.93-.89 2.07-.89 3 0l3 3" />
    </Icon>
  );
}
