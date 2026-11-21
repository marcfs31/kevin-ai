import { useEffect, useRef, useState, type ReactNode } from 'react';
import ReactMarkdown, { type Components } from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { useI18n } from '../i18n';

interface MdNode {
  type: string;
  value?: string;
  children?: MdNode[];
}

/** Raw HTML in a reply is shown as literal text, never parsed or dropped. */
export function remarkHtmlAsText() {
  const walk = (node: MdNode) => {
    if (node.type === 'html') node.type = 'text';
    node.children?.forEach(walk);
  };
  return (tree: MdNode) => walk(tree);
}

/** Prefers a second Copy when the block is taller than one viewport thumb-reach. */
export const DUAL_COPY_MIN_HEIGHT_PX = 320;

export function needsDualCopy(heightPx: number): boolean {
  return heightPx >= DUAL_COPY_MIN_HEIGHT_PX;
}

function textOf(node: ReactNode): string {
  if (typeof node === 'string' || typeof node === 'number') return String(node);
  if (Array.isArray(node)) return node.map(textOf).join('');
  if (node && typeof node === 'object' && 'props' in node) return textOf((node.props as { children?: ReactNode }).children);
  return '';
}

export function CopyButton({ text, label }: { text: string; label: string }) {
  const { t } = useI18n();
  const [copied, setCopied] = useState(false);
  const copy = () => {
    void navigator.clipboard?.writeText(text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  };
  return (
    <button type="button" className="copy" onClick={copy}>
      {copied ? t('message.copied') : label}
    </button>
  );
}

function Pre({ children }: { children?: ReactNode }) {
  const { t } = useI18n();
  const preRef = useRef<HTMLPreElement>(null);
  const [dual, setDual] = useState(false);
  const text = textOf(children).replace(/\n$/, '');
  const label = t('message.copyCode');

  useEffect(() => {
    const pre = preRef.current;
    if (!pre) return;
    const measure = () => setDual(needsDualCopy(pre.getBoundingClientRect().height));
    measure();
    if (typeof ResizeObserver === 'undefined') return;
    const ro = new ResizeObserver(measure);
    ro.observe(pre);
    return () => ro.disconnect();
  }, [children]);

  return (
    <div className="code-block">
      <CopyButton text={text} label={label} />
      <pre ref={preRef}>{children}</pre>
      {dual && <CopyButton text={text} label={label} />}
    </div>
  );
}

const components: Components = {
  pre: Pre,
  a: ({ href, children }) => (
    <a href={href} target="_blank" rel="noopener noreferrer">
      {children}
    </a>
  ),
  // Never load images named in a reply: they could be remote trackers.
  img: ({ alt, src }) => <span className="img-alt">[{alt || (typeof src === 'string' ? src : '')}]</span>,
};

export function Markdown({ text }: { text: string }) {
  return (
    <ReactMarkdown remarkPlugins={[remarkGfm, remarkHtmlAsText]} components={components}>
      {text}
    </ReactMarkdown>
  );
}
