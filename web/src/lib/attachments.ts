// Attachments: classification, text insertion, PDF page selection and the wire format.
// Pure functions only; browser decoding lives in image.ts and pdf.ts.
import type { ContentPart, ServerInfo } from './api';
import { estimateImageTokens, estimateTextTokens } from './estimate';

export const MAX_BODY_BYTES = 24 * 1024 * 1024;
export const MAX_IMAGES = 8;
/** Server refuses a decoded image larger than this on either side before resizing (Phase 3). */
export const MAX_DECODE_SIDE = 4096;
/** lm-serve --image-max-pixels default when /v1/models omits max_pixels. */
export const DEFAULT_MAX_PIXELS = 1_048_576;
export const MAX_TEXT_FILE_BYTES = 2 * 1024 * 1024;
export const MAX_IMAGE_FILE_BYTES = 30 * 1024 * 1024;
export const MAX_PDF_FILE_BYTES = 60 * 1024 * 1024;
export const IMAGE_PRESETS = [512, 768, 1024] as const;
export type ImagePreset = (typeof IMAGE_PRESETS)[number];

export type AttachErrorCode = 'unsupported_type' | 'binary_file' | 'too_large' | 'read_failed' | 'vision_off' | 'empty_pdf';

export class AttachError extends Error {
  readonly code: AttachErrorCode;
  readonly fileName: string;
  constructor(code: AttachErrorCode, fileName: string) {
    super(`${code}: ${fileName}`);
    this.name = 'AttachError';
    this.code = code;
    this.fileName = fileName;
  }
}

export interface ImageData {
  dataUrl: string;
  width: number;
  height: number;
}

export interface PdfPage {
  number: number;
  /** Text of the page, or null when it has no text layer. */
  text: string | null;
  /** The page rendered to PNG, only for pages without text and only when vision is ok. */
  image: ImageData | null;
}

export type Attachment =
  | { id: string; kind: 'text'; name: string; text: string }
  | ({ id: string; kind: 'image'; name: string } & ImageData)
  | { id: string; kind: 'pdf'; name: string; pages: PdfPage[]; range: string };

const TEXT_EXTENSIONS: Record<string, string> = {
  txt: '', md: 'markdown', markdown: 'markdown', json: 'json', csv: 'csv', tsv: 'tsv', log: '',
  c: 'c', h: 'c', cc: 'cpp', cpp: 'cpp', hpp: 'cpp', cs: 'csharp', go: 'go', rs: 'rust', py: 'python',
  rb: 'ruby', php: 'php', java: 'java', kt: 'kotlin', swift: 'swift', js: 'javascript', mjs: 'javascript',
  cjs: 'javascript', jsx: 'jsx', ts: 'typescript', tsx: 'tsx', css: 'css', scss: 'scss', html: 'html',
  xml: 'xml', svg: 'xml', yaml: 'yaml', yml: 'yaml', toml: 'toml', ini: 'ini', sh: 'bash', bash: 'bash',
  zsh: 'bash', sql: 'sql', lua: 'lua', r: 'r', m: 'objectivec', scala: 'scala', dart: 'dart', vue: 'vue',
  tex: 'latex', diff: 'diff', patch: 'diff', mk: 'makefile', cmake: 'cmake', proto: 'protobuf', env: '',
};
const TEXT_NAMES = new Set(['makefile', 'dockerfile', 'readme', 'license', 'notice']);
const IMAGE_TYPES = new Set(['image/png', 'image/jpeg', 'image/webp']);
const IMAGE_EXTENSIONS = new Set(['png', 'jpg', 'jpeg', 'webp']);

export type FileKind = 'text' | 'pdf' | 'image';

function extension(name: string): string {
  const dot = name.lastIndexOf('.');
  return dot > 0 ? name.slice(dot + 1).toLowerCase() : '';
}

/** What a file is from its name and MIME type, or null when it is not accepted. */
export function classifyFile(name: string, mime: string): FileKind | null {
  const ext = extension(name);
  if (mime === 'application/pdf' || ext === 'pdf') return 'pdf';
  if (IMAGE_TYPES.has(mime) || IMAGE_EXTENSIONS.has(ext)) return 'image';
  if (ext in TEXT_EXTENSIONS || TEXT_NAMES.has(name.toLowerCase())) return 'text';
  return null;
}

export const ACCEPT_ATTRIBUTE = [
  '.pdf', '.png', '.jpg', '.jpeg', '.webp',
  ...Object.keys(TEXT_EXTENSIONS).map((e) => `.${e}`),
].join(',');

export function maxBytes(kind: FileKind): number {
  return kind === 'text' ? MAX_TEXT_FILE_BYTES : kind === 'image' ? MAX_IMAGE_FILE_BYTES : MAX_PDF_FILE_BYTES;
}

/** Decoded file text, refused when it looks binary (NUL bytes or invalid UTF-8). */
export function checkText(name: string, text: string): string {
  if (text.includes('\u0000') || text.includes('�')) throw new AttachError('binary_file', name);
  return text;
}

/** A fenced code block, with a fence longer than any backtick run inside the text. */
export function fence(name: string, text: string): string {
  const longest = Math.max(2, ...Array.from(text.matchAll(/`+/g), (m) => m[0].length));
  const ticks = '`'.repeat(longest + 1);
  const lang = TEXT_EXTENSIONS[extension(name)] ?? '';
  const body = text.endsWith('\n') ? text : `${text}\n`;
  return `${name}\n${ticks}${lang}\n${body}${ticks}`;
}

/** Parses "1-3, 5" into sorted unique page numbers within 1..count, or null when invalid. */
export function parsePageRange(input: string, count: number): number[] | null {
  const pages = new Set<number>();
  const parts = input.split(',').map((p) => p.trim()).filter(Boolean);
  if (!parts.length) return null;
  for (const part of parts) {
    const m = /^(\d+)(?:\s*-\s*(\d+))?$/.exec(part);
    if (!m) return null;
    const a = Number(m[1]);
    const b = m[2] === undefined ? a : Number(m[2]);
    if (a < 1 || b < a || b > count) return null;
    for (let p = a; p <= b; p++) pages.add(p);
  }
  return [...pages].sort((x, y) => x - y);
}

export interface TextItemLike {
  str?: string;
  hasEOL?: boolean;
}

/** A PDF page's text from pdf.js text items, one line per end-of-line mark. */
export function joinTextItems(items: TextItemLike[]): string {
  let out = '';
  for (const item of items) {
    if (typeof item.str !== 'string') continue;
    out += item.str;
    if (item.hasEOL) out += '\n';
  }
  return out.replace(/[ \t]+\n/g, '\n').trim();
}

export interface PdfSelection {
  text: string;
  images: ImageData[];
  /** Selected pages with no text that cannot be sent (vision is not ok). */
  skipped: number[];
}

/** The selected pages of a PDF as text with page markers, plus page images when allowed. */
export function selectPdf(att: Extract<Attachment, { kind: 'pdf' }>, visionOk: boolean): PdfSelection | null {
  const numbers = parsePageRange(att.range, att.pages.length);
  if (!numbers) return null;
  const blocks: string[] = [];
  const images: ImageData[] = [];
  const skipped: number[] = [];
  for (const n of numbers) {
    const page = att.pages[n - 1];
    if (!page) continue;
    if (page.text !== null) blocks.push(`--- page ${n} ---\n${page.text}`);
    else if (visionOk && page.image) {
      blocks.push(`--- page ${n} (image) ---`);
      images.push(page.image);
    } else skipped.push(n);
  }
  const text = blocks.length ? `${att.name}\n${blocks.join('\n\n')}` : '';
  return { text, images, skipped };
}

export interface Estimate {
  /** null when the server gives no way to count image tokens. */
  tokens: number | null;
  images: number;
}

export function estimateAttachment(att: Attachment, info: ServerInfo | null): Estimate {
  const visionOk = info?.vision === 'ok';
  if (att.kind === 'text') return { tokens: estimateTextTokens(fence(att.name, att.text)), images: 0 };
  if (att.kind === 'image') return { tokens: estimateImageTokens(att.width, att.height, info), images: 1 };
  const sel = selectPdf(att, visionOk);
  if (!sel) return { tokens: 0, images: 0 };
  let tokens: number | null = estimateTextTokens(sel.text);
  for (const img of sel.images) {
    const t = estimateImageTokens(img.width, img.height, info);
    tokens = tokens === null || t === null ? null : tokens + t;
  }
  return { tokens, images: sel.images.length };
}

/** OpenAI-style data URLs the server accepts (remote URLs are refused). */
export function isAllowedImageDataUrl(url: string): boolean {
  return /^data:image\/(?:png|jpeg|jpg);base64,/i.test(url);
}

/** The user message content: a plain string, or text and image parts when images are attached. */
export function buildUserContent(text: string, attachments: Attachment[], visionOk: boolean): string | ContentPart[] {
  const blocks: string[] = [];
  const images: ImageData[] = [];
  for (const att of attachments) {
    if (att.kind === 'text') blocks.push(fence(att.name, att.text));
    else if (att.kind === 'image') {
      // Keep old servers text-only: never put image parts on the wire without vision.
      // Only PNG/JPEG data URLs — never remote http(s) URLs.
      if (visionOk && isAllowedImageDataUrl(att.dataUrl)) images.push(att);
    } else {
      const sel = selectPdf(att, visionOk);
      if (sel?.text) blocks.push(sel.text);
      if (sel) images.push(...sel.images.filter((img) => isAllowedImageDataUrl(img.dataUrl)));
    }
  }
  if (text.trim()) blocks.push(text);
  const joined = blocks.join('\n\n');
  if (!images.length) return joined;
  return [
    ...images.map((img): ContentPart => ({ type: 'image_url', image_url: { url: img.dataUrl } })),
    { type: 'text', text: joined },
  ];
}

export function countImages(content: string | ContentPart[]): number {
  return typeof content === 'string' ? 0 : content.filter((p) => p.type === 'image_url').length;
}

/** Text of a message's content, for estimates and display. */
export function contentText(content: string | ContentPart[]): string {
  return typeof content === 'string' ? content : content.map((p) => (p.type === 'text' ? p.text : '')).join('');
}

/** Target size for an image: the longer side at most `preset`, the area at most `maxPixels`. */
export function fitSize(width: number, height: number, preset: number, maxPixels: number | null): { width: number; height: number } {
  let scale = Math.min(1, preset / Math.max(width, height));
  if (maxPixels && width * height * scale * scale > maxPixels) scale = Math.sqrt(maxPixels / (width * height));
  return { width: Math.max(1, Math.floor(width * scale)), height: Math.max(1, Math.floor(height * scale)) };
}

let counter = 0;
export function newId(): string {
  counter += 1;
  return `${Date.now().toString(36)}-${counter.toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
}

/** The most characters of a file the preview shows; longer ones are cut and say so. */
export const PREVIEW_MAX_CHARS = 50_000;

/** What was sent as text besides what the user typed: the attached text files and PDF pages. Empty when there were none. */
export function sentFileText(content: string | ContentPart[], typed: string): string {
  const text = contentText(content);
  const files = typed && text.endsWith(typed) ? text.slice(0, text.length - typed.length) : text;
  return files.trim();
}
