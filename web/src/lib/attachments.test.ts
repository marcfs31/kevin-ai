import { describe, expect, it } from 'vitest';
import type { ServerInfo } from './api';
import {
  AttachError, DEFAULT_MAX_PIXELS, MAX_BODY_BYTES, MAX_DECODE_SIDE, MAX_IMAGES, buildUserContent, checkText, classifyFile,
  estimateAttachment, fence, fitSize, isAllowedImageDataUrl, joinTextItems, parsePageRange, selectPdf, type Attachment,
} from './attachments';
import { checkContext, estimateImageTokens, estimateTextTokens, estimateWaitSeconds } from './estimate';
import { requestBody } from './api';
import { assess, requestBytes, toWire } from './request';
import { DEFAULT_SETTINGS, type Message } from './storage';

const info = (over: Partial<ServerInfo> = {}): ServerInfo => ({
  model: 'm', contextLength: null, vision: 'unknown', maxPixels: null, pixelsPerToken: null, tokensPerImage: null, models: [], ...over,
});
const pdf = (pages: Array<string | null>, range: string, image = true): Extract<Attachment, { kind: 'pdf' }> => ({
  id: 'p', kind: 'pdf', name: 'doc.pdf', range,
  pages: pages.map((text, i) => ({ number: i + 1, text, image: text === null && image ? { dataUrl: `data:image/png;base64,P${i + 1}`, width: 100, height: 200 } : null })),
});

describe('classifyFile', () => {
  it('accepts text, code, PDF and PNG/JPEG/WebP', () => {
    expect(classifyFile('notes.md', 'text/markdown')).toBe('text');
    expect(classifyFile('main.c', '')).toBe('text');
    expect(classifyFile('Makefile', '')).toBe('text');
    expect(classifyFile('paper.PDF', '')).toBe('pdf');
    expect(classifyFile('x.webp', 'image/webp')).toBe('image');
    expect(classifyFile('x.jpg', '')).toBe('image');
  });
  it('refuses other files (negative case)', () => {
    expect(classifyFile('song.mp3', 'audio/mpeg')).toBeNull();
    expect(classifyFile('archive.zip', 'application/zip')).toBeNull();
    expect(classifyFile('anim.gif', 'image/gif')).toBeNull();
  });
});

describe('checkText', () => {
  it('passes text and refuses binary content', () => {
    expect(checkText('a.txt', 'héllo')).toBe('héllo');
    expect(() => checkText('a.txt', 'a\u0000b')).toThrow(AttachError);
    expect(() => checkText('a.txt', 'bad � byte')).toThrow(AttachError);
  });
});

describe('fence', () => {
  it('names the file and picks a language', () => {
    expect(fence('a.py', 'print(1)')).toBe('a.py\n```python\nprint(1)\n```');
  });
  it('uses a fence longer than any backtick run inside', () => {
    expect(fence('r.md', 'x ```` y\n')).toBe('r.md\n`````markdown\nx ```` y\n`````');
  });
});

describe('parsePageRange', () => {
  it('parses lists and ranges, sorted and unique', () => {
    expect(parsePageRange('3, 1-2, 2', 5)).toEqual([1, 2, 3]);
    expect(parsePageRange(' 4 - 5 ', 5)).toEqual([4, 5]);
  });
  it('refuses out-of-range and malformed input', () => {
    expect(parsePageRange('0', 5)).toBeNull();
    expect(parsePageRange('6', 5)).toBeNull();
    expect(parsePageRange('3-1', 5)).toBeNull();
    expect(parsePageRange('a', 5)).toBeNull();
    expect(parsePageRange('', 5)).toBeNull();
  });
});

describe('selectPdf', () => {
  it('marks pages and sends pages without text as images when vision is ok', () => {
    const sel = selectPdf(pdf(['one', null, 'three'], '1-3'), true);
    expect(sel?.text).toBe('doc.pdf\n--- page 1 ---\none\n\n--- page 2 (image) ---\n\n--- page 3 ---\nthree');
    expect(sel?.images).toHaveLength(1);
    expect(sel?.skipped).toEqual([]);
  });
  it('skips pages without text when vision is not ok', () => {
    const sel = selectPdf(pdf(['one', null], '1-2'), false);
    expect(sel?.images).toEqual([]);
    expect(sel?.skipped).toEqual([2]);
  });
  it('respects the page range', () => {
    expect(selectPdf(pdf(['one', 'two', 'three'], '2'), false)?.text).toBe('doc.pdf\n--- page 2 ---\ntwo');
    expect(selectPdf(pdf(['one'], '2'), false)).toBeNull();
  });
});

describe('estimates', () => {
  it('counts text at 3.5 characters a token, rounded up', () => {
    expect(estimateTextTokens('')).toBe(0);
    expect(estimateTextTokens('abcdefg')).toBe(2);
    expect(estimateTextTokens('abcdefgh')).toBe(3);
  });
  it('counts images from tokens_per_image first, then pixels_per_token, else unknown', () => {
    expect(estimateImageTokens(512, 512, info({ tokensPerImage: 280, pixelsPerToken: 1024 }))).toBe(280);
    expect(estimateImageTokens(512, 512, info({ pixelsPerToken: 1024 }))).toBe(256);
    expect(estimateImageTokens(512, 512, info())).toBeNull();
    expect(estimateImageTokens(512, 512, null)).toBeNull();
  });
  it('waits at 12 prompt tokens a second', () => {
    expect(estimateWaitSeconds(1440)).toBe(120);
  });
  it('checks the context only when the server gives it', () => {
    expect(checkContext(100, info())).toEqual({ kind: 'unknown' });
    expect(checkContext(100, info({ contextLength: 100 }))).toEqual({ kind: 'ok' });
    expect(checkContext(101, info({ contextLength: 100 }))).toEqual({ kind: 'overflow', tokens: 101, limit: 100 });
  });
  it('estimates attachments', () => {
    const text: Attachment = { id: 't', kind: 'text', name: 'a.txt', text: 'x'.repeat(70) };
    expect(estimateAttachment(text, null).tokens).toBe(estimateTextTokens(fence('a.txt', 'x'.repeat(70))));
    const img: Attachment = { id: 'i', kind: 'image', name: 'a.png', dataUrl: 'data:image/png;base64,A', width: 10, height: 10 };
    expect(estimateAttachment(img, info({ vision: 'ok' }))).toEqual({ tokens: null, images: 1 });
    expect(estimateAttachment(pdf(['abc', null], '1-2'), info({ vision: 'ok', pixelsPerToken: 100 })).tokens).toBe(
      estimateTextTokens('doc.pdf\n--- page 1 ---\nabc\n\n--- page 2 (image) ---') + 200,
    );
  });
});

describe('requestBytes', () => {
  it('matches the UTF-8 size of the real request body', () => {
    const img = { type: 'image_url' as const, image_url: { url: `data:image/png;base64,${'A/+='.repeat(1000)}` } };
    const messages = [
      { role: 'system' as const, content: 'Sé breve "y" \\ claro\n' },
      { role: 'user' as const, content: [img, img, { type: 'text' as const, text: 'qué es <esto> 😀' }] },
      { role: 'assistant' as const, content: 'Un gato.' },
    ];
    for (const maxTokens of [null, 128]) {
      const real = new TextEncoder().encode(requestBody({ messages, temperature: 0.7, maxTokens })).length;
      expect(requestBytes(messages, { temperature: 0.7, maxTokens })).toBe(real);
    }
    expect(requestBytes([], { temperature: 0, maxTokens: null })).toBe(new TextEncoder().encode(requestBody({ messages: [], temperature: 0, maxTokens: null })).length);
  });
});

describe('joinTextItems', () => {
  it('breaks lines at end-of-line marks and trims trailing spaces', () => {
    expect(joinTextItems([{ str: 'Hello ' }, { str: 'world  ', hasEOL: true }, { str: 'next' }, {}])).toBe('Hello world\nnext');
  });
  it('is empty for a page with no text layer', () => {
    expect(joinTextItems([])).toBe('');
    expect(joinTextItems([{ str: '  ', hasEOL: true }])).toBe('');
  });
});

describe('fitSize', () => {
  it('fits the longer side to the preset and never upscales', () => {
    expect(fitSize(2048, 1024, 512, null)).toEqual({ width: 512, height: 256 });
    expect(fitSize(300, 200, 512, null)).toEqual({ width: 300, height: 200 });
  });
  it('also respects the server pixel limit', () => {
    const s = fitSize(1024, 1024, 1024, 512 * 512);
    expect(s.width * s.height).toBeLessThanOrEqual(512 * 512);
  });
});

describe('serve image contract constants', () => {
  it('matches lm-serve limits', () => {
    expect(MAX_IMAGES).toBe(8);
    expect(MAX_DECODE_SIDE).toBe(4096);
    expect(MAX_BODY_BYTES).toBe(24 * 1024 * 1024);
    expect(DEFAULT_MAX_PIXELS).toBe(1_048_576);
  });
  it('accepts only PNG/JPEG data URLs', () => {
    expect(isAllowedImageDataUrl('data:image/png;base64,AA')).toBe(true);
    expect(isAllowedImageDataUrl('data:image/jpeg;base64,AA')).toBe(true);
    expect(isAllowedImageDataUrl('data:image/jpg;base64,AA')).toBe(true);
    expect(isAllowedImageDataUrl('data:image/webp;base64,AA')).toBe(false);
    expect(isAllowedImageDataUrl('https://example.com/x.png')).toBe(false);
  });
});

describe('buildUserContent', () => {
  it('is a plain string without images, file blocks before the typed text', () => {
    const att: Attachment = { id: 't', kind: 'text', name: 'a.txt', text: 'body' };
    expect(buildUserContent('question', [att], false)).toBe('a.txt\n```\nbody\n```\n\nquestion');
  });
  it('is image parts then one text part with images', () => {
    const img: Attachment = { id: 'i', kind: 'image', name: 'a.png', dataUrl: 'data:image/png;base64,A', width: 1, height: 1 };
    expect(buildUserContent('what is this', [img], true)).toEqual([
      { type: 'image_url', image_url: { url: 'data:image/png;base64,A' } },
      { type: 'text', text: 'what is this' },
    ]);
  });
  it('drops images when vision is off (text-only servers)', () => {
    const img: Attachment = { id: 'i', kind: 'image', name: 'a.png', dataUrl: 'data:image/png;base64,A', width: 1, height: 1 };
    expect(buildUserContent('what is this', [img], false)).toBe('what is this');
  });
  it('refuses remote image URLs even when vision is ok', () => {
    const img: Attachment = { id: 'i', kind: 'image', name: 'a.png', dataUrl: 'https://evil.example/x.png', width: 1, height: 1 };
    expect(buildUserContent('x', [img], true)).toBe('x');
  });
});

describe('toWire and assess', () => {
  const history: Message[] = [
    { id: '1', role: 'user', content: 'hi', typed: 'hi', attachmentNames: [], createdAt: 0 },
    { id: '2', role: 'assistant', content: 'Hello **there**', reasoning: 'r', status: 'done', error: null, stats: null, createdAt: 0 },
    { id: '3', role: 'user', content: 'again', typed: 'again', attachmentNames: [], createdAt: 0 },
    { id: '4', role: 'assistant', content: '', reasoning: '', status: 'error', error: { code: null, message: 'x', status: 500 }, stats: null, createdAt: 0 },
    { id: '5', role: 'user', content: 'three', typed: 'three', attachmentNames: [], createdAt: 0 },
    { id: '6', role: 'assistant', content: 'cut sho', reasoning: '', status: 'error', error: { code: 'stream_cut', message: '', status: 200 }, stats: null, createdAt: 0 },
    { id: '7', role: 'user', content: 'four', typed: 'four', attachmentNames: [], createdAt: 0 },
    { id: '8', role: 'assistant', content: 'half', reasoning: '', status: 'stopped', error: null, stats: null, createdAt: 0 },
  ];

  it('re-sends answers exactly as received, without reasoning or failed replies, even partial ones', () => {
    expect(toWire('be brief', history)).toEqual([
      { role: 'system', content: 'be brief' },
      { role: 'user', content: 'hi' },
      { role: 'assistant', content: 'Hello **there**' },
      { role: 'user', content: 'again' },
      { role: 'user', content: 'three' },
      { role: 'user', content: 'four' },
      { role: 'assistant', content: 'half' },
    ]);
    expect(toWire('  ', [])).toEqual([]);
  });

  it('warns about the wait and refuses what overflows the context', () => {
    const long = 'x'.repeat(3.5 * 2000);
    const a = assess(DEFAULT_SETTINGS, [], long, [], long, info({ contextLength: 1000 }));
    expect(a.newTokens).toBe(2000);
    expect(a.waitSeconds).toBeGreaterThan(120);
    expect(a.problems).toEqual([{ kind: 'overflow', tokens: 2000, limit: 1000 }]);
    expect(assess(DEFAULT_SETTINGS, [], 'short', [], 'short', info({ contextLength: 1000 })).problems).toEqual([]);
  });

  it('refuses more than 8 images in one request, counting earlier ones', () => {
    const img = { type: 'image_url' as const, image_url: { url: 'data:image/png;base64,A' } };
    const earlier: Message[] = [{ id: 'u', role: 'user', content: Array(6).fill(img), typed: '', attachmentNames: [], createdAt: 0 }];
    const a = assess(DEFAULT_SETTINGS, earlier, 'x', [], [img, img, img, { type: 'text', text: 'x' }], info({ vision: 'ok' }));
    expect(a.problems).toContainEqual({ kind: 'too_many_images', count: 9, max: 8 });
  });

  it('refuses a request body over 24 MiB', () => {
    const big = 'y'.repeat(25 * 1024 * 1024);
    const a = assess(DEFAULT_SETTINGS, [], big, [], big, info());
    expect(a.problems.map((p) => p.kind)).toContain('too_large');
  });
});

describe('toWire with tool calls', () => {
  const asks = (over: Partial<Extract<Message, { role: 'assistant' }>> = {}): Message => ({
    id: 'a', role: 'assistant', content: '', reasoning: '', status: 'done', error: null, stats: null, createdAt: 0,
    toolCalls: [{ id: 'c1', name: 'calculator', arguments: '{"expression":"1"}' }], ...over,
  });
  const result: Message = { id: 't', role: 'tool', toolCallId: 'c1', name: 'calculator', content: '1', isError: false, createdAt: 0 };
  const hi: Message = { id: 'u', role: 'user', content: 'hi', typed: 'hi', attachmentNames: [], createdAt: 0 };

  it('sends a call with its result straight after it', () => {
    expect(toWire('', [hi, asks(), result])).toEqual([
      { role: 'user', content: 'hi' },
      { role: 'assistant', content: null, tool_calls: [{ id: 'c1', type: 'function', function: { name: 'calculator', arguments: '{"expression":"1"}' } }] },
      { role: 'tool', tool_call_id: 'c1', content: '1' },
    ]);
  });

  it('never sends a call nothing answered, a stopped or failed call, or a stray result', () => {
    expect(toWire('', [hi, asks()])).toEqual([{ role: 'user', content: 'hi' }]);
    expect(toWire('', [hi, asks({ content: 'text' })])).toEqual([{ role: 'user', content: 'hi' }, { role: 'assistant', content: 'text' }]);
    expect(toWire('', [hi, asks({ status: 'stopped' }), result])).toEqual([{ role: 'user', content: 'hi' }]);
    expect(toWire('', [hi, result])).toEqual([{ role: 'user', content: 'hi' }]);
  });
});
