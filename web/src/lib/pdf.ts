// Browser-only: PDF text extraction and rendering of pages that have no text layer.
// The worker is bundled by Vite from node_modules: no CDN, no remote code.
import { GlobalWorkerOptions, getDocument } from 'pdfjs-dist';
import workerUrl from 'pdfjs-dist/build/pdf.worker.min.mjs?url';
import { AttachError, fitSize, joinTextItems, type PdfPage, type TextItemLike } from './attachments';
import { canvasToPng } from './image';

GlobalWorkerOptions.workerSrc = workerUrl;

// pdf.js loads its JPEG 2000 and JBIG2 decoders (used by scanned pages) from wasmUrl. The JS
// builds of those decoders are copied there by vite.config.ts; WebAssembly stays off, so the
// CSP needs no 'wasm-unsafe-eval'.
const DECODERS_URL = `${import.meta.env.BASE_URL}pdfjs/`;

export async function readPdf(data: ArrayBuffer, name: string, renderImages: boolean, preset: number, maxPixels: number | null): Promise<PdfPage[]> {
  const task = getDocument({ data: new Uint8Array(data), enableXfa: false, useWasm: false, wasmUrl: DECODERS_URL });
  let doc;
  try {
    doc = await task.promise;
  } catch {
    void task.destroy();
    throw new AttachError('read_failed', name);
  }
  try {
    const pages: PdfPage[] = [];
    for (let n = 1; n <= doc.numPages; n++) {
      const page = await doc.getPage(n);
      const content = await page.getTextContent();
      const text = joinTextItems(content.items as TextItemLike[]);
      let image = null;
      if (!text && renderImages) {
        const base = page.getViewport({ scale: 1 });
        const size = fitSize(base.width, base.height, preset, maxPixels);
        const viewport = page.getViewport({ scale: size.width / base.width });
        const canvas = document.createElement('canvas');
        canvas.width = Math.floor(viewport.width);
        canvas.height = Math.floor(viewport.height);
        await page.render({ canvas, viewport }).promise;
        image = canvasToPng(canvas);
      }
      pages.push({ number: n, text: text || null, image });
      page.cleanup();
    }
    if (!pages.length) throw new AttachError('empty_pdf', name);
    return pages;
  } finally {
    await task.destroy();
  }
}
