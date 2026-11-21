// Browser-only: decode an image file and re-encode it as PNG at the chosen size.
import { AttachError, MAX_DECODE_SIDE, fitSize, type ImageData } from './attachments';

async function decode(blob: Blob, name: string): Promise<ImageBitmap> {
  try {
    return await createImageBitmap(blob);
  } catch {
    throw new AttachError('read_failed', name);
  }
}

export function canvasToPng(canvas: HTMLCanvasElement): ImageData {
  return { dataUrl: canvas.toDataURL('image/png'), width: canvas.width, height: canvas.height };
}

export async function imageToPng(blob: Blob, name: string, preset: number, maxPixels: number | null): Promise<ImageData> {
  const bitmap = await decode(blob, name);
  try {
    if (bitmap.width > MAX_DECODE_SIDE || bitmap.height > MAX_DECODE_SIDE) throw new AttachError('too_large', name);
    const size = fitSize(bitmap.width, bitmap.height, preset, maxPixels);
    const canvas = document.createElement('canvas');
    canvas.width = size.width;
    canvas.height = size.height;
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new AttachError('read_failed', name);
    ctx.drawImage(bitmap, 0, 0, size.width, size.height);
    return canvasToPng(canvas);
  } finally {
    bitmap.close();
  }
}
