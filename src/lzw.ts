/** Decode TIFF LZW with Development Seed's Rust/WASM implementation. */
export async function decodeLzw(
  data: Uint8Array,
  decodedSize: number,
): Promise<Uint8Array> {
  if (
    !Number.isSafeInteger(decodedSize) ||
    decodedSize <= 0 ||
    decodedSize >= 0xffffffff
  ) {
    throw new Error('LZW decoded size must be a positive 32-bit integer');
  }
  // The package initializes WASM with top-level await. Await the import before use.
  const { decompress } = await import('@developmentseed/lzw-tiff-decoder');
  // The decoder truncates at its output limit. One extra byte lets us detect
  // an oversized result instead of silently accepting a truncated chunk.
  const result = decompress(data, decodedSize + 1);
  if (result.byteLength !== decodedSize) {
    throw new Error(
      `Incorrect LZW decoded length: expected ${decodedSize}, received ${result.byteLength}`,
    );
  }
  return result;
}
