import assert from 'node:assert/strict';
import { test } from 'node:test';
import { decodeLzw } from '../lib/lzw.js';

// TIFF LZW codes [clear=256, 'A'=65, 'B'=66, end=257], MSB first.
const compressed = Uint8Array.of(128, 16, 72, 80, 16);

test('decodes a known TIFF LZW stream, including an offset input view', async () => {
  assert.deepEqual(await decodeLzw(compressed, 2), Uint8Array.of(65, 66));
  const padded = Uint8Array.of(255, ...compressed, 255);
  assert.deepEqual(await decodeLzw(padded.subarray(1, -1), 2), Uint8Array.of(65, 66));
});

test('rejects both undersized and oversized decoded chunks', async () => {
  await assert.rejects(decodeLzw(compressed, 1), /Incorrect LZW decoded length/);
  await assert.rejects(decodeLzw(compressed, 3), /Incorrect LZW decoded length/);
});

test('rejects invalid output bounds before allocating', async () => {
  for (const size of [0, -1, 1.5, NaN, Infinity, 0xffffffff]) {
    await assert.rejects(decodeLzw(compressed, size), /decoded size/);
  }
});
