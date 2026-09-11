import assert from 'node:assert/strict';
import { test } from 'node:test';
import * as zarr from 'zarrita';
import { openNode, readArray, listMembers } from '../lib/zarr.js';

export async function fixture() {
  const objects = new Map();
  const root = zarr.root(objects);
  await zarr.create(root, { attributes: { title: 'browser' } });
  const array = await zarr.create(root.resolve('a'), {
    shape: [4, 6], chunkShape: [2, 3], dtype: 'int32', dimensionNames: ['y', 'x'],
    codecs: [{ name: 'bytes', configuration: { endian: 'little' } }],
  });
  await zarr.set(array, null, { data: Int32Array.from({ length: 24 }, (_, i) => i), shape: [4, 6], stride: [6, 1] });
  const reads = [];
  const store = { async get(key) { reads.push(key); return objects.get('/' + key) ?? null; } };
  return { store, objects, reads, root };
}

test('Zarrita opens metadata and reads slices across chunks without shared memory', async () => {
  const { store, reads } = await fixture();
  const shared = globalThis.SharedArrayBuffer;
  globalThis.SharedArrayBuffer = undefined;
  try {
    assert.equal((await openNode(store, '')).attrs.title, 'browser');
    const meta = await openNode(store, 'a');
    assert.deepEqual(meta.shape, [4, 6]);
    assert.deepEqual(meta.chunks, [2, 3]);
    assert.equal(reads.some(k => k.includes('/c/')), false);
    const { result, data } = await readArray(store, 'a', [[1, 4, 2], [1, 6, 2]]);
    assert.deepEqual(result.shape, [2, 3]);
    assert.deepEqual([...new Int32Array(data.buffer)], [7, 9, 11, 19, 21, 23]);
    const point = await readArray(store, 'a', [2, 4]);
    assert.deepEqual(point.result.shape, []);
    assert.equal(new Int32Array(point.data.buffer)[0], 16);
    const empty = await readArray(store, 'a', [[3, 1, 1], [0, 6, 1]]);
    assert.deepEqual(empty.result.shape, [0, 6]);
    assert.equal(empty.data.length, 0);
  } finally { globalThis.SharedArrayBuffer = shared; }
});

test('LZW chunks decode through the browser codec registry', async () => {
  const { store, objects, root } = await fixture();
  await zarr.create(root.resolve('lzw'), { shape: [2], chunkShape: [2], dtype: 'uint8', codecs: [
    { name: 'bytes' }, { name: 'imagecodecs_lzw', configuration: {} },
  ] });
  objects.set('/lzw/c/0', Uint8Array.of(128, 16, 72, 80, 16));
  const { data } = await readArray(store, 'lzw', [[0, 2, 1]]);
  assert.deepEqual([...data], [65, 66]);
});

test('scalar, exact int64, fill values and unsupported operations', async () => {
  const { store, root } = await fixture();
  const scalar = await zarr.create(root.resolve('scalar'), { shape: [], chunkShape: [], dtype: 'int64', codecs: [{ name: 'bytes' }] });
  await zarr.set(scalar, [], 9007199254740993n);
  const point = await readArray(store, 'scalar', []);
  assert.deepEqual(point.result.shape, []);
  assert.equal(new BigInt64Array(point.data.buffer)[0], 9007199254740993n);
  await zarr.create(root.resolve('fill'), { shape: [3], chunkShape: [2], dtype: 'int32', fillValue: 42, codecs: [{ name: 'bytes' }] });
  assert.deepEqual([...new Int32Array((await readArray(store, 'fill', [[0, 3, 1]])).data.buffer)], [42, 42, 42]);
  await assert.rejects(readArray(store, 'a', [5, 0]), /bounds/);
  await assert.rejects(openNode(store, '../a'), /paths/);
  await assert.rejects(openNode(store, 'missing'));
});


test('group enumeration and dimension names are metadata-only', async () => {
  const { store, reads } = await fixture();
  store.listDir = async path => {
    assert.equal(path, '');
    return ['zarr.json', 'a'];
  };
  assert.deepEqual(await listMembers(store, ''), ['a']);
  assert.deepEqual((await openNode(store, 'a')).dimension_names, ['y', 'x']);
  assert.equal(reads.some(k => k.includes('/c/')), false);
  delete store.listDir;
  await assert.rejects(listMembers(store, ''), /listing/);
});

test('string coordinates preserve Unicode, scalar and empty selections', async () => {
  const { store, objects, root } = await fixture();
  await zarr.create(root.resolve('tile'), {
    shape: [3], chunkShape: [3], dtype: 'string', dimensionNames: ['tile'],
    codecs: [{ name: 'vlen-utf8' }],
  });
  const labels = ['50_-110', 'é🌍', ''];
  const encoded = labels.map(value => new TextEncoder().encode(value));
  const bytes = new Uint8Array(4 + encoded.reduce((n, value) => n + 4 + value.length, 0));
  const view = new DataView(bytes.buffer);
  view.setUint32(0, labels.length, true);
  let offset = 4;
  for (const value of encoded) {
    view.setUint32(offset, value.length, true);
    bytes.set(value, offset + 4);
    offset += 4 + value.length;
  }
  objects.set('/tile/c/0', bytes);
  assert.equal((await openNode(store, 'tile')).dtype, 'string');
  assert.deepEqual((await readArray(store, 'tile', [[0, 3, 1]])).result.values, labels);
  assert.deepEqual((await readArray(store, 'tile', [[0, 3, 2]])).result.values, [labels[0], '']);
  const point = await readArray(store, 'tile', [1]);
  assert.deepEqual(point.result.shape, []);
  assert.deepEqual(point.result.values, ['é🌍']);
  const empty = await readArray(store, 'tile', [[1, 1, 1]]);
  assert.deepEqual(empty.result.values, []);
});
