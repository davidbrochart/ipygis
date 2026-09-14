import assert from 'node:assert/strict';
import { test } from 'node:test';
import { ContentsStore } from '../lib/contents-store.js';
import { createGroup, createArray, writeArray } from '../lib/zarr-write.js';
import { openNode, readArray } from '../lib/zarr.js';
import { ZarrBridge } from '../lib/zarr-bridge.js';

function manager() {
  const files = new Map();
  const saves = [];
  return { files, saves,
    async get(path) {
      if (!files.has(path)) throw Object.assign(new Error('Missing'), { response: { status: 404 } });
      const model = files.get(path);
      if (model.type === 'directory') {
        const prefix = path + '/';
        return { ...model, content: [...files.keys()].filter(key => key.startsWith(prefix) && !key.slice(prefix.length).includes('/')).map(key => ({ name: key.slice(prefix.length), type: files.get(key).type })) };
      }
      return model;
    },
    async save(path, model) { saves.push(path); files.set(path, model); return model; },
  };
}

test('Contents writes valid Zarr numeric/string arrays using base64 and no shared memory', async () => {
  const contents = manager();
  const store = new ContentsStore(contents, 'exports/test.zarr');
  const shared = globalThis.SharedArrayBuffer;
  globalThis.SharedArrayBuffer = undefined;
  try {
    await store.create();
    await createGroup(store, { title: 'export' });
    for (const [dtype, data] of [['int64', BigInt64Array.of(9007199254740993n, -2n, 3n)],
                                ['float64', Float64Array.of(NaN, Infinity, 3)],
                                ['string', ['é🌍', '', 'tile']]]) {
      const definition = { dtype, shape: [3], chunks: [2], dimensions: ['x'], attributes: {} };
      await createArray(store, dtype, definition);
      for (let start = 0; start < 3; start += 2) {
        const values = data.slice(start, start + 2);
        await writeArray(store, dtype, [[start, Math.min(start + 2, 3)]], [values.length],
          dtype === 'string' ? values : undefined, dtype === 'string' ? [] : [values.buffer]);
      }
      assert.deepEqual((await openNode(store, dtype)).dimension_names, ['x']);
      const result = await readArray(store, dtype, [[0, 3, 1]]);
      const actual = dtype === 'string' ? result.result.values : [...new data.constructor(result.data.buffer)];
      assert.deepEqual(actual, [...data]);
    }
    await createArray(store, 'scalar', { dtype: 'float64', shape: [], chunks: [], dimensions: [], attributes: {} });
    await writeArray(store, 'scalar', [], [], undefined, [Float64Array.of(12).buffer]);
    assert.equal(new Float64Array((await readArray(store, 'scalar', [])).data.buffer)[0], 12);
    assert.equal(contents.files.get('exports/test.zarr/float64/c/0').format, 'base64');
    assert.equal(contents.files.get('exports/test.zarr/float64').type, 'directory');
    const count = contents.saves.length;
    await assert.rejects(new ContentsStore(contents, 'exports/test.zarr').create(), /exists/);
    assert.equal(contents.saves.length, count);
    await assert.rejects(store.set('../escape', new Uint8Array()), /relative/);
    store.close();
    await assert.rejects(store.set('x', new Uint8Array()), /closed/);
  } finally { globalThis.SharedArrayBuffer = shared; }
});

test('Contents permission failures propagate without writing', async () => {
  const contents = manager();
  contents.get = async () => { throw Object.assign(new Error('Forbidden'), { response: { status: 403 } }); };
  await assert.rejects(new ContentsStore(contents, 'test.zarr').create(), /Forbidden/);
  assert.equal(contents.saves.length, 0);
});

test('writer RPC uses Contents independently of Icechunk and releases stores on disposal', async () => {
  const contents = manager();
  const replies = [];
  const model = {
    on(_event, handler, context) { this.receive = (message, buffers) => handler.call(context, message, buffers); },
    off() {}, send(message) { replies.push(message); },
  };
  const bridge = new ZarrBridge(model, () => { throw new Error('Must not use Icechunk'); }, contents);
  const request = (operation, extra = {}, buffers = []) => model.receive({ type: 'zarr_request', id: operation,
    store_id: 'contents_create', path: 'a', operation, ...extra }, buffers);
  await request('contents_create', { path: 'out.zarr', attributes: {} });
  assert.deepEqual(replies.at(-1).result, { store_id: 'contents_create' });
  await request('zarr_create_array', { definition: { dtype: 'int32', shape: [2], chunks: [2], dimensions: ['x'], attributes: {} } });
  await request('zarr_set', { selection: [[0, 2]], shape: [2] }, [Int32Array.of(4, 7).buffer]);
  assert.equal(replies.at(-1).error, undefined);
  assert.ok(contents.files.has('out.zarr/a/c/0'));
  bridge.dispose();
  const count = contents.saves.length;
  await request('zarr_set', { selection: [[0, 2]], shape: [2] }, [Int32Array.of(8, 9).buffer]);
  assert.equal(contents.saves.length, count);
});

test('Contents reader reopens exported files through the array RPC without writes', async () => {
  const contents = manager();
  const writer = new ContentsStore(contents, 'out.zarr');
  await writer.create();
  await createGroup(writer, { title: 'reopened' });
  await createArray(writer, 'a', { dtype: 'int32', shape: [2], chunks: [2], dimensions: ['x'], attributes: {} });
  await writeArray(writer, 'a', [[0, 2]], [2], undefined, [Int32Array.of(4, 7).buffer]);
  writer.close();
  const saves = contents.saves.length;
  const replies = [];
  const model = {
    on(_event, handler, context) { this.receive = message => handler.call(context, message); },
    off() {}, send(message, _callbacks, buffers) { replies.push({ ...message, buffers }); },
  };
  const bridge = new ZarrBridge(model, () => { throw new Error('Must not use Icechunk'); }, contents);
  const request = (operation, extra = {}) => model.receive({ type: 'zarr_request', id: operation,
    store_id: 'contents_open', path: '', operation, ...extra });
  await request('contents_open', { path: 'out.zarr' });
  assert.deepEqual(replies.at(-1).result, { store_id: 'contents_open' });
  await request('zarr_members');
  assert.deepEqual(replies.at(-1).result, ['a']);
  await request('zarr_open');
  assert.equal(replies.at(-1).result.attrs.title, 'reopened');
  await request('zarr_get', { path: 'a', selection: [[0, 2, 1]] });
  assert.deepEqual([...new Int32Array(replies.at(-1).buffers[0])], [4, 7]);
  await request('zarr_create_array', { path: 'forbidden', definition: { dtype: 'int32', shape: [], chunks: [], dimensions: [], attributes: {} } });
  assert.match(replies.at(-1).error, /read-only/);
  assert.equal(contents.saves.length, saves);
  await request('contents_open', { path: 'missing.zarr' });
  assert.match(replies.at(-1).error, /Missing/);
  assert.equal(contents.saves.length, saves);
  bridge.dispose();
});
