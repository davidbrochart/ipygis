import assert from 'node:assert/strict';
import { test } from 'node:test';
import { ZarrBridge } from '../lib/zarr-bridge.js';

test('array RPC works with an independent byte store and checks its lifetime', async () => {
  const metadata = new TextEncoder().encode(JSON.stringify({
    zarr_format: 3, node_type: 'array', attributes: {}, shape: [], data_type: 'uint8',
    chunk_grid: { name: 'regular', configuration: { chunk_shape: [] } },
    chunk_key_encoding: { name: 'default', configuration: { separator: '/' } },
    fill_value: 0, codecs: [{ name: 'bytes' }],
  }));
  const objects = new Map([['a/zarr.json', metadata], ['a/c', Uint8Array.of(7)]]);
  const stores = new Map([['memory', { async get(key) { return objects.get(key) ?? null; } }]]);
  const replies = [];
  const model = {
    on(_event, handler, context) { this.receive = message => handler.call(context, message); },
    off() {},
    send(message, _callbacks, buffers) { replies.push({ ...message, buffers }); },
  };
  const bridge = new ZarrBridge(model, id => {
    if (!stores.has(id)) throw new Error('Store closed');
    return stores.get(id);
  });
  const request = (operation, extra = {}) => model.receive({
    type: 'zarr_request', id: 'request', store_id: 'memory', path: 'a', operation, ...extra,
  });
  await request('zarr_open');
  assert.equal(replies.at(-1).type, 'zarr_response');
  assert.equal(replies.at(-1).result.kind, 'array');
  await request('zarr_get', { selection: [] });
  assert.equal(new Uint8Array(replies.at(-1).buffers[0])[0], 7);
  const count = replies.length;
  await request('get', { type: 'icechunk_request' });
  assert.equal(replies.length, count);
  stores.delete('memory');
  await request('zarr_get', { selection: [] });
  assert.match(replies.at(-1).error, /closed/);
  bridge.dispose();
  await request('zarr_open');
  assert.equal(replies.length, count + 1);
});
