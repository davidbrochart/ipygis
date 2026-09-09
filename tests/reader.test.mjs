import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { test } from 'node:test';
import { ServerConnection } from '@jupyterlab/services';
import { openIcechunk, virtualChunkClient } from '../lib/icechunk.js';

const fixture = JSON.parse(await readFile(new URL('./fixtures/icechunk.json', import.meta.url)));
const objects = new Map(Object.entries(fixture.objects).map(([path, hex]) => [path, Uint8Array.from(Buffer.from(hex, 'hex'))]));
function contents() {
  const reads = [];
  const manager = {
    serverSettings: ServerConnection.makeSettings({
      baseUrl: 'http://localhost/',
      fetch: async (request) => {
        const path = new URL(request.url).pathname.replace('/files/repository/', '');
        reads.push(path);
        const data = objects.get(path);
        // Like some Jupyter file handlers, ignore Range and return a full object.
        return new Response(request.method === 'HEAD' ? null : data, { status: data ? 200 : 404 });
      },
    }),
    async getDownloadUrl(path) { return `http://localhost/files/${path}`; },
    async get(path) {
      const prefix = path.replace(/^repository\/?/, '').replace(/\/$/, '');
      const start = prefix ? `${prefix}/` : '';
      const entries = new Map();
      for (const object of objects.keys()) {
        if (!object.startsWith(start)) continue;
        const rest = object.slice(start.length);
        const name = rest.split('/')[0];
        entries.set(name, { path: `repository/${start}${name}`, type: rest.includes('/') ? 'directory' : 'file' });
      }
      return { content: [...entries.values()] };
    },
  };
  return { manager, reads };
}

test('Python repository: Jupyter storage, snapshots, sparse listings, ranges and no shared memory', async () => {
  const shared = globalThis.SharedArrayBuffer;
  globalThis.SharedArrayBuffer = undefined;
  try {
    const { manager, reads } = contents();
    const repo = await openIcechunk(manager, 'repository', '', ['https://example.test/']);
    const store = await repo.readonlySession({ branch: 'main' });
    assert.equal(store.snapshotId, fixture.snapshot);
    assert.deepEqual(await store.listDir(''), ['inline', 'scalar', 'sparse', 'zarr.json']);
    assert.equal(reads.some(path => path.startsWith('manifests/')), false, 'group discovery needs no manifests');
    assert.deepEqual(await store.list(), ['inline/c/0', 'inline/zarr.json', 'scalar/c', 'scalar/zarr.json', 'sparse/c/0', 'sparse/c/2', 'sparse/zarr.json', 'zarr.json']);
    assert.deepEqual(await store.listPrefix('sparse/c/'), ['sparse/c/0', 'sparse/c/2']);
    assert.deepEqual(await store.listDir('sparse/c/'), ['0', '2']);
    assert.equal(await store.exists('sparse/c/1'), false);
    assert.equal(await store.exists('sparse/c/2'), true);
    assert.equal(reads.some(path => path.startsWith('chunks/')), false, 'listing and exists must not read chunks');
    assert.equal(new TextDecoder().decode(await store.get('sparse/c/0')), '12345678');
    assert.equal(new TextDecoder().decode(await store.get('sparse/c/0', { offset: 2, length: 3 })), '345');
    assert.equal(new TextDecoder().decode(await store.get('sparse/c/0', { offset: 5 })), '678');
    assert.equal(new TextDecoder().decode(await store.get('sparse/c/0', { suffixLength: 2 })), '78');
    assert.deepEqual(await store.get('sparse/c/0', { suffixLength: 0 }), new Uint8Array(0));
    assert.equal(await store.get('sparse/c/1'), null);
    assert.equal(await store.get('missing'), null);
    assert.deepEqual(await store.get('scalar/c'), Uint8Array.of(7));
    assert.equal(new TextDecoder().decode(await store.get('inline/c/0')), 'ab');
    const snapshot = await repo.readonlySession({ snapshotId: fixture.snapshot });
    store.close();
    await assert.rejects(store.get('zarr.json'), /closed/);
    assert.ok(await snapshot.get('zarr.json'));
    snapshot.close();
  } finally { globalThis.SharedArrayBuffer = shared; }
});

test('virtual chunks retain permissions, proxy rewriting, ranges, and ETag validation', async (t) => {
  const calls = [];
  t.mock.method(globalThis, 'fetch', async (url, init) => {
    calls.push({ url, init });
    assert.equal(new URL(url).origin, 'https://proxy.test');
    assert.equal(new URL(url).searchParams.get('url'), 'https://example.test/raster.tif');
    assert.equal(init.headers.get('Range'), 'bytes=10-17');
    assert.equal(init.headers.has('If-Match'), false);
    return new Response('abcdefgh', { status: 206, headers: { 'Content-Range': 'bytes 10-17/100', ETag: '"fixture-etag"' } });
  });
  const repo = await openIcechunk(contents().manager, 'repository', 'https://proxy.test/', ['https://example.test/']);
  const store = await repo.readonlySession({});
  assert.equal(new TextDecoder().decode(await store.get('sparse/c/2')), 'abcdefgh');
  assert.equal(calls.length, 1);
  const denied = await openIcechunk(contents().manager, 'repository', '', []);
  const deniedStore = await denied.readonlySession({});
  await assert.rejects(deniedStore.get('sparse/c/2'), /Unauthorized/);
  assert.equal(calls.length, 1);
});

for (const [name, status, headers, error] of [
  ['full response', 200, {}, /206/],
  ['wrong range', 206, { 'Content-Range': 'bytes 1-8/100' }, /Content-Range/],
  ['missing etag', 206, { 'Content-Range': 'bytes 10-17/100' }, /ETag/],
  ['changed etag', 206, { 'Content-Range': 'bytes 10-17/100', ETag: 'changed' }, /ETag/],
]) {
  test(`rejects ${name} before consuming the response body`, async (t) => {
    let cancelled = false;
    t.mock.method(globalThis, 'fetch', async () => new Response(new ReadableStream({ cancel() { cancelled = true; } }), { status, headers }));
    const client = virtualChunkClient('', ['https://example.test/']);
    await assert.rejects(client.fetch('https://example.test/file', { headers: { Range: 'bytes=10-17', 'If-Match': 'expected' } }), error);
    assert.equal(cancelled, true);
  });
}

test('Last-Modified must be exposed and no newer than the reference', async (t) => {
  let modified = 'Mon, 01 Sep 2025 00:00:00 GMT';
  t.mock.method(globalThis, 'fetch', async () => new Response('a', { status: 206, headers: { 'Content-Range': 'bytes 0-0/1', 'Last-Modified': modified } }));
  const client = virtualChunkClient('', ['https://example.test/']);
  const init = { headers: { Range: 'bytes=0-0', 'If-Unmodified-Since': 'Tue, 02 Sep 2025 00:00:00 GMT' } };
  assert.equal(await (await client.fetch('https://example.test/file', init)).text(), 'a');
  modified = 'Wed, 03 Sep 2025 00:00:00 GMT';
  await assert.rejects(client.fetch('https://example.test/file', init), /Last-Modified/);
  modified = '';
  await assert.rejects(client.fetch('https://example.test/file', init), /Last-Modified/);
});

test('short virtual responses fail instead of returning truncated data', async (t) => {
  t.mock.method(globalThis, 'fetch', async () => new Response('abc', {
    status: 206, headers: { 'Content-Range': 'bytes 10-17/100', ETag: 'fixture-etag' },
  }));
  const repo = await openIcechunk(contents().manager, 'repository', '', ['https://example.test/']);
  const store = await repo.readonlySession({});
  await assert.rejects(store.get('sparse/c/2'), /size mismatch/);
});
