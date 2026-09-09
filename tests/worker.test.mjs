import assert from 'node:assert/strict';
import { test } from 'node:test';
import worker from './worker.js';

const target = 'https://data.hydrosheds.org/file/hydrosheds-v2/ACC/1s/tile.tif';
const url = `https://proxy.test/?url=${encodeURIComponent(target)}`;

test('Worker forwards range headers without browser metadata and preserves 206', async (t) => {
  t.mock.method(globalThis, 'fetch', async (destination, init) => {
    assert.equal(destination, target);
    assert.deepEqual(Object.fromEntries(init.headers), {
      'accept-encoding': 'identity',
      'if-match': 'etag',
      range: 'bytes=10-12',
    });
    return new Response('abc', {
      status: 206,
      headers: { 'Content-Range': 'bytes 10-12/100', 'x-bz-upload-timestamp': '1782952179003', ETag: 'etag' },
    });
  });
  const response = await worker.fetch(new Request(url, { headers: {
    Range: 'bytes=10-12', 'If-Match': 'etag', Origin: 'http://localhost:8888',
    Referer: 'http://localhost:8888/', 'Sec-Fetch-Site': 'cross-site',
    'Sec-Fetch-Mode': 'cors', Cookie: 'browser-cookie',
  } }));
  assert.equal(response.status, 206);
  assert.equal(response.headers.get('Content-Range'), 'bytes 10-12/100');
  assert.equal(response.headers.get('Last-Modified'), 'Thu, 02 Jul 2026 00:29:39 GMT');
  assert.equal(response.headers.get('Access-Control-Allow-Origin'), '*');
  assert.equal(await response.text(), 'abc');
});

test('Worker cancels a full-file response when a range was requested', async (t) => {
  let cancelled = false;
  t.mock.method(globalThis, 'fetch', async () => new Response(new ReadableStream({ cancel() { cancelled = true; } })));
  const response = await worker.fetch(new Request(url, { headers: { Range: 'bytes=10-12' } }));
  assert.equal(response.status, 502);
  assert.match(await response.text(), /Upstream ignored the Range/);
  assert.equal(cancelled, true);
});

test('Worker handles preflight without contacting the upstream', async (t) => {
  t.mock.method(globalThis, 'fetch', () => { throw new Error('Unexpected upstream request'); });
  const response = await worker.fetch(new Request(url, { method: 'OPTIONS' }));
  assert.equal(response.status, 204);
  assert.match(response.headers.get('Access-Control-Allow-Headers'), /Range/);
});

test('Worker preserves upstream errors and does not invent timestamps for other hosts', async (t) => {
  t.mock.method(globalThis, 'fetch', async () => new Response('missing', {
    status: 404, headers: { 'x-bz-upload-timestamp': '1782952179003' },
  }));
  const response = await worker.fetch(new Request('https://proxy.test/?url=https://example.test/file', { headers: { Range: 'bytes=0-1' } }));
  assert.equal(response.status, 404);
  assert.equal(response.headers.get('Last-Modified'), null);
  assert.equal(response.headers.get('Access-Control-Allow-Origin'), '*');
});
