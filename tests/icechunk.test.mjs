import assert from 'node:assert/strict';
import { test } from 'node:test';
import { IcechunkBridge } from '../lib/icechunk.js';

function fixture(readonlySession) {
  const responses = [];
  const model = {
    on(_event, handler, context) { this.receive = (...args) => handler.call(context, ...args); },
    off() {},
    send(content, _callbacks, buffers) { responses.push({ ...content, buffers }); },
  };
  const bridge = new IcechunkBridge(model, undefined);
  // Substitute the repository; exercise the real message routing and lifecycle.
  bridge.repositories.set('repo', { readonlySession });
  const request = (id, operation, extra = {}) => model.receive({
    type: 'icechunk_request', id, operation, store_id: 'repo', ...extra,
  });
  return { bridge, responses, request };
}

function store(snapshotId) {
  return { snapshotId, closed: false,
    async get() { return Uint8Array.of(7); },
    close() { this.closed = true; },
  };
}

test('sessions use their own selectors and close independently', async () => {
  const selectors = [];
  const stores = [];
  const { bridge, responses, request } = fixture(async (selector) => {
    selectors.push(selector);
    const result = store(`snapshot-${stores.length}`);
    stores.push(result);
    return result;
  });
  await request('s1', 'readonly_session', { branch: 'main' });
  await request('s2', 'readonly_session', { snapshot_id: 'previous' });
  assert.deepEqual(selectors, [{ branch: 'main' }, { snapshotId: 'previous' }]);
  assert.equal(responses[0].result.snapshot_id, 'snapshot-0');
  await request('close1', 'close', { store_id: 's1' });
  assert.equal(stores[0].closed, true);
  assert.equal(stores[1].closed, false);
  await request('get2', 'get', { store_id: 's2', key: 'key' });
  assert.deepEqual(new Uint8Array(responses.at(-1).buffers[0]), Uint8Array.of(7));
  await request('closeRepo', 'close');
  assert.equal(stores[1].closed, true);
  await request('getClosed', 'get', { store_id: 's2', key: 'key' });
  assert.match(responses.at(-1).error, /closed or missing/);
  bridge.dispose();
});

for (const target of ['session', 'repository']) {
  test(`closing ${target} during session creation releases the late session`, async () => {
    let finish;
    const lateStore = store('late');
    const { bridge, responses, request } = fixture(() => new Promise((resolve) => { finish = resolve; }));
    const opening = request('session', 'readonly_session');
    await request('cancel', 'close', { store_id: target === 'session' ? 'session' : 'repo' });
    finish(lateStore);
    await opening;
    assert.equal(lateStore.closed, true);
    assert.match(responses.find((response) => response.id === 'session').error, /closed/);
    bridge.dispose();
  });
}
