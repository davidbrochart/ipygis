"""Offline tests: no Jupyter server, proxy, or remote raster required."""

from unittest.mock import patch

import anyio
import numpy as np
import pytest
import xarray as xr
import zarr
from zarr.abc.store import OffsetByteRequest, RangeByteRequest, SuffixByteRequest
from zarr.core.buffer import default_buffer_prototype
from zarr.storage import MemoryStore

from ipygis.gis import GIS, GISWidget
from ipygis.icechunk import Repository, _IcechunkBackend, jupyter_storage
from ipygis.zarr.storage import BrowserStore


class FakeBrowser:
    def __init__(self, data):
        self.data = data
        self.calls = []

    async def _request(self, operation, **kwargs):
        await anyio.lowlevel.checkpoint()
        self.calls.append((operation, kwargs))
        if operation == 'close':
            return None, []
        if operation == 'exists':
            return kwargs['key'] in self.data, []
        if operation == 'get':
            data = self.data.get(kwargs['key'])
            query = kwargs.get('range')
            if data is not None and query:
                if 'suffixLength' in query:
                    data = (
                        data[-query['suffixLength'] :] if query['suffixLength'] else b''
                    )
                else:
                    start = query.get('offset', 0)
                    data = (
                        data[start : start + query['length']]
                        if 'length' in query
                        else data[start:]
                    )
            return {'missing': data is None}, [] if data is None else [memoryview(data)]
        if operation == 'list':
            return list(self.data), []
        if operation == 'list_prefix':
            return [k for k in self.data if k.startswith(kwargs['prefix'])], []
        if operation == 'list_dir':
            prefix = kwargs['prefix'].rstrip('/') + '/'
            return sorted(
                {
                    k[len(prefix) :].split('/')[0]
                    for k in self.data
                    if k.startswith(prefix)
                }
            ), []
        raise AssertionError(operation)

    def _notify_close(self, store_id):
        self.calls.append(('close', {'store_id': store_id}))


class MemoryBackend:
    """Byte backend without Icechunk, widget messages, or codec support."""

    def __init__(self, data):
        self.data = data
        self.calls = []
        self.closed = False

    async def get(self, key, byte_range=None):
        await anyio.lowlevel.checkpoint()
        self.calls.append(('get', {'key': key}))
        data = self.data.get(key)
        if data is not None and byte_range is not None:
            if 'suffixLength' in byte_range:
                length = byte_range['suffixLength']
                return data[-length:] if length else b''
            offset = byte_range.get('offset', 0)
            end = offset + byte_range['length'] if 'length' in byte_range else None
            return data[offset:end]
        return data

    async def exists(self, key):
        return key in self.data

    async def list(self):
        return list(self.data)

    async def list_prefix(self, prefix):
        return [key for key in self.data if key.startswith(prefix)]

    async def list_dir(self, prefix):
        prefix = prefix.rstrip('/') + '/' if prefix else ''
        return sorted(
            {
                key[len(prefix) :].split('/')[0]
                for key in self.data
                if key.startswith(prefix)
            }
        )

    def close(self):
        self.closed = True

    async def aclose(self):
        self.close()


@pytest.mark.anyio
class TestBrowserStore:
    @pytest.fixture(autouse=True)
    async def setup_store(self):
        memory = MemoryStore()
        group = await zarr.api.asynchronous.create_group(store=memory, zarr_format=3)
        array = await group.create_array(
            'temperature',
            shape=(4,),
            chunks=(2,),
            dtype='float64',
            dimension_names=('x',),
            compressors=[],
        )
        await array.setitem(slice(None), np.array([1.0, 2.0, 7.0, 8.0]))
        self.data = {
            key: (await memory.get(key, default_buffer_prototype())).to_bytes()
            async for key in memory.list()
        }
        self.browser = MemoryBackend(self.data)
        self.store = await BrowserStore.open(self.browser)
        assert all(args['key'].endswith('zarr.json') for _, args in self.browser.calls)
        self.browser.calls.clear()

        try:
            yield
        finally:
            await self.store.aclose()

    async def test_xarray_open_uses_only_metadata_then_async_loads_one_chunk(self):
        ds = xr.open_zarr(
            self.store, chunks=None, consolidated=False, create_default_indexes=False
        )
        assert self.browser.calls == []
        assert ds.sizes == {'x': 4}
        selected = await ds.temperature.isel(x=2).load_async()
        assert selected.item() == 7.0
        reads = [
            args['key'] for operation, args in self.browser.calls if operation == 'get'
        ]
        assert reads == ['temperature/c/1']

    async def test_ranges_missing_and_concurrent_reads(self):
        prototype = default_buffer_prototype()
        key = 'temperature/c/0'
        for query in [
            RangeByteRequest(2, 8),
            OffsetByteRequest(3),
            SuffixByteRequest(4),
        ]:
            actual = await self.store.get(key, prototype, query)
            assert actual.to_bytes() == self.store._slice(self.data[key], query)
        values = await self.store.get_partial_values(
            prototype, [(key, None), ('temperature/c/99', None)]
        )
        assert values[0].to_bytes() == self.data[key]
        assert values[1] is None
        assert await self.store.get('missing/zarr.json', prototype) is None
        with pytest.raises(ValueError):
            await self.store.get(key, prototype, RangeByteRequest(8, 2))

    async def test_sync_chunk_reads_fail_instead_of_deadlocking(self):
        transport = FakeBrowser(self.data)
        repository = Repository(transport, 'repository')
        backend = _IcechunkBackend(repository, 'fixture')
        self.store = BrowserStore(
            backend, self.store._metadata, self.store._directories
        )
        ds = xr.open_zarr(
            self.store, chunks=None, consolidated=False, create_default_indexes=False
        )
        with pytest.raises(RuntimeError, match='notebook event loop'):
            ds.temperature.isel(x=2).load()

    async def test_listing_read_only_and_close(self):
        assert set([key async for key in self.store.list_dir('')]) == {
            'zarr.json',
            'temperature',
        }
        assert self.browser.calls == []
        assert set([key async for key in self.store.list()]) == set(self.data)
        assert await self.store.exists('temperature/c/0')
        assert not await self.store.exists('missing/zarr.json')
        with pytest.raises(ValueError):
            await self.store.delete('zarr.json')
        with pytest.raises(ValueError):
            await self.store.set(
                'zarr.json', default_buffer_prototype().buffer.from_bytes(b'')
            )
        await self.store.aclose()
        with pytest.raises(RuntimeError, match='closed'):
            await self.store.get('zarr.json', default_buffer_prototype())

    @pytest.mark.parametrize('shared_connection', [False, True])
    @pytest.mark.parametrize('ready_before_open', [False, True])
    async def test_open_waits_for_frontend_ready_message(
        self, ready_before_open, shared_connection
    ):
        with patch('ipygis.gis.display'):
            gis = GIS()
        messages = []
        gis._widget.send = lambda content, buffers=None: messages.append(content)
        result = []

        async def open_store():
            result.append(
                await Repository.open_async(
                    jupyter_storage('example.icechunk'),
                    gis=gis if shared_connection else None,
                )
            )

        try:
            if ready_before_open:
                gis._on_response(None, {'type': 'icechunk_ready'}, [])
            with patch('ipygis.gis.GIS', return_value=gis):
                async with anyio.create_task_group() as tg:
                    tg.start_soon(open_store)
                    await anyio.wait_all_tasks_blocked()
                    if not ready_before_open:
                        assert messages == []
                        gis._on_response(None, {'type': 'icechunk_ready'}, [])
                        await anyio.wait_all_tasks_blocked()
                    assert len(messages) == 1
                    assert messages[0]['operation'] == 'open'
                    assert messages[0]['repository'] == 'example.icechunk'
                    gis._on_response(None, {'type': 'icechunk_ready'}, [])
                    gis._on_response(
                        None,
                        {
                            'type': 'icechunk_response',
                            'id': messages[0]['id'],
                            'result': {'repository_id': 'opened'},
                        },
                        [],
                    )
            assert len(result) == 1
            assert isinstance(result[0], Repository)
            assert not hasattr(result[0], 'snapshot_id')
            assert 'branch' not in messages[0]
            with patch.object(gis._widget, 'close') as close_widget:
                result[0].close()
                assert close_widget.call_count == (0 if shared_connection else 1)
        finally:
            gis._widget.close()

    async def test_generic_store_closes_backend_and_has_no_snapshot(self):
        assert not hasattr(self.store, 'snapshot_id')
        assert self.store.backend is self.browser
        await self.store.aclose()
        assert self.browser.closed

    async def test_missing_metadata_leaves_backend_open(self):
        backend = MemoryBackend({})
        with pytest.raises(ValueError, match='Missing Zarr metadata'):
            await BrowserStore.open(backend)
        assert not backend.closed

    async def test_browser_decoder_works_without_icechunk_store(self):
        gis = GIS.__new__(GIS)
        gis._ready = anyio.Event()
        gis._ready.set()
        gis._widget = GISWidget()
        gis._pending = {}
        messages = []
        gis._widget.send = lambda content, buffers=None: messages.append(
            (content, buffers)
        )
        results = []

        async def decode():
            results.append(await gis.decode_lzw(b'compressed', 2))

        try:
            async with anyio.create_task_group() as tg:
                tg.start_soon(decode)
                await anyio.wait_all_tasks_blocked()
                message, buffers = messages[0]
                assert message['operation'] == 'decode_lzw'
                assert 'store_id' not in message
                assert bytes(buffers[0]) == b'compressed'
                gis._on_response(
                    None,
                    {
                        'type': 'icechunk_response',
                        'id': message['id'],
                        'result': None,
                    },
                    [memoryview(b'AB')],
                )
            assert results == [b'AB']
        finally:
            gis._widget.close()

    async def test_open_rejects_writes_before_creating_connection(self):
        with patch('ipygis.gis.GIS') as connection:
            with pytest.raises(ValueError, match='read-only'):
                await BrowserStore.open(self.browser, read_only=False)
            connection.assert_not_called()

    async def test_open_timeout_closes_owned_connection(self):
        with patch('ipygis.gis.display'):
            gis = GIS()
        fail_after = anyio.fail_after
        try:
            with (
                patch('ipygis.gis.GIS', return_value=gis),
                patch.object(gis._widget, 'close') as close_widget,
                patch(
                    'ipygis.icechunk.anyio.fail_after', lambda seconds: fail_after(0)
                ),
            ):
                with pytest.raises(TimeoutError):
                    await Repository.open_async(jupyter_storage('example.icechunk'))
                close_widget.assert_called_once()
        finally:
            gis._widget.close()

    async def test_rpc_timeout_cleans_open_and_ignores_late_response(self):
        gis = GIS.__new__(GIS)
        gis._widget = GISWidget()
        gis._pending = {}
        messages = []
        gis._widget.send = lambda content, buffers=None: messages.append(content)
        fail_after = anyio.fail_after
        try:
            with patch('ipygis.gis.anyio.fail_after', lambda seconds: fail_after(0)):
                with pytest.raises(TimeoutError):
                    await gis._request('open')
            assert gis._pending == {}
            assert messages[-1]['operation'] == 'close'
            assert messages[-1]['store_id'] == messages[0]['id']
            gis._on_response(
                None,
                {
                    'type': 'icechunk_response',
                    'id': messages[0]['id'],
                    'result': {},
                },
                [],
            )
            assert gis._pending == {}
        finally:
            gis._widget.close()

    async def test_rpc_matches_out_of_order_responses_and_cleans_cancellation(self):
        gis = GIS.__new__(GIS)
        gis._widget = GISWidget()
        gis._pending = {}
        messages = []
        gis._widget.send = lambda content, buffers=None: messages.append(content)
        try:
            results = {}

            async def read(key):
                results[key] = await gis._request('get', key=key)

            async with anyio.create_task_group() as tg:
                tg.start_soon(read, 'first')
                tg.start_soon(read, 'second')
                await anyio.wait_all_tasks_blocked()
                for message in reversed(messages):
                    gis._on_response(
                        None,
                        {
                            'type': 'icechunk_response',
                            'id': message['id'],
                            'result': message['key'],
                        },
                        [memoryview(b'abc')],
                    )
            assert results['first'][0] == 'first'
            assert results['second'][0] == 'second'

            async def failed_read():
                with pytest.raises(RuntimeError, match='bad checksum'):
                    await gis._request('get')

            async with anyio.create_task_group() as tg:
                tg.start_soon(failed_read)
                await anyio.wait_all_tasks_blocked()
                gis._on_response(
                    None,
                    {
                        'type': 'icechunk_response',
                        'id': messages[-1]['id'],
                        'error': 'bad checksum',
                    },
                    [],
                )

            async def open_until_cancelled(*, task_status=anyio.TASK_STATUS_IGNORED):
                with anyio.CancelScope() as scope:
                    task_status.started(scope)
                    await gis._request('open')
                assert scope.cancelled_caught

            async with anyio.create_task_group() as tg:
                scope = await tg.start(open_until_cancelled)
                await anyio.wait_all_tasks_blocked()
                scope.cancel()
            assert gis._pending == {}
            assert messages[-1]['operation'] == 'close'
        finally:
            gis._widget.close()
