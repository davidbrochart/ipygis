"""Repository/session lifecycle tests with a fake browser transport."""

import anyio
import pytest
from zarr.abc.store import Store
from zarr.core.buffer import default_buffer_prototype

from ipygis.icechunk import Repository


class Connection:
    def __init__(self, data=None):
        self.data = (
            data
            if data is not None
            else {
                'zarr.json': b'{"zarr_format":3,"node_type":"group","attributes":{}}',
                'key': b'chunk',
            }
        )
        self.calls = []
        self.next_session = 0

    async def _request(self, operation, **arguments):
        await anyio.lowlevel.checkpoint()
        self.calls.append((operation, arguments))
        if operation == 'readonly_session':
            self.next_session += 1
            return {
                'store_id': f'session-{self.next_session}',
                'snapshot_id': arguments.get(
                    'snapshot_id', f'snapshot-{self.next_session}'
                ),
            }, []
        if operation == 'get':
            data = self.data.get(arguments['key'])
            return {'missing': data is None}, [] if data is None else [memoryview(data)]
        if operation == 'list_dir':
            prefix = arguments['prefix']
            prefix = prefix.rstrip('/') + '/' if prefix else ''
            entries = {
                key[len(prefix) :].split('/')[0]
                for key in self.data
                if key.startswith(prefix) and key != 'key'
            }
            return sorted(entries), []
        if operation == 'close':
            return None, []
        raise AssertionError(operation)

    def _notify_close(self, store_id):
        self.calls.append(('close', {'store_id': store_id}))


@pytest.mark.anyio
async def test_sessions_pin_independent_snapshots_and_close_independently():
    connection = Connection()
    repository = Repository(connection, 'repo')
    first = await repository.readonly_session_async('main')
    second = await repository.readonly_session_async(snapshot_id='previous')
    assert isinstance(first.store, Store)
    assert first.snapshot_id == 'snapshot-1'
    assert second.snapshot_id == 'previous'
    assert connection.calls[0] == (
        'readonly_session',
        {'store_id': 'repo', 'branch': 'main'},
    )
    assert [call for call in connection.calls if call[0] == 'readonly_session'][1] == (
        'readonly_session',
        {'store_id': 'repo', 'snapshot_id': 'previous'},
    )
    await first.aclose()
    with pytest.raises(RuntimeError, match='closed'):
        await first.store.get('key', default_buffer_prototype())
    assert (
        await second.store.get('key', default_buffer_prototype())
    ).to_bytes() == b'chunk'
    assert connection.calls[-1][1]['store_id'] == 'session-2'
    third = await repository.readonly_session_async()
    assert third.snapshot_id == 'snapshot-3'
    await repository.aclose()
    for session in (second, third):
        with pytest.raises(RuntimeError, match='closed'):
            await session.store.get('key', default_buffer_prototype())
        await session.aclose()
    with pytest.raises(RuntimeError, match='closed'):
        await repository.readonly_session_async()
    assert connection.calls[-1] == ('close', {'store_id': 'repo'})


@pytest.mark.anyio
async def test_session_selector_is_validated_before_sending():
    connection = Connection()
    repository = Repository(connection, 'repo')
    with pytest.raises(ValueError, match='either'):
        await repository.readonly_session_async('main', snapshot_id='snapshot')
    assert connection.calls == []
    repository.close()


def test_jupyter_storage_is_configuration_only():
    from unittest.mock import patch

    from ipygis.icechunk import Storage, jupyter_storage

    with patch('ipygis.gis.GIS') as connection:
        storage = jupyter_storage('examples/data.icechunk')
        assert isinstance(storage, Storage)
        assert storage.path == 'examples/data.icechunk'
        connection.assert_not_called()
    with pytest.raises(ValueError, match='paths'):
        jupyter_storage('../data.icechunk')


@pytest.mark.anyio
async def test_open_requires_storage_before_creating_connection():
    from unittest.mock import patch

    with patch('ipygis.gis.GIS') as connection:
        with pytest.raises(TypeError, match='jupyter_storage'):
            await Repository.open_async('example.icechunk')
        connection.assert_not_called()


@pytest.mark.anyio
async def test_session_store_opens_directly_in_xarray_with_optional_lzw():
    import json
    from types import SimpleNamespace
    from unittest.mock import patch

    import numpy as np
    import xarray as xr
    import zarr
    from zarr.registry import register_codec
    from zarr.storage import MemoryStore

    from ipygis.gis import _close_decoder_connections
    from ipygis.zarr.codecs import BrowserLzwCodec

    memory = MemoryStore()
    group = await zarr.api.asynchronous.create_group(store=memory, zarr_format=3)
    array = await group.create_array(
        'data',
        shape=(2,),
        chunks=(2,),
        dtype='float64',
        dimension_names=('x',),
        compressors=[],
    )
    await array.setitem(slice(None), np.array([7.0, 8.0]))
    data = {
        key: (await memory.get(key, default_buffer_prototype())).to_bytes()
        async for key in memory.list()
    }
    metadata = json.loads(data['data/zarr.json'])
    metadata['codecs'].append({'name': 'imagecodecs_lzw'})
    data['data/zarr.json'] = json.dumps(metadata).encode()
    decoded = []

    async def decode(chunk, size):
        decoded.append((chunk, size))
        return chunk  # Codec transport stub; real decompression has Node tests.

    connection = Connection(data)
    repository = Repository(connection, 'repo')
    session = await repository.readonly_session_async()
    register_codec('imagecodecs_lzw', BrowserLzwCodec)
    browser = SimpleNamespace(decode_lzw=decode, _widget=SimpleNamespace(comm=object()))
    browser._widget.close = lambda: setattr(browser._widget, 'comm', None)
    assert all(
        args['key'].endswith('zarr.json')
        for op, args in connection.calls
        if op == 'get'
    )
    connection.calls.clear()
    with (
        patch('ipygis.gis.GIS', return_value=browser),
        zarr.config.set(
            {'codecs.imagecodecs_lzw': 'ipygis.zarr.codecs.BrowserLzwCodec'}
        ),
    ):
        ds = xr.open_zarr(
            session.store, chunks=None, consolidated=False, create_default_indexes=False
        )
        assert connection.calls == []
        assert decoded == []
        result = await ds.data.isel(x=0).load_async()
        assert result.item() == 7.0
        assert decoded == [(data['data/c/0'], 16)]
    await repository.aclose()
    assert session.store._closed
    assert browser._widget.comm is not None
    _close_decoder_connections()
    assert session.store._metadata['data/zarr.json'] == data['data/zarr.json']


@pytest.mark.anyio
async def test_failed_metadata_prefetch_releases_session():
    connection = Connection({})
    repository = Repository(connection, 'repo')
    with pytest.raises(ValueError, match='Missing Zarr metadata'):
        await repository.readonly_session_async()
    assert connection.calls[-1] == ('close', {'store_id': 'session-1'})
    assert not repository._closed
    repository.close()


@pytest.mark.anyio
@pytest.mark.parametrize('backend', ['icechunk-js', '@earthmover/icechunk'])
async def test_backend_is_forwarded_when_opening_repository(backend):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from ipygis.icechunk import jupyter_storage

    ready = anyio.Event()
    ready.set()
    connection = SimpleNamespace(
        _ready=ready,
        _request=AsyncMock(return_value=({'repository_id': 'repo'}, [])),
    )
    repository = await Repository.open_async(
        jupyter_storage('example.icechunk'),
        gis=connection,
        backend=backend,
    )
    assert connection._request.call_args.kwargs['backend'] == backend
    await repository.aclose()


@pytest.mark.anyio
async def test_invalid_backend_fails_before_creating_widget():
    from unittest.mock import patch

    from ipygis.icechunk import jupyter_storage

    with patch('ipygis.gis.GIS') as connection:
        with pytest.raises(ValueError, match='backend'):
            await Repository.open_async(
                jupyter_storage('example.icechunk'), backend='unknown'
            )
        connection.assert_not_called()


@pytest.mark.anyio
async def test_session_composes_independent_array_transport():
    from ipygis.zarr import asynchronous as zarr

    class ArrayConnection(Connection):
        async def _request(self, operation, **arguments):
            if operation == 'zarr_open':
                self.calls.append((operation, arguments))
                return {'kind': 'group', 'attrs': {}}, []
            return await super()._request(operation, **arguments)

    connection = ArrayConnection()
    repository = Repository(connection, 'repo')
    session = await repository.readonly_session_async()
    group = await zarr.open_group(session.store)
    assert group.store is session.store
    assert connection.calls[-1] == (
        'zarr_open',
        {
            'store_id': 'session-1',
            'message_type': 'zarr_request',
            'path': '',
        },
    )
    await session.aclose()
    with pytest.raises(RuntimeError, match='closed'):
        await group.getitem('child')
    repository.close()
