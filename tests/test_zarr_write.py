from types import SimpleNamespace

import anyio
import numpy as np
import pytest
import xarray as xr
from test_xarray import Transport
from test_xarray import forbid_sync as forbid_sync

from ipygis.xarray import open_zarr_async, to_zarr_async

pytestmark = pytest.mark.anyio


@pytest.fixture
def connection(monkeypatch):
    class Connection:
        def __init__(self):
            self.messages = []
            self.closed = False
            self._ready = anyio.Event()
            self._ready.set()
            self._widget = SimpleNamespace(close=self.close)

        def close(self):
            self.closed = True

        async def _request(self, operation, **arguments):
            assert arguments['message_type'] == 'zarr_request'
            self.messages.append((operation, arguments))
            await anyio.lowlevel.checkpoint()
            return {'store_id': 'output'}, []

    instance = Connection()
    monkeypatch.setattr('ipygis.gis.GIS', lambda: instance)
    return instance


async def test_stream_lazy_dataset_and_metadata(connection):
    transport = Transport()
    ds = await open_zarr_async(transport)
    ds.attrs['title'] = 'export'
    await to_zarr_async(
        ds, 'exports/region.zarr', encoding={'elevation': {'chunks': (2, 3)}}
    )
    assert connection.closed
    assert connection.messages[0][1]['attributes']['title'] == 'export'
    definitions = {
        args['path']: args['definition']
        for op, args in connection.messages
        if op == 'zarr_create_array'
    }
    assert definitions['elevation']['dimensions'] == ['y', 'x']
    assert definitions['elevation']['chunks'] == [2, 3]
    result = np.empty((3, 4))
    writes = [
        args
        for op, args in connection.messages
        if op == 'zarr_set' and args['path'] == 'elevation'
    ]
    assert len(writes) == 4
    for args in writes:
        data = np.frombuffer(args['buffers'][0], dtype='<f8').reshape(args['shape'])
        result[tuple(slice(*s) for s in args['selection'])] = data
    np.testing.assert_equal(result, transport.values['elevation'])
    assert transport.reads.count('elevation') == 4


async def test_strings_scalars_empty_and_int64(connection):
    ds = xr.Dataset(
        {
            'labels': ('x', np.array(['é🌍', '', 'tile'], dtype=object)),
            'precise': xr.Variable((), np.int64(2**60 + 1)),
            'empty': ('z', np.array([], dtype='float64')),
        },
        coords={'x': [1, 2, 3]},
    )
    await to_zarr_async(ds, 'new.zarr')
    writes = {
        args['path']: args for op, args in connection.messages if op == 'zarr_set'
    }
    assert writes['labels']['values'] == ['é🌍', '', 'tile']
    assert writes['precise']['shape'] == []
    assert np.frombuffer(writes['precise']['buffers'][0], dtype='<i8')[0] == 2**60 + 1
    assert 'empty' not in writes


@pytest.mark.parametrize(
    'kwargs,match',
    [
        ({'mode': 'w'}, 'mode'),
        ({'store': '../escape'}, 'relative'),
        ({'encoding': {'a': {'compressor': 'gzip'}}}, 'chunks'),
        ({'encoding': {'a': {'chunks': (0,)}}}, 'chunks'),
    ],
)
async def test_preflight(connection, kwargs, match):
    arguments = dict(store='out.zarr') | kwargs
    with pytest.raises(ValueError, match=match):
        await to_zarr_async(xr.Dataset({'a': ('x', [1, 2])}), **arguments)
    assert not connection.messages


async def test_unsupported_dtype_before_creating_store(connection):
    ds = xr.Dataset({'time': ('x', np.array(['2000-01-01'], dtype='datetime64[D]'))})
    with pytest.raises(TypeError, match='dtype'):
        await to_zarr_async(ds, 'out.zarr')
    assert not connection.messages


async def test_write_failure_closes_connection(connection):
    original = connection._request

    async def fail(operation, **args):
        if operation == 'zarr_set':
            raise RuntimeError('Disk full')
        return await original(operation, **args)

    connection._request = fail
    with pytest.raises(RuntimeError, match='Disk full'):
        await to_zarr_async(xr.Dataset({'a': ('x', [1, 2])}), 'out.zarr')
    assert connection.closed


async def test_cancellation_closes_connection(connection):
    started = anyio.Event()

    async def block(operation, **args):
        started.set()
        await anyio.sleep_forever()

    connection._request = block
    async with anyio.create_task_group() as tg:
        tg.start_soon(to_zarr_async, xr.Dataset({'a': ('x', [1, 2])}), 'out.zarr')
        await started.wait()
        tg.cancel_scope.cancel()
    assert connection.closed
