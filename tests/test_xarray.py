import asyncio
import base64
import struct
import threading

import anyio
import numpy as np
import pytest
import xarray as xr

from ipygis.xarray import open_zarr_async

pytestmark = pytest.mark.anyio


class Transport:
    def __init__(self, v2=False):
        self.values = {
            'y': np.array([50.0, 49.0, 48.0]),
            'x': np.array([10.0, 20.0, 30.0, 40.0]),
            'elevation': np.arange(12, dtype='float64').reshape(3, 4),
        }
        self.dims = {'y': ['y'], 'x': ['x'], 'elevation': ['y', 'x']}
        self.attrs = {name: {} for name in self.values}
        self.reads = []
        self.v2 = v2
        self.closed = False
        self.loop = asyncio.get_running_loop()

    async def array_request(self, operation, *, path, **kwargs):
        assert asyncio.get_running_loop() is self.loop
        if self.closed:
            raise RuntimeError('Store closed')
        await anyio.lowlevel.checkpoint()
        if operation == 'zarr_members':
            return list(self.values), []
        if operation == 'zarr_open' and path in ('', 'nested'):
            return {'kind': 'group', 'attrs': {'title': 'terrain'}}, []
        name = path.rsplit('/', 1)[-1]
        value = self.values[name]
        if operation == 'zarr_open':
            attrs = dict(self.attrs[name])
            if self.v2:
                attrs['_ARRAY_DIMENSIONS'] = self.dims[name]
            return dict(
                kind='array',
                attrs=attrs,
                shape=value.shape,
                chunks=value.shape,
                dimension_names=None if self.v2 else self.dims[name],
                dtype='string' if value.dtype.kind == 'O' else str(value.dtype),
            ), []
        self.reads.append(name)
        key = tuple(
            slice(*s) if isinstance(s, list) else s for s in kwargs['selection']
        )
        result = np.asarray(value[key]).copy()
        if value.dtype.kind == 'O':
            return dict(
                shape=result.shape, dtype='string', values=result.ravel().tolist()
            ), []
        return dict(
            shape=result.shape,
            strides=[s // result.itemsize for s in result.strides],
            dtype=str(result.dtype),
            byteorder='<',
        ), [result.tobytes()]


@pytest.fixture(autouse=True)
def forbid_sync(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError('Unexpected synchronous Zarr API or thread')

    monkeypatch.setattr(xr, 'open_zarr', fail)
    monkeypatch.setattr(threading.Thread, 'start', fail)
    monkeypatch.setattr(asyncio, 'to_thread', fail)
    monkeypatch.setattr(asyncio.BaseEventLoop, 'run_in_executor', fail)
    import zarr.core.sync

    monkeypatch.setattr(zarr.core.sync, 'sync', fail)


@pytest.mark.parametrize('v2', [False, True])
async def test_coordinates_and_lazy_selection(v2):
    backend = Transport(v2)
    ds = await open_zarr_async(backend, group='nested')
    assert ds.attrs['title'] == 'terrain'
    assert ds.elevation.dims == ('y', 'x')
    assert backend.reads == ['y', 'x']
    assert '_ARRAY_DIMENSIONS' not in ds.elevation.attrs
    result = await ds.sel(y=slice(50, 49), x=slice(20, 30)).load_async()
    np.testing.assert_array_equal(result.elevation, [[1, 2], [5, 6]])
    point = await ds.sel(y=49, x=30).load_async()
    assert point.elevation.item() == 6
    fancy = await ds.sel(y=[48, 50], x=[40, 10]).load_async()
    np.testing.assert_array_equal(fancy.elevation, [[11, 8], [3, 0]])
    vector = await ds.isel(
        y=xr.DataArray([2, 0], dims='points'), x=xr.DataArray([3, 1], dims='points')
    ).load_async()
    np.testing.assert_array_equal(vector.elevation, [11, 1])
    with pytest.raises(RuntimeError, match='load_async'):
        ds.elevation.values


async def test_metadata_only_and_lifetime():
    backend = Transport()
    ds = await open_zarr_async(backend, create_default_indexes=False)
    assert backend.reads == []
    assert not ds.xindexes
    np.testing.assert_array_equal(
        (await ds.isel(y=1, x=slice(1, 3)).load_async()).elevation, [5, 6]
    )
    ds.close()
    assert not backend.closed
    backend.closed = True
    with pytest.raises(RuntimeError, match='closed'):
        await ds.isel(y=0, x=0).load_async()


async def test_cf_scale_mask_and_time():
    backend = Transport()
    backend.attrs['elevation'] = {
        'scale_factor': 2.0,
        'add_offset': 1.0,
        '_FillValue': 0.0,
    }
    backend.values['y'] = np.arange(3, dtype='int32')
    backend.attrs['y'] = {'units': 'days since 2000-01-01'}
    ds = await open_zarr_async(backend)
    assert np.issubdtype(ds.y.dtype, np.datetime64)
    result = await ds.sel(y='2000-01-01').load_async()
    np.testing.assert_array_equal(result.elevation, [np.nan, 3, 5, 7])
    raw = await open_zarr_async(backend, decode_cf=False, create_default_indexes=False)
    assert raw.y.dtype == np.dtype('int32')
    assert raw.elevation.attrs['scale_factor'] == 2.0


async def test_dimension_validation_and_drop():
    backend = Transport()
    backend.dims['elevation'] = None
    with pytest.raises(ValueError, match='dimension metadata'):
        await open_zarr_async(backend)
    ds = await open_zarr_async(backend, drop_variables='elevation')
    assert 'elevation' not in ds
    backend.dims['elevation'] = ['x', 'y']
    with pytest.raises(ValueError, match='Conflicting sizes'):
        await open_zarr_async(backend)


async def test_encoded_fill_value_and_auxiliary_coordinate():
    backend = Transport()
    backend.attrs['elevation']['_FillValue'] = base64.standard_b64encode(
        struct.pack('<d', 0.0)
    ).decode()
    backend.values['latitude'] = np.array([50.0, 49.0, 48.0])
    backend.dims['latitude'] = ['y']
    backend.attrs['latitude'] = {}
    backend.attrs['elevation']['coordinates'] = 'latitude'
    ds = await open_zarr_async(backend)
    assert 'latitude' in ds.coords
    assert backend.reads == ['y', 'x']
    result = await ds.isel(y=0).load_async()
    assert result.latitude.item() == 50.0
    np.testing.assert_array_equal(result.elevation, [np.nan, 1.0, 2.0, 3.0])


async def test_string_coordinate_selection():
    backend = Transport()
    backend.values['y'] = np.array(['50_-110', 'é🌍', ''], dtype=object)
    ds = await open_zarr_async(backend)
    assert ds.y.values.tolist() == ['50_-110', 'é🌍', '']
    assert backend.reads == ['y', 'x']
    point = await ds.sel(y='é🌍', x=30).load_async()
    assert point.elevation.item() == 6
    from ipygis.zarr.asynchronous import open_array

    array = await open_array(backend, path='y')
    assert await array.getitem(1) == 'é🌍'
    assert (await array.getitem(slice(1, 1))).shape == (0,)
