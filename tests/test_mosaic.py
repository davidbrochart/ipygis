import anyio
import numpy as np
import pytest
import xarray as xr
from test_xarray import Transport
from test_xarray import forbid_sync as forbid_sync  # Reuse the no-thread/no-sync guard.

from ipygis.xarray import mosaic_async, open_zarr_async

pytestmark = pytest.mark.anyio


async def source(top=3.5, left=0.5, base=0):
    transport = Transport()
    transport.values = {
        'y': top - np.arange(3.0),
        'x': left + np.arange(4.0),
        'elevation': base + np.arange(12.0).reshape(3, 4),
    }
    transport.requests = []
    original = transport.array_request

    async def request(operation, **kwargs):
        if operation == 'zarr_get' and kwargs['path'] == 'elevation':
            transport.requests.append(kwargs['selection'])
        return await original(operation, **kwargs)

    transport.array_request = request
    ds = await open_zarr_async(transport)
    transport.reads.clear()
    return ds.elevation, transport


async def test_seams_gaps_and_selection():
    a, ta = await source()
    b, tb = await source(left=4.5, base=20)
    c, tc = await source(top=0.5, base=40)
    mosaic = await mosaic_async([a, b, c], crs='EPSG:4326')
    assert mosaic.shape == (6, 8)
    assert not ta.reads and not tb.reads and not tc.reads
    expected = np.full((6, 8), np.nan)
    expected[:3, :4], expected[:3, 4:], expected[3:, :4] = (
        ta.values['elevation'],
        tb.values['elevation'],
        tc.values['elevation'],
    )
    for key in [
        dict(y=slice(1, 5), x=slice(2, 6)),
        dict(y=2, x=4),
        dict(y=slice(None, None, 2), x=slice(None, None, 3)),
        dict(y=slice(1, 1)),
        dict(y=[5, 0], x=[7, 1]),
        dict(y=slice(None, None, -1), x=slice(None, None, -2)),
    ]:
        result = await mosaic.isel(key).load_async()
        np.testing.assert_equal(
            result.values, xr.DataArray(expected, dims=('y', 'x')).isel(key).values
        )
    geographic = await mosaic.sel(y=slice(2.5, -0.5), x=slice(2.5, 5.5)).load_async()
    np.testing.assert_equal(geographic, expected[1:5, 2:6])
    vector = await mosaic.isel(
        y=xr.DataArray([0, 4], dims='points'), x=xr.DataArray([5, 1], dims='points')
    ).load_async()
    np.testing.assert_equal(vector.values, [expected[0, 5], expected[4, 1]])
    with pytest.raises(RuntimeError, match='load_async'):
        mosaic.values


async def test_reads_only_intersections_and_lifetime():
    a, ta = await source()
    b, tb = await source(left=4.5)
    mosaic = await mosaic_async([a, b], crs='EPSG:4326')
    await mosaic.isel(y=slice(1, 3), x=slice(1, 3)).load_async()
    assert ta.requests == [[[1, 3, 1], [1, 3, 1]]]
    assert tb.reads == []
    mosaic.close()
    assert not ta.closed
    ta.closed = True
    with pytest.raises(RuntimeError, match='closed'):
        await mosaic.isel(y=0, x=0).load_async()


async def test_validation_and_integer_fill():
    a, _ = await source()
    for sources, match in [
        ([], 'At least'),
        ([a, a], 'overlap'),
        ([a.isel(y=slice(None, None, -1))], 'ascending'),
        ([a.assign_coords(x=[0, 1, 2, 4])], 'regular'),
        ([a, a.assign_coords(x=a.x + 4.1)], 'aligned'),
        ([a, a.assign_coords(x=a.x * 2 + 5)], 'spacing'),
        ([a.assign_attrs(crs='other')], 'CRS'),
    ]:
        with pytest.raises(ValueError, match=match):
            await mosaic_async(sources, crs='EPSG:4326')
    integer = xr.DataArray(
        np.full((3, 4), 2**60 + 1, dtype='int64'), coords=a.coords, dims=a.dims
    )
    with pytest.raises(ValueError, match='fill_value'):
        await mosaic_async([integer], crs='EPSG:4326')
    result = await mosaic_async([integer], crs='EPSG:4326', fill_value=-1)
    assert (await result.isel(y=0, x=0).load_async()).item() == 2**60 + 1


async def test_cancelled_read():
    a, transport = await source()
    mosaic = await mosaic_async([a], crs='EPSG:4326')
    original = transport.array_request
    entered = anyio.Event()

    async def blocked(operation, **kwargs):
        if operation == 'zarr_get':
            entered.set()
            await anyio.sleep_forever()
        return await original(operation, **kwargs)

    transport.array_request = blocked
    async with anyio.create_task_group() as tg:
        tg.start_soon(mosaic.load_async)
        await entered.wait()
        tg.cancel_scope.cancel()
    transport.array_request = original
    assert (await mosaic.isel(y=0, x=0).load_async()).item() == 0


async def test_lazy_coordinates_and_rounding():
    transport = Transport()
    ds = await open_zarr_async(transport, create_default_indexes=False)
    # The standard fixture has x increasing by ten, y decreasing by one.
    mosaic = await mosaic_async([ds.elevation], crs='EPSG:4326')
    assert sorted(transport.reads) == ['x', 'y']
    assert (await mosaic.sel(y=49, x=30).load_async()).item() == 6
    a, _ = await source()
    b, _ = await source(left=4.5 + 3e-6)
    rounded = await mosaic_async([a, b], crs='EPSG:4326')
    assert rounded.shape == (3, 8)


async def test_mismatched_types_and_metadata():
    a, _ = await source()
    loaded = await a.copy(deep=False).load_async()
    with pytest.raises(ValueError, match='dtypes'):
        await mosaic_async([a, loaded.astype('float32')], crs='EPSG:4326')
    with pytest.raises(ValueError, match='integer'):
        await mosaic_async([loaded.astype('uint8')], crs='EPSG:4326', fill_value=-1)
    with pytest.raises(ValueError, match='two spatial'):
        await mosaic_async([loaded.expand_dims(band=[1])], crs='EPSG:4326')
    a = a.assign_attrs(units='m', model_tiepoint=[0, 0, 0, 0, 0, 0], crs='EPSG:4326')
    a.encoding['chunks'] = (2, 3)
    mosaic = await mosaic_async([a.transpose('x', 'y')], crs='EPSG:4326')
    assert mosaic.dims == ('y', 'x')
    assert mosaic.attrs == {'units': 'm', 'crs': 'EPSG:4326'}
    assert not mosaic.encoding


async def test_notebook_georeferencing():
    import ast
    import json
    from pathlib import Path

    examples = Path(__file__).parents[1] / 'examples'
    build = json.loads((examples / 'build_time.ipynb').read_text())
    cell = next(
        ''.join(c['source'])
        for c in build['cells']
        if 'def add_tile_coords' in ''.join(c['source'])
    )
    tree = ast.parse(cell)
    definition = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
    namespace = {'reference_grid': None}
    exec(
        compile(ast.Module(body=[definition], type_ignores=[]), 'preprocess', 'exec'),
        namespace,
    )
    attrs = dict(
        model_tiepoint=[2, 3, 0, 11, 19.25, 0],
        model_pixel_scale=[0.5, 0.25, 0],
        geographic_type=4326,
        raster_type=1,
    )
    tile = xr.Dataset(
        {'0': xr.DataArray(np.arange(12.0).reshape(3, 4), dims=('y', 'x'), attrs=attrs)}
    )
    ds = namespace['add_tile_coords'](tile)
    assert ds.tile.values.tolist() == ['20_10']
    assert ds.tile_x.item() == 10 and ds.tile_y.item() == 20
    runtime = json.loads((examples / 'run_time.ipynb').read_text())
    code = next(
        ''.join(c['source'])
        for c in runtime['cells']
        if 'await mosaic_async(' in ''.join(c['source'])
    )
    datasets = {}
    for name, dtype, fill in [('acc', 'float64', -9999.0), ('dir', 'uint8', 255)]:
        first = ds.astype(dtype)
        first['0'].attrs['_FillValue'] = fill
        # Leave a tile-sized gap to exercise each dataset's nodata value.
        second = first.assign_coords(tile=['20_14'], tile_x=('tile', [14.0]))
        datasets[name] = xr.concat([first, second], dim='tile')
    namespace.update(ds=datasets, np=np, mosaic_async=mosaic_async)
    await eval(
        compile(code, 'mosaic example', 'exec', flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT),
        namespace,
    )
    assert set(namespace['mosaic']) == {'acc', 'dir'}
    for name, mosaic in namespace['mosaic'].items():
        np.testing.assert_equal(mosaic.longitude.values, 10.25 + np.arange(12) * 0.5)
        np.testing.assert_equal(mosaic.latitude.values, [19.875, 19.625, 19.375])
        assert mosaic.dtype == datasets[name]['0'].dtype
        point = await mosaic.sel(latitude=19.625, longitude=11.25).load_async()
        assert point.item() == 6
        gap = await mosaic.isel(longitude=slice(4, 8)).load_async()
        assert np.all(gap.values == datasets[name]['0'].attrs['_FillValue'])
    for changed, match in [
        ({'geographic_type': 3857}, 'WGS84'),
        ({'raster_type': 2}, 'PixelIsArea'),
        ({'model_pixel_scale': [1, 1, 0]}, 'spacing'),
    ]:
        invalid = tile.copy()
        invalid['0'].attrs = attrs | changed
        with pytest.raises(ValueError, match=match):
            namespace['add_tile_coords'](invalid)
