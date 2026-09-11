"""Asynchronous xarray opening backed by browser-side Zarrita arrays."""
from __future__ import annotations

from collections.abc import Mapping

import xarray as xr
from xarray import conventions
from xarray.backends import BackendArray
from xarray.backends.zarr import FillValueCoder
from xarray.core import indexing

from .zarr import asynchronous as zarr


class _BrowserArray(BackendArray):
    def __init__(self, array):
        self.array = array
        self.shape = array.shape
        self.dtype = array.dtype

    def __getitem__(self, key):
        raise RuntimeError('Browser data requires async I/O; use await .load_async()')

    async def async_getitem(self, key):
        # Xarray translates advanced indexing into a basic bounding selection,
        # followed by in-memory indexing. Only that bounding selection crosses
        # the browser connection.
        return await indexing.async_explicit_indexing_adapter(
            key, self.shape, indexing.IndexingSupport.BASIC, self.array.getitem,
        )


async def open_zarr_async(
    store, *, group=None, create_default_indexes=True, decode_cf=True,
    mask_and_scale=True, decode_times=True, decode_timedelta=None,
    decode_coords=True, drop_variables=None,
) -> xr.Dataset:
    """Open a group as a lazy xarray Dataset using ipygis's async Zarr API.

    Metadata and coordinate reads run on the caller's event loop. This does not
    call xarray.open_zarr or Zarr-Python's synchronous API. Select data with
    ``isel`` or ``sel`` and load it using ``await selection.load_async()``.

    Dimension coordinates are loaded to construct indexes by default; disable
    ``create_default_indexes`` for metadata-only opening. CF datetime variables
    are currently loaded before decoding, even with indexes disabled; use
    ``decode_times=False`` to leave them lazy. CF masks/scales remain lazy.

    Dimensions come from Zarr v3 dimension_names or v2 _ARRAY_DIMENSIONS.
    Array fill values alone are not interpreted as missing values; use CF
    _FillValue/missing_value attributes. Numeric and Zarr v3 string browser arrays are
    supported. Advanced selections may fetch a larger bounding region.

    The returned dataset borrows the store. Closing it does not close the
    session; keep the session open until all async reads have finished.
    """
    root = await zarr.open_group(store, path=group)
    dropped = {drop_variables} if isinstance(drop_variables, str) else set(drop_variables or ())
    variables = {}
    sizes = {}
    async for name in root.keys():
        if name in dropped:
            continue
        array = await root.getitem(name)
        if not isinstance(array, zarr.AsyncArray):
            continue
        attrs = dict(array.attrs)
        # Xarray stores floating CF fill values as base64 strings in Zarr v3.
        if array.dimension_names is not None and array.dtype.kind == 'f' and isinstance(attrs.get('_FillValue'), str):
            attrs['_FillValue'] = FillValueCoder.decode(attrs['_FillValue'], array.dtype)
        dims = array.dimension_names
        if dims is None:
            dims = attrs.get('_ARRAY_DIMENSIONS')
        attrs.pop('_ARRAY_DIMENSIONS', None)
        if dims is None or isinstance(dims, str) or len(dims) != array.ndim or any(not isinstance(d, str) for d in dims):
            raise ValueError(f'Array {name!r} is missing valid dimension metadata')
        dims = tuple(dims)
        for dim, size in zip(dims, array.shape):
            if dim in sizes and sizes[dim] != size:
                raise ValueError(f'Conflicting sizes for dimension {dim!r}')
            sizes[dim] = size
        variable = xr.Variable(
            dims, indexing.LazilyIndexedArray(_BrowserArray(array)), attrs,
            encoding={'chunks': array.chunks, 'preferred_chunks': dict(zip(dims, array.chunks))},
        )
        times = decode_times.get(name, True) if isinstance(decode_times, Mapping) else decode_times
        units = attrs.get('units')
        if decode_cf and times and isinstance(units, str) and 'since' in units:
            # Xarray's datetime decoder probes values synchronously for dtype
            # inference. Await the raw data before invoking that decoder.
            await variable.load_async()
        variables[name] = variable
    variables, attrs, coord_names = conventions.decode_cf_variables(
        variables, dict(root.attrs),
        mask_and_scale=mask_and_scale if decode_cf else False,
        decode_times=decode_times if decode_cf else False,
        decode_timedelta=decode_timedelta if decode_cf else False,
        decode_coords=decode_coords if decode_cf else False,
        concat_characters=False,
    )
    coordinates = {name: var for name, var in variables.items()
                   if name in coord_names or var.dims == (name,)}
    dataset = xr.Dataset(
        {name: var for name, var in variables.items() if name not in coordinates},
        coords=xr.Coordinates(coordinates, indexes={}), attrs=attrs,
    )
    if create_default_indexes:
        for name in coordinates:
            if dataset.variables[name].dims == (name,):
                await dataset.variables[name].load_async()
                dataset = dataset.set_xindex(name)
    return dataset
