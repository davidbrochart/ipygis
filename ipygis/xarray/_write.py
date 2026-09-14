"""Asynchronous xarray exports through browser-side Zarrita and Jupyter Contents."""

from __future__ import annotations

import itertools
import json
import math
import operator

import anyio
import numpy as np
import xarray as xr
from xarray.backends.zarr import FillValueCoder
from xarray.conventions import encode_dataset_coordinates


def _json(value):
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        raise TypeError(f'Attribute is not JSON serializable: {type(obj).__name__}')

    return json.loads(json.dumps(value, default=convert, allow_nan=False))


def _prepare(dataset, path, mode, encoding):
    if not isinstance(dataset, xr.Dataset):
        raise TypeError('dataset must be an xarray Dataset')
    if mode != 'w-':
        raise ValueError("Only mode='w-' (create a new store) is supported")
    if (
        not isinstance(path, str)
        or not path
        or path.startswith('/')
        or '\\' in path
        or any(part in {'', '.', '..'} for part in path.split('/'))
    ):
        raise ValueError(
            'store must be a nonempty path relative to the Jupyter contents root'
        )
    encoding = encoding or {}
    if set(encoding) - set(dataset.variables):
        raise ValueError('encoding contains unknown variables')
    variables, attributes = encode_dataset_coordinates(dataset)
    prepared = []
    reserved = {'zarr.json', '.zgroup', '.zattrs', '.zarray', '.zmetadata'}
    for name, variable in variables.items():
        if (
            not isinstance(name, str)
            or not name
            or name in {'.', '..'} | reserved
            or '/' in name
            or '\\' in name
        ):
            raise ValueError(f'Unsupported Zarr variable name: {name!r}')
        if any(not isinstance(dim, str) for dim in variable.dims):
            raise ValueError('Dimension names must be strings')
        dtype = variable.dtype
        if (
            dtype.kind not in 'iufUO'
            or (dtype.kind in 'iuf' and dtype.itemsize not in (1, 2, 4, 8))
            or (dtype.kind == 'f' and dtype.itemsize not in (4, 8))
        ):
            raise TypeError(
                f'Unsupported output dtype for {name!r}: {dtype}; encode it explicitly before writing'
            )
        options = encoding.get(name, {})
        if set(options) - {'chunks'}:
            raise ValueError('Only the chunks encoding option is supported')
        if 'chunks' in options:
            chunks = tuple(operator.index(size) for size in options['chunks'])
            if len(chunks) != variable.ndim or any(size <= 0 for size in chunks):
                raise ValueError(f'Invalid chunks for {name!r}')
        else:
            chunks = [max(1, size) for size in variable.shape]
            itemsize = max(8, dtype.itemsize) if dtype.kind in 'iuf' else 16
            while math.prod(chunks) * itemsize > 1024 * 1024:
                axis = max(range(len(chunks)), key=chunks.__getitem__)
                chunks[axis] = max(1, (chunks[axis] + 1) // 2)
            chunks = tuple(chunks)
        attrs = dict(variable.attrs)
        if (
            dtype.kind == 'f'
            and '_FillValue' in attrs
            and not isinstance(attrs['_FillValue'], str)
        ):
            attrs['_FillValue'] = FillValueCoder.encode(attrs['_FillValue'], dtype)
        definition = dict(
            shape=list(variable.shape),
            chunks=list(chunks),
            dtype='string' if dtype.kind in 'UO' else dtype.name,
            dimensions=list(variable.dims),
            attributes=_json(attrs),
        )
        prepared.append((name, variable, definition))
    return _json(attributes), prepared


async def to_zarr_async(
    dataset: xr.Dataset, store: str, *, mode='w-', encoding=None
) -> None:
    """Write a new Zarr v3 directory through Jupyter's browser contents manager.

    ``store`` is relative to the Jupyter contents root. In JupyterLab files live
    on the server; in JupyterLite they use its configured browser storage.
    Existing destinations are rejected. The contents API does not provide atomic
    exclusive creation: callers must not write the same path concurrently.

    Numeric and string variables, dimensions, coordinates and JSON attributes
    are supported. Object arrays must contain only strings. Datetimes, bools,
    complex values, append/region writes and consolidated metadata are not yet
    supported. Source encodings are not copied: decoded values are exported.
    Set output chunk shapes with ``encoding={name: {'chunks': (...)}}``.
    Default numeric chunks are approximately 1 MiB or smaller; strings vary.

    Each source chunk is loaded with ``load_async`` and sent to Zarrita to encode
    in the browser. There are no synchronous Zarr calls or Python filesystem
    writes. Source arrays must be in memory or support xarray's async reads.
    Cancellation or failure can leave a partial directory; it is not removed
    automatically. Returns None after all writes have completed.
    """
    attributes, prepared = _prepare(dataset, store, mode, encoding)
    from ..gis import GIS

    connection = GIS()
    try:
        with anyio.fail_after(30):
            await connection._ready.wait()
        result, _ = await connection._request(
            'contents_create',
            message_type='zarr_request',
            path=store,
            attributes=attributes,
        )
        store_id = result['store_id']
        for name, variable, definition in prepared:
            await connection._request(
                'zarr_create_array',
                message_type='zarr_request',
                store_id=store_id,
                path=name,
                definition=definition,
            )
            ranges = [
                range(0, size, chunk)
                for size, chunk in zip(variable.shape, definition['chunks'])
            ]
            for start in itertools.product(*ranges):
                selection = tuple(
                    slice(offset, min(offset + chunk, size))
                    for offset, chunk, size in zip(
                        start, definition['chunks'], variable.shape
                    )
                )
                loaded = await variable[selection].load_async()
                values = np.asarray(loaded.values)
                arguments = dict(
                    message_type='zarr_request',
                    store_id=store_id,
                    path=name,
                    selection=[[s.start, s.stop] for s in selection],
                    shape=list(values.shape),
                )
                if definition['dtype'] == 'string':
                    strings = values.ravel().tolist()
                    if not all(isinstance(value, str) for value in strings):
                        raise TypeError(
                            f'Object array {name!r} contains non-string values'
                        )
                    arguments['values'] = strings
                else:
                    values = np.ascontiguousarray(
                        values, dtype=values.dtype.newbyteorder('<')
                    )
                    arguments['buffers'] = [memoryview(values.tobytes())]
                await connection._request('zarr_set', **arguments)
    finally:
        connection._widget.close()
