"""Read-only Zarr-Python-like async interface backed by Zarrita in the browser.

Supports fixed-width integer and floating-point arrays. Reads return NumPy
arrays (or NumPy scalars for point selections). Writing, fancy indexing,
new axes, and negative slice steps are not yet supported.
"""
from __future__ import annotations

import math
import operator
from typing import Protocol

import numpy as np


class ArrayBackend(Protocol):
    """Transport for metadata and decoded array reads in a browser byte store."""

    async def array_request(self, operation: str, **arguments): ...


class BrowserArrayBackend:
    """Zarrita operations over an injected, lifetime-aware browser transport."""

    def __init__(self, request):
        self._request = request

    async def array_request(self, operation: str, **arguments):
        if operation not in {'zarr_open', 'zarr_get'}:
            raise ValueError(f'Unknown array operation: {operation}')
        return await self._request(operation, message_type='zarr_request', **arguments)


def _path(path):
    if not isinstance(path, str):
        raise TypeError('path must be a string')
    parts = path.split('/')
    if any(p in {'.', '..'} for p in parts):
        raise ValueError('Paths must not contain . or ..')
    return '/'.join(p for p in parts if p)


def _selection(selection, shape):
    keys = selection if isinstance(selection, tuple) else (selection,)
    ellipses = sum(key is Ellipsis for key in keys)
    if ellipses > 1:
        raise IndexError('Only one ellipsis is allowed')
    if ellipses:
        position = next(i for i, key in enumerate(keys) if key is Ellipsis)
        count = len(shape) - len(keys) + 1
        if count < 0:
            raise IndexError('Too many indices')
        keys = keys[:position] + (slice(None),) * count + keys[position + 1:]
    if len(keys) > len(shape):
        raise IndexError('Too many indices')
    keys += (slice(None),) * (len(shape) - len(keys))
    result, output_shape = [], []
    for key, size in zip(keys, shape):
        if isinstance(key, slice):
            start, stop, step = key.indices(size)
            if step < 0:
                raise NotImplementedError('Negative slice steps are not supported')
            result.append([start, stop, step])
            output_shape.append(len(range(start, stop, step)))
        else:
            if isinstance(key, (bool, np.bool_)):
                raise TypeError('Boolean indexing is not supported')
            try:
                index = operator.index(key)
            except TypeError:
                raise TypeError('Only integers, slices, and ellipses are supported') from None
            if index < 0:
                index += size
            if index < 0 or index >= size:
                raise IndexError('Array index out of bounds')
            result.append(index)
    return result, tuple(output_shape)


class AsyncGroup:
    """Group metadata and async child lookup; lifetime is owned by the store."""

    def __init__(self, store, backend, path, metadata):
        self.store = store
        self._backend = backend
        self.path = path
        self.attrs = dict(metadata['attrs'])

    async def getitem(self, key: str):
        """Open a child array or group, including nested paths."""
        if not isinstance(key, str):
            raise TypeError('Group keys must be strings')
        path = key if key.startswith('/') else '/'.join(filter(None, (self.path, key)))
        return await _open(self.store, path, 'r', None)


class AsyncArray:
    """Numeric array metadata and async basic indexing, executed by Zarrita."""

    def __init__(self, store, backend, path, metadata):
        self.store = store
        self._backend = backend
        self.path = path
        self.attrs = dict(metadata['attrs'])
        self.shape = tuple(metadata['shape'])
        self.chunks = tuple(metadata['chunks'])
        self.dtype = np.dtype(metadata['dtype'])
        self.ndim = len(self.shape)
        self.size = math.prod(self.shape)

    async def getitem(self, selection):
        """Read integers/slices without a synchronous Zarr API or worker thread."""
        selection, expected_shape = _selection(selection, self.shape)
        result, buffers = await self._backend.array_request(
            'zarr_get', path=self.path, selection=selection,
        )
        if tuple(result['shape']) != expected_shape or np.dtype(result['dtype']) != self.dtype:
            raise RuntimeError('Browser returned inconsistent array metadata')
        if len(buffers) != 1 or len(result['strides']) != len(expected_shape):
            raise RuntimeError('Browser returned an invalid array buffer')
        if result['byteorder'] not in {'<', '>'}:
            raise RuntimeError('Browser returned an invalid byte order')
        dtype = self.dtype.newbyteorder(result['byteorder'])
        strides = tuple(operator.index(s) * dtype.itemsize for s in result['strides'])
        if any(s < 0 for s in strides):
            raise RuntimeError('Browser returned invalid array strides')
        try:
            data = np.ndarray(expected_shape, dtype=dtype, buffer=buffers[0], strides=strides)
        except (TypeError, ValueError) as error:
            raise RuntimeError('Browser returned an invalid array buffer') from error
        data = data.astype(self.dtype, order='C', copy=True)
        return data[()] if expected_shape == () else data


async def _open(store, path, mode, kind):
    if mode != 'r':
        raise ValueError('Browser arrays are read-only; use mode="r"')
    path = _path(path or '')
    backend = getattr(store, 'array_backend', store)
    if not callable(getattr(backend, 'array_request', None)):
        raise TypeError('store must provide a browser array backend (e.g. session.store)')
    metadata, _ = await backend.array_request('zarr_open', path=path)
    actual = metadata.get('kind')
    if kind is not None and actual != kind:
        raise TypeError(f'Expected a Zarr {kind}, found {actual}')
    if actual == 'group':
        return AsyncGroup(store, backend, path, metadata)
    if actual == 'array':
        return AsyncArray(store, backend, path, metadata)
    raise RuntimeError('Browser returned an unknown Zarr node type')


async def open_group(store, *, mode='r', path=None) -> AsyncGroup:
    """Open an existing group; store may be an Icechunk session.store.

    This subset of Zarr-Python's async API reads and decodes arrays in JavaScript.
    It does not transfer ownership: close the session when finished reading.
    """
    return await _open(store, path, mode, 'group')


async def open_array(store, *, mode='r', path=None) -> AsyncArray:
    """Open an existing numeric array at path without fetching its chunk data."""
    return await _open(store, path, mode, 'array')
