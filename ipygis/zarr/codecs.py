"""TIFF LZW codec registered with Zarr Python's standard codec registry."""

from __future__ import annotations

import math
from dataclasses import dataclass

from zarr.abc.codec import BytesBytesCodec
from zarr.registry import register_codec

from ..gis import _decode_lzw


@dataclass(frozen=True)
class BrowserLzwCodec(BytesBytesCodec):
    """Read-only TIFF LZW codec using an async browser decoder.

    Registered as ``imagecodecs_lzw``. A shared browser connection is acquired
    automatically at read time; metadata contains no runtime connection data.
    """

    is_fixed_size = False

    @classmethod
    def from_dict(cls, data):
        if data.get('name') != 'imagecodecs_lzw' or data.get('configuration'):
            raise ValueError('Unsupported TIFF LZW codec configuration')
        return cls()

    def to_dict(self):
        return {'name': 'imagecodecs_lzw', 'configuration': {}}

    async def _decode_single(self, chunk_bytes, chunk_spec):
        dtype = chunk_spec.dtype.to_native_dtype()
        if dtype.hasobject or dtype.itemsize <= 0:
            raise ValueError('Browser LZW requires a fixed-width dtype')
        size = math.prod(chunk_spec.shape) * dtype.itemsize
        data = await _decode_lzw(chunk_bytes.to_bytes(), size)
        if len(data) != size:
            raise RuntimeError('Incorrect decoded LZW buffer length from the browser')
        return chunk_spec.prototype.buffer.from_bytes(data)

    async def _encode_single(self, chunk_bytes, chunk_spec):
        raise NotImplementedError('Browser LZW encoding is not supported')

    def compute_encoded_size(self, input_byte_length, chunk_spec):
        raise NotImplementedError('LZW encoded size is variable')


register_codec('imagecodecs_lzw', BrowserLzwCodec)
