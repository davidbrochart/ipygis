"""Read-only Zarr v3 adapter for asynchronous byte-storage backends."""
from __future__ import annotations

import json
import anyio
from collections.abc import AsyncIterator, Iterable
from typing import Protocol

from zarr.abc.store import (
    Store, ByteRequest, RangeByteRequest, OffsetByteRequest, SuffixByteRequest,
)
from zarr.core.buffer import Buffer, BufferPrototype


class StorageBackend(Protocol):
    """Byte storage contract; ranges use offset/length or suffixLength.

    The backend handles its transport, concurrency, and event-loop requirements.
    """
    async def get(self, key: str, byte_range: dict | None = None) -> bytes | None: ...
    async def exists(self, key: str) -> bool: ...
    async def list(self) -> list[str]: ...
    async def list_prefix(self, prefix: str) -> list[str]: ...
    async def list_dir(self, prefix: str) -> list[str]: ...
    async def aclose(self) -> None: ...
    def close(self) -> None: ...


class BrowserStore(Store):
    """Read-only Zarr adapter with prefetched metadata for synchronous opening.

    Chunk reads delegate to the supplied backend. Zarr resolves codecs from metadata.
    Closing the adapter closes its backend.
    """
    supports_writes = False
    supports_deletes = False
    supports_listing = True
    supports_consolidated_metadata = False

    @classmethod
    async def open(cls, backend: StorageBackend, *, read_only=True, array_backend=None) -> BrowserStore:
        """Prefetch Zarr v3 metadata from an already-open backend.

        On failure the backend remains owned by the caller. On success the
        returned store takes responsibility for closing it. An optional
        ``array_backend`` handles browser array reads independently of byte I/O;
        its transport must enforce the same lifetime as the byte backend.
        """
        if not read_only:
            raise ValueError('BrowserStore only supports read-only access')
        metadata = {}
        directories = {}

        async def visit(path):
            key = f'{path}/zarr.json' if path else 'zarr.json'
            data = await backend.get(key)
            if data is None:
                raise ValueError(f'Missing Zarr metadata: {key}')
            metadata[key] = data
            if json.loads(data).get('node_type') == 'group':
                entries = await backend.list_dir(path)
                directories[path] = entries
                for entry in entries:
                    if entry != 'zarr.json':
                        await visit(f'{path}/{entry}' if path else entry)

        await visit('')
        return cls(backend, metadata, directories, array_backend=array_backend)

    def __init__(self, backend: StorageBackend, metadata: dict[str, bytes],
                 directories: dict[str, list[str]], *, array_backend=None):
        super().__init__(read_only=True)
        self.backend = backend
        self.array_backend = array_backend
        self._metadata = dict(metadata)
        self._directories = {key: list(value) for key, value in directories.items()}
        self._closed = False
        self._is_open = True

    def __eq__(self, other: object) -> bool:
        return self is other

    def _check_open(self):
        if self._closed:
            raise RuntimeError('Browser store is closed; reopen the store')

    @staticmethod
    def _metadata_key(key: str) -> bool:
        return key.rsplit('/', 1)[-1] in {'zarr.json', '.zarray', '.zgroup', '.zattrs', '.zmetadata'}

    @staticmethod
    def _range(byte_range: ByteRequest | None) -> dict | None:
        if byte_range is None:
            return None
        if isinstance(byte_range, RangeByteRequest):
            result = {'offset': byte_range.start, 'length': byte_range.end - byte_range.start}
        elif isinstance(byte_range, OffsetByteRequest):
            result = {'offset': byte_range.offset}
        elif isinstance(byte_range, SuffixByteRequest):
            result = {'suffixLength': byte_range.suffix}
        else:
            raise TypeError(f'Unsupported byte range: {byte_range!r}')
        if any(not isinstance(value, int) or value < 0 or value > 2**53 - 1 for value in result.values()):
            raise ValueError('Byte ranges must be nonnegative safe JavaScript integers')
        return result

    @staticmethod
    def _slice(data: bytes, byte_range: ByteRequest | None) -> bytes:
        if isinstance(byte_range, RangeByteRequest):
            return data[byte_range.start:byte_range.end]
        if isinstance(byte_range, OffsetByteRequest):
            return data[byte_range.offset:]
        if isinstance(byte_range, SuffixByteRequest):
            return data[-byte_range.suffix:] if byte_range.suffix else b''
        return data

    async def get(self, key: str, prototype: BufferPrototype, byte_range: ByteRequest | None = None) -> Buffer | None:
        self._check_open()
        query = self._range(byte_range)
        if self._metadata_key(key):
            data = self._metadata.get(key)
            return None if data is None else prototype.buffer.from_bytes(self._slice(data, byte_range))
        data = await self.backend.get(key, query)
        return None if data is None else prototype.buffer.from_bytes(data)

    async def get_partial_values(self, prototype: BufferPrototype, key_ranges: Iterable[tuple[str, ByteRequest | None]]) -> list[Buffer | None]:
        return list(await anyio.gather(*(self.get(key, prototype, byte_range) for key, byte_range in key_ranges)))

    async def exists(self, key: str) -> bool:
        self._check_open()
        if self._metadata_key(key):
            return key in self._metadata
        return await self.backend.exists(key)

    async def list(self) -> AsyncIterator[str]:
        self._check_open()
        result = await self.backend.list()
        for key in result:
            yield key

    async def list_prefix(self, prefix: str) -> AsyncIterator[str]:
        self._check_open()
        result = await self.backend.list_prefix(prefix)
        for key in result:
            yield key

    async def list_dir(self, prefix: str) -> AsyncIterator[str]:
        self._check_open()
        prefix = prefix.rstrip('/')
        if prefix in self._directories:
            entries = self._directories[prefix]
        else:
            entries = await self.backend.list_dir(prefix)
        for entry in entries:
            yield entry

    async def set(self, key: str, value: Buffer) -> None:
        self._check_writable()

    async def delete(self, key: str) -> None:
        self._check_writable()

    def with_read_only(self, read_only: bool = False) -> BrowserStore:
        if not read_only:
            raise ValueError('BrowserStore only supports read-only access')
        self._check_open()
        return self

    async def aclose(self):
        if not self._closed:
            try:
                await self.backend.aclose()
            finally:
                self._closed = True
                super().close()

    def close(self):
        if not self._closed:
            try:
                self.backend.close()
            finally:
                self._closed = True
                super().close()

    async def __aenter__(self):
        self._check_open()
        return self

    async def __aexit__(self, *exc):
        await self.aclose()
