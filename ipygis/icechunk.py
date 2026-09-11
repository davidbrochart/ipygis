"""Browser access to Icechunk repositories and read-only sessions."""
from __future__ import annotations

from typing import TYPE_CHECKING, Literal
from dataclasses import dataclass
from weakref import WeakValueDictionary
import anyio
from anyio.lowlevel import current_token

if TYPE_CHECKING:
    from .gis import GIS
    from .zarr.storage import BrowserStore


@dataclass(frozen=True)
class Storage:
    """Repository files accessed through Jupyter's contents service.

    ``path`` is relative to the Jupyter server root, not the Python filesystem.
    Creating this configuration performs no I/O.
    """

    path: str

    def __post_init__(self):
        if not isinstance(self.path, str):
            raise TypeError('Storage path must be a string')
        if any(part in {'.', '..'} for part in self.path.split('/')):
            raise ValueError('Repository paths must not contain . or ..')


def jupyter_storage(path: str) -> Storage:
    """Configure repository storage using Jupyter's shared contents manager."""
    return Storage(path)


class Repository:
    """An open Icechunk repository. Create sessions to select snapshots."""

    @classmethod
    async def open_async(cls, storage: Storage, *, proxy_url="", virtual_chunk_prefixes=None,
                   gis: GIS | None = None,
                   backend: Literal["icechunk-js", "@earthmover/icechunk"] = "icechunk-js") -> Repository:
        """Open an existing repository with the selected browser implementation.

        ``icechunk-js`` (default) runs without shared memory.
        ``@earthmover/icechunk`` uses WASM and requires COOP/COEP headers.
        The choice applies to this repository and all sessions created from it.
        """
        if backend not in ("icechunk-js", "@earthmover/icechunk"):
            raise ValueError(f"Unknown Icechunk backend: {backend}")
        if not isinstance(storage, Storage):
            raise TypeError('storage must be a Storage; use jupyter_storage(path)')
        from .gis import GIS

        owns_gis = gis is None
        if gis is None:
            gis = GIS()
        try:
            with anyio.fail_after(30):
                await gis._ready.wait()
            result, _ = await gis._request(
                'open', repository=storage.path, proxy_url=proxy_url, backend=backend,
                virtual_chunk_prefixes=list(virtual_chunk_prefixes or []),
            )
            try:
                repo = cls(gis, result['repository_id'])
            except BaseException:
                gis._notify_close(result['repository_id'])
                raise
            repo._owns_gis = owns_gis
            return repo
        except BaseException:
            if owns_gis:
                gis._widget.close()
            raise

    def __init__(self, gis: GIS, repository_id: str):
        self.connection = gis
        self._repository_id = repository_id
        self._owns_gis = False
        self._closed = False
        self._loop_token = current_token()
        self._requests = anyio.Semaphore(8)
        self._stores = WeakValueDictionary()

    def _check_open(self):
        if self._closed:
            raise RuntimeError('Repository is closed; reopen it')

    async def _request(self, operation, *, resource_id=None, **arguments):
        self._check_open()
        if current_token() != self._loop_token:
            raise RuntimeError(
                'Browser I/O requires the notebook event loop; use await .load_async()'
            )
        async with self._requests:
            return await self.connection._request(
                operation, store_id=resource_id or self._repository_id, **arguments,
            )

    async def readonly_session_async(self, branch: str | None = None, *,
                               snapshot_id: str | None = None) -> Session:
        """Pin a snapshot and prepare its Zarr store, fetching metadata only.

        ``session.store`` can be passed directly to Zarr or xarray. Configure
        browser codecs separately through ``ipygis.zarr.codecs``.
        """
        from .zarr.storage import BrowserStore
        from .zarr.asynchronous import BrowserArrayBackend

        if branch is not None and snapshot_id is not None:
            raise ValueError('Specify either branch or snapshot_id')
        selector = {'snapshot_id': snapshot_id} if snapshot_id is not None else {'branch': branch or 'main'}
        result, _ = await self._request('readonly_session', **selector)
        backend = _IcechunkBackend(self, result['store_id'])
        store = None
        try:
            self._check_open()
            store = await BrowserStore.open(
                backend, array_backend=BrowserArrayBackend(backend._remote),
            )
            self._check_open()
            self._stores[result['store_id']] = store
            return Session(self, result['snapshot_id'], store)
        except BaseException:
            if store is not None:
                store.close()
            else:
                backend.close()
            raise

    def _close_stores(self):
        self._closed = True
        for store in list(self._stores.values()):
            store.close()
        self._stores.clear()

    async def aclose(self):
        if not self._closed:
            try:
                await self._request('close')
            finally:
                self._close_stores()
                if self._owns_gis:
                    self.connection._widget.close()

    def close(self):
        if not self._closed:
            self.connection._notify_close(self._repository_id)
            self._close_stores()
            if self._owns_gis:
                self.connection._widget.close()

    async def __aenter__(self):
        self._check_open()
        return self

    async def __aexit__(self, *exc):
        await self.aclose()


class Session:
    """A read-only snapshot whose store implements Zarr Python's Store API."""

    def __init__(self, repository: Repository, snapshot_id: str, store: BrowserStore):
        self.repository = repository
        self.snapshot_id = snapshot_id
        self.store = store

    async def aclose(self):
        await self.store.aclose()

    def close(self):
        self.store.close()

    async def __aenter__(self):
        self.store._check_open()
        return self

    async def __aexit__(self, *exc):
        await self.aclose()


class _IcechunkBackend:
    """Byte access for one session; closing it leaves the repository open."""

    def __init__(self, repository: Repository, store_id: str):
        self._repository = repository
        self._store_id = store_id
        self._closed = False

    def _check_open(self):
        if self._closed:
            raise RuntimeError('Session store is closed; open a new session')
        self._repository._check_open()

    async def _remote(self, operation: str, **arguments):
        self._check_open()
        return await self._repository._request(
            operation, resource_id=self._store_id, **arguments,
        )

    async def get(self, key: str, byte_range: dict | None = None) -> bytes | None:
        result, buffers = await self._remote('get', key=key, range=byte_range)
        if result['missing']:
            return None
        if len(buffers) != 1:
            raise RuntimeError('Expected one binary buffer from browser Icechunk')
        return bytes(buffers[0])

    async def exists(self, key: str) -> bool:
        result, _ = await self._remote('exists', key=key)
        return result

    async def list(self) -> list[str]:
        result, _ = await self._remote('list')
        return result

    async def list_prefix(self, prefix: str) -> list[str]:
        result, _ = await self._remote('list_prefix', prefix=prefix)
        return result

    async def list_dir(self, prefix: str) -> list[str]:
        result, _ = await self._remote('list_dir', prefix=prefix)
        return result

    async def aclose(self):
        if not self._closed:
            try:
                if not self._repository._closed:
                    await self._remote('close')
            finally:
                self._closed = True

    def close(self):
        if not self._closed:
            if not self._repository._closed:
                self._repository.connection._notify_close(self._store_id)
            self._closed = True
