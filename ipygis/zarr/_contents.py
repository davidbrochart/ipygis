"""Array reads from Jupyter Contents over the independent Zarr bridge."""

import anyio
from anyio.lowlevel import current_token


class ContentsArrayBackend:
    @classmethod
    async def open(cls, path):
        if (
            not path
            or path.startswith('/')
            or '\\' in path
            or any(part in {'', '.', '..'} for part in path.split('/'))
        ):
            raise ValueError(
                'Use a nonempty path relative to the Jupyter contents root'
            )
        from ..gis import GIS

        connection = GIS()
        try:
            with anyio.fail_after(30):
                await connection._ready.wait()
            result, _ = await connection._request(
                'contents_open',
                message_type='zarr_request',
                path=path,
            )
            return cls(connection, result['store_id'])
        except BaseException:
            connection._widget.close()
            raise

    def __init__(self, connection, store_id):
        self._connection = connection
        self._store_id = store_id
        self._token = current_token()
        self._closed = False
        self._requests = anyio.Semaphore(8)

    async def array_request(self, operation, **arguments):
        if operation not in {'zarr_open', 'zarr_get', 'zarr_members'}:
            raise ValueError(f'Unknown array operation: {operation}')
        if self._closed:
            raise RuntimeError('Contents store is closed; reopen the dataset')
        if current_token() != self._token:
            raise RuntimeError('Browser I/O requires the notebook event loop')
        async with self._requests:
            if self._closed:
                raise RuntimeError('Contents store is closed; reopen the dataset')
            return await self._connection._request(
                operation,
                message_type='zarr_request',
                store_id=self._store_id,
                **arguments,
            )

    def close(self):
        if not self._closed:
            self._closed = True
            self._connection._widget.close()
