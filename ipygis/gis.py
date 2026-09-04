from __future__ import annotations

import anyio
from uuid import uuid4
import atexit
from anyio.lowlevel import current_token

from IPython.display import display
from ipywidgets import DOMWidget
from traitlets import Unicode

from ._frontend import module_name, module_version


class GISWidget(DOMWidget):
    """The widget that allows communicating with GIS libraries in the browser."""

    _model_name = Unicode("GISModel").tag(sync=True)
    _model_module = Unicode(module_name).tag(sync=True)
    _model_module_version = Unicode(module_version).tag(sync=True)
    _view_name = Unicode("GISView").tag(sync=True)
    _view_module = Unicode(module_name).tag(sync=True)
    _view_module_version = Unicode(module_version).tag(sync=True)


class GIS:
    def __init__(self):
        self._ready = anyio.Event()
        self._widget = GISWidget()
        self._pending = {}
        self._widget.on_msg(self._on_response)
        display(self._widget)

    def _on_response(self, widget, content, buffers):
        if content.get("type") == "icechunk_ready":
            self._ready.set()
            return
        if content.get("type") != "icechunk_response":
            return
        future = self._pending.get(content.get("id"))
        if future is None or future.status is not anyio.Future.Status.PENDING:
            return
        if "error" in content:
            future.exception = RuntimeError(content["error"])
        else:
            future.return_value = (content.get("result"), buffers)

    def _notify_close(self, store_id):
        self._widget.send({"type": "icechunk_request", "id": uuid4().hex,
                           "operation": "close", "store_id": store_id})

    async def _request(self, operation, *, buffers=None, **arguments):
        request_id = uuid4().hex
        future = anyio.Future()
        self._pending[request_id] = future
        try:
            self._widget.send({"type": "icechunk_request", "id": request_id,
                               "operation": operation, **arguments}, buffers=buffers or [])
            with anyio.fail_after(120):
                await future.wait()
            if future.exception is not None:
                raise future.exception
            return future.return_value
        except BaseException:
            if operation in {"open", "readonly_session"}:
                self._notify_close(request_id)
            raise
        finally:
            self._pending.pop(request_id, None)
            future.cancel()

    async def decode_lzw(self, data: bytes, decoded_size: int) -> bytes:
        """Decode TIFF LZW bytes in the browser, independently of storage."""
        if self._widget.comm is None:
            raise RuntimeError("Browser connection is closed")
        with anyio.fail_after(30):
            await self._ready.wait()
        _, buffers = await self._request('decode_lzw', decoded_size=decoded_size,
                                        buffers=[memoryview(data)])
        if len(buffers) != 1 or len(buffers[0]) != decoded_size:
            raise RuntimeError('Incorrect decoded LZW buffer length from the browser')
        return bytes(buffers[0])


# A decoder connection is shared by reads on the same event loop. It is separate
# from repository connections, so closing a repository does not interrupt it.
_decoder_connections = {}


def _get_decoder_connection() -> GIS:
    token = current_token()
    connection = _decoder_connections.get(token)
    if connection is None or connection._widget.comm is None:
        connection = GIS()
        _decoder_connections[token] = connection
    return connection


async def _decode_lzw(data: bytes, decoded_size: int) -> bytes:
    connection = _get_decoder_connection()
    try:
        return await connection.decode_lzw(data, decoded_size)
    except TimeoutError:
        # Retry initialization on the next read if this frontend never became ready.
        if not connection._ready.is_set():
            token = current_token()
            if _decoder_connections.get(token) is connection:
                _decoder_connections.pop(token)
                connection._widget.close()
        raise


def _close_decoder_connections():
    for connection in list(_decoder_connections.values()):
        connection._widget.close()
    _decoder_connections.clear()


atexit.register(_close_decoder_connections)
