"""Lifecycle of the decoder connection, independent of repositories."""
from types import SimpleNamespace
from unittest.mock import Mock

import anyio
import pytest

from ipygis.gis import _get_decoder_connection, _decode_lzw, _close_decoder_connections


def connection():
    widget = SimpleNamespace(comm=object())
    widget.close = Mock(side_effect=lambda: setattr(widget, 'comm', None))
    return SimpleNamespace(_widget=widget, _ready=anyio.Event())


@pytest.fixture(autouse=True)
def cleanup():
    _close_decoder_connections()
    yield
    _close_decoder_connections()


@pytest.mark.anyio
async def test_closed_connection_is_replaced(monkeypatch):
    first, second = connection(), connection()
    factory = Mock(side_effect=[first, second])
    monkeypatch.setattr('ipygis.gis.GIS', factory)
    assert _get_decoder_connection() is first
    first._widget.close()
    assert _get_decoder_connection() is second
    assert _get_decoder_connection() is second
    assert factory.call_count == 2


@pytest.mark.anyio
async def test_readiness_timeout_releases_connection_for_retry(monkeypatch):
    first, second = connection(), connection()

    async def timeout(data, size):
        raise TimeoutError('frontend never became ready')

    async def decode(data, size):
        return b'AB'

    first.decode_lzw = timeout
    second.decode_lzw = decode
    second._ready.set()
    factory = Mock(side_effect=[first, second])
    monkeypatch.setattr('ipygis.gis.GIS', factory)
    with pytest.raises(TimeoutError):
        await _decode_lzw(b'compressed', 2)
    first._widget.close.assert_called_once()
    assert await _decode_lzw(b'compressed', 2) == b'AB'
    assert factory.call_count == 2


def test_different_event_loops_get_separate_connections(monkeypatch):
    factory = Mock(side_effect=connection)
    monkeypatch.setattr('ipygis.gis.GIS', factory)

    async def get():
        return _get_decoder_connection()

    first = anyio.run(get)
    second = anyio.run(get)
    assert first is not second
    assert factory.call_count == 2
    _close_decoder_connections()
    first._widget.close.assert_called_once()
    second._widget.close.assert_called_once()
