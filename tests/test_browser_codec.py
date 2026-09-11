"""Zarr registry integration with automatically managed browser decoding."""

import json
from types import SimpleNamespace
from unittest.mock import patch

import anyio
import numpy as np
import pytest
import xarray as xr
import zarr
from test_browser_store import MemoryBackend
from zarr.core.buffer import default_buffer_prototype
from zarr.registry import get_codec_class
from zarr.storage import MemoryStore

from ipygis.gis import _close_decoder_connections
from ipygis.zarr.codecs import BrowserLzwCodec
from ipygis.zarr.storage import BrowserStore


@pytest.mark.anyio
class TestBrowserCodec:
    @pytest.fixture(autouse=True)
    async def setup_store(self):
        memory = MemoryStore()
        group = await zarr.api.asynchronous.create_group(store=memory, zarr_format=3)
        array = await group.create_array(
            'data',
            shape=(2,),
            chunks=(2,),
            dtype='float64',
            dimension_names=('x',),
            compressors=[],
        )
        await array.setitem(slice(None), np.array([7.0, 8.0]))
        self.data = {
            key: (await memory.get(key, default_buffer_prototype())).to_bytes()
            async for key in memory.list()
        }
        metadata = json.loads(self.data['data/zarr.json'])
        metadata['codecs'].append({'name': 'imagecodecs_lzw'})
        self.data['data/zarr.json'] = json.dumps(metadata).encode()
        self.store = await BrowserStore.open(MemoryBackend(self.data))
        self.decodes = []

        async def decode(data, size):
            await anyio.lowlevel.checkpoint()
            self.decodes.append((data, size))
            return self.data['data/c/0']

        self.connection = SimpleNamespace(
            decode_lzw=decode,
            _ready=anyio.Event(),
            _widget=SimpleNamespace(comm=object()),
        )
        self.connection._ready.set()
        self.connection._widget.close = lambda: setattr(
            self.connection._widget, 'comm', None
        )
        with (
            patch('ipygis.gis.GIS', return_value=self.connection) as self.factory,
            zarr.config.set(
                {'codecs.imagecodecs_lzw': 'ipygis.zarr.codecs.BrowserLzwCodec'}
            ),
        ):
            try:
                yield
            finally:
                await self.store.aclose()
                _close_decoder_connections()

    def open_dataset(self, store=None):
        return xr.open_zarr(
            self.store if store is None else store,
            chunks=None,
            consolidated=False,
            create_default_indexes=False,
        )

    async def test_registry_selects_browser_codec_without_metadata_changes(self):
        assert get_codec_class('imagecodecs_lzw') is BrowserLzwCodec
        with patch.dict('sys.modules', {'virtual_tiff': None, 'imagecodecs': None}):
            ds = self.open_dataset()
            assert self.decodes == []
            self.factory.assert_not_called()
            value = await ds.data.isel(x=0).load_async()
            self.factory.assert_called_once()
        assert value.item() == 7.0
        assert self.decodes == [(self.data['data/c/0'], 16)]
        assert self.store._metadata['data/zarr.json'] == self.data['data/zarr.json']

    async def test_sequential_reads_reuse_the_decoder_connection(self):
        ds = self.open_dataset()
        for index in (0, 1):
            result = await ds.data.isel(x=index).load_async()
            assert result.item() == (7.0 if index == 0 else 8.0)
        self.factory.assert_called_once()
        assert len(self.decodes) == 2

    async def test_concurrent_reads_share_one_connection(self):
        ds = self.open_dataset()
        results = []

        async def read(index):
            result = await ds.data.isel(x=index).load_async()
            results.append(result.item())

        async with anyio.create_task_group() as tg:
            tg.start_soon(read, 0)
            tg.start_soon(read, 1)
        assert sorted(results) == [7.0, 8.0]
        self.factory.assert_called_once()
        assert len(self.decodes) == 2

    async def test_codec_configuration_roundtrips_without_runtime_data(self):
        codec = BrowserLzwCodec.from_dict({'name': 'imagecodecs_lzw'})
        assert codec.to_dict() == {'name': 'imagecodecs_lzw', 'configuration': {}}
        assert BrowserLzwCodec.from_dict(codec.to_dict()) == codec
        with pytest.raises(ValueError, match='configuration'):
            BrowserLzwCodec.from_dict(
                {'name': 'imagecodecs_lzw', 'configuration': {'unsupported': True}}
            )

    async def test_standard_gzip_codec_is_unaffected(self):
        import gzip

        data = dict(self.data)
        metadata = json.loads(data['data/zarr.json'])
        metadata['codecs'][-1] = {'name': 'gzip', 'configuration': {'level': 1}}
        data['data/zarr.json'] = json.dumps(metadata).encode()
        data['data/c/0'] = gzip.compress(data['data/c/0'])
        store = await BrowserStore.open(MemoryBackend(data))
        try:
            ds = self.open_dataset(store)
            result = await ds.data.isel(x=0).load_async()
            assert result.item() == 7.0
            assert self.decodes == []
        finally:
            await store.aclose()

    async def test_decoder_output_length_is_checked(self):
        async def decode(data, size):
            return b''

        self.connection.decode_lzw = decode
        ds = self.open_dataset()
        with pytest.raises(RuntimeError, match='decoded LZW buffer length'):
            await ds.data.isel(x=0).load_async()
