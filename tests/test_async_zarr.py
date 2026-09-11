import numpy as np
import pytest

from ipygis.zarr import asynchronous as zarr

pytestmark = pytest.mark.anyio


class Backend:
    def __init__(self):
        self.calls = []
        self.closed = False
        self.data = np.arange(24, dtype='int32').reshape(4, 6)

    async def array_request(self, operation, **args):
        if self.closed:
            raise RuntimeError('Session store is closed')
        self.calls.append((operation, args))
        if operation == 'zarr_members':
            return ['a'], []
        if operation == 'zarr_open':
            if not args['path']:
                return {'kind': 'group', 'attrs': {'title': 'test'}}, []
            return {
                'kind': 'array',
                'attrs': {},
                'shape': [4, 6],
                'chunks': [2, 3],
                'dtype': 'int32',
            }, []
        selection = tuple(
            slice(*s) if isinstance(s, list) else s for s in args['selection']
        )
        data = np.asarray(self.data[selection]).copy()
        return {
            'shape': data.shape,
            'strides': [s // data.itemsize for s in data.strides],
            'dtype': 'int32',
            'byteorder': '<',
        }, [memoryview(data.tobytes())]


async def test_group_and_array_metadata():
    backend = Backend()
    group = await zarr.open_group(backend)
    array = await group.getitem('a')
    assert group.attrs == {'title': 'test'}
    assert array.shape == (4, 6)
    assert array.chunks == (2, 3)
    assert array.dtype == np.dtype('int32')
    assert array.ndim == 2 and array.size == 24
    assert all(operation == 'zarr_open' for operation, _ in backend.calls)
    with pytest.raises(TypeError, match='Expected'):
        await zarr.open_array(backend)
    with pytest.raises(ValueError, match='read-only'):
        await zarr.open_group(backend, mode='w')


@pytest.mark.parametrize(
    'key',
    [
        (slice(1, 4, 2), slice(1, 6, 2)),
        (-1, 2),
        (Ellipsis, -1),
        (),
        slice(None),
        (slice(3, 1), slice(None)),
        (slice(-100, 100), 0),
    ],
)
async def test_slices_match_numpy(key):
    backend = Backend()
    array = await zarr.open_array(backend, path='a')
    actual = await array.getitem(key)
    np.testing.assert_array_equal(actual, backend.data[key])
    if np.ndim(actual) == 0:
        assert isinstance(actual, np.generic)


@pytest.mark.parametrize(
    'key,error',
    [
        ((4, 0), IndexError),
        ((-5, 0), IndexError),
        ((0, 0, 0), IndexError),
        ((Ellipsis, Ellipsis), IndexError),
        ((slice(None, None, 0),), ValueError),
        ((slice(None, None, -1),), NotImplementedError),
        (([1, 2],), TypeError),
        ((None,), TypeError),
        ((True,), TypeError),
    ],
)
async def test_rejects_invalid_selection_without_io(key, error):
    backend = Backend()
    array = await zarr.open_array(backend, path='a')
    with pytest.raises(error):
        await array.getitem(key)
    assert len(backend.calls) == 1


async def test_store_lifecycle_and_paths():
    backend = Backend()
    group = await zarr.open_group(backend)
    array = await group.getitem('nested/a')
    assert array.path == 'nested/a'
    with pytest.raises(ValueError):
        await group.getitem('../escape')
    backend.closed = True
    with pytest.raises(RuntimeError, match='closed'):
        await array.getitem(Ellipsis)


async def test_strided_big_endian_reply():
    class Strided(Backend):
        async def array_request(self, operation, **args):
            if operation == 'zarr_open':
                return await super().array_request(operation, **args)
            data = np.arange(24, dtype='>i4').reshape(4, 6).copy(order='F')
            return {
                'shape': [4, 6],
                'strides': [1, 4],
                'dtype': 'int32',
                'byteorder': '>',
            }, [data.tobytes(order='F')]

    array = await zarr.open_array(Strided(), path='a')
    np.testing.assert_array_equal(
        await array.getitem(Ellipsis), np.arange(24).reshape(4, 6)
    )


async def test_array_transport_is_independent_of_byte_backend():
    from types import SimpleNamespace

    from ipygis.zarr.asynchronous import BrowserArrayBackend

    backend = Backend()
    messages = []

    async def request(operation, *, message_type, **arguments):
        messages.append(message_type)
        return await backend.array_request(operation, **arguments)

    # A byte backend does not need to know about any Zarr operations.
    store = SimpleNamespace(
        backend=object(), array_backend=BrowserArrayBackend(request)
    )
    array = await zarr.open_array(store, path='a')
    assert await array.getitem((1, 2)) == 8
    assert messages == ['zarr_request', 'zarr_request']
    group = await zarr.open_group(store)
    assert [name async for name in group.keys()] == ['a']
    assert [(name, array.path) async for name, array in group.arrays()] == [('a', 'a')]
    assert all(message == 'zarr_request' for message in messages)
    backend.closed = True
    with pytest.raises(RuntimeError, match='closed'):
        await array.getitem((0, 0))
