import pytest


@pytest.fixture
def anyio_backend():
    # Zarr/xarray currently use asyncio internally.
    return "asyncio"
