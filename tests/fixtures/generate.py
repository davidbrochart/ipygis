"""Regenerate icechunk.json with native icechunk; no remote I/O is performed."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import anyio
import icechunk
from zarr.core.buffer import default_buffer_prototype


async def generate(root):
    config = icechunk.RepositoryConfig(inline_chunk_threshold_bytes=4)
    config.set_virtual_chunk_container(icechunk.VirtualChunkContainer(
        url_prefix="https://example.test/", store=icechunk.http_store(),
    ))
    repo = icechunk.Repository.create(
        icechunk.local_filesystem_storage(root), config=config,
        authorize_virtual_chunk_access={"https://example.test/": icechunk.credentials.HttpAccess},
    )
    session = repo.writable_session("main")
    buffer = default_buffer_prototype().buffer

    async def put(key, value):
        await session.store.set(key, buffer.from_bytes(value))

    await put("zarr.json", json.dumps({
        "zarr_format": 3, "node_type": "group", "attributes": {},
    }).encode())
    for name in ("sparse", "inline", "scalar"):
        metadata = {
            "zarr_format": 3, "node_type": "array", "attributes": {},
            "shape": [] if name == "scalar" else [32], "data_type": "uint8",
            "chunk_grid": {"name": "regular", "configuration": {
                "chunk_shape": [] if name == "scalar" else [8],
            }},
            "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
            "fill_value": 0, "codecs": [{"name": "bytes"}],
        }
        await put(f"{name}/zarr.json", json.dumps(metadata).encode())
    await put("sparse/c/0", b"12345678")
    await put("inline/c/0", b"ab")
    await put("scalar/c", b"\x07")
    session.store.set_virtual_ref(
        "sparse/c/2", "https://example.test/raster.tif",
        offset=10, length=8, checksum="fixture-etag",
    )
    snapshot = session.commit("Reader compatibility fixture")
    objects = {
        str(path.relative_to(root)): path.read_bytes().hex()
        for path in Path(root).rglob("*") if path.is_file()
    }
    Path(__file__).with_name("icechunk.json").write_text(
        json.dumps({"snapshot": snapshot, "objects": objects}, indent=2) + "\n",
    )


if __name__ == "__main__":
    with TemporaryDirectory() as directory:
        anyio.run(generate, directory)
