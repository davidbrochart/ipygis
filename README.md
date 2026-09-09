# ipygis

A bridge to GIS libraries running in the browser.

## Development installation

```bash
micromamba create -n ipygis
micromamba activate ipygis
micromamba install xeus-python pip "nodejs<25"
pip install jupyterlab -e .
jupyter labextension develop --overwrite .
```

When making changes to the JavaScript code, you can just recompile that part:

```bash
jlpm run build
```

## Running the example notebooks

Start JupyterLab from the repository root:

```bash
jupyter lab
```

The browser uses [icechunk-js](https://github.com/EarthyScience/icechunk-js), a
TypeScript Icechunk reader. No COOP/COEP headers or shared memory are required.

### HydroSHEDS browser read

`examples/run_time.ipynb` opens `examples/hydrosheds.icechunk` on branch `main`
as a Python Zarr store, then reads it with xarray. Use the xeus-python kernel and
start Jupyter from this directory; repository paths are relative to the server
root. Run `examples/build_time.ipynb` first if the store does not exist.

Install the Python integration:

```bash
pip install -e .
```

```python
import xarray as xr
import zarr
from ipygis.icechunk import Repository, jupyter_storage
from ipygis.zarr.codecs import BrowserLzwCodec  # registers imagecodecs_lzw

# Select the browser implementation if other packages also provide this codec.
zarr.config.set({"codecs.imagecodecs_lzw": "ipygis.zarr.codecs.BrowserLzwCodec"})

storage = jupyter_storage("examples/hydrosheds.icechunk")
repository = await Repository.open_async(
    storage,
    proxy_url="https://my-proxy.david-brochart.workers.dev/",
    virtual_chunk_prefixes=["https://data.hydrosheds.org/file/hydrosheds-v2/ACC/1s/"],
)
session = await repository.readonly_session_async("main")
ds = xr.open_zarr(
    session.store, chunks=None, consolidated=False, create_default_indexes=False,
)
result = await ds["0"].isel(tile=10, y=100, x=200).load_async()
print(result.item())  # 7.0
await session.aclose()
await repository.aclose()  # releases the repository and its connection
```

`jupyter_storage(path)` configures access to repository files through Jupyter;
it performs no I/O. `Repository.open_async(storage)` opens the repository without
selecting a snapshot. Proxy settings and virtual-chunk permissions are supplied
separately when opening the repository.
`await repository.readonly_session_async("main")` pins the branch's current snapshot;
use `snapshot_id="..."` instead to select a specific snapshot. The session exposes
`snapshot_id` and a ready-to-use Zarr `store`. Session creation prefetches its
metadata. Browser codec configuration is a separate, optional step. Closing a session store leaves
the repository open;
closing the repository releases all of its sessions. If metadata loading fails during session creation,
the session is released automatically.
Internally, `BrowserStore` adapts byte storage to Zarr and caches metadata.
You do not need to wrap `session.store` yourself. For other storage backends,
`BrowserStore.open(backend)` remains available.
It does not fetch raster chunks. That cache lets
xarray's synchronous open step run without browser communication. Subsequent
async reads send requests to the browser over widget comms and return binary
buffers to Python. Zarr Python's async LZW codec delegates decompression back to
the browser using `@developmentseed/lzw-tiff-decoder` (Rust/WASM); Python handles
byte interpretation and xarray constructs the result. Neither `virtual-tiff`
nor native `imagecodecs` is required at read time. `virtual-tiff` is still used
by the build-time notebook to create the virtual references.

Importing `BrowserLzwCodec` registers it under the existing `imagecodecs_lzw`
name using Zarr's codec registry. Zarr's `codecs.imagecodecs_lzw` configuration
selects it when multiple implementations are installed. Repository metadata and
the store's cached metadata stay unchanged.

The codec lazily creates a shared browser decoder connection on the first read.
Reads on the same event loop reuse it, independently of repository connections.
Closing a repository does not close the decoder connection. Closed connections
are recreated on demand, and the shared connections are closed at Python shutdown.
No connection context or runtime token is needed in user code or array metadata.
`BrowserLzwCodec` supports fixed-width TIFF LZW
chunks; other TIFF codecs and predictors require separate implementations. In a native kernel, compressed bytes make an
extra round trip to the browser for decoding; this prototype does not optimize
that transfer yet.

The browser reads repository files using JupyterLab's `ContentsManager` and
`ServerConnection`. `icechunk-js` resolves virtual chunks, and browser `fetch`
requests their TIFF byte ranges through the configured proxy. No remote raster
I/O occurs in the Python kernel.

This initial integration supports read-only Zarr v3 repositories. Keep
`chunks=None` and `create_default_indexes=False`, and use `load_async()` for data
and coordinate reads. Synchronous reads such as `.compute()`, `.load()`, or
`.values` raise an error when they need browser I/O. Load coordinates explicitly
before operations that need their values, such as creating indexes for `.sel()`.
Close stores before refreshing the page, and reopen them after a refresh.

Validated with a native xeus-python kernel and browser-owned I/O. Running Python
itself in the browser remains untested: Zarr's synchronous initialization and
other codecs may still use Python threads, which need separate runtime integration.
The new LZW codec itself does not use Python threads.

The proxy must preserve Range requests, return HTTP 206, and expose Content-Range
and the checksum headers required by the repository (ETag/Last-Modified) via CORS.

Run the offline Python integration tests with:

```bash
python -m pytest tests -v
jlpm test
```

These use a local synthetic Zarr dataset and a fake browser transport; they do
not contact HydroSHEDS or the proxy.

Install and build the frontend with:

```bash
jlpm install
jlpm build
```

No sibling Icechunk checkout or custom WASM build is needed. Restart the notebook
kernel and refresh JupyterLab after updating the widget.

The reader is pinned to `icechunk-js` 0.6.0: full Zarr key listing currently uses
its internal cached manifest loader because its public API only lists nodes.
The integration tests cover sparse chunk listings; revisit this adapter when
upgrading the reader or when it exposes a public chunk-reference iterator.

The current HydroSHEDS host supplies a Backblaze `x-bz-upload-timestamp` in
milliseconds instead of `Last-Modified`. The existing store uses a modification
time checksum. For these HydroSHEDS responses, the proxy can conservatively use
the object upload time as `Last-Modified` and expose it, along with Content-Range:

```js
const headers = new Headers(upstream.headers);
const uploaded = headers.get('x-bz-upload-timestamp');
if (!headers.has('Last-Modified') && uploaded && /^\d+$/.test(uploaded)) {
  const date = new Date(Number(uploaded));
  if (Number.isFinite(date.getTime())) {
    headers.set('Last-Modified', date.toUTCString());
  }
}
headers.set('Access-Control-Allow-Origin', '*');
headers.set('Access-Control-Expose-Headers', 'Content-Range, ETag, Last-Modified');
return new Response(upstream.body, {status: upstream.status, headers});
```

Apply that timestamp fallback only to this trusted HydroSHEDS/Backblaze upstream.
A later upload may cause Icechunk to reject the reference even when the bytes
are unchanged; refresh the virtual references in that case. Do not substitute
the request date or disable checksum validation.
