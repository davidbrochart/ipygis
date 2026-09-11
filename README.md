[![Build Status](https://github.com/davidbrochart/ipygis/actions/workflows/test.yml/badge.svg?query=branch%3Amain++)](https://github.com/davidbrochart/ipygis/actions/workflows/test.yml/badge.svg?query=branch%3Amain++)

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

Run `examples/build_time.ipynb` first to create the Icechunk repository. This notebook
cannot run in JupyterLite because of incompatible libraries, but the created repository
is the only thing that will be needed by `examples/run_time.ipynb`.

That notebook uses a read-only API modeled on Zarr-Python’s asynchronous interface.
[Zarrita](https://github.com/manzt/zarrita.js) reads, decodes, and slices arrays in the
browser, returning NumPy arrays to Python without Zarr-Python’s synchronous bridge.
The browser uses
[icechunk-js](https://github.com/EarthyScience/icechunk-js) by default. This backend
requires no COOP/COEP headers or shared memory. You can change the backend to
`"@earthmover/icechunk"` but this is a WASM library that needs `SharedArrayBuffer` and
so the server must send COOP/COEP headers. In this case JupyterLab must use a special
configuration:

```bash
jupyter lab --config=./jupyter_server_config.json
```

Both readers are installed from npm. No local Icechunk checkout or WASM preparation
step is required. The published `@earthmover/icechunk` 2.0.3 package does not include
the browser HTTP virtual-chunk callback; use `backend="icechunk-js"` for the
HydroSHEDS runtime notebook.

## Asynchronous array reads

```python
from ipygis.zarr import asynchronous as zarr

group = await zarr.open_group(session.store, mode="r")
array = await group.getitem("0")
print(array.shape, array.dtype, array.chunks)
region = await array.getitem((10, slice(100, 200), slice(200, 300)))
point = await array.getitem((10, 100, 200))
```

You can also use `await zarr.open_array(session.store, path="0", mode="r")`.
Reads support integers (including negative indices), ellipses, and positive-step
slices. Integer and floating-point dtypes from 8-bit integers through 64-bit
integers/floats are supported; 64-bit integers are transferred as binary data.
Point selections return NumPy scalars; other selections return NumPy arrays.
Metadata attributes are local snapshots, not a write API. Writing, fancy/boolean
indexing, new axes, negative slice steps, and nonnumeric dtypes are unsupported.
TIFF LZW decoding is registered in JavaScript automatically. Other codecs must be
supported by Zarrita; this API does not apply xarray's CF decoding or geospatial
coordinate selection. Close the session after finishing all reads.

The existing `session.store` remains usable with Zarr-Python. The new array API
uses its browser backend directly; Python does not fetch or decode the chunks.
