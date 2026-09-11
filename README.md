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

Install the Python test and lint tools with `pip install -e '.[test]'`, then run:

```bash
python -m ruff check ipygis tests
python -m ruff format --check ipygis tests
python -m pytest tests
```

Ruff checks basic Python errors and import ordering. To apply its automatic
fixes, use `python -m ruff check ipygis tests --fix`. Format the Python code with
`python -m ruff format ipygis tests`; the formatter is configured to use single
quotes.

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
Zarr v3 variable-length strings are returned as NumPy object arrays (or Python
strings for point selections). Numeric point selections return NumPy scalars;
other numeric selections return NumPy arrays.
Metadata attributes are local snapshots, not a write API. Writing, fancy/boolean
indexing, new axes, negative slice steps, and other nonnumeric dtypes are unsupported.
TIFF LZW decoding is registered in JavaScript automatically. Other codecs must be
supported by Zarrita; this API does not apply xarray's CF decoding or geospatial
coordinate selection. Close the session after finishing all reads.

The existing `session.store` remains usable with Zarr-Python. The new array API
uses its browser backend directly; Python does not fetch or decode the chunks.

## Asynchronous xarray reads

Use ipygis's async opener to construct an xarray dataset from the same store:

```python
from ipygis.xarray import open_zarr_async

ds = await open_zarr_async(session.store)
region = await ds.isel(y=slice(100, 200), x=slice(200, 300)).load_async()
# If the dataset contains x and y coordinate arrays:
point = await ds.sel(x=10.5, y=48.5, method="nearest").load_async()
```

Use your dataset's dimension names in place of `x` and `y`. Dimensions must be
stored in Zarr v3 `dimension_names` or Zarr v2 `_ARRAY_DIMENSIONS` attributes.
Coordinate arrays must already exist; the opener does not derive them from TIFF
georeferencing.

This uses Zarrita for array reads, without calling `xarray.open_zarr()` or
Zarr-Python's synchronous API. Raster data stays lazy until `load_async()`;
synchronous operations such as accessing unloaded `.values` raise an error.
Coordinate arrays for dimension indexes are read asynchronously while opening
so that `.sel()` can work. Pass `create_default_indexes=False` to skip those
reads and use `.isel()` instead.

CF masks and scale factors are decoded lazily. CF datetime arrays are currently
loaded asynchronously before decoding because xarray's datetime decoder probes
values synchronously; use `decode_times=False` to leave those arrays lazy.
The browser API supports numeric arrays and Zarr v3 strings. Advanced indexing may read a larger
bounding region before selecting the requested values in Python. Keep the
session open until all reads finish; closing the dataset does not close it.

## Geographic mosaics

`mosaic_async` combines georeferenced xarray DataArrays into one lazy array:

```python
from ipygis.xarray import mosaic_async

mosaic = await mosaic_async(
    sources, x="longitude", y="latitude", crs="EPSG:4326",
)
region = await mosaic.sel(
    longitude=slice(-110, -109), latitude=slice(50, 49),
).load_async()
```

Each source must be a numeric 2-D array with one-dimensional pixel-center
coordinates, ascending in x and descending in y. Coordinates must have at least
two values per axis, the same regular spacing, and aligned centers. Sources must
share their dtype and CRS, and must not overlap. The `crs` argument declares their
common CRS; source `crs` attributes, when present, must match it exactly. There is
no reprojection or resampling. Chunk boundaries can differ between sources.

Opening loads only coordinates. Reads fetch intersecting source slices and
assemble the requested output in Python. Missing areas are filled with NaN;
integer arrays require an explicit integer `fill_value` to preserve precision.
The result supports xarray selection and `load_async()`; synchronous reads of
unloaded values raise an error. Advanced indexing may fetch a larger bounding
region. Keep all source stores open until reads finish.

The HydroSHEDS notebooks retain the virtual tile collection in Icechunk and
construct this mosaic at runtime from the tile origins and pixel spacing. A
single stored virtual grid cannot join these TIFFs because their edges fall
inside the mosaic's chunks. The mosaic reader handles those boundaries after
Zarrita decodes the selected source chunks.
