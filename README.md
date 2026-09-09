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

That notebook will read the repository as a Python Zarr store which can be opened by
xarray, and it can run in JupyterLite. The browser uses
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
