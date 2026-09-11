"""Lazy mosaics of aligned xarray arrays; no storage-specific dependencies."""
from __future__ import annotations

import numpy as np
import xarray as xr
from xarray.backends import BackendArray
from xarray.core import indexing

# TIFF tiepoints can be rounded independently between files. Allow a tiny
# fraction of a pixel, without accepting shifted or resampled grids.
_TOLERANCE = 1e-5


class _MosaicArray(BackendArray):
    def __init__(self, sources, shape, dtype, fill_value, y, x):
        self.sources = sources
        self.shape = shape
        self.dtype = dtype
        self.fill_value = fill_value
        self.y, self.x = y, x

    def __getitem__(self, key):
        raise RuntimeError('Mosaic data requires async I/O; use await .load_async()')

    async def async_getitem(self, key):
        return await indexing.async_explicit_indexing_adapter(
            key, self.shape, indexing.IndexingSupport.BASIC, self._read,
        )

    async def _read(self, key):
        axes = []
        scalar_axes = []
        for axis, (item, size) in enumerate(zip(key, self.shape)):
            if isinstance(item, slice):
                axes.append(np.arange(*item.indices(size)))
            else:
                scalar_axes.append(axis)
                axes.append(np.array([int(item) % size]))
        output = np.full(tuple(len(a) for a in axes), self.fill_value, dtype=self.dtype)
        for source, offsets in self.sources:
            positions, selections, reverse = [], {}, []
            for axis, dim in enumerate((self.y, self.x)):
                indices = axes[axis]
                offset = offsets[axis]
                pos = np.flatnonzero((indices >= offset) & (indices < offset + source.sizes[dim]))
                if not len(pos):
                    break
                local = indices[pos] - offset
                step = abs(int(local[1] - local[0])) if len(local) > 1 else 1
                selections[dim] = slice(int(local.min()), int(local.max()) + 1, step)
                positions.append(pos)
                if len(local) > 1 and local[0] > local[-1]:
                    reverse.append(axis)
            else:
                selected = await source.isel(selections).load_async()
                values = selected.values
                for axis in reverse:
                    values = np.flip(values, axis=axis)
                output[np.ix_(*positions)] = values
        return np.squeeze(output, axis=tuple(scalar_axes))


async def mosaic_async(sources, *, x='x', y='y', crs, fill_value=np.nan) -> xr.DataArray:
    """Create a lazy mosaic of numeric 2-D DataArrays on the same pixel grid.

    Sources must have ascending x and descending y pixel-center coordinates,
    equal spacing and dtype, and no overlapping footprints. Each spatial axis
    needs at least two centers to infer its spacing. Coordinate reads
    are awaited; raster reads occur only with ``await selection.load_async()``.
    ``crs`` declares the common CRS; no reprojection or CRS inference is done.
    Existing source ``crs`` attributes must match it exactly.

    Gaps use fill_value. Integer arrays require an explicit integer fill value.
    Source stores remain owned by the caller. Advanced indexing may fetch a
    larger bounding region. Only coordinates and requested raster regions are
    allocated, never the entire mosaic during construction.
    """
    sources = list(sources)
    if not sources:
        raise ValueError('At least one source is required')
    if not isinstance(crs, str) or not crs.strip():
        raise ValueError('crs must be a nonempty shared CRS identifier')
    if x == y:
        raise ValueError('x and y must name different dimensions')
    if not all(isinstance(source, xr.DataArray) for source in sources):
        raise TypeError('sources must contain xarray DataArrays')
    dtype = sources[0].dtype
    if dtype.kind not in 'iuf':
        raise ValueError('Mosaic sources must have numeric integer or floating dtypes')
    if np.ndim(fill_value) != 0:
        raise ValueError('fill_value must be scalar')
    if dtype.kind in 'iu':
        limits = np.iinfo(dtype)
        if not isinstance(fill_value, (int, np.integer)) or not limits.min <= fill_value <= limits.max:
            raise ValueError('Integer sources require an explicit representable integer fill_value')
    with np.errstate(over='ignore', invalid='ignore'):
        fill = np.asarray(fill_value, dtype=dtype).item()
    if np.isfinite(fill_value) and not np.isfinite(fill):
        raise ValueError('fill_value is not representable in the source dtype')

    prepared, starts, ends, spacing = [], [], [], None
    for source in sources:
        if source.ndim != 2 or set(source.dims) != {x, y}:
            raise ValueError('Each source must have exactly the two spatial dimensions')
        if source.dtype != dtype:
            raise ValueError('Source dtypes must match')
        if source.attrs.get('crs', crs) != crs:
            raise ValueError('Source CRS conflicts with the declared crs')
        source = source.transpose(y, x)
        origin, last, steps = [], [], []
        for dim, direction in ((y, -1), (x, 1)):
            if dim not in source.coords or source.coords[dim].dims != (dim,):
                raise ValueError(f'{dim!r} requires a one-dimensional pixel-center coordinate')
            coord = await source.coords[dim].variable.copy(deep=False).load_async()
            if coord.dtype.kind not in 'iuf':
                raise ValueError('Spatial coordinates must be numeric')
            values = np.asarray(coord.values, dtype=np.float64)
            if len(values) < 2 or not np.all(np.isfinite(values)):
                raise ValueError('Each spatial coordinate needs at least two finite centers')
            step = (values[-1] - values[0]) / (len(values) - 1)
            if step * direction <= 0:
                raise ValueError('Coordinates must be ascending in x and descending in y')
            expected = values[0] + np.arange(len(values)) * step
            if not np.allclose(values, expected, rtol=0, atol=abs(step) * _TOLERANCE):
                raise ValueError('Coordinates must have regular pixel spacing')
            origin.append(values[0])
            last.append(values[-1])
            steps.append(step)
        if spacing is None:
            spacing = np.array(steps)
        elif not np.allclose(steps, spacing, rtol=_TOLERANCE, atol=0):
            raise ValueError('Source pixel spacing must match')
        prepared.append(source)
        starts.append(origin)
        ends.append(last)

    starts = np.array(starts)
    origin = np.array([starts[:, 0].max(), starts[:, 1].min()])
    sources_with_offsets = []
    shape = np.zeros(2, dtype=np.int64)
    rectangles = []
    for source, start, last in zip(prepared, starts, ends):
        offset = (start - origin) / spacing
        rounded = np.rint(offset).astype(np.int64)
        if (not np.allclose(offset, rounded, rtol=0, atol=_TOLERANCE)
                or not np.allclose((np.array(last) - origin) / spacing,
                                   rounded + np.array(source.shape) - 1,
                                   rtol=0, atol=_TOLERANCE)):
            raise ValueError('Source pixel centers are not aligned')
        end = rounded + source.shape
        for previous_start, previous_end in rectangles:
            if np.all(rounded < previous_end) and np.all(previous_start < end):
                raise ValueError('Source footprints overlap')
        rectangles.append((rounded, end))
        sources_with_offsets.append((source, rounded))
        shape = np.maximum(shape, end)

    backend = _MosaicArray(sources_with_offsets, tuple(shape), dtype, fill, y, x)
    excluded = {'crs', 'model_tiepoint', 'model_pixel_scale', 'model_transformation',
                'transform', 'GeoTransform', 'coordinates', '_ARRAY_DIMENSIONS', '_FillValue',
                'scale_factor', 'add_offset'}
    attrs = {key: value for key, value in prepared[0].attrs.items()
             if key not in excluded and all(key in s.attrs and np.array_equal(value, s.attrs[key])
                                            for s in prepared[1:])}
    attrs['crs'] = crs
    name = prepared[0].name if all(s.name == prepared[0].name for s in prepared) else None
    return xr.DataArray(
        xr.Variable((y, x), indexing.LazilyIndexedArray(backend), attrs=attrs),
        coords={dim: origin[axis] + np.arange(shape[axis]) * spacing[axis]
                for axis, dim in enumerate((y, x))},
        name=name,
    )
