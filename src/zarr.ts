import * as zarr from 'zarrita';
import { decodeLzw } from './lzw.js';

/** The browser byte-store contract, independent of the repository implementation. */
export interface ByteStore {
  get(key: string): Promise<Uint8Array | null>;
}

const sizes: Record<string, number> = {
  int8: 1,
  uint8: 1,
  int16: 2,
  uint16: 2,
  int32: 4,
  uint32: 4,
  int64: 8,
  uint64: 8,
  float32: 4,
  float64: 8,
};

zarr.registry.set('imagecodecs_lzw', async () => ({
  fromConfig(config, meta) {
    if (config && Object.keys(config).length) {
      throw new Error('Unsupported imagecodecs_lzw configuration');
    }
    const itemsize = sizes[meta.dataType];
    if (!itemsize) throw new Error(`Unsupported TIFF dtype: ${meta.dataType}`);
    const size = meta.shape.reduce((a, b) => a * b, itemsize);
    return {
      kind: 'bytes_to_bytes',
      decode: (bytes: Uint8Array) => decodeLzw(bytes, size),
      encode() {
        throw new Error('Browser arrays are read-only');
      },
    };
  },
}));

function location(store: ByteStore, path: string) {
  if (path.split('/').some((p) => p === '.' || p === '..')) {
    throw new Error('Array paths must not contain . or ..');
  }
  return zarr
    .root({
      async get(key: string) {
        return (await store.get(key.replace(/^\//, ''))) ?? undefined;
      },
    })
    .resolve(path);
}

export async function openNode(store: ByteStore, path: string) {
  const node = await zarr.open(location(store, path));
  if (node.kind === 'group') return { kind: node.kind, attrs: node.attrs };
  if (!sizes[node.dtype])
    throw new Error(`Unsupported browser array dtype: ${node.dtype}`);
  return {
    kind: node.kind,
    attrs: node.attrs,
    shape: node.shape,
    chunks: node.chunks,
    dtype: node.dtype,
  };
}

export type Selection = (number | [number, number, number])[];

export async function readArray(
  store: ByteStore,
  path: string,
  selection: Selection,
) {
  const array = await zarr.open(location(store, path), { kind: 'array' });
  if (!sizes[array.dtype])
    throw new Error(`Unsupported browser array dtype: ${array.dtype}`);
  if (selection.length !== array.shape.length)
    throw new Error('Selection rank mismatch');
  // Keep integer-selected axes until after reading, so even point selections
  // travel as binary data (including exact int64 values), never JSON numbers.
  const slices = selection.map((s, axis) => {
    const size = array.shape[axis];
    if (typeof s === 'number') {
      if (!Number.isSafeInteger(s) || s < 0 || s >= size)
        throw new Error('Index out of bounds');
      return zarr.slice(s, s + 1);
    }
    const [start, stop, step] = s;
    if (
      ![start, stop, step].every(Number.isSafeInteger) ||
      start < 0 ||
      stop < 0 ||
      start > size ||
      stop > size ||
      step <= 0
    ) {
      throw new Error('Invalid slice');
    }
    return zarr.slice(start, stop, step);
  });
  const littleEndian = new Uint8Array(new Uint16Array([1]).buffer)[0] === 1;
  const shape: number[] = [];
  for (const s of selection) {
    if (typeof s !== 'number')
      shape.push(Math.max(0, Math.ceil((s[1] - s[0]) / s[2])));
  }
  if (shape.includes(0)) {
    // Zarrita currently rejects empty selections; return the NumPy-compatible
    // empty shape without requesting any chunks.
    return {
      result: {
        shape,
        strides: shape.map(() => 0),
        dtype: array.dtype,
        byteorder: littleEndian ? '<' : '>',
      },
      data: new Uint8Array(0),
    };
  }
  const result =
    array.shape.length === 0
      ? await array.getChunk([], undefined, { useSharedArrayBuffer: false })
      : await zarr.get(array, slices, { useSharedArrayBuffer: false });
  if (
    typeof result !== 'object' ||
    result === null ||
    !('data' in result) ||
    !ArrayBuffer.isView(result.data)
  ) {
    throw new Error('Expected a numeric array result');
  }
  const data = new Uint8Array(
    result.data.buffer,
    result.data.byteOffset,
    result.data.byteLength,
  ).slice();
  const keep = (_: number, axis: number) => typeof selection[axis] !== 'number';
  return {
    result: {
      shape: result.shape.filter(keep),
      strides: result.stride.filter(keep),
      dtype: array.dtype,
      byteorder: littleEndian ? '<' : '>',
    },
    data,
  };
}
