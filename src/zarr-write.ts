import * as zarr from 'zarrita';
import type { ContentsStore } from './contents-store';

// Zarrita decodes vlen-utf8, but its built-in codec does not yet encode it.
const loadUtf8 = zarr.registry.get('vlen-utf8')!;
zarr.registry.set('vlen-utf8', async () => {
  const codec = await loadUtf8();
  return {
    fromConfig(config, meta) {
      const original = codec.fromConfig(config, meta);
      return {
        kind: 'array_to_bytes',
        decode: original.decode.bind(original),
        encode(chunk: { data: string[]; shape: number[]; stride: number[] }) {
          const values: Uint8Array[] = [];
          const encoder = new TextEncoder();
          const visit = (axis: number, offset: number) => {
            if (axis === chunk.shape.length) {
              values.push(encoder.encode(chunk.data[offset]));
              return;
            }
            for (let i = 0; i < chunk.shape[axis]; i++) {
              visit(axis + 1, offset + i * chunk.stride[axis]);
            }
          };
          visit(0, 0);
          const bytes = new Uint8Array(
            4 + values.reduce((n, value) => n + 4 + value.length, 0),
          );
          const view = new DataView(bytes.buffer);
          view.setUint32(0, values.length, true);
          let offset = 4;
          for (const value of values) {
            view.setUint32(offset, value.length, true);
            bytes.set(value, offset + 4);
            offset += 4 + value.length;
          }
          return bytes;
        },
      };
    },
  };
});

const constructors = {
  int8: Int8Array,
  uint8: Uint8Array,
  int16: Int16Array,
  uint16: Uint16Array,
  int32: Int32Array,
  uint32: Uint32Array,
  int64: BigInt64Array,
  uint64: BigUint64Array,
  float32: Float32Array,
  float64: Float64Array,
};
type NumericType = keyof typeof constructors;

export interface ArrayDefinition {
  shape: number[];
  chunks: number[];
  dtype: NumericType | 'string';
  dimensions: string[];
  attributes: Record<string, unknown>;
}

export async function createGroup(
  store: ContentsStore,
  attributes: Record<string, unknown>,
) {
  await zarr.create(store, { attributes });
}

export async function createArray(
  store: ContentsStore,
  path: string,
  definition: ArrayDefinition,
) {
  if (definition.dtype !== 'string' && !(definition.dtype in constructors)) {
    throw new Error('Unsupported output dtype');
  }
  await zarr.create(zarr.root(store).resolve(path), {
    shape: definition.shape,
    chunkShape: definition.chunks,
    dtype: definition.dtype,
    dimensionNames: definition.dimensions,
    attributes: definition.attributes,
    fillValue: definition.dtype === 'string' ? '' : 0,
    codecs:
      definition.dtype === 'string'
        ? [{ name: 'vlen-utf8', configuration: {} }]
        : [{ name: 'bytes', configuration: { endian: 'little' } }],
  });
}

export async function writeArray(
  store: ContentsStore,
  path: string,
  selection: number[][],
  shape: number[],
  values: string[] | undefined,
  buffers: (ArrayBuffer | ArrayBufferView)[],
) {
  const array = await zarr.open(zarr.root(store).resolve(path), {
    kind: 'array',
  });
  const count = shape.reduce((a, b) => a * b, 1);
  let data;
  if (array.dtype === 'string') {
    if (
      !values ||
      values.length !== count ||
      !values.every((value) => typeof value === 'string')
    ) {
      throw new Error('Invalid string write buffer');
    }
    data = values;
  } else {
    const Ctor = constructors[array.dtype as NumericType];
    if (!Ctor || buffers.length !== 1) {
      throw new Error('Invalid numeric write buffer');
    }
    const buffer = buffers[0];
    const bytes = (
      ArrayBuffer.isView(buffer)
        ? new Uint8Array(buffer.buffer, buffer.byteOffset, buffer.byteLength)
        : new Uint8Array(buffer)
    ).slice();
    if (bytes.length !== count * Ctor.BYTES_PER_ELEMENT) {
      throw new Error('Incorrect write buffer size');
    }
    // The Python transport always sends little-endian values.
    if (new Uint8Array(new Uint16Array([1]).buffer)[0] !== 1) {
      for (let i = 0; i < bytes.length; i += Ctor.BYTES_PER_ELEMENT) {
        bytes.subarray(i, i + Ctor.BYTES_PER_ELEMENT).reverse();
      }
    }
    data = new Ctor(bytes.buffer);
  }
  const stride = shape.map((_, i) =>
    shape.slice(i + 1).reduce((a, b) => a * b, 1),
  );
  await zarr.set(
    array,
    selection.map(([start, stop]) => zarr.slice(start, stop)),
    shape.length ? { data, shape, stride } : data[0],
    { useSharedArrayBuffer: false },
  );
}
