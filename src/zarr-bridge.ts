import type { Contents } from '@jupyterlab/services';
import { ContentsStore } from './contents-store.js';
import type { ArrayDefinition } from './zarr-write';
import type { DOMWidgetModel } from '@jupyter-widgets/base';
import type { ByteStore, Selection } from './zarr.js';

interface ArrayRequest {
  type: string;
  id: string;
  operation: string;
  store_id: string;
  path: string;
  selection: Selection;
  attributes: Record<string, unknown>;
  definition: ArrayDefinition;
  shape: number[];
  values?: string[];
}

/** Array RPC over any registered browser byte store. */
export class ZarrBridge {
  private closed = false;
  private contentsStores = new Map<string, ContentsStore>();

  constructor(
    private model: DOMWidgetModel,
    private getStore: (id: string) => ByteStore,
    private contents?: Contents.IManager,
  ) {
    model.on('msg:custom', this.handle, this);
  }

  dispose() {
    this.closed = true;
    for (const store of this.contentsStores.values()) {
      store.close();
    }
    this.contentsStores.clear();
    this.model.off('msg:custom', this.handle, this);
  }

  private async handle(
    message: ArrayRequest,
    buffers: (ArrayBuffer | ArrayBufferView)[] = [],
  ) {
    if (message.type !== 'zarr_request' || this.closed) {
      return;
    }
    try {
      if (
        message.operation === 'contents_create' ||
        message.operation === 'contents_open'
      ) {
        if (!this.contents) {
          throw new Error('Jupyter Contents service is unavailable');
        }
        const store = new ContentsStore(this.contents, message.path);
        this.contentsStores.set(message.id, store);
        try {
          if (message.operation === 'contents_open') {
            await store.open();
          } else {
            await store.create();
            const { createGroup } = await import('./zarr-write.js');
            await createGroup(store, message.attributes);
          }
          if (!this.closed) {
            this.model.send({
              type: 'zarr_response',
              id: message.id,
              result: { store_id: message.id },
            });
          }
        } catch (error) {
          store.close();
          this.contentsStores.delete(message.id);
          throw error;
        }
        return;
      }
      if (
        message.operation === 'zarr_create_array' ||
        message.operation === 'zarr_set'
      ) {
        const store = this.contentsStores.get(message.store_id);
        if (!store) {
          throw new Error('Writable contents store is closed or unknown');
        }
        const { createArray, writeArray } = await import('./zarr-write.js');
        if (message.operation === 'zarr_create_array') {
          await createArray(store, message.path, message.definition);
        } else {
          await writeArray(
            store,
            message.path,
            message.selection as number[][],
            message.shape,
            message.values,
            buffers,
          );
        }
        if (!this.closed) {
          this.model.send({
            type: 'zarr_response',
            id: message.id,
            result: null,
          });
        }
        return;
      }
      const contentsStore = this.contentsStores.get(message.store_id);
      const store: ByteStore = contentsStore
        ? {
            get: async (key: string) => (await contentsStore.get(key)) ?? null,
            listDir: (prefix: string) => contentsStore.listDir(prefix),
          }
        : this.getStore(message.store_id);
      const { openNode, readArray, listMembers } = await import('./zarr.js');
      if (message.operation === 'zarr_open') {
        const result = await openNode(store, message.path);
        if (!this.closed) {
          this.model.send({
            type: 'zarr_response',
            id: message.id,
            result: JSON.parse(JSON.stringify(result)),
          });
        }
      } else if (message.operation === 'zarr_members') {
        const result = await listMembers(store, message.path);
        if (!this.closed) {
          this.model.send({ type: 'zarr_response', id: message.id, result });
        }
      } else if (message.operation === 'zarr_get') {
        const { result, data } = await readArray(
          store,
          message.path,
          message.selection,
        );
        if (!this.closed) {
          this.model.send(
            { type: 'zarr_response', id: message.id, result },
            undefined,
            [data.buffer],
          );
        }
      } else {
        throw new Error('Unknown Zarr request operation');
      }
    } catch (error) {
      if (!this.closed) {
        this.model.send({
          type: 'zarr_response',
          id: message.id,
          error: error instanceof Error ? error.message : String(error),
        });
      }
    }
  }
}
