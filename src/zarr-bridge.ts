import type { DOMWidgetModel } from '@jupyter-widgets/base';
import type { ByteStore, Selection } from './zarr.js';

interface ArrayRequest {
  type: string;
  id: string;
  operation: string;
  store_id: string;
  path: string;
  selection: Selection;
}

/** Array RPC over any registered browser byte store. */
export class ZarrBridge {
  private closed = false;

  constructor(
    private model: DOMWidgetModel,
    private getStore: (id: string) => ByteStore,
  ) {
    model.on('msg:custom', this.handle, this);
  }

  dispose() {
    this.closed = true;
    this.model.off('msg:custom', this.handle, this);
  }

  private async handle(message: ArrayRequest) {
    if (message.type !== 'zarr_request' || this.closed) return;
    try {
      const store = this.getStore(message.store_id);
      const { openNode, readArray } = await import('./zarr.js');
      if (message.operation === 'zarr_open') {
        const result = await openNode(store, message.path);
        if (!this.closed)
          this.model.send({
            type: 'zarr_response',
            id: message.id,
            result: JSON.parse(JSON.stringify(result)),
          });
      } else if (message.operation === 'zarr_get') {
        const { result, data } = await readArray(
          store,
          message.path,
          message.selection,
        );
        if (!this.closed)
          this.model.send(
            { type: 'zarr_response', id: message.id, result },
            undefined,
            [data.buffer],
          );
      } else {
        throw new Error('Unknown Zarr request operation');
      }
    } catch (error) {
      if (!this.closed)
        this.model.send({
          type: 'zarr_response',
          id: message.id,
          error: error instanceof Error ? error.message : String(error),
        });
    }
  }
}
