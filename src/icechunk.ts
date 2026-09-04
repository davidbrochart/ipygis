import type { DOMWidgetModel } from '@jupyter-widgets/base';
import { decodeLzw } from './lzw.js';

import type {
  StorageGetObjectRangeArgs,
  StorageGetObjectConditionalArgs,
  StorageListInfo,
  RangeQuery,
  ReadonlySessionOptions,
} from '@earthmover/icechunk';
import { createHttpVirtualChunkFetcher } from '@earthmover/icechunk/http-virtual-chunks';
import { ServerConnection } from '@jupyterlab/services';
import type { Contents } from '@jupyterlab/services';

interface StoreRequest {
  type: 'icechunk_request';
  id: string;
  operation:
    | 'open'
    | 'readonly_session'
    | 'get'
    | 'exists'
    | 'list'
    | 'list_prefix'
    | 'list_dir'
    | 'close'
    | 'decode_lzw';
  store_id: string;
  repository: string;
  proxy_url: string;
  branch?: string;
  snapshot_id?: string;
  virtual_chunk_prefixes: string[];
  key: string;
  prefix: string;
  range?: RangeQuery;
  decoded_size: number;
}

/** One handler per model, independent of how many views display the widget. */
export class IcechunkBridge {
  private repositories = new Map<string, BrowserIcechunkRepository>();
  private owners = new Map<string, string>();
  private stores = new Map<string, BrowserIcechunkStore>();
  private closed = false;
  private cancelled = new Set<string>();
  private opening = new Set<string>();

  constructor(
    private model: DOMWidgetModel,
    private contents: Contents.IManager | undefined,
  ) {
    model.on('msg:custom', this.handle, this);
  }

  dispose() {
    this.closed = true;
    this.model.off('msg:custom', this.handle, this);
    for (const store of this.stores.values()) {
      store.close();
    }
    this.stores.clear();
    this.repositories.clear();
    this.owners.clear();
  }

  private async handle(message: StoreRequest, buffers: DataView[] = []) {
    if (message.type !== 'icechunk_request') {
      return;
    }
    try {
      if (message.operation === 'open') {
        if (!this.contents) {
          throw new Error('The JupyterLab contents manager is not available');
        }
        this.opening.add(message.id);
        const repository = await openIcechunk(
          this.contents,
          message.repository,
          message.proxy_url,
          message.virtual_chunk_prefixes,
        );
        if (this.closed || this.cancelled.delete(message.id)) {
          throw new Error('Widget closed while opening the repository');
        }
        this.repositories.set(message.id, repository);
        this.model.send({
          type: 'icechunk_response',
          id: message.id,
          result: {
            repository_id: message.id,
          },
        });
        return;
      }
      if (message.operation === 'readonly_session') {
        const repository = this.repositories.get(message.store_id);
        if (!repository) {
          throw new Error('Repository is closed or missing');
        }
        this.opening.add(message.id);
        const store = await repository.readonlySession(
          message.snapshot_id === undefined
            ? { branch: message.branch ?? 'main' }
            : { snapshotId: message.snapshot_id },
        );
        if (
          this.closed ||
          this.cancelled.delete(message.id) ||
          !this.repositories.has(message.store_id)
        ) {
          store.close();
          throw new Error('Repository closed while opening the session');
        }
        this.stores.set(message.id, store);
        this.owners.set(message.id, message.store_id);
        this.model.send({
          type: 'icechunk_response',
          id: message.id,
          result: { store_id: message.id, snapshot_id: store.snapshotId },
        });
        return;
      }
      if (
        message.operation === 'close' &&
        this.repositories.has(message.store_id)
      ) {
        this.repositories.delete(message.store_id);
        for (const [id, owner] of this.owners) {
          if (owner === message.store_id) {
            this.stores.get(id)?.close();
            this.stores.delete(id);
            this.owners.delete(id);
          }
        }
        this.model.send({
          type: 'icechunk_response',
          id: message.id,
          result: null,
        });
        return;
      }
      if (message.operation === 'decode_lzw') {
        if (buffers.length !== 1) {
          throw new Error('LZW decoding requires one binary input buffer');
        }
        const input = buffers[0];
        const decoded = await decodeLzw(
          new Uint8Array(input.buffer, input.byteOffset, input.byteLength),
          message.decoded_size,
        );
        this.model.send(
          { type: 'icechunk_response', id: message.id, result: null },
          undefined,
          [decoded.slice().buffer],
        );
        return;
      }
      const store = this.stores.get(message.store_id);
      if (!store && message.operation === 'close') {
        if (this.opening.has(message.store_id)) {
          this.cancelled.add(message.store_id);
        }
        this.model.send({
          type: 'icechunk_response',
          id: message.id,
          result: null,
        });
        return;
      }
      if (!store) {
        throw new Error(
          'Browser Icechunk session is closed or missing; reopen the store',
        );
      }
      if (message.operation === 'get') {
        const data = await store.get(message.key, message.range);
        this.model.send(
          {
            type: 'icechunk_response',
            id: message.id,
            result: { missing: data === null },
          },
          undefined,
          data === null ? [] : [data.slice().buffer],
        );
        return;
      }
      let result: boolean | string[] | null;
      switch (message.operation) {
        case 'exists':
          result = await store.exists(message.key);
          break;
        case 'list':
          result = await store.list();
          break;
        case 'list_prefix':
          result = await store.listPrefix(message.prefix);
          break;
        case 'list_dir':
          result = await store.listDir(message.prefix);
          break;
        case 'close':
          store.close();
          this.stores.delete(message.store_id);
          this.owners.delete(message.store_id);
          result = null;
          break;
        default:
          throw new Error('Unknown Icechunk request operation');
      }
      this.model.send({ type: 'icechunk_response', id: message.id, result });
    } catch (error) {
      this.model.send({
        type: 'icechunk_response',
        id: message.id,
        error: error instanceof Error ? error.message : String(error),
      });
    } finally {
      if (
        message.operation === 'open' ||
        message.operation === 'readonly_session'
      ) {
        this.opening.delete(message.id);
        this.cancelled.delete(message.id);
      }
    }
  }
}

function repositoryPath(path: string): string {
  const parts = path.split('/').filter(Boolean);
  if (parts.some((part) => part === '.' || part === '..')) {
    throw new Error('Repository paths must not contain . or ..');
  }
  return parts.join('/');
}

/** Read repository objects over Jupyter HTTP; virtual TIFF bytes stay in the browser. */
export async function openIcechunk(
  contents: Contents.IManager,
  repository: string,
  proxyUrl: string,
  virtualChunkPrefixes: string[] = [],
) {
  const { Repository, Storage } = await import('@earthmover/icechunk');
  const repoPath = repositoryPath(repository);
  const request = async (
    path: string,
    headers: HeadersInit = {},
    method = 'GET',
  ) => {
    const url = await contents.getDownloadUrl(
      `${repoPath}/${repositoryPath(path)}`,
    );
    const response = await ServerConnection.makeRequest(
      url,
      { headers, method },
      contents.serverSettings,
    );
    if (response.status === 404) {
      throw new Error('ObjectNotFound');
    }
    if (!response.ok) {
      throw new Error(
        `Repository request failed: HTTP ${response.status} (${path})`,
      );
    }
    return response;
  };
  const list = async (prefix: string): Promise<StorageListInfo[]> => {
    const directory = prefix.slice(0, prefix.lastIndexOf('/') + 1);
    async function walk(relative: string): Promise<StorageListInfo[]> {
      let directory: Contents.IModel;
      try {
        directory = await contents.get(
          `${repoPath}/${repositoryPath(relative)}`,
          {
            type: 'directory',
            content: true,
          },
        );
      } catch (error) {
        if (
          error instanceof ServerConnection.ResponseError &&
          error.response.status === 404
        ) {
          return [];
        }
        throw error;
      }
      const entries: Contents.IModel[] = directory.content;
      const results = await Promise.all(
        entries.map(async (entry) => {
          const path = entry.path.slice(repoPath.length + 1);
          if (entry.type === 'directory') {
            return walk(path);
          }
          return path.startsWith(prefix)
            ? [
                {
                  id: path.slice(prefix.length),
                  sizeBytes: entry.size ?? 0,
                  createdAt: new Date(entry.last_modified),
                },
              ]
            : [];
        }),
      );
      return ([] as StorageListInfo[]).concat(...results);
    }
    return walk(directory);
  };
  const noWrite = async () => {
    throw new Error('This repository connection is read-only');
  };
  const backend = {
    canWrite: async () => false,
    getObjectRange: async (
      _err: null,
      { path, rangeStart, rangeEnd }: StorageGetObjectRangeArgs,
    ) => {
      const response = await request(
        path,
        rangeStart === undefined
          ? {}
          : {
              Range: `bytes=${rangeStart}-${rangeEnd === undefined ? '' : rangeEnd - 1}`,
            },
      );
      let data = new Uint8Array(await response.arrayBuffer());
      if (response.status === 200 && rangeStart !== undefined) {
        data = data.slice(rangeStart, rangeEnd);
      }
      return {
        data,
        version: { etag: response.headers.get('etag') ?? undefined },
      };
    },
    getObjectConditional: async (
      _err: null,
      { path }: StorageGetObjectConditionalArgs,
    ) => {
      const response = await request(path);
      return {
        kind: 'modified',
        data: new Uint8Array(await response.arrayBuffer()),
        newVersion: { etag: response.headers.get('etag') ?? undefined },
      };
    },
    listObjects: async (_err: null, prefix: string) => list(prefix),
    getObjectLastModified: async (_err: null, path: string) =>
      new Date((await request(path, {}, 'HEAD')).headers.get('last-modified')!),
    putObject: noWrite,
    copyObject: noWrite,
    deleteBatch: noWrite,
  };
  // Generated declarations omit the error-first argument and still call Uint8Array Buffer.
  const storage = Storage.newCustom(
    backend as unknown as Parameters<typeof Storage.newCustom>[0],
  );
  const authorization: Record<string, { type: string }> = {};
  for (const prefix of virtualChunkPrefixes) {
    authorization[prefix] = { type: 'HttpAccess' };
  }
  const repo = await Repository.open(storage, undefined, authorization);
  const fetchChunk = createHttpVirtualChunkFetcher((url, options) => {
    if (!proxyUrl) {
      return fetch(url, options);
    }
    const proxy = new URL(proxyUrl);
    proxy.searchParams.set('url', String(url));
    return fetch(proxy, options);
  });
  let transportError: unknown;
  await repo.setHttpVirtualChunkFetcher(async (err, request) => {
    try {
      const response = await fetchChunk(err, request);
      if (request.etag !== undefined && response.etag === undefined) {
        throw new Error(
          'The TIFF response must include an ETag header exposed via CORS to validate this virtual chunk',
        );
      }
      if (
        request.lastModified !== undefined &&
        response.lastModified === undefined
      ) {
        throw new Error(
          'The TIFF response must include a Last-Modified header exposed via CORS to validate this virtual chunk',
        );
      }
      return response;
    } catch (error) {
      transportError = error;
      throw error;
    }
  });
  // All sessions share the callback, so serialize reads across the repository.
  let pending: Promise<unknown> = Promise.resolve();
  return {
    readonlySession: async (options: ReadonlySessionOptions) => {
      const session = await repo.readonlySession(options);
      // Serialize reads so a callback error is attributed to the right request.
      const get = (key: string, range?: RangeQuery) => {
        const result = pending.then(async () => {
          transportError = undefined;
          try {
            return range
              ? await session.store.getRange(key, range)
              : await session.store.get(key);
          } catch (error) {
            throw transportError ?? error;
          }
        });
        pending = result.catch(() => undefined);
        return result;
      };
      return {
        get,
        list: () => session.store.list(),
        listPrefix: (prefix: string) => session.store.listPrefix(prefix),
        listDir: (prefix: string) => session.store.listDir(prefix),
        exists: (key: string) => session.store.exists(key),
        snapshotId: session.snapshotId,
        // The contents manager belongs to JupyterLab and must remain alive.
        close: () => {},
      };
    },
  };
}

export type BrowserIcechunkRepository = Awaited<
  ReturnType<typeof openIcechunk>
>;
export type BrowserIcechunkStore = Awaited<
  ReturnType<BrowserIcechunkRepository['readonlySession']>
>;
