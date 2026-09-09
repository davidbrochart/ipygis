import type { DOMWidgetModel } from '@jupyter-widgets/base';
import { decodeLzw } from './lzw.js';

import { encodeObjectId12 } from 'icechunk-js';
import type {
  Storage,
  ReadSession,
  Manifest,
  NodeSnapshot,
  FetchClient,
  RangeQuery as IcechunkRangeQuery,
} from 'icechunk-js';
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
  const { Repository, NotFoundError } = await import('icechunk-js');
  const repoPath = repositoryPath(repository);
  const request = async (path: string, init: RequestInit = {}) => {
    const url = await contents.getDownloadUrl(
      [repoPath, repositoryPath(path)].filter(Boolean).join('/'),
    );
    const response = await ServerConnection.makeRequest(
      url,
      init,
      contents.serverSettings,
    );
    if (response.status === 404) {
      throw new NotFoundError(path);
    }
    if (!response.ok) {
      throw new Error(
        `Repository request failed: HTTP ${response.status} (${path})`,
      );
    }
    return response;
  };
  const storage: Storage = {
    async getObject(path, range, options) {
      if (range && range.start === range.end) {
        return new Uint8Array(0);
      }
      const response = await request(path, {
        signal: options?.signal,
        headers: range
          ? { Range: `bytes=${range.start}-${range.end - 1}` }
          : {},
      });
      const data = new Uint8Array(await response.arrayBuffer());
      return range && response.status === 200
        ? data.slice(range.start, range.end)
        : data;
    },
    async exists(path, options) {
      try {
        await request(path, { method: 'HEAD', signal: options?.signal });
        return true;
      } catch (error) {
        if (error instanceof NotFoundError) {
          return false;
        }
        throw error;
      }
    },
    async *listPrefix(prefix) {
      const directory = prefix.slice(0, prefix.lastIndexOf('/') + 1);
      async function* walk(relative: string): AsyncGenerator<string> {
        let model: Contents.IModel;
        try {
          model = await contents.get(
            [repoPath, repositoryPath(relative)].filter(Boolean).join('/'),
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
            return;
          }
          throw error;
        }
        for (const entry of model.content as Contents.IModel[]) {
          const path = repoPath
            ? entry.path.slice(repoPath.length + 1)
            : entry.path;
          if (entry.type === 'directory') {
            yield* walk(path);
          } else if (path.startsWith(prefix)) {
            yield path;
          }
        }
      }
      yield* walk(directory);
    },
  };
  const repo = await Repository.open({ storage });
  const fetchClient = virtualChunkClient(proxyUrl, virtualChunkPrefixes);
  return {
    readonlySession: async (options: ReadonlySessionOptions) => {
      const session =
        options.snapshotId !== undefined
          ? await repo.checkoutSnapshot(options.snapshotId)
          : await repo.checkoutBranch(options.branch ?? 'main');
      return sessionStore(session, fetchClient);
    },
  };
}

/** Keep permission, byte-range and checksum checks at the virtual HTTP boundary. */
export function virtualChunkClient(
  proxyUrl: string,
  prefixes: string[],
): FetchClient {
  return {
    async fetch(url, init) {
      if (
        !/^https?:/.test(url) ||
        !prefixes.some((prefix) => url.startsWith(prefix))
      ) {
        throw new Error(`Unauthorized virtual chunk URL: ${url}`);
      }
      const headers = new Headers(init?.headers);
      const etag = headers.get('If-Match');
      const modified = headers.get('If-Unmodified-Since');
      // Validate returned metadata ourselves, as the WASM adapter did. These
      // conditional headers trigger preflights and may be ignored by proxies.
      headers.delete('If-Match');
      headers.delete('If-Unmodified-Since');
      let target = url;
      if (proxyUrl) {
        const proxy = new URL(proxyUrl);
        proxy.searchParams.set('url', url);
        target = proxy.href;
      }
      const response = await fetch(target, {
        ...init,
        headers,
        credentials: 'omit',
        redirect: 'error',
      });
      try {
        if (response.status !== 206) {
          throw new Error(
            `Virtual chunk request requires HTTP 206, received ${response.status}`,
          );
        }
        const requested = /^bytes=(\d+)-(\d+)$/.exec(
          headers.get('Range') ?? '',
        );
        const returned = /^bytes (\d+)-(\d+)\/(\d+|\*)$/.exec(
          response.headers.get('Content-Range') ?? '',
        );
        if (
          !requested ||
          !returned ||
          requested[1] !== returned[1] ||
          requested[2] !== returned[2]
        ) {
          throw new Error(
            'Virtual chunk response has a missing or mismatched Content-Range header',
          );
        }
        const stripQuotes = (value: string) => value.replace(/^"|"$/g, '');
        if (etag !== null) {
          const actual = response.headers.get('ETag');
          if (actual === null || stripQuotes(actual) !== stripQuotes(etag)) {
            throw new Error(
              'Virtual chunk ETag is missing or does not match the reference',
            );
          }
        }
        if (modified !== null) {
          const actual = Date.parse(
            response.headers.get('Last-Modified') ?? '',
          );
          if (!Number.isFinite(actual) || actual > Date.parse(modified)) {
            throw new Error(
              'Virtual chunk Last-Modified is missing, invalid, or newer than the reference',
            );
          }
        }
        return response;
      } catch (error) {
        await response.body?.cancel();
        throw error;
      }
    },
  };
}

/** Adapt snapshot metadata and chunk references to the Python Zarr store contract. */
export function sessionStore(session: ReadSession, fetchClient: FetchClient) {
  const abort = new AbortController();
  const checkOpen = () => abort.signal.throwIfAborted();
  const nodes = session.listNodes();
  const metadataKeys = nodes.map(
    (node) => `${node.path === '/' ? '' : node.path.slice(1) + '/'}zarr.json`,
  );
  const readOptions = {
    fetchClient,
    validateChecksums: true,
    signal: abort.signal,
  };
  // icechunk-js 0.6.0 exposes hierarchy listing, but not stored chunk listing.
  // Use its cached manifest loader until a public chunk-reference iterator is
  // available. The dependency is pinned and sparse listing is covered by tests.
  const manifests = session as unknown as {
    loadManifest(
      id: Uint8Array,
      options: { signal: AbortSignal },
    ): Promise<Manifest>;
  };
  const chunkKeys = async (node: NodeSnapshot) => {
    if (node.nodeData.type !== 'array') {
      return [];
    }
    const result = new Set<string>();
    const metadata = session.getMetadata(node.path) as ArrayMetadata;
    for (const ref of node.nodeData.manifests) {
      const manifest = await manifests.loadManifest(ref.objectId, {
        signal: abort.signal,
      });
      const array = manifest.arrays.find((entry) =>
        entry.nodeId.every((byte, i) => byte === node.id[i]),
      );
      if (!array) {
        continue;
      }
      for (let i = 0; i < array.numRefs; i++) {
        const coords = array.refIndex(i);
        if (
          !coords.every(
            (coord, dim) =>
              coord >= ref.extents[dim].from && coord < ref.extents[dim].to,
          )
        ) {
          continue;
        }
        const separator = metadata.chunk_key_encoding.configuration.separator;
        const encoded =
          metadata.chunk_key_encoding.name === 'default'
            ? ['c', ...coords].join(separator)
            : coords.length
              ? coords.join(separator)
              : '0';
        result.add(
          `${node.path === '/' ? '' : node.path.slice(1) + '/'}${encoded}`,
        );
      }
    }
    return [...result];
  };
  const listPrefix = async (prefix: string) => {
    checkOpen();
    const keys = metadataKeys.filter((key) => key.startsWith(prefix));
    for (const node of nodes) {
      const path = node.path === '/' ? '' : `${node.path.slice(1)}/`;
      if (path.startsWith(prefix) || prefix.startsWith(path)) {
        keys.push(
          ...(await chunkKeys(node)).filter((key) => key.startsWith(prefix)),
        );
      }
    }
    return keys.sort();
  };
  const get = async (
    key: string,
    range?: RangeQuery,
  ): Promise<Uint8Array | null> => {
    checkOpen();
    if (metadataKeys.includes(key)) {
      const path =
        key === 'zarr.json' ? '/' : `/${key.slice(0, -'/zarr.json'.length)}`;
      const data = session.getRawMetadata(path);
      return data === null ? null : sliceRange(data, range);
    }
    // Match against array metadata so both default and v2 chunk-key encodings
    // work, including root arrays and scalar arrays.
    for (const node of nodes) {
      if (node.nodeData.type !== 'array') {
        continue;
      }
      const prefix = node.path === '/' ? '' : `${node.path.slice(1)}/`;
      if (!key.startsWith(prefix)) {
        continue;
      }
      const metadata = session.getMetadata(node.path) as ArrayMetadata;
      const separator = metadata.chunk_key_encoding.configuration.separator;
      let encoded = key.slice(prefix.length);
      if (metadata.chunk_key_encoding.name === 'default') {
        if (encoded === 'c') {
          encoded = '';
        } else if (encoded.startsWith(`c${separator}`)) {
          encoded = encoded.slice(2);
        } else {
          continue;
        }
      }
      const parts =
        encoded === '' || (metadata.shape.length === 0 && encoded === '0')
          ? []
          : encoded.split(separator);
      if (
        parts.length !== metadata.shape.length ||
        parts.some((part) => !/^(0|[1-9]\d*)$/.test(part))
      ) {
        continue;
      }
      const coords = parts.map(Number);
      if (!coords.every(Number.isSafeInteger)) {
        return null;
      }
      if (
        range &&
        (('suffixLength' in range && range.suffixLength === 0) ||
          ('length' in range && range.length === 0))
      ) {
        return (await exists(key)) ? new Uint8Array(0) : null;
      }
      if (range && ('suffixLength' in range || range.length !== undefined)) {
        return session.getChunkRange(
          node.path,
          coords,
          range as IcechunkRangeQuery,
          readOptions,
        );
      }
      const data = await session.getChunk(node.path, coords, readOptions);
      return data === null ? null : sliceRange(data, range);
    }
    return null;
  };
  const exists = async (key: string) => (await listPrefix(key)).includes(key);
  return {
    get,
    exists,
    list: () => listPrefix(''),
    listPrefix,
    async listDir(prefix: string) {
      checkOpen();
      prefix = prefix.replace(/\/$/, '');
      const node = session.getNode(prefix ? `/${prefix}` : '/');
      if (node?.nodeData.type === 'group') {
        return [
          'zarr.json',
          ...session
            .listChildren(node.path)
            .filter((child) => child.path !== node.path)
            .map((child) => child.path.split('/').pop()!),
        ].sort();
      }
      const start = prefix ? `${prefix}/` : '';
      return [
        ...new Set(
          (await listPrefix(start)).map(
            (key) => key.slice(start.length).split('/')[0],
          ),
        ),
      ].sort();
    },
    snapshotId: encodeObjectId12(session.getSnapshotId()),
    close: () => abort.abort(new Error('Browser Icechunk session is closed')),
  };
}

function sliceRange(data: Uint8Array, range?: RangeQuery) {
  if (!range) {
    return data;
  }
  if ('suffixLength' in range) {
    return range.suffixLength === 0
      ? data.slice(0, 0)
      : data.slice(-range.suffixLength);
  }
  return data.slice(
    range.offset,
    range.length === undefined ? undefined : range.offset + range.length,
  );
}

interface ArrayMetadata {
  shape: number[];
  chunk_key_encoding: { name: string; configuration: { separator: string } };
}
type RangeQuery =
  { offset: number; length?: number } | { suffixLength: number };
interface ReadonlySessionOptions {
  branch?: string;
  snapshotId?: string;
}

export type BrowserIcechunkRepository = Awaited<
  ReturnType<typeof openIcechunk>
>;
export type BrowserIcechunkStore = ReturnType<typeof sessionStore>;
