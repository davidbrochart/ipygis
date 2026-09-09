import type {
  StorageGetObjectRangeArgs,
  StorageGetObjectConditionalArgs,
  StorageListInfo,
  RangeQuery,
  ReadonlySessionOptions,
} from '@earthmover/icechunk';
import { ServerConnection } from '@jupyterlab/services';
import type { Contents } from '@jupyterlab/services';
function repositoryPath(path: string): string {
  const parts = path.split('/').filter(Boolean);
  if (parts.some((part) => part === '.' || part === '..')) {
    throw new Error('Repository paths must not contain . or ..');
  }
  return parts.join('/');
}

/** Read repository objects over Jupyter HTTP; virtual TIFF bytes stay in the browser. */
export async function openWasmIcechunk(
  contents: Contents.IManager,
  repository: string,
  proxyUrl: string,
  virtualChunkPrefixes: string[] = [],
) {
  if (proxyUrl || virtualChunkPrefixes.length) {
    throw new Error(
      'The published @earthmover/icechunk package does not support browser HTTP virtual chunks; use backend="icechunk-js" for virtual TIFF data',
    );
  }
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
  const repo = await Repository.open(storage);
  return {
    readonlySession: async (options: ReadonlySessionOptions) => {
      const session = await repo.readonlySession(options);
      const get = (key: string, range?: RangeQuery) =>
        range ? session.store.getRange(key, range) : session.store.get(key);
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
