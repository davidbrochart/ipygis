import type { Contents } from '@jupyterlab/services';

export function contentsPath(path: string): string {
  if (
    typeof path !== 'string' ||
    !path ||
    path.startsWith('/') ||
    path
      .split('/')
      .some(
        (part) => !part || part === '.' || part === '..' || part.includes('\\'),
      )
  ) {
    throw new Error(
      'Use a nonempty path relative to the Jupyter contents root',
    );
  }
  return path;
}

function notFound(error: unknown): boolean {
  return (
    (error as { response?: { status?: number } })?.response?.status === 404
  );
}

/** A new Zarr directory, written through the application's shared contents manager. */
export class ContentsStore {
  private closed = false;
  private readOnly = false;
  private directories = new Set<string>();

  constructor(
    private contents: Contents.IManager,
    private path: string,
  ) {
    contentsPath(path);
  }

  close() {
    this.closed = true;
  }
  private check() {
    if (this.closed) {
      throw new Error('Contents store is closed');
    }
  }

  async open() {
    this.check();
    this.readOnly = true;
    const model = await this.contents.get(this.path, { content: false });
    if (model.type !== 'directory') {
      throw new Error(`Not a directory: ${this.path}`);
    }
  }

  async listDir(prefix: string): Promise<string[]> {
    this.check();
    const model = await this.contents.get(
      prefix ? this.key(prefix) : this.path,
      {
        type: 'directory',
        content: true,
      },
    );
    if (model.type !== 'directory' || !Array.isArray(model.content)) {
      throw new Error('Expected a directory listing');
    }
    return model.content.map((entry: Contents.IModel) => entry.name);
  }

  async create() {
    this.check();
    if (this.readOnly) {
      throw new Error('Contents store is read-only');
    }
    try {
      await this.contents.get(this.path, { content: false });
    } catch (error) {
      if (!notFound(error)) {
        throw error;
      }
      await this.directory(this.path, true);
      return;
    }
    throw new Error(`Destination already exists: ${this.path}`);
  }

  private async directory(path: string, exclusive = false): Promise<void> {
    this.check();
    if (!path || this.directories.has(path)) {
      return;
    }
    const parent = path.slice(
      0,
      path.lastIndexOf('/') < 0 ? 0 : path.lastIndexOf('/'),
    );
    await this.directory(parent);
    try {
      const model = await this.contents.get(path, { content: false });
      if (exclusive) {
        throw new Error(`Destination already exists: ${path}`);
      }
      if (model.type !== 'directory') {
        throw new Error(`Not a directory: ${path}`);
      }
    } catch (error) {
      if (!notFound(error)) {
        throw error;
      }
      this.check();
      await this.contents.save(path, { type: 'directory' });
    }
    this.directories.add(path);
  }

  private key(key: string) {
    return `${this.path}/${contentsPath(key.replace(/^\//, ''))}`;
  }

  async get(key: string): Promise<Uint8Array | undefined> {
    this.check();
    try {
      const model = await this.contents.get(this.key(key), {
        type: 'file',
        format: 'base64',
        content: true,
      });
      if (model.format !== 'base64') {
        throw new Error('Expected base64 file contents');
      }
      return Uint8Array.from(atob(model.content), (char) => char.charCodeAt(0));
    } catch (error) {
      if (notFound(error)) {
        return undefined;
      }
      throw error;
    }
  }

  async set(key: string, bytes: Uint8Array): Promise<void> {
    this.check();
    if (this.readOnly) {
      throw new Error('Contents store is read-only');
    }
    const path = this.key(key);
    await this.directory(path.slice(0, path.lastIndexOf('/')));
    let binary = '';
    for (let i = 0; i < bytes.length; i += 8192) {
      binary += String.fromCharCode(...bytes.subarray(i, i + 8192));
    }
    this.check();
    await this.contents.save(path, {
      type: 'file',
      format: 'base64',
      content: btoa(binary),
    });
  }
}
