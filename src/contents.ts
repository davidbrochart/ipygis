import { ServerConnection } from '@jupyterlab/services';
import type { Contents } from '@jupyterlab/services';

/** Read repository objects through the configured Jupyter contents drive. */
export async function requestContents(
  contents: Contents.IManager,
  path: string,
  init: RequestInit = {},
): Promise<Response> {
  init.signal?.throwIfAborted();
  let model: Contents.IModel;
  try {
    model = await contents.get(
      path,
      init.method === 'HEAD'
        ? { content: false }
        : { type: 'file', format: 'base64', content: true },
    );
  } catch (error) {
    if (error instanceof ServerConnection.ResponseError) {
      return new Response(null, { status: error.response.status });
    }
    throw error;
  }
  // The contents API cannot cancel an in-flight read, but callers can still
  // discard its result when their session closes.
  init.signal?.throwIfAborted();
  const headers = {
    'Last-Modified': new Date(model.last_modified).toUTCString(),
  };
  if (init.method === 'HEAD') {
    return new Response(null, { headers });
  }
  if (model.format !== 'base64' || typeof model.content !== 'string') {
    throw new Error(`Expected base64 file content for ${path}`);
  }
  const bytes = Uint8Array.from(atob(model.content), (char) =>
    char.charCodeAt(0),
  );
  // Contents.get returns the whole file. The adapters slice requested ranges
  // locally when they receive this full (HTTP 200) response.
  return new Response(bytes, { headers });
}
