// Copy this module into the Cloudflare Worker editor.
const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET, HEAD, OPTIONS',
  'Access-Control-Allow-Headers': 'Range, If-Match, If-Unmodified-Since',
  'Access-Control-Expose-Headers': 'Content-Range, ETag, Last-Modified',
};

function errorResponse(message, status) {
  return new Response(message, { status, headers: corsHeaders });
}

export default {
  async fetch(request) {
    if (request.method === 'OPTIONS') {
      return new Response(null, { status: 204, headers: corsHeaders });
    }
    if (request.method !== 'GET' && request.method !== 'HEAD') {
      return errorResponse('Only GET and HEAD reads are supported', 405);
    }
    const target = new URL(request.url).searchParams.get('url');
    let targetUrl;
    try {
      targetUrl = new URL(target);
      if (!['http:', 'https:'].includes(targetUrl.protocol)) throw new Error();
    } catch {
      return errorResponse("Provide an HTTP(S) target in the 'url' parameter", 400);
    }
    // Browser fetch metadata describes the request to this proxy, not the
    // Worker-to-origin request. Forwarding it can make the upstream ignore Range.
    const headers = new Headers({ 'Accept-Encoding': 'identity' });
    for (const name of ['Range', 'If-Match', 'If-Unmodified-Since']) {
      const value = request.headers.get(name);
      if (value !== null) headers.set(name, value);
    }
    try {
      const response = await fetch(targetUrl.href, {
        method: request.method, headers, redirect: 'follow',
      });
      // Never relay a full TIFF when the client requested a byte range.
      if (request.method === 'GET' && headers.has('Range') && response.status === 200) {
        await response.body?.cancel();
        return errorResponse('Upstream ignored the Range request and returned HTTP 200', 502);
      }
      const responseHeaders = new Headers(response.headers);
      for (const [name, value] of Object.entries(corsHeaders)) {
        responseHeaders.set(name, value);
      }
      // Use the upload time conservatively, only for this trusted upstream.
      const upstreamUrl = new URL(response.url || targetUrl.href);
      if (
        upstreamUrl.protocol === 'https:' &&
        upstreamUrl.hostname === 'data.hydrosheds.org' &&
        upstreamUrl.pathname.startsWith('/file/hydrosheds-v2/') &&
        !responseHeaders.has('Last-Modified')
      ) {
        const uploaded = response.headers.get('x-bz-upload-timestamp');
        if (uploaded && /^\d+$/.test(uploaded)) {
          const date = new Date(Number(uploaded));
          if (Number.isFinite(date.getTime())) {
            responseHeaders.set('Last-Modified', date.toUTCString());
          }
        }
      }
      return new Response(response.body, {
        status: response.status, statusText: response.statusText, headers: responseHeaders,
      });
    } catch (error) {
      return errorResponse(`Error fetching target: ${error.message}`, 502);
    }
  },
};
