import type { FileSpec } from "src/models/adapter";
import { ForecastError } from "src/types";

export const CACHE_NAME = "forecast-web-v1";

export interface ModelFiles {
  config: string;
  model: string;
  /** fp16 weights to expand and pass to onnxruntime as external data named `location`. */
  halfData?: { url: string; location: string };
}

const isUrl = (s: string) => /^(https?:|blob:|data:)/.test(s) || /^\.{0,2}\//.test(s);

/**
 * Resolves the base URL for a weights location. Repo ids map to the Hugging
 * Face resolve endpoint; anything URL-shaped is used as-is, resolved against
 * `relativeTo` so it still works after being handed to a worker.
 */
export function resolveBase(weights: string, revision = "main", relativeTo?: string): string {
  let base: string;
  if (isUrl(weights)) {
    base = relativeTo ? new URL(weights, relativeTo).href : weights;
  } else {
    if (!/^[\w.-]+\/[\w.-]+$/.test(weights))
      throw new ForecastError("LOAD_FAILED", `"${weights}" is neither a Hugging Face repo id nor a URL`);
    base = `https://huggingface.co/${weights}/resolve/${encodeURIComponent(revision)}/`;
  }
  return base.endsWith("/") ? base : base + "/";
}

export function resolveFiles(base: string, spec: FileSpec): ModelFiles {
  return {
    config: base + "config.json",
    model: base + spec.graph,
    halfData: spec.half && { url: base + spec.half.file, location: spec.half.location },
  };
}

const aborted = () => new ForecastError("ABORTED", "load was aborted");

async function openCache(enabled: boolean): Promise<Cache | null> {
  if (!enabled || typeof caches === "undefined") return null;
  try {
    return await caches.open(CACHE_NAME);
  } catch {
    return null;
  }
}

/**
 * Downloads a file, reporting bytes as they arrive. Serves from and writes to
 * the Cache API when available; a full or unavailable cache never fails the load.
 * `cacheKey` stores the entry under a different URL (for content-hash keys).
 * `networkFirst` always tries the network and uses the cache only as an offline fallback.
 */
export async function fetchBytes(
  url: string,
  opts: {
    cache?: boolean;
    cacheKey?: string;
    networkFirst?: boolean;
    signal?: AbortSignal;
    onProgress?: (loaded: number, total: number, cached: boolean) => void;
  } = {},
): Promise<Uint8Array> {
  const { signal, onProgress } = opts;
  if (signal?.aborted) throw aborted();
  const cache = await openCache(opts.cache ?? true);
  const key = opts.cacheKey ?? url;

  const fromCache = async (): Promise<Uint8Array | null> => {
    if (!cache) return null;
    try {
      const hit = await cache.match(key);
      if (!hit) return null;
      const bytes = new Uint8Array(await hit.arrayBuffer());
      onProgress?.(bytes.length, bytes.length, true);
      return bytes;
    } catch {
      return null;
    }
  };

  if (!opts.networkFirst) {
    const hit = await fromCache();
    if (hit) return hit;
  }

  let res: Response;
  try {
    res = await fetch(url, { signal });
  } catch (e) {
    if (signal?.aborted) throw aborted();
    const hit = opts.networkFirst ? await fromCache() : null;
    if (hit) return hit;
    throw new ForecastError("LOAD_FAILED", `could not fetch ${url}`, { cause: e });
  }
  if (!res.ok) {
    const hit = opts.networkFirst ? await fromCache() : null;
    if (hit) return hit;
    throw new ForecastError("LOAD_FAILED", `${url} returned HTTP ${res.status}`);
  }

  let total = Number(res.headers.get("content-length")) || 0;
  const chunks: Uint8Array[] = [];
  let loaded = 0;
  try {
    if (res.body) {
      const reader = res.body.getReader();
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        loaded += value.length;
        // Compressed responses report the encoded length; never show > 100%.
        total = Math.max(total, loaded);
        onProgress?.(loaded, total, false);
      }
    } else {
      const b = new Uint8Array(await res.arrayBuffer());
      chunks.push(b);
      loaded = total = b.length;
      onProgress?.(loaded, total, false);
    }
  } catch (e) {
    if (signal?.aborted) throw aborted();
    throw new ForecastError("LOAD_FAILED", `download of ${url} was interrupted`, { cause: e });
  }

  const bytes = new Uint8Array(loaded);
  let o = 0;
  for (const c of chunks) {
    bytes.set(c, o);
    o += c.length;
  }
  if (cache) {
    try {
      await cache.put(key, new Response(bytes, { headers: { "content-length": String(loaded) } }));
    } catch {
      /* quota exceeded or private mode: keep going without caching */
    }
  }
  return bytes;
}
