import { TimesFMError, type ResolvedDtype } from "src/types";

/**
 * Default weights repo. Replace once the converted weights are published;
 * until then pass `model` explicitly (a repo id or a self-hosted base URL).
 */
export const DEFAULT_MODEL = "bakshinghampalace/timesfm-2.5-200m-onnx";
export const CACHE_NAME = "timesfm-web-v1";

/**
 * fp32 and q8 use Transformers.js file names. fp16 is a custom compact pair:
 * an fp32 graph whose weights are external data, shipped as fp16 and expanded
 * by the loader into the buffer the graph names `location`.
 */
const FILES: Record<ResolvedDtype, { graph: string; data?: { file: string; location: string } }> = {
  fp32: { graph: "onnx/model.onnx" },
  fp16: { graph: "onnx/model_fp16w.onnx", data: { file: "onnx/model_fp16w.f16", location: "model_fp16w.onnx_data" } },
  q8: { graph: "onnx/model_quantized.onnx" },
};

export interface ModelFiles {
  config: string;
  model: string;
  /** fp16 weights to expand and pass to onnxruntime as external data named `location`. */
  halfData?: { url: string; location: string };
}

const isUrl = (s: string) => /^(https?:|blob:|data:)/.test(s) || /^\.{0,2}\//.test(s);

/**
 * Resolves the base URL for a model. Repo ids map to the Hugging Face resolve
 * endpoint; anything URL-shaped is used as-is, resolved against `relativeTo`
 * so it still works after being handed to a worker.
 */
export function resolveBase(model = DEFAULT_MODEL, revision = "main", relativeTo?: string): string {
  let base: string;
  if (isUrl(model)) {
    base = relativeTo ? new URL(model, relativeTo).href : model;
  } else {
    if (!/^[\w.-]+\/[\w.-]+$/.test(model))
      throw new TimesFMError("LOAD_FAILED", `"${model}" is neither a Hugging Face repo id nor a URL`);
    base = `https://huggingface.co/${model}/resolve/${encodeURIComponent(revision)}/`;
  }
  return base.endsWith("/") ? base : base + "/";
}

export function resolveFiles(base: string, dtype: ResolvedDtype = "q8"): ModelFiles {
  const f = FILES[dtype];
  if (!f) throw new TimesFMError("LOAD_FAILED", `unknown dtype "${dtype}"`);
  return {
    config: base + "config.json",
    model: base + f.graph,
    halfData: f.data && { url: base + f.data.file, location: f.data.location },
  };
}

const aborted = () => new TimesFMError("ABORTED", "load was aborted");

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
 */
export async function fetchBytes(
  url: string,
  opts: {
    cache?: boolean;
    signal?: AbortSignal;
    onProgress?: (loaded: number, total: number, cached: boolean) => void;
  } = {},
): Promise<Uint8Array> {
  const { signal, onProgress } = opts;
  if (signal?.aborted) throw aborted();
  const cache = await openCache(opts.cache ?? true);

  if (cache) {
    try {
      const hit = await cache.match(url);
      if (hit) {
        const bytes = new Uint8Array(await hit.arrayBuffer());
        onProgress?.(bytes.length, bytes.length, true);
        return bytes;
      }
    } catch {
      /* fall through to the network */
    }
  }

  let res: Response;
  try {
    res = await fetch(url, { signal });
  } catch (e) {
    if (signal?.aborted) throw aborted();
    throw new TimesFMError("LOAD_FAILED", `could not fetch ${url}`, { cause: e });
  }
  if (!res.ok) throw new TimesFMError("LOAD_FAILED", `${url} returned HTTP ${res.status}`);

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
    throw new TimesFMError("LOAD_FAILED", `download of ${url} was interrupted`, { cause: e });
  }

  const bytes = new Uint8Array(loaded);
  let o = 0;
  for (const c of chunks) {
    bytes.set(c, o);
    o += c.length;
  }
  if (cache) {
    try {
      await cache.put(url, new Response(bytes, { headers: { "content-length": String(loaded) } }));
    } catch {
      /* quota exceeded or private mode: keep going without caching */
    }
  }
  return bytes;
}
