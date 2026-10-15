import { CACHE_NAME } from "src/core/assets";
import { getAdapter } from "src/models/index";
import { pinOf } from "src/pins";
import { getTabularSpec, TABULAR_MODELS, type TabularModelId } from "src/tabular/models";
import type { ModelId } from "src/types";

const repoOf = (id: string): string =>
  Object.hasOwn(TABULAR_MODELS, id) ? getTabularSpec(id as TabularModelId).weights : getAdapter(id).weights;

/**
 * Deletes model files forecast-web stored with the Cache API. With a model id, deletes that
 * model's published weights (every format) and anything cached from its Hugging Face repo;
 * without one, deletes everything, including self-hosted weights. Resolves true if anything
 * was deleted, false when nothing was stored or the Cache API is unavailable.
 *
 * @example
 * ```ts
 * await clearCache("timesfm-2.5"); // one model
 * await clearCache(); // everything
 * ```
 */
export async function clearCache(model?: ModelId | TabularModelId): Promise<boolean> {
  const repo = model === undefined ? undefined : repoOf(model);
  if (typeof caches === "undefined") return false;
  try {
    if (!repo) return await caches.delete(CACHE_NAME);
    const cache = await caches.open(CACHE_NAME);
    const pinned = new Set(Object.values(pinOf(repo).files).map((f) => `https://forecast-web.invalid/sha256/${f.sha256}`));
    const prefix = `https://huggingface.co/${repo}/`;
    let deleted = false;
    for (const req of await cache.keys()) if (pinned.has(req.url) || req.url.startsWith(prefix)) deleted = (await cache.delete(req)) || deleted;
    return deleted;
  } catch {
    return false;
  }
}
