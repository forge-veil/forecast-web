import { fetchBytes, resolveFiles } from "src/core/assets";
import { hasWebGPU } from "src/core/gpu";
import { expandHalfToFloat } from "src/core/half";
import type { GraphSession, SessionOptions } from "src/core/session";
import type { RunGraph } from "src/models/adapter";
import { getAdapter } from "src/models/index";
import { getTabularSpec, isTabularId, type GraphSpec, type TabularModelId } from "src/tabular/models";
import {
  ForecastError,
  type Backend,
  type Device,
  type Dtype,
  type Forecast,
  type ForecastOptions,
  type ModelConfig,
  type ModelId,
  type ModelInfo,
  type Progress,
  type ResolvedDtype,
  type Series,
} from "src/types";

/** Options after the main thread has resolved defaults and the base URL. Structured-clone safe. */
export interface CoreOptions {
  model: ModelId | TabularModelId;
  base: string;
  dtype: Dtype;
  device: Device;
  cache: boolean;
  wasmPaths?: string;
  /** Size and SHA-256 per weight file, built into the package for pinned weights; replaces config.json's list. */
  hashes?: Record<string, { bytes: number; sha256: string }>;
}

export interface Core {
  backend: Backend;
  dtype: ResolvedDtype;
  config: ModelConfig;
  forecast(series: Series[], opts?: ForecastOptions): Promise<Forecast[]>;
  /** Runs the raw graph on named feeds (tabular models). */
  run: RunGraph;
  release(): Promise<void>;
}

/** The forecasting adapter or tabular graph spec for a model id; throws UNKNOWN_MODEL. */
function specFor(id: string): GraphSpec & { forecaster?: ReturnType<typeof getAdapter>["forecaster"] } {
  return isTabularId(id) ? getTabularSpec(id) : getAdapter(id);
}

export type SessionFactory = (bytes: Uint8Array, opts: SessionOptions) => Promise<GraphSession>;

/**
 * Chooses weights and backend before anything is downloaded. q8 cannot run on
 * onnxruntime-web's WebGPU backend, so it always lands on WASM.
 */
export function planLoad(
  info: Pick<ModelInfo, "name" | "dtypes" | "defaults">,
  dtype: Dtype,
  device: Device,
  gpuAvailable: boolean,
): { dtype: ResolvedDtype; device: Backend; fallback: boolean } {
  const gpu = device === "webgpu" || (device === "auto" && gpuAvailable);
  const resolved: ResolvedDtype = dtype === "auto" ? info.defaults[gpu ? "webgpu" : "wasm"] : dtype;
  if (!info.dtypes.includes(resolved))
    throw new ForecastError(
      "LOAD_FAILED",
      `${info.name} has no "${resolved}" weights; choose one of ${info.dtypes.join(", ")}`,
    );
  if (resolved === "q8" && gpu) {
    if (device === "webgpu")
      throw new ForecastError("LOAD_FAILED", 'dtype "q8" runs on WASM only; use "fp16" or "fp32" with WebGPU');
    return { dtype: resolved, device: "wasm", fallback: false };
  }
  return { dtype: resolved, device: gpu ? "webgpu" : "wasm", fallback: gpu && device === "auto" };
}

/** Downloads config and graph, creates a session, returns a ready forecaster. */
export async function loadCore(
  opts: CoreOptions,
  createSession: SessionFactory,
  onProgress?: (p: Progress) => void,
  signal?: AbortSignal,
  gpuCheck: () => Promise<boolean> = hasWebGPU,
): Promise<Core> {
  const adapter = specFor(opts.model);
  const needsGpuCheck = opts.device === "auto";
  const plan = planLoad(adapter.info, opts.dtype, opts.device, needsGpuCheck ? await gpuCheck() : false);
  const files = resolveFiles(opts.base, adapter.files(plan.dtype));

  // Always fetch config fresh (it is small and carries the weight hashes); the cache is the offline fallback.
  const cfgBytes = await fetchBytes(files.config, { cache: opts.cache, networkFirst: true, signal });
  let config: ModelConfig;
  try {
    config = JSON.parse(new TextDecoder().decode(cfgBytes));
  } catch (e) {
    throw new ForecastError("LOAD_FAILED", `${files.config} is not valid JSON`, { cause: e });
  }
  adapter.checkConfig(config);

  const urls = [files.model, ...(files.halfData ? [files.halfData.url] : [])];
  const meta = urls.map((u) => {
    const path = u.slice(opts.base.length);
    if (!opts.hashes) return config.files?.[path];
    // Pinned weights trust only the hashes built into the package, and fail closed without one.
    const pinned = opts.hashes[path];
    if (!pinned) throw new ForecastError("LOAD_FAILED", `no pinned SHA-256 for ${path}`);
    return pinned;
  });

  // Progress is the sum over every weight file. Totals come from config.json when listed;
  // otherwise nothing is reported until every file has announced its size, so it never jumps back.
  const state = urls.map((_, i) => ({ loaded: 0, total: meta[i]?.bytes ?? 0, cached: true, known: !!meta[i] }));
  const report = () => {
    if (!state.every((s) => s.known)) return;
    const loaded = state.reduce((a, s) => a + s.loaded, 0);
    const total = Math.max(state.reduce((a, s) => a + s.total, 0), loaded);
    onProgress?.({ loaded, total, cached: state.every((s) => s.cached) });
  };

  // One failed file cancels the others instead of letting them download in the background.
  const inner = new AbortController();
  const forward = () => inner.abort();
  signal?.addEventListener("abort", forward, { once: true });
  let graph: Uint8Array;
  let half: Uint8Array | undefined;
  try {
    [graph, half] = await Promise.all(
      urls.map((url, i) =>
        fetchBytes(url, {
          cache: opts.cache,
          // Keyed by content alone, so a new revision with the same bytes reuses the cached copy.
          cacheKey: meta[i] ? `https://forecast-web.invalid/sha256/${meta[i]!.sha256}` : undefined,
          sha256: meta[i]?.sha256,
          signal: inner.signal,
          onProgress: (loaded, total, cached) => {
            const s = state[i];
            s.loaded = loaded;
            s.total = Math.max(s.total, total);
            s.cached = cached;
            s.known = true;
            report();
          },
        }).catch((e) => {
          inner.abort();
          throw e;
        }),
      ),
    );
  } catch (e) {
    if (signal?.aborted) throw new ForecastError("ABORTED", "load was aborted");
    throw e;
  } finally {
    signal?.removeEventListener("abort", forward);
  }
  if (signal?.aborted) throw new ForecastError("ABORTED", "load was aborted");

  const externalData =
    half && files.halfData ? [{ path: files.halfData.location, data: expandHalfToFloat(half) }] : undefined;
  const session = await createSession(graph, {
    device: plan.device,
    fallback: plan.fallback,
    wasmPaths: opts.wasmPaths,
    externalData,
  });
  if (signal?.aborted) {
    await session.release();
    throw new ForecastError("ABORTED", "load was aborted");
  }
  return {
    backend: session.backend,
    dtype: plan.dtype,
    config,
    forecast: adapter.forecaster
      ? adapter.forecaster(session.run, config)
      : () => Promise.reject(new ForecastError("LOAD_FAILED", `${adapter.info.name} is not a forecasting model`)),
    run: session.run,
    release: () => session.release(),
  };
}
