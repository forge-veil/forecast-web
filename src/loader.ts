import { fetchBytes, resolveFiles } from "src/assets";
import { createForecaster } from "src/forecaster";
import { hasWebGPU } from "src/gpu";
import { expandHalfToFloat } from "src/half";
import type { GraphSession, SessionOptions } from "src/session";
import {
  TimesFMError,
  type Backend,
  type Device,
  type Dtype,
  type Forecast,
  type ForecastOptions,
  type ModelConfig,
  type Progress,
  type ResolvedDtype,
  type Series,
} from "src/types";

/** Options after the main thread has resolved defaults and the base URL. Structured-clone safe. */
export interface CoreOptions {
  base: string;
  dtype: Dtype;
  device: Device;
  cache: boolean;
  wasmPaths?: string;
}

export interface Core {
  backend: Backend;
  dtype: ResolvedDtype;
  config: ModelConfig;
  forecast(series: Series[], opts?: ForecastOptions): Promise<Forecast[]>;
  release(): Promise<void>;
}

export type SessionFactory = (bytes: Uint8Array, opts: SessionOptions) => Promise<GraphSession>;

/**
 * Chooses weights and backend before anything is downloaded. q8 cannot run on
 * onnxruntime-web's WebGPU backend, so it always lands on WASM.
 */
export function planLoad(
  dtype: Dtype,
  device: Device,
  gpuAvailable: boolean,
): { dtype: ResolvedDtype; device: Backend; fallback: boolean } {
  const gpu = device === "webgpu" || (device === "auto" && gpuAvailable);
  const resolved: ResolvedDtype = dtype === "auto" ? (gpu ? "fp16" : "q8") : dtype;
  if (resolved === "q8" && gpu) {
    if (device === "webgpu")
      throw new TimesFMError("LOAD_FAILED", 'dtype "q8" runs on WASM only; use "fp16" or "fp32" with WebGPU');
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
  const needsGpuCheck = opts.device === "auto";
  const plan = planLoad(opts.dtype, opts.device, needsGpuCheck ? await gpuCheck() : false);
  const files = resolveFiles(opts.base, plan.dtype);

  const cfgBytes = await fetchBytes(files.config, { cache: opts.cache, signal });
  let config: ModelConfig;
  try {
    config = JSON.parse(new TextDecoder().decode(cfgBytes));
  } catch (e) {
    throw new TimesFMError("LOAD_FAILED", `${files.config} is not valid JSON`, { cause: e });
  }
  if (config.context !== 512 || config.horizon !== 128)
    throw new TimesFMError(
      "LOAD_FAILED",
      `unsupported model shape: context ${config.context}, horizon ${config.horizon} (expected 512 and 128)`,
    );

  // Progress is the sum over every weight file being downloaded.
  const urls = [files.model, ...(files.halfData ? [files.halfData.url] : [])];
  const state = urls.map(() => ({ loaded: 0, total: 0, cached: true }));
  const report = () =>
    onProgress?.({
      loaded: state.reduce((a, s) => a + s.loaded, 0),
      total: state.reduce((a, s) => a + s.total, 0),
      cached: state.every((s) => s.cached),
    });
  const [graph, half] = await Promise.all(
    urls.map((url, i) =>
      fetchBytes(url, {
        cache: opts.cache,
        signal,
        onProgress: (loaded, total, cached) => {
          state[i] = { loaded, total, cached };
          report();
        },
      }),
    ),
  );
  if (signal?.aborted) throw new TimesFMError("ABORTED", "load was aborted");

  const externalData =
    half && files.halfData ? [{ path: files.halfData.location, data: expandHalfToFloat(half) }] : undefined;
  const session = await createSession(graph, {
    device: plan.device,
    fallback: plan.fallback,
    wasmPaths: opts.wasmPaths,
    externalData,
  });
  return {
    backend: session.backend,
    dtype: plan.dtype,
    config,
    forecast: createForecaster(session.run, config.context),
    release: () => session.release(),
  };
}
