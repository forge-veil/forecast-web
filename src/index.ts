import { resolveBase } from "src/core/assets";
import { connectWorker } from "src/core/client";
import { loadCore, type CoreOptions } from "src/core/loader";
import { DEFAULT_MODEL_ID, getAdapter } from "src/models/index";
import { ForecastError, type Forecaster, type LoadOptions } from "src/types";

export { DEFAULT_MODEL_ID, MODELS } from "src/models/index";
export { cleanSeries } from "src/models/timesfm/preprocess";
export * from "src/types";

/**
 * Downloads a forecasting model and prepares it. By default the model runs in
 * a Web Worker on WebGPU, falling back to WASM.
 */
export async function loadForecaster(options: LoadOptions = {}): Promise<Forecaster> {
  const adapter = getAdapter(options.model ?? DEFAULT_MODEL_ID);
  const page = typeof location !== "undefined" ? location.href : undefined;
  const opts: CoreOptions = {
    model: adapter.info.id,
    base: resolveBase(options.weights ?? adapter.weights, options.revision, page),
    dtype: options.dtype ?? "auto",
    device: options.device ?? "auto",
    cache: options.cache ?? true,
    wasmPaths: options.wasmPaths,
  };
  const useWorker = options.worker ?? typeof Worker !== "undefined";

  if (useWorker) {
    // Keep this expression literal: bundlers (Vite, webpack 5) detect it and emit the worker file.
    const worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
    return connectWorker(worker, opts, options.onProgress, options.signal);
  }

  const { createSession } = await import("src/core/session");
  const core = await loadCore(opts, createSession, options.onProgress, options.signal);
  let disposed = false;
  return {
    model: adapter.info.id,
    info: adapter.info,
    backend: core.backend,
    dtype: core.dtype,
    config: core.config,
    forecast(series, fopts) {
      if (disposed) return Promise.reject(new ForecastError("DISPOSED", "this model has been disposed"));
      return core.forecast(series, fopts);
    },
    async dispose() {
      if (disposed) return;
      disposed = true;
      await core.release();
    },
  };
}
