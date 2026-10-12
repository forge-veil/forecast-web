import { resolveBase } from "src/assets";
import { connectWorker } from "src/client";
import { loadCore, type CoreOptions } from "src/loader";
import { TimesFMError, type LoadOptions, type TimesFM } from "src/types";

export { DEFAULT_MODEL } from "src/assets";
export { cleanSeries } from "src/preprocess";
export * from "src/types";

/**
 * Downloads TimesFM 2.5 and prepares it for forecasting. By default the model
 * runs in a Web Worker on WebGPU, falling back to WASM.
 */
export async function loadTimesFM(options: LoadOptions = {}): Promise<TimesFM> {
  const page = typeof location !== "undefined" ? location.href : undefined;
  const opts: CoreOptions = {
    base: resolveBase(options.model, options.revision, page),
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

  const { createSession } = await import("src/session");
  const core = await loadCore(opts, createSession, options.onProgress, options.signal);
  let disposed = false;
  return {
    backend: core.backend,
    dtype: core.dtype,
    config: core.config,
    forecast(series, fopts) {
      if (disposed) return Promise.reject(new TimesFMError("DISPOSED", "this model has been disposed"));
      return core.forecast(series, fopts);
    },
    async dispose() {
      if (disposed) return;
      disposed = true;
      await core.release();
    },
  };
}
