/**
 * Time series forecasting and anomaly detection in the browser. Load a model with
 * {@link loadForecaster}, then call `forecast` or `detectAnomalies` on it.
 *
 * ```ts
 * import { loadForecaster } from "forecast-web";
 * ```
 *
 * @module forecast-web
 */
import { withAnalysis } from "src/anomaly/index";
import { resolveBase } from "src/core/assets";
import { resolvePin } from "src/pins";
import { connectWorker } from "src/core/client";
import { loadCore, type CoreOptions } from "src/core/loader";
import { DEFAULT_MODEL_ID, getAdapter } from "src/models/index";
import { ForecastError, type Forecaster, type LoadOptions } from "src/types";

export { clearCache } from "src/cache";
export { DEFAULT_MODEL_ID, MODELS } from "src/models/index";
export { ForecastError } from "src/types";
export type {
  AnomalyOptions,
  AnomalyPoint,
  AnomalyResult,
  AnomalyThresholds,
  Backend,
  Calibration,
  Device,
  Dtype,
  ErrorCode,
  Forecast,
  Forecaster,
  ForecastOptions,
  LoadOptions,
  ModelId,
  ModelInfo,
  Progress,
  ResolvedDtype,
  Sensitivity,
  SensitivityLevel,
  Series,
} from "src/types";
export type { TabularModelId } from "src/tabular/models";

/**
 * Downloads a forecasting model and prepares it. By default the model runs in
 * a Web Worker on WebGPU, falling back to WASM.
 *
 * @example
 * ```ts
 * const model = await loadForecaster({ model: "chronos-bolt-small" });
 * const [forecast] = await model.forecast([dailySignups], { horizon: 30 });
 * const { anomalies } = await model.detectAnomalies(dailySignups);
 * ```
 */
export async function loadForecaster(options: LoadOptions = {}): Promise<Forecaster> {
  const adapter = getAdapter(options.model ?? DEFAULT_MODEL_ID);
  const page = typeof location !== "undefined" ? location.href : undefined;
  // Published weights default to the commit this release was verified against; custom repos to main.
  const pin = resolvePin(adapter.weights, options.weights, options.revision);
  const opts: CoreOptions = {
    model: adapter.info.id,
    base: resolveBase(options.weights ?? adapter.weights, pin.revision, page),
    hashes: pin.hashes,
    dtype: options.dtype ?? "auto",
    device: options.device ?? "auto",
    cache: options.cache ?? true,
    wasmPaths: options.wasmPaths,
  };
  const useWorker = options.worker ?? typeof Worker !== "undefined";

  if (useWorker) {
    // Keep this expression literal: bundlers (Vite, webpack 5) detect it and emit the worker file.
    const worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
    return withAnalysis(await connectWorker(worker, opts, options.onProgress, options.signal));
  }

  const { createSession } = await import("src/core/session");
  const core = await loadCore(opts, createSession, options.onProgress, options.signal);
  let disposed = false;
  return withAnalysis({
    model: adapter.info.id,
    info: adapter.info,
    backend: core.backend,
    dtype: core.dtype,
    forecast(series, fopts) {
      if (disposed) return Promise.reject(new ForecastError("DISPOSED", "this model has been disposed"));
      return core.forecast(series, fopts);
    },
    async dispose() {
      if (disposed) return;
      disposed = true;
      await core.release();
    },
  });
}
