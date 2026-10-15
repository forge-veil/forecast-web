/**
 * Tabular classification and regression with TabICLv2, no training step. Load a
 * model with {@link loadTabular}, `fit` it to a table, then predict.
 *
 * ```ts
 * import { loadTabular } from "forecast-web/tabular";
 * ```
 *
 * @module forecast-web/tabular
 */
import { resolveBase } from "src/core/assets";
import { resolvePin } from "src/pins";
import { connectGraph } from "src/core/client";
import { loadCore, type CoreOptions } from "src/core/loader";
import { createTabular, type Tabular, type TabularOptions } from "src/tabular/tabular";
import { getTabularSpec, tabularModelFor, type TabularTask } from "src/tabular/models";
import type { Device, Dtype, Progress } from "src/types";

export { LIMITS } from "src/tabular/tabular";
export type { FittedClassifier, FittedFor, FittedRegressor, FittedTabular, RegressionOptions, Tabular, TabularOptions } from "src/tabular/tabular";
export type { Cell } from "src/tabular/encode";
export { TABULAR_MODELS } from "src/tabular/models";
export type { TabularInfo, TabularModelId, TabularTask } from "src/tabular/models";
export { clearCache } from "src/cache";
export { ForecastError } from "src/types";
export type { Backend, Device, Dtype, ErrorCode, Progress, ResolvedDtype } from "src/types";

/** Options for `loadTabular`. */
export interface LoadTabularOptions<T extends TabularTask = TabularTask> extends TabularOptions {
  task: T;
  /** Hugging Face repo id or base URL. Defaults to the published TabICL weights for the task. */
  weights?: string;
  /**
   * Hugging Face revision. Ignored for base URLs. Defaults to the commit this release was verified
   * against, or "main" for a custom `weights` repo.
   */
  revision?: string;
  dtype?: Dtype;
  device?: Device;
  /** Run inference in a Web Worker. Defaults to true where Worker exists. */
  worker?: boolean;
  cache?: boolean;
  wasmPaths?: string;
  onProgress?: (p: Progress) => void;
  signal?: AbortSignal;
}

/**
 * Downloads TabICL for a task and returns a model you can fit to a table and
 * predict with. Fitting is instant (in-context learning); prediction runs the
 * network, by default in a Web Worker on WebGPU with a WASM fallback.
 *
 * @example
 * ```ts
 * const clf = await loadTabular({ task: "classification" });
 * const labels = await clf.fit(trainRows, trainLabels).predict(testRows);
 * ```
 */
export async function loadTabular<T extends TabularTask>(options: LoadTabularOptions<T>): Promise<Tabular<T>> {
  const id = tabularModelFor(options.task);
  const spec = getTabularSpec(id);
  const page = typeof location !== "undefined" ? location.href : undefined;
  const pin = resolvePin(spec.weights, options.weights, options.revision);
  const opts: CoreOptions = {
    model: id,
    base: resolveBase(options.weights ?? spec.weights, pin.revision, page),
    hashes: pin.hashes,
    dtype: options.dtype ?? "auto",
    device: options.device ?? "auto",
    cache: options.cache ?? true,
    wasmPaths: options.wasmPaths,
  };
  const useWorker = options.worker ?? typeof Worker !== "undefined";
  if (useWorker) {
    // Keep this expression literal: bundlers detect it and emit the worker file.
    const worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
    const graph = await connectGraph(worker, opts, options.onProgress, options.signal);
    return createTabular<T>(graph, spec.info, options);
  }
  const { createSession } = await import("src/core/session");
  const core = await loadCore(opts, createSession, options.onProgress, options.signal);
  return createTabular<T>({ run: core.run, dispose: () => core.release(), backend: core.backend, dtype: core.dtype }, spec.info, options);
}
