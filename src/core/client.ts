import { validateHorizon } from "src/core/validate";
import { getAdapter } from "src/models/index";
import type { CoreOptions } from "src/core/loader";
import type { Request, Response } from "src/core/protocol";
import {
  ForecastError,
  type Backend,
  type Forecast,
  type ForecastOptions,
  type Forecaster,
  type ModelConfig,
  type Progress,
  type ResolvedDtype,
  type Series,
} from "src/types";

/** The subset of Worker the client needs, so tests can supply an in-process fake. */
export interface WorkerLike {
  postMessage(msg: Request, transfer?: Transferable[]): void;
  addEventListener(type: "message", fn: (e: MessageEvent<Response>) => void): void;
  addEventListener(type: "error", fn: (e: ErrorEvent) => void): void;
  terminate(): void;
}

/** Copies a series into a transferable Float64Array, with NaN for missing values. */
export function toFloat64(s: Series): Float64Array {
  const out = new Float64Array(s.length);
  for (let i = 0; i < s.length; i++) {
    const v = s[i];
    out[i] = v === null || v === undefined ? NaN : v;
  }
  return out;
}

/** Loads the model inside `worker` and returns a Forecaster that proxies to it. */
export function connectWorker(
  worker: WorkerLike,
  opts: CoreOptions,
  onProgress?: (p: Progress) => void,
  signal?: AbortSignal,
): Promise<Forecaster> {
  let nextId = 0;
  let disposed = false;
  const pending = new Map<number, { resolve: (r: Response) => void; reject: (e: unknown) => void }>();

  const failAll = (e: unknown) => {
    for (const p of pending.values()) p.reject(e);
    pending.clear();
  };

  worker.addEventListener("message", (e) => {
    const msg = e.data;
    if (msg.type === "progress") return onProgress?.(msg.progress);
    const p = pending.get(msg.id);
    if (!p) return;
    pending.delete(msg.id);
    if (msg.type === "error") p.reject(new ForecastError(msg.code, msg.message));
    else p.resolve(msg);
  });
  worker.addEventListener("error", (e) => {
    failAll(new ForecastError("LOAD_FAILED", `worker crashed: ${e.message ?? "unknown error"}`));
  });

  const call = <T extends Response["type"]>(
    req: Request,
    transfer: Transferable[] = [],
  ): Promise<Extract<Response, { type: T }>> =>
    new Promise((resolve, reject) => {
      pending.set(req.id, { resolve: resolve as (r: Response) => void, reject });
      worker.postMessage(req, transfer);
    });

  const onAbort = () => {
    disposed = true;
    worker.terminate();
    failAll(new ForecastError("ABORTED", "load was aborted"));
  };
  if (signal?.aborted) {
    worker.terminate();
    return Promise.reject(new ForecastError("ABORTED", "load was aborted"));
  }
  signal?.addEventListener("abort", onAbort, { once: true });

  return call<"loaded">({ type: "load", id: nextId++, opts })
    .finally(() => signal?.removeEventListener("abort", onAbort))
    .then(({ backend, dtype, config }) => {
      const info = getAdapter(opts.model).info;
      const api: Forecaster = {
        model: info.id,
        info,
        backend: backend as Backend,
        dtype: dtype as ResolvedDtype,
        config: config as ModelConfig,
        async forecast(series: Series[], fopts: ForecastOptions = {}): Promise<Forecast[]> {
          if (disposed) throw new ForecastError("DISPOSED", "this model has been disposed");
          if (fopts.horizon !== undefined) validateHorizon(fopts.horizon, info.maxHorizon);
          if (!series || series.length === 0) throw new ForecastError("EMPTY_INPUT", "pass at least one series");
          const arrays = series.map(toFloat64);
          const r = await call<"result">(
            { type: "forecast", id: nextId++, series: arrays, horizon: fopts.horizon },
            arrays.map((a) => a.buffer),
          );
          return r.forecasts;
        },
        async dispose() {
          if (disposed) return;
          disposed = true;
          try {
            await call<"disposed">({ type: "dispose", id: nextId++ });
          } finally {
            worker.terminate();
            failAll(new ForecastError("DISPOSED", "this model has been disposed"));
          }
        },
      };
      return api;
    });
}
