import type { Core, CoreOptions } from "src/loader";
import {
  TimesFMError,
  type Backend,
  type ErrorCode,
  type Forecast,
  type ModelConfig,
  type Progress,
  type ResolvedDtype,
} from "src/types";

export type Request =
  | { type: "load"; id: number; opts: CoreOptions }
  | { type: "forecast"; id: number; series: Float64Array[]; horizon?: number }
  | { type: "dispose"; id: number };

export type Response =
  | { type: "progress"; id: number; progress: Progress }
  | { type: "loaded"; id: number; backend: Backend; dtype: ResolvedDtype; config: ModelConfig }
  | { type: "result"; id: number; forecasts: Forecast[] }
  | { type: "disposed"; id: number }
  | { type: "error"; id: number; code: ErrorCode; message: string };

export function toErrorResponse(id: number, e: unknown): Response {
  if (e instanceof TimesFMError) return { type: "error", id, code: e.code, message: e.message };
  return { type: "error", id, code: "LOAD_FAILED", message: e instanceof Error ? e.message : String(e) };
}

/**
 * Worker-side request handler, independent of the actual Worker global so it
 * can be tested in-process. `load` must resolve to a Core.
 */
export function createHandler(load: (opts: CoreOptions, onProgress: (p: Progress) => void) => Promise<Core>) {
  let core: Promise<Core> | null = null;

  return async (msg: Request, post: (r: Response) => void): Promise<void> => {
    try {
      switch (msg.type) {
        case "load": {
          core = load(msg.opts, (progress) => post({ type: "progress", id: msg.id, progress }));
          const c = await core;
          post({ type: "loaded", id: msg.id, backend: c.backend, dtype: c.dtype, config: c.config });
          return;
        }
        case "forecast": {
          if (!core) throw new TimesFMError("LOAD_FAILED", "model is not loaded");
          const c = await core;
          const forecasts = await c.forecast(msg.series, { horizon: msg.horizon });
          post({ type: "result", id: msg.id, forecasts });
          return;
        }
        case "dispose": {
          const c = core;
          core = null;
          if (c) await (await c).release();
          post({ type: "disposed", id: msg.id });
          return;
        }
      }
    } catch (e) {
      post(toErrorResponse(msg.id, e));
    }
  };
}
