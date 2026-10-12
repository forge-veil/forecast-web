import { fetchBytes, resolveFiles } from "src/assets";
import { createForecaster } from "src/forecaster";
import type { GraphSession } from "src/session";
import {
  TimesFMError,
  type Backend,
  type Device,
  type Dtype,
  type Forecast,
  type ForecastOptions,
  type ModelConfig,
  type Progress,
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
  config: ModelConfig;
  forecast(series: Series[], opts?: ForecastOptions): Promise<Forecast[]>;
  release(): Promise<void>;
}

export type SessionFactory = (
  bytes: Uint8Array,
  opts: { device: Device; wasmPaths?: string },
) => Promise<GraphSession>;

/** Downloads config and graph, creates a session, returns a ready forecaster. */
export async function loadCore(
  opts: CoreOptions,
  createSession: SessionFactory,
  onProgress?: (p: Progress) => void,
  signal?: AbortSignal,
): Promise<Core> {
  const files = resolveFiles(opts.base, opts.dtype);
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

  const bytes = await fetchBytes(files.model, {
    cache: opts.cache,
    signal,
    onProgress: (loaded, total, cached) => onProgress?.({ loaded, total, cached }),
  });
  if (signal?.aborted) throw new TimesFMError("ABORTED", "load was aborted");

  const session = await createSession(bytes, { device: opts.device, wasmPaths: opts.wasmPaths });
  return {
    backend: session.backend,
    config,
    forecast: createForecaster(session.run, config.context),
    release: () => session.release(),
  };
}
