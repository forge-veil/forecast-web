import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { loadForecaster } from "src/index";
import {
  ForecastError,
  type Backend,
  type Forecast,
  type ForecastOptions,
  type LoadOptions,
  type ModelId,
  type Progress,
  type Series,
  type Forecaster,
} from "src/types";

export type Status = "idle" | "loading" | "ready" | "error";

export interface ForecasterState {
  status: Status;
  progress: Progress | null;
  backend: Backend | null;
  error: Error | null;
  /** Starts the download if needed and resolves when the model is ready. */
  load(): Promise<Forecaster>;
  /** Forecasts, loading the model first if it is not loaded yet. */
  forecast(series: Series[], opts?: ForecastOptions): Promise<Forecast[]>;
}

const Ctx = createContext<ForecasterState | null>(null);

export interface ForecastProviderProps extends Omit<LoadOptions, "onProgress" | "signal"> {
  children?: ReactNode;
  /** Load on mount instead of on first use. Defaults to false. */
  eager?: boolean;
}

/**
 * Owns one model instance for its subtree. Load options are read when the load
 * starts; changing them later does not reload. To switch models, remount with
 * key={model}.
 */
export function ForecastProvider({ children, eager = false, ...options }: ForecastProviderProps) {
  const [status, setStatus] = useState<Status>("idle");
  const [progress, setProgress] = useState<Progress | null>(null);
  const [backend, setBackend] = useState<Backend | null>(null);
  const [error, setError] = useState<Error | null>(null);

  const optionsRef = useRef(options);
  optionsRef.current = options;
  const promiseRef = useRef<Promise<Forecaster> | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const genRef = useRef(0);

  const load = useCallback((): Promise<Forecaster> => {
    if (promiseRef.current) return promiseRef.current;
    const gen = ++genRef.current;
    const ac = new AbortController();
    abortRef.current = ac;
    setStatus("loading");
    setError(null);
    const p = loadForecaster({
      ...optionsRef.current,
      signal: ac.signal,
      onProgress: (pr) => gen === genRef.current && setProgress(pr),
    });
    promiseRef.current = p;
    p.then(
      (m) => {
        if (gen !== genRef.current) return;
        setBackend(m.backend);
        setStatus("ready");
      },
      (e: unknown) => {
        if (gen !== genRef.current) return;
        promiseRef.current = null; // allow a retry
        setError(e instanceof Error ? e : new Error(String(e)));
        setStatus("error");
      },
    );
    return p;
  }, []);

  const forecast = useCallback(
    async (series: Series[], opts?: ForecastOptions) => (await load()).forecast(series, opts),
    [load],
  );

  useEffect(() => {
    if (eager) void load().catch(() => {});
  }, [eager, load]);

  const teardownRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(() => {
    // StrictMode unmounts and remounts immediately; a pending teardown is cancelled so
    // in-flight loads and forecasts started by children survive.
    if (teardownRef.current !== null) {
      clearTimeout(teardownRef.current);
      teardownRef.current = null;
    }
    return () => {
      teardownRef.current = setTimeout(() => {
        teardownRef.current = null;
        genRef.current++;
        abortRef.current?.abort();
        const p = promiseRef.current;
        promiseRef.current = null;
        p?.then((m) => m.dispose()).catch(() => {});
      }, 0);
    };
  }, []);

  const value = useMemo<ForecasterState>(
    () => ({ status, progress, backend, error, load, forecast }),
    [status, progress, backend, error, load, forecast],
  );
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useForecaster(): ForecasterState {
  const v = useContext(Ctx);
  if (!v) throw new Error("useForecaster must be used inside <ForecastProvider>");
  return v;
}

export { MODELS } from "src/models/index";
export { ForecastError };
export type { Forecast, ForecastOptions, Forecaster, LoadOptions, ModelId, Progress, Series };
