import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { loadTimesFM } from "src/index";
import {
  ForecastError,
  type Backend,
  type Forecast,
  type ForecastOptions,
  type LoadOptions,
  type Progress,
  type Series,
  type TimesFM,
} from "src/types";

export type Status = "idle" | "loading" | "ready" | "error";

export interface TimesFMState {
  status: Status;
  progress: Progress | null;
  backend: Backend | null;
  error: Error | null;
  /** Starts the download if needed and resolves when the model is ready. */
  load(): Promise<TimesFM>;
  /** Forecasts, loading the model first if it is not loaded yet. */
  forecast(series: Series[], opts?: ForecastOptions): Promise<Forecast[]>;
}

const Ctx = createContext<TimesFMState | null>(null);

export interface TimesFMProviderProps extends Omit<LoadOptions, "onProgress" | "signal"> {
  children?: ReactNode;
  /** Load on mount instead of on first use. Defaults to false. */
  eager?: boolean;
}

/**
 * Owns one TimesFM instance for its subtree. Load options are read when the
 * load starts; changing them later does not reload the model.
 */
export function TimesFMProvider({ children, eager = false, ...options }: TimesFMProviderProps) {
  const [status, setStatus] = useState<Status>("idle");
  const [progress, setProgress] = useState<Progress | null>(null);
  const [backend, setBackend] = useState<Backend | null>(null);
  const [error, setError] = useState<Error | null>(null);

  const optionsRef = useRef(options);
  optionsRef.current = options;
  const promiseRef = useRef<Promise<TimesFM> | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const genRef = useRef(0);

  const load = useCallback((): Promise<TimesFM> => {
    if (promiseRef.current) return promiseRef.current;
    const gen = ++genRef.current;
    const ac = new AbortController();
    abortRef.current = ac;
    setStatus("loading");
    setError(null);
    const p = loadTimesFM({
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

  const value = useMemo<TimesFMState>(
    () => ({ status, progress, backend, error, load, forecast }),
    [status, progress, backend, error, load, forecast],
  );
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useTimesFM(): TimesFMState {
  const v = useContext(Ctx);
  if (!v) throw new Error("useTimesFM must be used inside <TimesFMProvider>");
  return v;
}

export { ForecastError };
export type { Forecast, ForecastOptions, LoadOptions, Progress, Series, TimesFM };
