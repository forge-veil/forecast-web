// @vitest-environment jsdom
import { act, render, renderHook, waitFor } from "@testing-library/react";
import { StrictMode, useEffect, type ReactNode } from "react";
import { MODELS } from "src/models/index";
import type { Forecaster, LoadOptions } from "src/types";

const loadMock = vi.fn<(o: LoadOptions) => Promise<Forecaster>>();
vi.mock("src/index", () => ({ loadForecaster: (o: LoadOptions) => loadMock(o) }));

const { ForecastProvider, useForecaster } = await import("src/react");
const { ForecastError } = await import("src/types");

function fakeModel(): Forecaster & { dispose: ReturnType<typeof vi.fn> } {
  return {
    model: "timesfm-2.5" as const,
    info: MODELS["timesfm-2.5"],
    backend: "wasm",
    dtype: "q8",
    forecast: vi.fn(async () => []),
    detectAnomalies: vi.fn(async () => ({ points: [], anomalies: [], scale: 1, threshold: 3 })),
    dispose: vi.fn(async () => {}),
  };
}

const wrap =
  (props: Record<string, unknown> = {}) =>
  ({ children }: { children: ReactNode }) => <ForecastProvider {...props}>{children}</ForecastProvider>;

beforeEach(() => {
  loadMock.mockClear();
  loadMock.mockImplementation(() => Promise.reject(new Error("loadMock not configured")));
});

describe("ForecastProvider + useForecaster", () => {
  it("is lazy by default", () => {
    const { result } = renderHook(() => useForecaster(), { wrapper: wrap() });
    expect(result.current.status).toBe("idle");
    expect(loadMock).not.toHaveBeenCalled();
  });

  it("loads once on the first forecast, reporting progress", async () => {
    const m = fakeModel();
    loadMock.mockImplementation(async (o) => {
      o.onProgress?.({ loaded: 1, total: 2, cached: false });
      return m;
    });
    const { result } = renderHook(() => useForecaster(), { wrapper: wrap({ dtype: "fp16" }) });
    await act(async () => {
      await Promise.all([result.current.forecast([[1, 2]]), result.current.forecast([[3]])]);
    });
    expect(loadMock).toHaveBeenCalledTimes(1);
    expect(loadMock.mock.calls[0][0].dtype).toBe("fp16");
    expect(m.forecast).toHaveBeenCalledTimes(2);
    expect(result.current.status).toBe("ready");
    expect(result.current.backend).toBe("wasm");
    expect(result.current.progress).toEqual({ loaded: 1, total: 2, cached: false });
  });

  it("detectAnomalies loads the model first and forwards options", async () => {
    const m = fakeModel();
    loadMock.mockResolvedValue(m);
    const { result } = renderHook(() => useForecaster(), { wrapper: wrap() });
    let res: unknown;
    await act(async () => {
      res = await result.current.detectAnomalies([1, 2, 3], { sensitivity: "high" });
    });
    expect(loadMock).toHaveBeenCalledTimes(1);
    expect(m.detectAnomalies).toHaveBeenCalledWith([1, 2, 3], { sensitivity: "high" });
    expect(res).toMatchObject({ threshold: 3 });
  });

  it("loads on mount when eager", async () => {
    loadMock.mockResolvedValue(fakeModel());
    const { result } = renderHook(() => useForecaster(), { wrapper: wrap({ eager: true }) });
    await waitFor(() => expect(result.current.status).toBe("ready"));
    expect(loadMock).toHaveBeenCalledTimes(1);
  });

  it("exposes load errors and allows a retry", async () => {
    loadMock.mockRejectedValueOnce(new ForecastError("LOAD_FAILED", "HTTP 404"));
    const { result } = renderHook(() => useForecaster(), { wrapper: wrap() });
    await act(async () => {
      await result.current.load().catch(() => {});
    });
    expect(result.current.status).toBe("error");
    expect(result.current.error?.message).toBe("HTTP 404");

    loadMock.mockResolvedValueOnce(fakeModel());
    await act(async () => {
      await result.current.load();
    });
    expect(result.current.status).toBe("ready");
    expect(result.current.error).toBeNull();
  });

  it("disposes the model on unmount", async () => {
    const m = fakeModel();
    loadMock.mockResolvedValue(m);
    const { result, unmount } = renderHook(() => useForecaster(), { wrapper: wrap() });
    await act(async () => {
      await result.current.load();
    });
    unmount();
    await waitFor(() => expect(m.dispose).toHaveBeenCalledTimes(1));
  });

  it("aborts an in-flight load on unmount", async () => {
    let signal: AbortSignal | undefined;
    loadMock.mockImplementation((o) => {
      signal = o.signal;
      return new Promise(() => {});
    });
    const { result, unmount } = renderHook(() => useForecaster(), { wrapper: wrap() });
    act(() => void result.current.load());
    unmount();
    await waitFor(() => expect(signal?.aborted).toBe(true));
  });

  it("survives StrictMode's double mount with eager loading", async () => {
    const models: ReturnType<typeof fakeModel>[] = [];
    loadMock.mockImplementation(async () => {
      const m = fakeModel();
      models.push(m);
      return m;
    });
    let state: ReturnType<typeof useForecaster> | undefined;
    function Probe() {
      state = useForecaster();
      return null;
    }
    render(
      <StrictMode>
        <ForecastProvider eager>
          <Probe />
        </ForecastProvider>
      </StrictMode>,
    );
    await waitFor(() => expect(state?.status).toBe("ready"));
    // The simulated unmount is cancelled by the remount: one load, still alive.
    await new Promise((r) => setTimeout(r, 10));
    expect(models).toHaveLength(1);
    expect(models[0].dispose).not.toHaveBeenCalled();
  });

  it("lets a child forecast on mount under StrictMode without an abort", async () => {
    const m = fakeModel();
    loadMock.mockImplementation(() => new Promise((r) => setTimeout(() => r(m), 5)));
    const outcomes: string[] = [];
    function Child() {
      const { forecast } = useForecaster();
      useEffect(() => {
        forecast([[1, 2, 3]]).then(
          () => outcomes.push("ok"),
          (e) => outcomes.push(`error:${e.code ?? e.message}`),
        );
      }, [forecast]);
      return null;
    }
    render(
      <StrictMode>
        <ForecastProvider>
          <Child />
        </ForecastProvider>
      </StrictMode>,
    );
    await waitFor(() => expect(outcomes.length).toBe(2));
    expect(outcomes).toEqual(["ok", "ok"]);
    expect(loadMock).toHaveBeenCalledTimes(1);
    expect(m.dispose).not.toHaveBeenCalled();
  });

  it("remounting with a new key disposes the old model and loads again", async () => {
    const a = fakeModel();
    const b = fakeModel();
    loadMock.mockResolvedValueOnce(a).mockResolvedValueOnce(b);
    function Child() {
      const { load } = useForecaster();
      useEffect(() => void load(), [load]);
      return null;
    }
    const { rerender } = render(
      <ForecastProvider key="first">
        <Child />
      </ForecastProvider>,
    );
    await waitFor(() => expect(loadMock).toHaveBeenCalledTimes(1));
    rerender(
      <ForecastProvider key="second">
        <Child />
      </ForecastProvider>,
    );
    await waitFor(() => expect(a.dispose).toHaveBeenCalledTimes(1));
    expect(loadMock).toHaveBeenCalledTimes(2);
    expect(b.dispose).not.toHaveBeenCalled();
  });

  it("throws a clear error outside the provider", () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    expect(() => renderHook(() => useForecaster())).toThrow(/inside <ForecastProvider>/);
  });
});
