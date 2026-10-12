// @vitest-environment jsdom
import { act, render, renderHook, waitFor } from "@testing-library/react";
import { StrictMode, useEffect, type ReactNode } from "react";
import type { LoadOptions, TimesFM } from "src/types";

const loadMock = vi.fn<(o: LoadOptions) => Promise<TimesFM>>();
vi.mock("src/index", () => ({ loadTimesFM: (o: LoadOptions) => loadMock(o) }));

const { TimesFMProvider, useTimesFM } = await import("src/react");
const { TimesFMError } = await import("src/types");

function fakeModel(): TimesFM & { dispose: ReturnType<typeof vi.fn> } {
  return {
    backend: "wasm",
    dtype: "q8",
    config: { model_type: "timesfm", context: 512, horizon: 128, patch: 32, quantiles: [] },
    forecast: vi.fn(async () => []),
    dispose: vi.fn(async () => {}),
  };
}

const wrap =
  (props: Record<string, unknown> = {}) =>
  ({ children }: { children: ReactNode }) => <TimesFMProvider {...props}>{children}</TimesFMProvider>;

beforeEach(() => {
  loadMock.mockClear();
  loadMock.mockImplementation(() => Promise.reject(new Error("loadMock not configured")));
});

describe("TimesFMProvider + useTimesFM", () => {
  it("is lazy by default", () => {
    const { result } = renderHook(() => useTimesFM(), { wrapper: wrap() });
    expect(result.current.status).toBe("idle");
    expect(loadMock).not.toHaveBeenCalled();
  });

  it("loads once on the first forecast, reporting progress", async () => {
    const m = fakeModel();
    loadMock.mockImplementation(async (o) => {
      o.onProgress?.({ loaded: 1, total: 2, cached: false });
      return m;
    });
    const { result } = renderHook(() => useTimesFM(), { wrapper: wrap({ dtype: "fp16" }) });
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

  it("loads on mount when eager", async () => {
    loadMock.mockResolvedValue(fakeModel());
    const { result } = renderHook(() => useTimesFM(), { wrapper: wrap({ eager: true }) });
    await waitFor(() => expect(result.current.status).toBe("ready"));
    expect(loadMock).toHaveBeenCalledTimes(1);
  });

  it("exposes load errors and allows a retry", async () => {
    loadMock.mockRejectedValueOnce(new TimesFMError("LOAD_FAILED", "HTTP 404"));
    const { result } = renderHook(() => useTimesFM(), { wrapper: wrap() });
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
    const { result, unmount } = renderHook(() => useTimesFM(), { wrapper: wrap() });
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
    const { result, unmount } = renderHook(() => useTimesFM(), { wrapper: wrap() });
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
    let state: ReturnType<typeof useTimesFM> | undefined;
    function Probe() {
      state = useTimesFM();
      return null;
    }
    render(
      <StrictMode>
        <TimesFMProvider eager>
          <Probe />
        </TimesFMProvider>
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
      const { forecast } = useTimesFM();
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
        <TimesFMProvider>
          <Child />
        </TimesFMProvider>
      </StrictMode>,
    );
    await waitFor(() => expect(outcomes.length).toBe(2));
    expect(outcomes).toEqual(["ok", "ok"]);
    expect(loadMock).toHaveBeenCalledTimes(1);
    expect(m.dispose).not.toHaveBeenCalled();
  });

  it("throws a clear error outside the provider", () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    expect(() => renderHook(() => useTimesFM())).toThrow(/inside <TimesFMProvider>/);
  });
});
