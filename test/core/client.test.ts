import { connectGraph, connectWorker, toFloat64, type WorkerLike } from "src/core/client";
import { loadCore, type Core, type CoreOptions } from "src/core/loader";
import { createHandler, type Request, type Response } from "src/core/protocol";
import { ForecastError, type Forecast, type Progress, type Series } from "src/types";

const opts: CoreOptions = { model: "timesfm-2.5", base: "https://x/", dtype: "q8", device: "wasm", cache: false };

function fakeForecast(series: Series[]): Forecast[] {
  return series.map((s) => {
    const last = s[s.length - 1] as number;
    const v = [last];
    return { horizon: 1, mean: v, median: v, q10: v, q20: v, q30: v, q40: v, q50: v, q60: v, q70: v, q80: v, q90: v };
  });
}

/** Connects the real worker-side handler to the real client in-process. */
function setup(load?: (o: CoreOptions, p: (p: Progress) => void) => Promise<Core>) {
  const released = vi.fn(async () => {});
  const delays: number[] = [];
  const core: Core = {
    backend: "wasm",
    dtype: "q8",
    config: { model_type: "timesfm", context: 512, horizon: 128, patch: 32, quantiles: [] },
    async forecast(series) {
      const d = delays.shift() ?? 0;
      await new Promise((r) => setTimeout(r, d));
      return fakeForecast(series);
    },
    run: vi.fn(async () => ({ out: new Float32Array([1, 2]) })),
    release: released,
  };
  const handle = createHandler(
    load ??
      (async (_o, onProgress) => {
        onProgress({ loaded: 5, total: 10, cached: false });
        onProgress({ loaded: 10, total: 10, cached: false });
        return core;
      }),
  );
  const listeners: Record<string, ((e: any) => void)[]> = { message: [], error: [] };
  const posted: Request[] = [];
  const worker: WorkerLike & { terminated: boolean } = {
    terminated: false,
    postMessage(msg: Request) {
      posted.push(msg);
      // Structured clone drops prototypes; emulate it so tests catch non-cloneable payloads.
      const cloned = structuredClone(msg);
      void handle(cloned, (r: Response) => listeners.message.forEach((fn) => fn({ data: structuredClone(r) })));
    },
    addEventListener(type: string, fn: (e: any) => void) {
      listeners[type].push(fn);
    },
    terminate() {
      this.terminated = true;
    },
  };
  return { worker, released, delays, posted, crash: (m: string) => listeners.error.forEach((fn) => fn({ message: m })) };
}

describe("toFloat64", () => {
  it("maps null and undefined to NaN", () => {
    expect(Array.from(toFloat64([1, null, undefined, 4]))).toEqual([1, NaN, NaN, 4]);
  });
});

describe("connectWorker", () => {
  it("loads, reports progress, and exposes the backend", async () => {
    const { worker } = setup();
    const seen: Progress[] = [];
    const m = await connectWorker(worker, opts, (p) => seen.push(p));
    expect(m.backend).toBe("wasm");
    expect(m.dtype).toBe("q8");
    expect(seen.map((p) => p.loaded)).toEqual([5, 10]);
  });

  it("routes concurrent forecasts back to the right caller", async () => {
    const { worker, delays } = setup();
    const m = await connectWorker(worker, opts);
    delays.push(30, 0); // first request finishes last
    const [a, b] = await Promise.all([m.forecast([[1, 2, 3]]), m.forecast([[7, 8, 9]])]);
    expect(a[0].median).toEqual([3]);
    expect(b[0].median).toEqual([9]);
  });

  it("validates horizon on the main thread before posting", async () => {
    const { worker, posted } = setup();
    const m = await connectWorker(worker, opts);
    const before = posted.length;
    await expect(m.forecast([[1]], { horizon: 500 })).rejects.toMatchObject({ code: "HORIZON_TOO_LONG" });
    await expect(m.forecast([])).rejects.toMatchObject({ code: "EMPTY_INPUT" });
    expect(posted.length).toBe(before);
  });

  it("rejects forecasts after dispose and releases the session once", async () => {
    const { worker, released } = setup();
    const m = await connectWorker(worker, opts);
    await m.dispose();
    await m.dispose();
    expect(released).toHaveBeenCalledTimes(1);
    expect(worker.terminated).toBe(true);
    await expect(m.forecast([[1]])).rejects.toMatchObject({ code: "DISPOSED" });
  });

  it("exposes the model id and info", async () => {
    const { worker } = setup();
    const m = await connectWorker(worker, opts);
    expect(m.model).toBe("timesfm-2.5");
    expect(m.info.maxHorizon).toBe(128);
  });

  it("rejects a horizon over the model limit on the main thread", async () => {
    const { worker, posted } = setup();
    const m = await connectWorker(worker, opts);
    const before = posted.length;
    await expect(m.forecast([[1, 2]], { horizon: 129 })).rejects.toMatchObject({ code: "HORIZON_TOO_LONG" });
    expect(posted.length).toBe(before);
  });

  it("carries UNKNOWN_MODEL from the worker", async () => {
    const { worker } = setup((o, p) =>
      loadCore(o, async () => {
        throw new Error("unused");
      }, p),
    );
    await expect(connectWorker(worker, { ...opts, model: "nope" as never })).rejects.toMatchObject({
      code: "UNKNOWN_MODEL",
    });
  });

  it("surfaces load errors with their code", async () => {
    const { worker } = setup(async () => {
      throw new ForecastError("LOAD_FAILED", "HTTP 404");
    });
    await expect(connectWorker(worker, opts)).rejects.toMatchObject({ code: "LOAD_FAILED", message: "HTTP 404" });
  });

  it("aborts a load in progress and terminates the worker", async () => {
    const { worker } = setup(() => new Promise(() => {}));
    const ac = new AbortController();
    const p = connectWorker(worker, opts, undefined, ac.signal);
    ac.abort();
    await expect(p).rejects.toMatchObject({ code: "ABORTED" });
    expect(worker.terminated).toBe(true);
  });

  it("fails pending calls when the worker crashes", async () => {
    const { worker, crash } = setup(() => new Promise(() => {}));
    const p = connectWorker(worker, opts);
    crash("out of memory");
    await expect(p).rejects.toThrow(/out of memory/);
  });
});

describe("connectGraph", () => {
  it("runs named feeds in the worker and returns named outputs", async () => {
    const { worker } = setup();
    const g = await connectGraph(worker, opts);
    const out = await g.run({ X: { data: new Float32Array([1, 2, 3]), dims: [1, 3] } });
    expect(Array.from(out.out)).toEqual([1, 2]);
    await g.dispose();
    await expect(g.run({ X: { data: new Float32Array([1]), dims: [1] } })).rejects.toMatchObject({ code: "DISPOSED" });
  });
});

describe("protocol run", () => {
  it("transfers output buffers back to the caller", async () => {
    const out = new Float32Array([1, 2]);
    const handle = createHandler(async () => ({
      backend: "wasm",
      dtype: "fp32",
      config: { model_type: "tabicl" },
      forecast: async () => [],
      run: async () => ({ out }),
      release: async () => {},
    }));
    const posted: [Response, Transferable[] | undefined][] = [];
    await handle({ type: "load", id: 0, opts }, (r, t) => posted.push([r, t]));
    await handle({ type: "run", id: 1, feeds: {} }, (r, t) => posted.push([r, t]));
    const [res, transfer] = posted.at(-1)!;
    expect(res.type).toBe("outputs");
    expect(transfer).toEqual([out.buffer]);
  });
});

describe("worker message failures", () => {
  it("rejects and forgets a request whose postMessage throws", async () => {
    const { worker } = setup();
    const g = await connectGraph(worker, opts);
    const original = worker.postMessage.bind(worker);
    worker.postMessage = () => {
      throw new DOMException("could not clone", "DataCloneError");
    };
    await expect(g.run({ X: { data: new Float32Array([1]), dims: [1] } })).rejects.toThrow(/could not clone/);
    worker.postMessage = original;
    const out = await g.run({ X: { data: new Float32Array([1]), dims: [1] } });
    expect(Array.from(out.out)).toEqual([1, 2]);
  });
});


describe("anomaly detection over the worker", () => {
  it("runs the backtest through the worker and rejects after dispose", async () => {
    const { withAnalysis } = await import("src/anomaly/index");
    const { worker } = setup();
    const m = withAnalysis(await connectWorker(worker, opts));
    const series = Array<number>(100).fill(5);
    series[90] = 50;
    const res = await m.detectAnomalies(series);
    expect(res.points.length).toBeGreaterThan(50);
    expect(res.anomalies.map((a) => a.index)).toContain(90);
    await m.dispose();
    await expect(m.detectAnomalies(series)).rejects.toMatchObject({ code: "DISPOSED" });
  });
});
