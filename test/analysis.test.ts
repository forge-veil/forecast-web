import { Z80 } from "src/anomaly/backtest";
import type { Forecast, ForecastOptions, Series } from "src/types";

/** Fake core: median = last value, band ±Z80·0.1, so 0.5-amplitude noise makes the bands too narrow. */
const calls: { series: Series[]; opts?: ForecastOptions }[] = [];
const fakeForecast = vi.fn(async (series: Series[], opts?: ForecastOptions): Promise<Forecast[]> => {
  calls.push({ series, opts });
  const horizon = opts?.horizon ?? 128;
  return series.map((s) => {
    const last = Number(s[s.length - 1] ?? 0);
    const med = Array(horizon).fill(last);
    const off = (k: number) => med.map((v) => v + k * Z80 * 0.1);
    return { horizon, median: med, q50: med, q10: off(-1), q20: off(-0.66), q30: off(-0.41), q40: off(-0.2), q60: off(0.2), q70: off(0.41), q80: off(0.66), q90: off(1) };
  });
});

vi.mock("src/core/session", () => ({ createSession: vi.fn() }));
vi.mock("src/core/loader", () => ({
  loadCore: async () => ({ backend: "wasm", dtype: "fp32", config: { model_type: "timesfm" }, forecast: fakeForecast, release: async () => {} }),
}));
const { loadForecaster } = await import("src/index");

const series = (n: number) => Array.from({ length: n }, (_, i) => 10 + 0.5 * Math.sin(i * 1.7));

beforeEach(() => {
  calls.length = 0;
  fakeForecast.mockClear();
});

describe("loadForecaster analysis", () => {
  it("passes a plain forecast through in one call", async () => {
    const m = await loadForecaster({ worker: false });
    const [f] = await m.forecast([series(100)], { horizon: 8 });
    expect(fakeForecast).toHaveBeenCalledTimes(1);
    expect(calls[0].opts).toEqual({ horizon: 8 });
    expect(f.calibration).toBeUndefined();
  });

  it("calibrates with one backtest call and leaves calibrate out of the model's options", async () => {
    const m = await loadForecaster({ worker: false });
    const [f] = await m.forecast([series(200)], { horizon: 8, calibrate: true });
    expect(fakeForecast).toHaveBeenCalledTimes(2);
    for (const c of calls) expect(c.opts).toEqual({ horizon: 8 });
    expect(f.calibration!.scale).toBeGreaterThan(1);
    expect(f.calibration!.points).toBeGreaterThan(10);
  });

  it("detects anomalies with the model's thresholds", async () => {
    const m = await loadForecaster({ worker: false });
    const v = series(150);
    v[120] = 40;
    const res = await m.detectAnomalies(v);
    expect(res.threshold).toBe(m.info.anomalyThresholds.medium);
    expect(res.anomalies.map((a) => a.index)).toContain(120);
    expect(fakeForecast).toHaveBeenCalledTimes(1);
  });

  it("rejects after dispose", async () => {
    const m = await loadForecaster({ worker: false });
    await m.dispose();
    await expect(m.detectAnomalies(series(100))).rejects.toMatchObject({ code: "DISPOSED" });
  });
});
