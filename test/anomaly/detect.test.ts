import { Z80, type ForecastFn } from "src/anomaly/backtest";
import { calibrateForecasts } from "src/anomaly/calibrate";
import { detectAnomalies, resolveThreshold } from "src/anomaly/detect";
import { MODELS } from "src/models/index";
import type { Forecast, ModelInfo } from "src/types";

const info: ModelInfo = { ...MODELS["timesfm-2.5"], anomalyThresholds: { low: 4, medium: 3, high: 2.5 } };

/** Fake model: median = last value, band ± Z80·w (sigma w). */
const fake = (w = 1): ForecastFn => async (contexts, { horizon }) =>
  contexts.map((c) => {
    const last = [...c].reverse().find((v) => v !== null && Number.isFinite(v)) ?? 0;
    const med = Array(horizon).fill(last);
    const off = (k: number) => med.map((v) => v + k * Z80 * w);
    return { horizon, median: med, q50: med, q10: off(-1), q20: off(-0.66), q30: off(-0.41), q40: off(-0.2), q60: off(0.2), q70: off(0.41), q80: off(0.66), q90: off(1) } as Forecast;
  });

/** Small deterministic wiggle so most residuals are below 1 sigma. */
const series = (n: number) => Array.from({ length: n }, (_, i) => 10 + 0.5 * Math.sin(i * 1.7));

describe("resolveThreshold", () => {
  it("uses the model's presets, numbers and per-call overrides", () => {
    expect(resolveThreshold(info, undefined)).toBe(3);
    expect(resolveThreshold(info, "high")).toBe(2.5);
    expect(resolveThreshold(info, 3.7)).toBe(3.7);
    expect(() => resolveThreshold(info, -1)).toThrow(expect.objectContaining({ code: "INVALID_INPUT" }));
  });
});

describe("detectAnomalies", () => {
  it("flags a planted spike with its direction, band and score", async () => {
    const v = series(120);
    v[100] = 30;
    v[110] = -10;
    const res = await detectAnomalies(v, fake(), info);
    const idx = res.anomalies.map((a) => a.index);
    expect(idx).toContain(100);
    expect(idx).toContain(110);
    const up = res.anomalies.find((a) => a.index === 100)!;
    expect(up.direction).toBe("high");
    expect(res.anomalies.find((a) => a.index === 110)!.direction).toBe("low");
    expect(up.upper - up.expected).toBeCloseTo(Z80 * res.scale * 1);
    expect(up.score).toBeGreaterThan(res.threshold);
    expect(res.points.length).toBeGreaterThan(res.anomalies.length);
  });

  it("is quiet on a clean series", async () => {
    const res = await detectAnomalies(series(200), fake(), info);
    expect(res.anomalies.length).toBe(0);
  });

  it("only checks points after enough observed history when the series starts with gaps", async () => {
    const v: (number | null)[] = [...Array(200).fill(null), ...series(100)];
    const res = await detectAnomalies(v, fake(), info);
    expect(Math.min(...res.points.map((p) => p.index))).toBeGreaterThanOrEqual(200 + 16);
    expect(res.scale).toBeLessThan(10);
    expect(res.anomalies.length).toBe(0);
  });

  it("rejects series without enough observed values", async () => {
    await expect(detectAnomalies([...Array(1000).fill(null), ...series(10)], fake(), info)).rejects.toMatchObject({ code: "INVALID_INPUT", message: expect.stringMatching(/observed/) });
    await expect(detectAnomalies(Array(100).fill(null), fake(), info)).rejects.toMatchObject({ code: "INVALID_INPUT" });
  });

  it("gives scale 1 and no flags on a constant series forecast to within float error", async () => {
    const jitter: ForecastFn = async (contexts, { horizon }) =>
      (await fake(1e-13)(contexts, { horizon })).map((f, k) => ({ ...f, median: f.median.map((m, s) => m + ((k + s) % 3) * 1e-12) }));
    const res = await detectAnomalies(Array(150).fill(100.1), jitter, info, { sensitivity: "high" });
    expect(res.scale).toBe(1);
    expect(res.anomalies.length).toBe(0);
  });

  it("rejects a series too short to check", async () => {
    await expect(detectAnomalies(series(10), fake(), info)).rejects.toMatchObject({ code: "INVALID_INPUT" });
  });
});

describe("calibrateForecasts", () => {
  it("stretches quantiles around the median by the backtest scale and reports raw coverage", async () => {
    const v = series(200);
    // The fake model's bands are 4x too narrow for this noise: sigma 0.1 vs residuals ~0.5.
    const model = fake(0.1);
    const [raw] = await model([v], { horizon: 8 });
    const [cal] = await calibrateForecasts([v], [raw], model, info, 8);
    expect(cal.calibration!.scale).toBeGreaterThan(1);
    expect(cal.calibration!.coverage).toBeLessThan(0.7);
    expect(cal.calibration!.coverage).toBeLessThan(0.8);
    expect(cal.median).toEqual(raw.median);
    const s = cal.calibration!.scale;
    expect(cal.q90[0] - cal.median[0]).toBeCloseTo(s * (raw.q90[0] - raw.median[0]));
    expect(cal.q10[3] - cal.median[3]).toBeCloseTo(s * (raw.q10[3] - raw.median[3]));
  });

  it("returns the forecast unchanged when history is too short", async () => {
    const v = series(12);
    const model = fake();
    const [raw] = await model([v], { horizon: 8 });
    const [cal] = await calibrateForecasts([v], [raw], model, info, 8);
    expect(cal.q90).toEqual(raw.q90);
    expect(cal.calibration).toEqual({ scale: 1, coverage: null, points: 0 });
  });

  it("leaves bands alone when they held 70 to 90% of past values", async () => {
    const v = series(200);
    // sigma 0.4 against residuals up to 0.75: the raw band holds most but not all past values.
    const model = fake(0.4);
    const [raw] = await model([v], { horizon: 8 });
    const [cal] = await calibrateForecasts([v], [raw], model, info, 8);
    expect(cal.calibration!.coverage).toBeGreaterThanOrEqual(0.7);
    expect(cal.calibration!.coverage).toBeLessThanOrEqual(0.9);
    expect(cal.calibration!.scale).toBe(1);
    expect(cal.q90).toEqual(raw.q90);
  });

  it("backtests one model pass even for longer horizons", async () => {
    const horizons: number[] = [];
    const base = fake();
    const spy: ForecastFn = (c, o) => (horizons.push(o.horizon), base(c, o));
    const short = { ...info, horizon: 8, maxHorizon: 64 };
    const [raw] = await base([series(200)], { horizon: 32 });
    await calibrateForecasts([series(200)], [raw], spy, short, 32);
    expect(horizons).toEqual([8]);
  });
});
