import { backtest, calibrationScale, Z80, type ForecastFn } from "src/anomaly/backtest";
import { ForecastError, type AnomalyOptions, type AnomalyPoint, type AnomalyResult, type ModelInfo, type Sensitivity, type Series } from "src/types";

/** Most backtest windows detection runs. */
export const DETECT_WINDOWS = 128;

/** Context before the first checked point: a quarter of the series, between 16 and 64. */
export const minContextFor = (n: number) => Math.max(16, Math.min(64, Math.floor(n / 4)));

const positive = (t: unknown): number => {
  if (typeof t !== "number" || !Number.isFinite(t) || t <= 0) throw new ForecastError("INVALID_INPUT", `anomaly threshold must be a positive number, got ${String(t)}`);
  return t;
};

/** The score threshold a sensitivity resolves to for this model. */
export function resolveThreshold(info: Pick<ModelInfo, "anomalyThresholds">, sensitivity: Sensitivity = "medium"): number {
  if (typeof sensitivity === "number") return positive(sensitivity);
  if (sensitivity !== "low" && sensitivity !== "medium" && sensitivity !== "high") throw new ForecastError("INVALID_INPUT", `unknown sensitivity "${String(sensitivity)}"`);
  return positive(info.anomalyThresholds[sensitivity]);
}

/**
 * Flag points the model could not have predicted from the history before them:
 * backtest the series against itself, calibrate the bands, and score each point's
 * distance from the median in calibrated standard deviations.
 */
export async function detectAnomalies(series: Series, forecast: ForecastFn, info: ModelInfo, opts: AnomalyOptions = {}): Promise<AnomalyResult> {
  const threshold = resolveThreshold(info, opts.sensitivity);
  const values = Array.from(series, (v) => (v == null ? null : v));
  const first = values.findIndex((v) => v !== null && Number.isFinite(v));
  if (first < 0) throw new ForecastError("INVALID_INPUT", "series has no observed values to check");
  const span = values.length - first;
  const m = minContextFor(span);
  if (span < m + 4) throw new ForecastError("INVALID_INPUT", `series too short to check: ${span} points from the first observed value, need at least ${m + 4}`);
  const h = Math.min(info.horizon, Math.max(1, Math.ceil((span - m) / DETECT_WINDOWS)));
  const scored = await backtest(values, forecast, { h, windows: DETECT_WINDOWS, minContext: m, maxContext: info.maxContext });
  const scale = calibrationScale(scored);
  const points: AnomalyPoint[] = scored.map((p) => {
    const half = Z80 * scale * p.sigma;
    const score = Math.abs(p.r) / scale;
    return { index: p.index, value: p.value, expected: p.median, lower: p.median - half, upper: p.median + half, score, direction: p.r >= 0 ? "high" : "low", anomaly: score > threshold };
  });
  return { points, anomalies: points.filter((p) => p.anomaly), scale, threshold };
}
