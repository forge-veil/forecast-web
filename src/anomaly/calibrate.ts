import { backtest, calibrationScale, type ForecastFn } from "src/anomaly/backtest";
import { minContextFor } from "src/anomaly/detect";
import type { Forecast, ModelInfo, Series } from "src/types";

/** Most backtest windows calibration runs. */
export const CALIBRATE_WINDOWS = 64;

const QUANTILES = ["q10", "q20", "q30", "q40", "q50", "q60", "q70", "q80", "q90"] as const;

/**
 * Rescale each forecast's quantiles around its median by how far the model's
 * bands were off over a backtest of that series' own history.
 */
export async function calibrateForecasts(series: Series[], forecasts: Forecast[], forecast: ForecastFn, info: ModelInfo, horizon: number): Promise<Forecast[]> {
  const h = Math.max(1, Math.min(horizon, info.maxHorizon));
  const out: Forecast[] = [];
  // One series at a time: an inline model runs one session, so calls never overlap.
  for (let i = 0; i < series.length; i++) {
    const values = Array.from(series[i], (v) => (v == null ? null : v));
    const points = await backtest(values, forecast, { h, windows: CALIBRATE_WINDOWS, minContext: minContextFor(values.length), maxContext: info.maxContext });
    const scale = calibrationScale(points);
    const coverage = points.length < 10 ? NaN : points.filter((p) => p.value >= p.q10 && p.value <= p.q90).length / points.length;
    const f = forecasts[i];
    const next: Forecast = { ...f, calibration: { scale, coverage, points: points.length } };
    if (scale !== 1) for (const k of QUANTILES) next[k] = f[k].map((q, s) => f.median[s] + scale * (q - f.median[s]));
    out.push(next);
  }
  return out;
}
