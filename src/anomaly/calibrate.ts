import { backtest, calibrationScale, type ForecastFn } from "src/anomaly/backtest";
import { minContextFor } from "src/anomaly/detect";
import type { Forecast, ModelInfo, Series } from "src/types";

/** Most backtest windows calibration runs. */
export const CALIBRATE_WINDOWS = 64;

const QUANTILES = ["q10", "q20", "q30", "q40", "q50", "q60", "q70", "q80", "q90"] as const;

/** Raw 80% band coverage inside this range counts as calibrated already; the bands are left alone. */
export const CALIBRATED = { min: 0.7, max: 0.9 };

/**
 * Rescale each forecast's quantiles around its median when the model's bands
 * were clearly off over a backtest of that series' own history.
 */
export async function calibrateForecasts(series: Series[], forecasts: Forecast[], forecast: ForecastFn, info: ModelInfo, horizon: number): Promise<Forecast[]> {
  // One model pass: longer horizons would multiply the backtest's cost (Chronos rolls out nine paths per pass).
  const h = Math.max(1, Math.min(horizon, info.horizon));
  const out: Forecast[] = [];
  // One series at a time: an inline model runs one session, so calls never overlap.
  for (let i = 0; i < series.length; i++) {
    const values = Array.from(series[i], (v) => (v == null ? null : v));
    const first = values.findIndex((v) => v !== null && Number.isFinite(v));
    const minContext = minContextFor(first < 0 ? 0 : values.length - first);
    const points = await backtest(values, forecast, { h, windows: CALIBRATE_WINDOWS, minContext, maxContext: info.maxContext });
    const coverage = points.length < 10 ? null : points.filter((p) => p.value >= p.q10 && p.value <= p.q90).length / points.length;
    // Measured on real models: rescaling bands that already held 70 to 90% made new values' coverage worse.
    const scale = coverage === null || (coverage >= CALIBRATED.min && coverage <= CALIBRATED.max) ? 1 : calibrationScale(points);
    const f = forecasts[i];
    const next: Forecast = { ...f, calibration: { scale, coverage, points: points.length } };
    if (scale !== 1) for (const k of QUANTILES) next[k] = f[k].map((q, s) => f.median[s] + scale * (q - f.median[s]));
    out.push(next);
  }
  return out;
}
