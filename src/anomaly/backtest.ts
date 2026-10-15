import type { Forecast } from "src/types";

/** Batched forecast over several series; how the engine calls a model. */
export type ForecastFn = (series: (number | null)[][], opts: { horizon: number }) => Promise<Forecast[]>;

/** One history point the model forecast without seeing it. */
export interface ScoredPoint {
  index: number;
  value: number;
  median: number;
  q10: number;
  q90: number;
  /** Steps ahead of its window's cut (1 = next point). */
  step: number;
  /** Implied standard deviation from the 80% band. */
  sigma: number;
  /** Normalized residual (value - median) / sigma. */
  r: number;
}

/** The 80% band of a normal distribution is ±Z80 standard deviations. */
export const Z80 = 1.2816;

const isNum = (v: number | null | undefined): v is number => typeof v === "number" && Number.isFinite(v);

/**
 * Rolling backtest: cut the series at n-h, n-2h, ... (newest first, at most
 * `windows` cuts, none before `minContext`), forecast h steps from each cut in
 * one batched call, and score every point the windows cover.
 */
export async function backtest(
  values: (number | null)[],
  forecast: ForecastFn,
  opts: { h: number; windows: number; minContext: number; maxContext: number },
): Promise<ScoredPoint[]> {
  const n = values.length;
  const cuts: number[] = [];
  for (let c = n - opts.h; c >= opts.minContext && cuts.length < opts.windows; c -= opts.h) cuts.push(c);
  if (cuts.length === 0) return [];
  const contexts = cuts.map((c) => values.slice(Math.max(0, c - opts.maxContext), c));
  const forecasts = await forecast(contexts, { horizon: opts.h });

  let absSum = 0;
  let count = 0;
  for (const v of values) if (isNum(v)) (absSum += Math.abs(v)), count++;
  const floor = 1e-9 * (1 + (count ? absSum / count : 0));

  const points: ScoredPoint[] = [];
  cuts.forEach((c, k) => {
    const f = forecasts[k];
    for (let s = 0; s < opts.h && c + s < n; s++) {
      const value = values[c + s];
      if (!isNum(value)) continue;
      const median = f.median[s];
      const sigma = Math.max((f.q90[s] - f.q10[s]) / (2 * Z80), floor);
      points.push({ index: c + s, value, median, q10: f.q10[s], q90: f.q90[s], step: s + 1, sigma, r: (value - median) / sigma });
    }
  });
  return points.sort((a, b) => a.index - b.index);
}

/** Linear-interpolation quantile of sorted values. */
function quantile(sorted: number[], q: number): number {
  const pos = q * (sorted.length - 1);
  const lo = Math.floor(pos);
  const hi = Math.min(lo + 1, sorted.length - 1);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

/**
 * How much wider (>1) or narrower (<1) the model's bands should be for this
 * series: the 0.8 quantile of |r| over Z80. Robust to a few anomalies; 1 when
 * fewer than 10 points were scored.
 */
export function calibrationScale(points: ScoredPoint[]): number {
  if (points.length < 10) return 1;
  const abs = points.map((p) => Math.abs(p.r)).sort((a, b) => a - b);
  const s = quantile(abs, 0.8) / Z80;
  return s > 0 && Number.isFinite(s) ? s : 1;
}
