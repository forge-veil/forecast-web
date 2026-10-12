import { CONTEXT, type Series, type SeriesStats } from "./types";

const TOLERANCE = 1e-6;

const isMissing = (v: number | null | undefined): boolean =>
  v === null || v === undefined || !Number.isFinite(v);

/**
 * Mirrors TimesFM's strip_leading_nans + linear_interpolation: drop leading
 * gaps, fill interior gaps linearly, hold the last value across trailing gaps.
 * An all-missing series becomes zeros of the same length.
 */
export function cleanSeries(series: Series): Float64Array {
  const n = series.length;
  let start = 0;
  while (start < n && isMissing(series[start])) start++;
  if (start === n) return new Float64Array(n);

  const out = new Float64Array(n - start);
  let prev = -1;
  for (let i = start; i < n; i++) {
    const v = series[i];
    const j = i - start;
    if (isMissing(v)) continue;
    out[j] = v as number;
    if (prev >= 0 && j - prev > 1) {
      const a = out[prev];
      const step = (out[j] - a) / (j - prev);
      for (let k = prev + 1; k < j; k++) out[k] = a + step * (k - prev);
    }
    prev = j;
  }
  for (let k = prev + 1; k < out.length; k++) out[k] = out[prev];
  return out;
}

/**
 * Builds the graph inputs for a batch. Row 2b holds series b normalized,
 * row 2b+1 holds its negation (for flip invariance). Shapes are [2B, context].
 */
export function prepareBatch(
  series: Series[],
  context = CONTEXT,
): { inputs: Float32Array; masks: Float32Array; stats: SeriesStats[] } {
  const B = series.length;
  const inputs = new Float32Array(2 * B * context);
  const masks = new Float32Array(2 * B * context);
  const stats: SeriesStats[] = [];

  series.forEach((s, b) => {
    const clean = cleanSeries(s);
    const v = clean.length > context ? clean.subarray(clean.length - context) : clean;
    const pad = context - v.length;
    const x = new Float64Array(context);
    x.set(v, pad);

    let mu = 0;
    for (let i = 0; i < context; i++) mu += x[i];
    mu /= context;
    let ss = 0;
    let isPositive = true;
    for (let i = 0; i < context; i++) {
      ss += (x[i] - mu) ** 2;
      if (x[i] < 0) isPositive = false;
    }
    const sigma = Math.sqrt(ss / (context - 1));
    const div = sigma < TOLERANCE ? 1 : sigma;

    const r0 = 2 * b * context;
    const r1 = r0 + context;
    for (let i = 0; i < context; i++) {
      const z = (x[i] - mu) / div;
      const m = i < pad ? 1 : 0;
      inputs[r0 + i] = z;
      inputs[r1 + i] = -z;
      masks[r0 + i] = m;
      masks[r1 + i] = m;
    }
    stats.push({ mu, sigma, isPositive });
  });

  return { inputs, masks, stats };
}
