import { CONTEXT, EPS } from "src/models/chronos-bolt/constants";
import type { Series } from "src/types";

export interface Scaling {
  loc: number;
  scale: number;
}

/** The last CONTEXT points as float64. Null, undefined and non-finite values become NaN (missing). */
export function lastContext(series: Series): Float64Array {
  const start = Math.max(0, series.length - CONTEXT);
  const out = new Float64Array(series.length - start);
  for (let i = start; i < series.length; i++) {
    const v = series[i];
    out[i - start] = v === null || v === undefined || !Number.isFinite(v) ? NaN : v;
  }
  return out;
}

/** chronos_bolt.py InstanceNorm: mean and population std over observed values. */
export function instanceNorm(x: Float64Array): Scaling {
  let n = 0;
  let sum = 0;
  for (const v of x) {
    if (Number.isNaN(v)) continue;
    n++;
    sum += v;
  }
  if (n === 0) return { loc: 0, scale: 1 };
  const loc = sum / n;
  let ss = 0;
  for (const v of x) if (!Number.isNaN(v)) ss += (v - loc) ** 2;
  const scale = Math.sqrt(ss / n);
  return { loc, scale: scale === 0 ? EPS : scale };
}

/**
 * Normalizes each row on its own and writes it, left-padded, into fixed
 * [rows, CONTEXT] context and mask buffers. Missing and padded positions are 0 with mask 0.
 */
export function packBatch(rows: Float64Array[]): { context: Float32Array; mask: Float32Array; stats: Scaling[] } {
  const context = new Float32Array(rows.length * CONTEXT);
  const mask = new Float32Array(rows.length * CONTEXT);
  const stats = rows.map((full, r) => {
    const x = full.length > CONTEXT ? full.subarray(full.length - CONTEXT) : full;
    const s = instanceNorm(x);
    const base = r * CONTEXT + CONTEXT - x.length;
    for (let i = 0; i < x.length; i++) {
      if (Number.isNaN(x[i])) continue;
      context[base + i] = (x[i] - s.loc) / s.scale;
      mask[base + i] = 1;
    }
    return s;
  });
  return { context, mask, stats };
}
