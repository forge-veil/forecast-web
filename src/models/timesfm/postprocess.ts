import { CHANNELS, MAX_HORIZON, type SeriesStats } from "src/models/timesfm/constants";
import type { Forecast } from "src/types";

const MEDIAN = 5;
const LOWER = [4, 3, 2, 1];
const UPPER = [6, 7, 8, 9];
const SPREAD = [1, 2, 3, 4, 6, 7, 8, 9];
/** flip_quantile_fn: channel 0 stays, channel q swaps with 10 - q. */
const flip = (q: number) => (q === 0 ? 0 : CHANNELS - q);

/**
 * Turns raw graph outputs ([2B, 128, 10] each) into forecasts, mirroring the
 * TimesFM 2.5 compiled decode: flip-invariance average, continuous quantile
 * head, quantile crossing fix, renormalization, positivity clamp.
 */
export function postprocess(
  point: Float32Array,
  qspread: Float32Array,
  stats: SeriesStats[],
  horizon: number,
  graphHorizon = MAX_HORIZON,
): Forecast[] {
  const rowLen = graphHorizon * CHANNELS;
  const row = new Float64Array(CHANNELS);
  const qs = new Float64Array(CHANNELS);

  return stats.map(({ mu, sigma, isPositive }, b) => {
    const r0 = 2 * b * rowLen;
    const r1 = r0 + rowLen;
    const ch: number[][] = Array.from({ length: CHANNELS }, () => new Array(horizon));

    for (let t = 0; t < horizon; t++) {
      const o = t * CHANNELS;
      for (let q = 0; q < CHANNELS; q++) {
        row[q] = (point[r0 + o + q] - point[r1 + o + flip(q)]) / 2;
        qs[q] = (qspread[r0 + o + q] - qspread[r1 + o + flip(q)]) / 2;
      }
      for (const q of SPREAD) row[q] = qs[q] - qs[MEDIAN] + row[MEDIAN];
      for (const q of LOWER) row[q] = Math.min(row[q], row[q + 1]);
      for (const q of UPPER) row[q] = Math.max(row[q], row[q - 1]);
      for (let q = 0; q < CHANNELS; q++) {
        const y = row[q] * sigma + mu;
        ch[q][t] = isPositive ? Math.max(y, 0) : y;
      }
    }

    return {
      horizon,
      mean: ch[0],
      median: ch[5],
      q10: ch[1],
      q20: ch[2],
      q30: ch[3],
      q40: ch[4],
      q50: ch[5],
      q60: ch[6],
      q70: ch[7],
      q80: ch[8],
      q90: ch[9],
    };
  });
}
