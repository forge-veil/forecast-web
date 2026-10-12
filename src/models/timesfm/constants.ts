/** Fixed shape of the exported TimesFM 2.5 graph. */
export const CONTEXT = 512;
export const MAX_HORIZON = 128;
/** Output channels: 0 = mean head, 1..9 = q10..q90. */
export const CHANNELS = 10;
export const QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] as const;

export interface SeriesStats {
  mu: number;
  sigma: number;
  isPositive: boolean;
}
