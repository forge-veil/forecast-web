/** Fixed shape of the exported Chronos-Bolt graphs. */
export const CONTEXT = 2048;
export const PATCH = 16;
/** Steps per forward pass. */
export const BLOCK = 64;
export const QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] as const;
export const NQ = QUANTILES.length;
/** Longest horizon served by repeated passes; every pass after the first runs 9x the batch. */
export const MAX_HORIZON = 512;
/** InstanceNorm's floor for a zero scale, as in chronos_bolt.py. */
export const EPS = 1e-5;
