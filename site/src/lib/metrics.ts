/** Mean absolute error over the steps both arrays share; missing actuals are skipped. */
export function mae(pred: number[], actual: (number | null)[]): number {
  let sum = 0;
  let n = 0;
  for (let i = 0; i < Math.min(pred.length, actual.length); i++) {
    const a = actual[i];
    if (a === null || !Number.isFinite(a)) continue;
    sum += Math.abs(pred[i] - a);
    n++;
  }
  return n ? sum / n : NaN;
}

export const naiveForecast = (train: number[], h: number): number[] =>
  Array.from({ length: h }, () => train[train.length - 1]);

/** Repeats the last full season; falls back to the naive forecast when history is too short. */
export function seasonalNaive(train: number[], h: number, season: number): number[] {
  if (season < 2 || train.length < season) return naiveForecast(train, h);
  const start = train.length - season;
  return Array.from({ length: h }, (_, i) => train[start + (i % season)]);
}

/** Share of actuals inside [lo, hi], inclusive, over the shared steps. */
export function coverage(lo: number[], hi: number[], actual: (number | null)[]): number {
  let inside = 0;
  let n = 0;
  for (let i = 0; i < Math.min(lo.length, hi.length, actual.length); i++) {
    const a = actual[i];
    if (a === null || !Number.isFinite(a)) continue;
    if (a >= lo[i] && a <= hi[i]) inside++;
    n++;
  }
  return n ? inside / n : NaN;
}

/** 1 - model/baseline: 0.3 means 30% less error than the baseline, negative means worse. */
export const skill = (modelMae: number, baselineMae: number): number =>
  baselineMae > 0 ? 1 - modelMae / baselineMae : NaN;

export interface BacktestScore {
  steps: number;
  mae: number;
  baselineMae: number;
  baseline: "last value" | "same point last season";
  skill: number;
  coverage: number;
  /** The baseline's own forecast for the scored steps, for drawing. */
  baselineValues: number[];
}

export function scoreBacktest(args: {
  train: number[];
  test: (number | null)[];
  median: number[];
  q10: number[];
  q90: number[];
  season?: number;
}): BacktestScore {
  const steps = Math.min(args.test.length, args.median.length);
  const test = args.test.slice(0, steps);
  const seasonal = !!args.season && args.season >= 2 && args.train.length >= args.season;
  const base = seasonal
    ? seasonalNaive(args.train, steps, args.season!)
    : naiveForecast(args.train, steps);
  const m = mae(args.median, test);
  const b = mae(base, test);
  return {
    steps,
    mae: m,
    baselineMae: b,
    baseline: seasonal ? "same point last season" : "last value",
    skill: skill(m, b),
    coverage: coverage(args.q10, args.q90, test),
    baselineValues: base,
  };
}
