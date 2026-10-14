import type { Norm } from "src/tabular/ensemble";
import { variance, yeoJohnson, yeoJohnsonLambda } from "src/tabular/yeojohnson";

const EPS = 2.220446049250313e-16;
type Cols = Float64Array[];

const mean = (x: ArrayLike<number>) => {
  let s = 0;
  for (let i = 0; i < x.length; i++) s += x[i];
  return s / x.length;
};

/** UniqueFeatureFilter(threshold=1): indices of columns with more than one distinct value. */
export function fitUniqueFilter(cols: Cols): number[] {
  const n = cols[0]?.length ?? 0;
  if (n <= 1) return cols.map((_, i) => i);
  const keep = cols.map((c, i) => (new Set(c).size > 1 ? i : -1)).filter((i) => i >= 0);
  return keep.length ? keep : [0];
}

/** CustomStandardScaler: (x - mean) / (std + 1e-6), clipped to [-100, 100]. */
export function standardScale(cols: Cols): { mean: number[]; scale: number[]; train: Cols; transform(c: Cols): Cols } {
  const m = cols.map(mean);
  const s = cols.map((c) => Math.sqrt(variance(c)) + 1e-6);
  const transform = (cs: Cols) => cs.map((c, j) => c.map((v) => Math.min(100, Math.max(-100, (v - m[j]) / s[j]))));
  return { mean: m, scale: s, train: transform(cols), transform };
}

/** sklearn _is_constant_feature. */
const isConstant = (v: number, m: number, n: number) => v <= n * EPS * v + (n * m * EPS) ** 2;

/** sklearn _incremental_mean_and_var for one batch: mean and variance with the correction term. */
function meanVar(c: Float64Array): { m: number; v: number } {
  const n = c.length;
  const m = mean(c);
  let sq = 0;
  let corr = 0;
  for (const x of c) {
    const t = x - m;
    sq += t * t;
    corr += t;
  }
  return { m, v: (sq - (corr * corr) / n) / n };
}

/** PowerTransformer(method="yeo-johnson", standardize=True), sklearn 1.9 semantics. */
function fitPower(cols: Cols): { lambdas: number[]; train: Cols; transform(c: Cols): Cols } {
  const n = cols[0].length;
  const lambdas: number[] = [];
  const transformed = cols.map((c) => {
    if (isConstant(variance(c), mean(c), n)) {
      lambdas.push(1.0);
      return Float64Array.from(c);
    }
    const l = yeoJohnsonLambda(c);
    lambdas.push(l);
    return yeoJohnson(c, l);
  });
  const stats = transformed.map(meanVar);
  const scale = stats.map(({ m, v }) => (isConstant(v, m, n) ? 1 : Math.sqrt(v)));
  const standardize = (cs: Cols) => cs.map((c, j) => c.map((x) => (x - stats[j].m) / scale[j]));
  return {
    lambdas,
    train: standardize(transformed),
    transform: (cs) => standardize(cs.map((c, j) => yeoJohnson(c, lambdas[j]))),
  };
}

/** OutlierRemover(threshold): two-stage bounds, then a soft log clip. */
function fitOutliers(cols: Cols, threshold = 4.0): { lower: number[]; upper: number[]; transform(c: Cols): Cols } {
  const n = cols[0].length;
  const ddof = n > 1 ? 1 : 0;
  const stats = (c: ArrayLike<number>) => {
    const vals = Array.from(c).filter((v) => !Number.isNaN(v));
    const m = mean(vals);
    let s = 0;
    for (const v of vals) s += (v - m) ** 2;
    return { m, sd: Math.max(Math.sqrt(s / (vals.length - ddof)), 1e-6) };
  };
  const lower: number[] = [];
  const upper: number[] = [];
  for (const c of cols) {
    const first = stats(c);
    const lo = first.m - threshold * first.sd;
    const hi = first.m + threshold * first.sd;
    const second = stats(c.map((v) => (v < lo || v > hi ? NaN : v)));
    lower.push(second.m - threshold * second.sd);
    upper.push(second.m + threshold * second.sd);
  }
  const transform = (cs: Cols) =>
    cs.map((c, j) =>
      c.map((v) => {
        const a = Math.max(-Math.log1p(Math.abs(v)) + lower[j], v);
        return Math.min(Math.log1p(Math.abs(a)) + upper[j], a);
      }),
    );
  return { lower, upper, transform };
}

export interface Pipeline {
  trainTransformed: Cols;
  transform(cols: Cols): Cols;
}

/** PreprocessingPipeline: standard scale, optional Yeo-Johnson, outlier squash. Fit on training columns. */
export function fitPipeline(cols: Cols, norm: Norm, outlierThreshold = 4.0): Pipeline {
  const scaler = standardScale(cols);
  let normalized = scaler.train;
  let normalize: (c: Cols) => Cols = (c) => c;
  if (norm === "power") {
    const min = scaler.train.map((c) => Math.min(...c));
    const max = scaler.train.map((c) => Math.max(...c));
    const power = fitPower(scaler.train);
    normalized = power.train;
    normalize = (c) => {
      const out = power.transform(c);
      // Upstream retries with values clipped to the training range when the transform fails.
      if (out.every((col) => col.every(Number.isFinite))) return out;
      return power.transform(c.map((col, j) => col.map((v) => Math.min(max[j], Math.max(min[j], v)))));
    };
  }
  const outliers = fitOutliers(normalized, outlierThreshold);
  return {
    trainTransformed: outliers.transform(normalized),
    transform: (c) => outliers.transform(normalize(scaler.transform(c))),
  };
}
