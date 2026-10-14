import { fminbound } from "src/tabular/fminbound";

const EPS = 2.220446049250313e-16;
const TINY = 2.2250738585072014e-308;
const MAX = 1.7976931348623157e308;

const mean = (x: ArrayLike<number>) => {
  let s = 0;
  for (let i = 0; i < x.length; i++) s += x[i];
  return s / x.length;
};

/** np.var (population). */
export function variance(x: ArrayLike<number>): number {
  const m = mean(x);
  let s = 0;
  for (let i = 0; i < x.length; i++) s += (x[i] - m) ** 2;
  return s / x.length;
}

/** scipy.special.logsumexp with weights; returns log|sum(b * exp(a))|. */
function logsumexp(a: ArrayLike<number>, b?: ArrayLike<number>): number {
  let m = -Infinity;
  for (let i = 0; i < a.length; i++) if (a[i] > m) m = a[i];
  if (!Number.isFinite(m)) m = 0;
  let s = 0;
  for (let i = 0; i < a.length; i++) s += (b ? b[i] : 1) * Math.exp(a[i] - m);
  return Math.log(Math.abs(s)) + m;
}

/** scipy.stats._morestats._log_var: log of the variance of x, from log(x). */
function logVar(logx: Float64Array): number {
  const n = logx.length;
  const logmean = logsumexp(logx) - Math.log(n);
  const logxmu = new Float64Array(n);
  for (let i = 0; i < n; i++) logxmu[i] = 2 * logsumexp([logx[i], logmean], [1, -1]);
  return logsumexp(logxmu) - Math.log(n);
}

/** scipy.stats._morestats._yeojohnson_transform. */
export function yeoJohnson(x: ArrayLike<number>, lmbda: number): Float64Array {
  const out = new Float64Array(x.length);
  for (let i = 0; i < x.length; i++) {
    const v = x[i];
    if (v >= 0) out[i] = Math.abs(lmbda) < EPS ? Math.log1p(v) : Math.expm1(lmbda * Math.log1p(v)) / lmbda;
    else out[i] = Math.abs(lmbda - 2) > EPS ? -Math.expm1((2 - lmbda) * Math.log1p(-v)) / (2 - lmbda) : -Math.log1p(-v);
  }
  return out;
}

/** scipy.stats._morestats._yeojohnson_llf for 1-D data. */
function llf(data: Float64Array, lmb: number): number {
  const n = data.length;
  let allPos = true;
  let allNeg = true;
  for (const v of data) {
    if (v >= 0) allNeg = false;
    else allPos = false;
  }
  let logvar: number;
  if (allPos) {
    const l = data.map((v) => Math.log1p(v));
    logvar = Math.abs(lmb) < EPS ? Math.log(variance(l)) : logVar(l.map((v) => lmb * v)) - 2 * Math.log(Math.abs(lmb));
  } else if (allNeg) {
    const l = data.map((v) => Math.log1p(-v));
    logvar = Math.abs(lmb - 2) < EPS ? Math.log(variance(l)) : logVar(l.map((v) => (2 - lmb) * v)) - 2 * Math.log(Math.abs(2 - lmb));
  } else {
    const sigma = variance(yeoJohnson(data, lmb));
    logvar = sigma >= TINY ? Math.log(sigma) : -Infinity;
  }
  let s = 0;
  for (const v of data) s += Math.sign(v) * Math.log1p(Math.abs(v));
  return (-n / 2) * logvar + (lmb - 1) * s;
}

/** scipy.stats.yeojohnson_normmax (no brack), as used by scikit-learn 1.9 PowerTransformer. */
export function yeoJohnsonLambda(column: ArrayLike<number>): number {
  const x = Float64Array.from(Array.from(column).filter((v) => !Number.isNaN(v)));
  if (x.every((v) => v === 0)) return 1.0;
  const negLlf = (l: number) => {
    const v = llf(x, l);
    return Number.isFinite(v) || Number.isNaN(v) ? -v : Infinity;
  };
  let maxAbs = 0;
  for (const v of x) maxAbs = Math.max(maxAbs, Math.abs(v));
  const log1pMax = Math.log1p(20 * maxAbs);
  const logEps = Math.log(EPS);
  const logTiny = (Math.log(TINY) - logEps) / 2;
  const logMax = (Math.log(MAX) + logEps) / 2;
  let lb = logTiny / log1pMax;
  let ub = logMax / log1pMax;
  if (x.every((v) => v < 0)) [lb, ub] = [2 - ub, 2 - lb];
  else if (x.some((v) => v < 0)) [lb, ub] = [Math.max(2 - ub, lb), Math.min(2 - lb, ub)];
  return fminbound(negLlf, lb, ub, 1.48e-8);
}
