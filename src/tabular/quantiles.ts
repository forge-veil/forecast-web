const TOL = 1e-6;
const MIN_BETA = 0.01;
const MAX_BETA = 100;
const TAIL = 20;
const f = Math.fround;

/**
 * torch.linspace(0, 1, 1001, dtype=float32)[1:-1]: a float32 step, each value
 * computed in double and rounded to float32, from the start below the integer
 * halfway point and from the end above it (matches torch bit for bit).
 */
export const ALPHAS999 = (() => {
  const steps = 1001;
  const step = f(1 / (steps - 1));
  const halfway = Math.floor(steps / 2);
  const out = new Float32Array(999);
  for (let i = 1; i < steps - 1; i++) out[i - 1] = f(i < halfway ? i * step : 1 - (steps - 1 - i) * step);
  return out;
})();

const mean = (x: ArrayLike<number>) => {
  let s = 0;
  for (let i = 0; i < x.length; i++) s += x[i];
  return s / x.length;
};

/** estimate_exp_tail_params: slopes of Q against log(alpha) on the 20 outermost quantiles. */
function tailBetas(q: Float64Array): [number, number] {
  const k = Math.min(TAIL, Math.floor(q.length / 4));
  const fit = (xs: number[], ys: number[]) => {
    const xm = mean(xs);
    const ym = mean(ys);
    let cov = 0;
    let v = 0;
    for (let i = 0; i < xs.length; i++) {
      cov += (ys[i] - ym) * (xs[i] - xm);
      v += (xs[i] - xm) ** 2;
    }
    return cov / xs.length / Math.max(v / xs.length, TOL);
  };
  const left = fit(Array.from(ALPHAS999.slice(0, k), (a) => Math.log(Math.max(a, TOL))), Array.from(q.slice(0, k)));
  const right = -fit(Array.from(ALPHAS999.slice(-k), (a) => Math.log(Math.max(1 - a, TOL))), Array.from(q.slice(-k)));
  const clamp = (b: number) => Math.min(MAX_BETA, Math.max(MIN_BETA, Math.abs(b)));
  return [clamp(left), clamp(right)];
}

/**
 * QuantileDistribution (exp tails, sort for crossing) over one row of 999 raw
 * quantiles: mean of the raw values, and icdf at the requested alphas.
 */
export function quantileStats(raw: ArrayLike<number>, alphas: number[]): { mean: number; median: number; quantiles: number[] } {
  const q = Float64Array.from(raw).sort();
  const [betaL, betaR] = tailBetas(q);
  const aL = ALPHAS999[0];
  const aR = ALPHAS999[998];
  const tailBL = q[0] - betaL * Math.log(Math.max(aL, TOL));
  const tailAR = -betaR;
  const tailBR = q[998] - tailAR * Math.log(1 - Math.min(aR, 1 - TOL));
  const icdf = (a: number) => {
    if (a < aL) return betaL * Math.log(Math.max(a, TOL)) + tailBL;
    if (a > aR) return tailAR * Math.log(Math.max(1 - a, TOL)) + tailBR;
    // searchsorted(alpha_lo, a, right=True) - 1, clamped to a segment.
    let lo = 0;
    let hi = 998;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (ALPHAS999[mid] <= a) lo = mid + 1;
      else hi = mid;
    }
    const seg = Math.min(997, Math.max(0, lo - 1));
    const t = (a - ALPHAS999[seg]) / Math.max(ALPHAS999[seg + 1] - ALPHAS999[seg], TOL);
    return q[seg] + Math.min(1, Math.max(0, t)) * (q[seg + 1] - q[seg]);
  };
  return { mean: mean(raw), median: icdf(0.5), quantiles: alphas.map(icdf) };
}
