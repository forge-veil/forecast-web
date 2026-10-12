import { createForecaster } from "src/models/timesfm/forecaster";
import { CHANNELS, MAX_HORIZON, type SeriesStats } from "src/models/timesfm/constants";
import { postprocess } from "src/models/timesfm/postprocess";
import { ForecastError } from "src/types";

const ROW = MAX_HORIZON * CHANNELS;

/** Graph output for one series: row 0 (normal) and row 1 (negated), same values at every step. */
function graphOut(normal: number[], negated: number[], series = 1) {
  const a = new Float32Array(2 * series * ROW);
  for (let s = 0; s < series; s++)
    for (let t = 0; t < MAX_HORIZON; t++)
      for (let q = 0; q < CHANNELS; q++) {
        a[(2 * s) * ROW + t * CHANNELS + q] = normal[q];
        a[(2 * s + 1) * ROW + t * CHANNELS + q] = negated[q];
      }
  return a;
}

const unit: SeriesStats = { mu: 0, sigma: 1, isPositive: false };
const ramp = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
const zeros = new Array(10).fill(0);

describe("postprocess", () => {
  it("averages the normal and negated runs with the quantile flip q -> 10 - q", () => {
    // Negated run mirrors the normal one exactly, so the average reproduces it.
    const neg = [-0, -9, -8, -7, -6, -5, -4, -3, -2, -1];
    const p = graphOut(ramp, neg);
    const [f] = postprocess(p, p, [unit], 1);
    expect(f.mean![0]).toBeCloseTo(0);
    expect(f.q10[0]).toBeCloseTo(1);
    expect(f.median[0]).toBeCloseTo(5);
    expect(f.q90[0]).toBeCloseTo(9);
  });

  it("keeps channel 0 unflipped", () => {
    const p = graphOut([4, ...zeros.slice(1)], [-2, ...zeros.slice(1)]);
    const [f] = postprocess(p, p, [unit], 1);
    expect(f.mean![0]).toBeCloseTo(3);
  });

  it("rebuilds quantiles from the continuous head anchored on the point median", () => {
    const point = graphOut([0, 0, 0, 0, 0, 10, 0, 0, 0, 0], zeros);
    const qs = graphOut([0, 2, 4, 6, 8, 10, 12, 14, 16, 18], zeros);
    const [f] = postprocess(point, qs, [unit], 1);
    // halved by the flip average: point median 5, spread 1..9 around qs median 5
    expect(f.median[0]).toBeCloseTo(5);
    expect(f.q10[0]).toBeCloseTo(1 - 5 + 5);
    expect(f.q90[0]).toBeCloseTo(9 - 5 + 5);
  });

  it("fixes quantile crossing outward from the median", () => {
    const qs = graphOut([0, 2, 8, 6, 8, 10, 12, 10, 16, 18], zeros);
    const point = graphOut([0, 0, 0, 0, 0, 10, 0, 0, 0, 0], zeros);
    const [f] = postprocess(point, qs, [unit], 1);
    const qsAt = [f.q10, f.q20, f.q30, f.q40, f.q50, f.q60, f.q70, f.q80, f.q90].map((q) => q[0]);
    for (let i = 1; i < qsAt.length; i++) expect(qsAt[i]).toBeGreaterThanOrEqual(qsAt[i - 1]);
    // After halving: q20 = 4 sits above q30 = 3, and q70 = 5 sits below q60 = 6.
    expect(f.q20[0]).toBeCloseTo(3);
    expect(f.q70[0]).toBeCloseTo(6);
  });

  it("renormalizes with the raw sigma and mean", () => {
    const p = graphOut([2, ...zeros.slice(1)], zeros);
    const [f] = postprocess(p, p, [{ mu: 100, sigma: 10, isPositive: false }], 1);
    expect(f.mean![0]).toBeCloseTo(100 + 1 * 10);
  });

  it("clamps at zero only for positive series", () => {
    const p = graphOut([-4, ...zeros.slice(1)], zeros);
    const [pos] = postprocess(p, p, [{ mu: 0, sigma: 1, isPositive: true }], 1);
    const [neg] = postprocess(p, p, [unit], 1);
    expect(pos.mean![0]).toBe(0);
    expect(neg.mean![0]).toBeCloseTo(-2);
  });

  it("returns exactly horizon steps per field", () => {
    const p = graphOut(ramp, zeros);
    const [f] = postprocess(p, p, [unit], 7);
    expect(f.horizon).toBe(7);
    for (const k of ["mean", "median", "q10", "q50", "q90"] as const) expect(f[k]).toHaveLength(7);
  });

  it("gives a constant forecast with no NaN for a zero-sigma series", () => {
    const p = graphOut(ramp, zeros);
    const [f] = postprocess(p, p, [{ mu: 5, sigma: 0, isPositive: true }], 3);
    expect(f.median).toEqual([5, 5, 5]);
    expect(f.q90).toEqual([5, 5, 5]);
  });

  it("reads each series from its own pair of rows", () => {
    const p = new Float32Array(4 * ROW);
    p[2 * ROW + 5] = 8; // series 1, normal row, t=0, median
    const [a, b] = postprocess(p, p, [unit, unit], 1);
    expect(a.median[0]).toBeCloseTo(0);
    expect(b.median[0]).toBeCloseTo(4);
  });
});

describe("createForecaster", () => {
  const fake = () => {
    const calls: number[] = [];
    const run = async (_i: Float32Array, _m: Float32Array, batch: number) => {
      calls.push(batch);
      return { point: new Float32Array(batch * ROW), qspread: new Float32Array(batch * ROW) };
    };
    return { calls, forecast: createForecaster(run) };
  };

  it("runs the graph once with a doubled batch and defaults to horizon 128", async () => {
    const { calls, forecast } = fake();
    const out = await forecast([[1, 2, 3], [4, 5, 6]]);
    expect(calls).toEqual([4]);
    expect(out).toHaveLength(2);
    expect(out[0].horizon).toBe(MAX_HORIZON);
  });

  it.each([[0], [-1], [1.5], [NaN]])("rejects invalid horizon %s", async (h) => {
    const { forecast } = fake();
    await expect(forecast([[1, 2]], { horizon: h })).rejects.toMatchObject({ code: "INVALID_HORIZON" });
  });

  it("rejects horizons over 128", async () => {
    const { forecast } = fake();
    const err = await forecast([[1, 2]], { horizon: 129 }).catch((e) => e);
    expect(err).toBeInstanceOf(ForecastError);
    expect(err.code).toBe("HORIZON_TOO_LONG");
  });

  it("rejects an empty series list", async () => {
    const { forecast } = fake();
    await expect(forecast([])).rejects.toMatchObject({ code: "EMPTY_INPUT" });
  });
});
