import { backtest, calibrationScale, Z80, type ForecastFn } from "src/anomaly/backtest";
import type { Forecast } from "src/types";

/** Fake model: median = last observed value of the context, band = median ± Z80 (sigma 1). */
function fake(width = Z80) {
  const calls: { contexts: (number | null)[][]; horizon: number }[] = [];
  const fn: ForecastFn = async (contexts, { horizon }) => {
    calls.push({ contexts, horizon });
    return contexts.map((c) => {
      let last = 0;
      for (let i = c.length - 1; i >= 0; i--) {
        const v = c[i];
        if (v !== null && Number.isFinite(v)) {
          last = v;
          break;
        }
      }
      const med = Array(horizon).fill(last);
      const f = { horizon, median: med, q50: med, q10: med.map((v) => v - width), q90: med.map((v) => v + width) } as unknown as Forecast;
      return f;
    });
  };
  return { fn, calls };
}

const ramp = (n: number) => Array.from({ length: n }, (_, i) => i);

describe("backtest", () => {
  it("cuts windows from the end, newest first, capped, in one batched call", async () => {
    const { fn, calls } = fake();
    await backtest(ramp(100), fn, { h: 10, windows: 3, minContext: 16, maxContext: 512 });
    expect(calls).toHaveLength(1);
    expect(calls[0].horizon).toBe(10);
    expect(calls[0].contexts.map((c) => c.length)).toEqual([90, 80, 70]);
  });

  it("assigns each point its window and step, and slices contexts to maxContext", async () => {
    const { fn, calls } = fake();
    const pts = await backtest(ramp(50), fn, { h: 5, windows: 100, minContext: 20, maxContext: 8 });
    expect(Math.max(...calls[0].contexts.map((c) => c.length))).toBe(8);
    const p = pts.find((x) => x.index === 47)!;
    expect(p.step).toBe(3); // window starting at 45
    expect(p.median).toBe(44); // last value before the cut
    expect(p.r).toBeCloseTo(3); // (47 - 44) / sigma 1
    expect(pts[0].index).toBe(20); // oldest scored point = oldest cut
  });

  it("skips missing actuals and keeps scores finite", async () => {
    const v: (number | null)[] = ramp(40);
    v[35] = null;
    const { fn } = fake();
    const pts = await backtest(v, fn, { h: 4, windows: 100, minContext: 16, maxContext: 512 });
    expect(pts.some((p) => p.index === 35)).toBe(false);
    expect(pts.every((p) => Number.isFinite(p.r))).toBe(true);
  });

  it("handles a constant series with zero-width bands", async () => {
    const { fn } = fake(0);
    const pts = await backtest(Array(40).fill(5), fn, { h: 4, windows: 100, minContext: 16, maxContext: 512 });
    expect(pts.every((p) => p.r === 0)).toBe(true);
  });

  it("scores a spike at the very last point", async () => {
    const v = Array(60).fill(10);
    v[59] = 30;
    const { fn } = fake();
    const pts = await backtest(v, fn, { h: 7, windows: 100, minContext: 16, maxContext: 512 });
    expect(pts.at(-1)!.index).toBe(59);
    expect(pts.at(-1)!.r).toBeCloseTo(20);
  });
});

describe("calibrationScale", () => {
  const pt = (r: number) => ({ index: 0, value: 0, median: 0, q10: 0, q90: 0, step: 1, sigma: 1, r });
  it("is 1 with fewer than 10 points", () => {
    expect(calibrationScale([pt(5), pt(5)])).toBe(1);
  });
  it("is the 0.8 quantile of |r| divided by Z80", () => {
    const pts = Array.from({ length: 11 }, (_, i) => pt(i % 2 ? i : -i)); // |r| = 0..10
    expect(calibrationScale(pts)).toBeCloseTo(8 / Z80);
  });
});

describe("backtest with leading gaps", () => {
  it("starts cuts minContext observed points in", async () => {
    const { fn } = fake();
    const v: (number | null)[] = [...Array(50).fill(null), ...Array.from({ length: 60 }, (_, i) => i)];
    const pts = await backtest(v, fn, { h: 5, windows: 100, minContext: 20, maxContext: 512 });
    expect(Math.min(...pts.map((p) => p.index))).toBe(70);
  });
});

