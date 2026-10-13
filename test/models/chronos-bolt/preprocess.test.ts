import { CONTEXT, EPS } from "src/models/chronos-bolt/constants";
import { instanceNorm, lastContext, packBatch } from "src/models/chronos-bolt/preprocess";

describe("lastContext", () => {
  it("keeps the last 2048 points and marks non-finite values as NaN", () => {
    const long = Array.from({ length: 3000 }, (_, i) => i);
    const x = lastContext(long);
    expect(x.length).toBe(CONTEXT);
    expect(x[0]).toBe(952);
    expect(Array.from(lastContext([1, null, undefined, NaN, Infinity, 2]))).toEqual([1, NaN, NaN, NaN, NaN, 2]);
  });
});

describe("instanceNorm", () => {
  it("uses the mean and population std of observed values", () => {
    const { loc, scale } = instanceNorm(Float64Array.from([1, NaN, 3]));
    expect(loc).toBe(2);
    expect(scale).toBe(1);
  });

  it("floors a zero scale at 1e-5", () => {
    expect(instanceNorm(Float64Array.from([5, 5, 5]))).toEqual({ loc: 5, scale: EPS });
  });

  it("uses loc 0 and scale 1 when nothing is observed", () => {
    expect(instanceNorm(Float64Array.from([NaN, NaN]))).toEqual({ loc: 0, scale: 1 });
  });
});

describe("packBatch", () => {
  it("left-pads each row on its own and masks missing values", () => {
    const { context, mask, stats } = packBatch([Float64Array.from([1, NaN, 3]), Float64Array.from([10, 20])]);
    expect(context.length).toBe(2 * CONTEXT);
    expect(stats).toEqual([{ loc: 2, scale: 1 }, { loc: 15, scale: 5 }]);
    const r0 = context.subarray(CONTEXT - 3, CONTEXT);
    expect(Array.from(r0)).toEqual([-1, 0, 1]);
    expect(Array.from(mask.subarray(CONTEXT - 3, CONTEXT))).toEqual([1, 0, 1]);
    expect(mask[CONTEXT - 4]).toBe(0);
    expect(Array.from(context.subarray(2 * CONTEXT - 2))).toEqual([-1, 1]);
  });

  it("truncates rows longer than the context to their last 2048 points", () => {
    const row = Float64Array.from({ length: 2100 }, (_, i) => i);
    const { mask, stats } = packBatch([row]);
    expect(mask.every((m) => m === 1)).toBe(true);
    expect(stats[0].loc).toBeCloseTo((52 + 2099) / 2);
  });

  it("handles a row shorter than one patch and an all-missing row", () => {
    const { mask, stats } = packBatch([Float64Array.from([4, 5, 6]), Float64Array.from([NaN])]);
    expect(mask.subarray(0, CONTEXT).reduce((a, b) => a + b, 0)).toBe(3);
    expect(mask.subarray(CONTEXT).reduce((a, b) => a + b, 0)).toBe(0);
    expect(stats[1]).toEqual({ loc: 0, scale: 1 });
  });
});
