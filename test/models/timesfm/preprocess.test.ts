import { cleanSeries, prepareBatch } from "src/models/timesfm/preprocess";
import { CONTEXT } from "src/models/timesfm/constants";

const arr = (a: Float64Array | Float32Array) => Array.from(a);

describe("cleanSeries", () => {
  it("strips leading missing values", () => {
    expect(arr(cleanSeries([NaN, null, undefined, 1, 2]))).toEqual([1, 2]);
  });

  it("linearly interpolates interior gaps without shifting the timeline", () => {
    expect(arr(cleanSeries([0, NaN, NaN, 3, null, 5]))).toEqual([0, 1, 2, 3, 4, 5]);
  });

  it("holds the last value flat across trailing gaps", () => {
    expect(arr(cleanSeries([1, 2, NaN, null]))).toEqual([1, 2, 2, 2]);
  });

  it("treats Infinity as missing", () => {
    expect(arr(cleanSeries([1, Infinity, 3]))).toEqual([1, 2, 3]);
  });

  it("turns an all-missing series into zeros of the same length", () => {
    expect(arr(cleanSeries([NaN, null, NaN]))).toEqual([0, 0, 0]);
  });

  it("keeps an empty series empty", () => {
    expect(cleanSeries([]).length).toBe(0);
  });
});

describe("prepareBatch", () => {
  it("builds a doubled batch: normalized row then negated row", () => {
    const { inputs, masks, stats } = prepareBatch([[1, 2, 3]]);
    expect(inputs.length).toBe(2 * CONTEXT);
    expect(masks.length).toBe(2 * CONTEXT);
    for (let i = 0; i < CONTEXT; i++) {
      expect(inputs[CONTEXT + i]).toBeCloseTo(-inputs[i], 6);
      expect(masks[CONTEXT + i]).toBe(masks[i]);
    }
    expect(stats).toHaveLength(1);
  });

  it("left-pads short series with zeros and marks padding in the mask", () => {
    const { masks } = prepareBatch([[1, 2, 3]]);
    for (let i = 0; i < CONTEXT - 3; i++) expect(masks[i]).toBe(1);
    for (let i = CONTEXT - 3; i < CONTEXT; i++) expect(masks[i]).toBe(0);
  });

  it("computes mean and unbiased std over the full padded window, like Python", () => {
    const { stats } = prepareBatch([[2, 4]]);
    const n = CONTEXT;
    const mu = 6 / n;
    const ss = (n - 2) * mu * mu + (2 - mu) ** 2 + (4 - mu) ** 2;
    expect(stats[0].mu).toBeCloseTo(mu, 12);
    expect(stats[0].sigma).toBeCloseTo(Math.sqrt(ss / (n - 1)), 12);
  });

  it("normalizes real values by (x - mu) / sigma", () => {
    const { inputs, stats } = prepareBatch([[2, 4]]);
    const { mu, sigma } = stats[0];
    expect(inputs[CONTEXT - 1]).toBeCloseTo((4 - mu) / sigma, 5);
  });

  it("keeps only the last 512 points of a long series", () => {
    const long = Array.from({ length: 700 }, (_, i) => i);
    const { masks, stats } = prepareBatch([long]);
    expect(Array.from(masks.subarray(0, CONTEXT)).every((m) => m === 0)).toBe(true);
    expect(stats[0].mu).toBeCloseTo((188 + 699) / 2, 9);
  });

  it("divides by 1 when sigma is below 1e-6, producing finite inputs", () => {
    const flat = Array.from({ length: CONTEXT }, () => 5);
    const { inputs, stats } = prepareBatch([flat]);
    expect(stats[0].sigma).toBeLessThan(1e-6);
    expect(Array.from(inputs).every(Number.isFinite)).toBe(true);
    expect(inputs[0]).toBeCloseTo(0, 9);
  });

  it("flags non-negative windows as positive, padding included", () => {
    const { stats } = prepareBatch([[0, 1, 2], [1, -1, 2]]);
    expect(stats[0].isPositive).toBe(true);
    expect(stats[1].isPositive).toBe(false);
  });

  it("places each series in its own pair of rows", () => {
    const { inputs, masks } = prepareBatch([[1, 2, 3], [10, 20]]);
    expect(inputs.length).toBe(4 * CONTEXT);
    expect(masks[2 * CONTEXT + CONTEXT - 3]).toBe(1);
    expect(masks[2 * CONTEXT + CONTEXT - 2]).toBe(0);
  });

  it("interpolates gaps before windowing, so a gap does not shift later points", () => {
    const a = prepareBatch([[1, NaN, 3, 4]]);
    const b = prepareBatch([[1, 2, 3, 4]]);
    expect(Array.from(a.inputs)).toEqual(Array.from(b.inputs));
  });
});
