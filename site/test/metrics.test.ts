import { coverage, mae, naiveForecast, scoreBacktest, seasonalNaive, skill } from "src/lib/metrics";

describe("mae", () => {
  it("averages absolute errors over the overlapping steps", () => {
    expect(mae([1, 2, 3, 99], [2, 2, 5])).toBe(1);
  });

  it("skips missing actuals", () => {
    expect(mae([1, 2], [2, null])).toBe(1);
  });

  it("is NaN with nothing to compare", () => {
    expect(mae([1], [])).toBeNaN();
  });
});

describe("baselines", () => {
  it("naive repeats the last value", () => {
    expect(naiveForecast([1, 2, 7], 3)).toEqual([7, 7, 7]);
  });

  it("seasonal naive repeats the last season", () => {
    expect(seasonalNaive([1, 2, 3, 10, 20, 30], 5, 3)).toEqual([10, 20, 30, 10, 20]);
  });

  it("seasonal naive falls back to naive when history is shorter than a season", () => {
    expect(seasonalNaive([1, 2], 2, 12)).toEqual([2, 2]);
  });
});

describe("coverage", () => {
  it("is the share of actuals inside the band, inclusive", () => {
    expect(coverage([0, 0, 0, 0], [2, 2, 2, 2], [1, 2, 3, -1])).toBe(0.5);
  });
});

describe("skill", () => {
  it("is positive when the model beats the baseline", () => {
    expect(skill(1, 4)).toBe(0.75);
    expect(skill(4, 2)).toBe(-1);
  });

  it("is NaN when the baseline is perfect", () => {
    expect(skill(1, 0)).toBeNaN();
  });
});

describe("scoreBacktest", () => {
  it("scores only the steps the forecast and holdout share", () => {
    const s = scoreBacktest({
      train: [1, 2, 3, 4],
      test: [5, 6],
      median: [5, 7, 100],
      q10: [4, 4, 0],
      q90: [6, 6.5, 0],
    });
    expect(s.steps).toBe(2);
    expect(s.mae).toBe(0.5);
    expect(s.baselineMae).toBe(1.5); // naive 4,4 vs 5,6
    expect(s.skill).toBeCloseTo(1 - 0.5 / 1.5);
    expect(s.coverage).toBe(1);
    expect(s.baseline).toBe("last value");
    expect(s.baselineValues).toEqual([4, 4]);
  });

  it("uses seasonal naive when a season length fits the history", () => {
    const s = scoreBacktest({
      train: [1, 2, 1, 2],
      test: [1, 2],
      median: [1, 2],
      q10: [0, 0],
      q90: [3, 3],
      season: 2,
    });
    expect(s.baseline).toBe("same point last season");
    expect(s.baselineMae).toBe(0);
    expect(s.baselineValues).toEqual([1, 2]);
  });
});
