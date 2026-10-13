import { formatMB, formatNumber, niceTicks, shortLabel } from "src/lib/format";

describe("formatNumber", () => {
  it.each([
    [654831, "654.8K"],
    [1234, "1234"],
    [426.07, "426"],
    [12.345, "12.3"],
    [1.2345, "1.23"],
    [0.000123, "0.000123"],
    [0, "0"],
    [NaN, "–"],
  ])("%s -> %s", (v, s) => expect(formatNumber(v)).toBe(s));
});

describe("shortLabel", () => {
  it("shortens months, days and hours", () => {
    expect(shortLabel("2026-08")).toBe("Aug 2026");
    expect(shortLabel("2026-10-09")).toBe("Oct 9, 2026");
    expect(shortLabel("2026-10-12 07:00")).toBe("Oct 12 07:00");
    expect(shortLabel("+3")).toBe("+3");
  });
});

describe("niceTicks", () => {
  it("covers the range with round steps", () => {
    expect(niceTicks(0, 10)).toEqual([0, 2, 4, 6, 8, 10]);
    expect(niceTicks(312, 431, 4)).toEqual([320, 340, 360, 380, 400, 420]);
  });

  it("handles a flat range", () => {
    expect(niceTicks(5, 5)).toEqual([5]);
  });
});

describe("formatMB", () => {
  it("shows decimal megabytes, or nothing when unknown", () => {
    expect(formatMB(463_368_499)).toBe("463 MB");
    expect(formatMB(9_995_263)).toBe("10 MB");
    expect(formatMB(undefined)).toBe("");
  });
});
