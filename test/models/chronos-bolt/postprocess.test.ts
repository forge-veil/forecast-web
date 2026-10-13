import { quantileLinear } from "src/models/chronos-bolt/postprocess";

describe("quantileLinear (torch.quantile, linear)", () => {
  const sorted = Float64Array.from({ length: 81 }, (_, i) => i * 2); // 0, 2, ..., 160

  it("hits sample points when q*(n-1) is an integer", () => {
    expect(quantileLinear(sorted, 0.5)).toBe(80);
    expect(quantileLinear(sorted, 0.1)).toBeCloseTo(16, 10);
    expect(quantileLinear(sorted, 0.9)).toBeCloseTo(144, 10);
  });

  it("interpolates between neighbors", () => {
    expect(quantileLinear([0, 10], 0.25)).toBe(2.5);
    expect(quantileLinear([1, 2, 4], 0.75)).toBe(3);
  });

  it("returns the ends for q = 0 and q = 1", () => {
    expect(quantileLinear([3, 7, 9], 0)).toBe(3);
    expect(quantileLinear([3, 7, 9], 1)).toBe(9);
  });
});
