import type { AnomalyPoint } from "forecast-web";
import { anomalySentence, calibrationSentence, flagged } from "src/lib/anomaly";

const pt = (o: Partial<AnomalyPoint>): AnomalyPoint => ({ index: 0, value: 138, expected: 100, lower: 90, upper: 110, score: 4.2, direction: "high", anomaly: true, ...o });

describe("anomalySentence", () => {
  it("says how far above or below expected, with the score", () => {
    expect(anomalySentence(pt({}), "2026-03-03", "")).toBe("Mar 3, 2026: 38% above expected, score 4.2");
    expect(anomalySentence(pt({ value: 50, direction: "low", score: 6.04 }), "2026-03", "")).toBe("Mar 2026: 50% below expected, score 6.0");
  });

  it("falls back to the difference when the expected value is near zero or the ratio is huge", () => {
    expect(anomalySentence(pt({ value: 12, expected: 0 }), "Week 3", "°C")).toBe("Week 3: 12.0 °C above expected, score 4.2");
    expect(anomalySentence(pt({ value: 5000, expected: 10 }), "x", "")).toBe("x: 4990 above expected, score 4.2");
  });
});

describe("flagged", () => {
  it("applies a threshold to every point and lists newest first", () => {
    const pts = [pt({ index: 1, score: 2 }), pt({ index: 2, score: 3.5 }), pt({ index: 3, score: 5 })];
    expect(flagged(pts, 3).map((p) => p.index)).toEqual([3, 2]);
    expect(flagged(pts, 6)).toEqual([]);
  });
});

describe("calibrationSentence", () => {
  it("describes widening, narrowing, no change and too little history", () => {
    expect(calibrationSentence({ scale: 1.42, coverage: 0.62, points: 120 })).toBe("Bands widened 1.4x: the raw 80% range held 62% of past values");
    expect(calibrationSentence({ scale: 0.7, coverage: 0.95, points: 120 })).toBe("Bands narrowed to 0.7x: the raw 80% range held 95% of past values");
    expect(calibrationSentence({ scale: 1, coverage: 0.84, points: 120 })).toBe("Bands unchanged: the raw 80% range already held 84% of past values");
    expect(calibrationSentence({ scale: 1, coverage: null, points: 3 })).toBe("Bands unchanged: too little history to check them");
  });
});
