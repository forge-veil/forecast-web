import type { AnomalyPoint, Calibration } from "forecast-web";
import { formatNumber, formatPercent, shortLabel } from "src/lib/format";

/** "Mar 3, 2026: 38% above expected, score 4.2"; the difference itself when a ratio would mislead. */
export function anomalySentence(p: AnomalyPoint, label: string, unit: string): string {
  const diff = p.value - p.expected;
  const ratio = Math.abs(p.expected) > 1e-9 ? Math.abs(diff / p.expected) : Infinity;
  const size = ratio <= 10 ? formatPercent(ratio) : `${formatNumber(Math.abs(diff))}${unit ? ` ${unit}` : ""}`;
  return `${shortLabel(label)}: ${size} ${diff >= 0 ? "above" : "below"} expected, score ${p.score.toFixed(1)}`;
}

/** Points scoring above the threshold, newest first. Scores do not depend on the threshold, so this needs no rerun. */
export const flagged = (points: AnomalyPoint[], threshold: number) => points.filter((p) => p.score > threshold).sort((a, b) => b.index - a.index);

export function calibrationSentence(c: Calibration): string {
  if (c.coverage === null) return "Bands unchanged: too little history to check them";
  const held = `the raw 80% range ${c.scale === 1 ? "already " : ""}held ${formatPercent(c.coverage)} of past values`;
  if (c.scale === 1) return `Bands unchanged: ${held}`;
  return c.scale > 1 ? `Bands widened ${c.scale.toFixed(1)}x: ${held}` : `Bands narrowed to ${c.scale.toFixed(1)}x: ${held}`;
}
