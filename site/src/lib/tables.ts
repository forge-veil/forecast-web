import type { TableCell } from "src/lib/csv";

export type Task = "classification" | "regression";

/** Text targets, or up to 10 distinct integers, read as classes; anything else as a number to predict. */
export function suggestTask(values: TableCell[]): Task {
  const present = values.filter((v) => v !== null);
  if (present.some((v) => typeof v === "string")) return "classification";
  const distinct = new Set(present);
  return present.every((v) => Number.isInteger(v)) && distinct.size <= 10 ? "classification" : "regression";
}

/** Deterministic train/test split of `n` row indices (mulberry32-seeded Fisher-Yates). */
export function holdoutSplit(n: number, testFraction: number, seed: number): { train: number[]; test: number[] } {
  let s = seed >>> 0;
  const rand = () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const idx = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [idx[i], idx[j]] = [idx[j], idx[i]];
  }
  const k = Math.max(1, Math.round(n * testFraction));
  return { test: idx.slice(0, k).sort((a, b) => a - b), train: idx.slice(k).sort((a, b) => a - b) };
}

export const accuracy = (actual: TableCell[], predicted: (string | number)[]) =>
  actual.filter((a, i) => a === predicted[i]).length / Math.max(1, actual.length);

export const mae = (actual: number[], predicted: number[]) =>
  actual.reduce((acc, a, i) => acc + Math.abs(a - predicted[i]), 0) / Math.max(1, actual.length);

export const coverage = (actual: number[], bands: number[][]) =>
  actual.filter((a, i) => a >= bands[i][0] && a <= bands[i][1]).length / Math.max(1, actual.length);

/** Best first: classification by accuracy (higher), regression by error (lower); unscored keep their order at the end. */
export function rankVariants<T extends { score?: number }>(rows: T[], task: Task): T[] {
  const scored = rows.filter((r) => r.score !== undefined && Number.isFinite(r.score));
  scored.sort((a, b) => (task === "classification" ? b.score! - a.score! : a.score! - b.score!));
  return [...scored, ...rows.filter((r) => !scored.includes(r))];
}

/** How TabICL will read a column: numbers stay numeric, anything with text becomes categories. */
export const columnKind = (values: TableCell[]): "number" | "category" =>
  values.some((v) => typeof v === "string") ? "category" : "number";

/** Page `page` of `n` rows at `size` per page, clamped to the last page. */
export function pageOf(n: number, page: number, size: number): { page: number; pages: number; start: number; end: number } {
  const pages = Math.max(1, Math.ceil(n / size));
  const p = Math.min(Math.max(0, page), pages - 1);
  return { page: p, pages, start: p * size, end: Math.min(n, (p + 1) * size) };
}

/** Accuracy of always guessing the most common training label. */
export function majorityBaseline(trainLabels: TableCell[], testLabels: TableCell[]): { label: TableCell; accuracy: number } {
  const counts = new Map<TableCell, number>();
  for (const l of trainLabels) counts.set(l, (counts.get(l) ?? 0) + 1);
  let label: TableCell = null;
  let best = -1;
  for (const [l, c] of counts) if (c > best) [label, best] = [l, c];
  return { label, accuracy: testLabels.filter((l) => l === label).length / Math.max(1, testLabels.length) };
}

/** Average error of always guessing the training mean. */
export function meanBaselineError(trainY: number[], testY: number[]): number {
  const m = trainY.reduce((a, b) => a + b, 0) / Math.max(1, trainY.length);
  return testY.reduce((a, y) => a + Math.abs(y - m), 0) / Math.max(1, testY.length);
}

/** Right and total per actual class, most frequent first. */
export function perClass(actual: TableCell[], predicted: TableCell[]): { label: TableCell; right: number; total: number }[] {
  const by = new Map<TableCell, { label: TableCell; right: number; total: number }>();
  actual.forEach((a, i) => {
    const e = by.get(a) ?? { label: a, right: 0, total: 0 };
    e.total++;
    if (predicted[i] === a) e.right++;
    by.set(a, e);
  });
  return [...by.values()].sort((x, y) => y.total - x.total);
}
