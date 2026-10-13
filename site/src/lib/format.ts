const compact = new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 1 });

/** Readable number at any magnitude: 1.2K, 654.8K, 12.34, 0.0123. */
export function formatNumber(v: number | null | undefined): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return "–";
  const a = Math.abs(v);
  if (a >= 10_000) return compact.format(v);
  if (a >= 100) return v.toFixed(0);
  if (a >= 10) return v.toFixed(1);
  if (a >= 1) return v.toFixed(2);
  if (a === 0) return "0";
  return v.toPrecision(3);
}

/** Download size in decimal megabytes, or "" when unknown. */
export const formatMB = (bytes: number | undefined) => (bytes ? `${Math.round(bytes / 1e6)} MB` : "");

export const formatPercent = (v: number) => (Number.isFinite(v) ? `${Math.round(v * 100)}%` : "–");

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

/** Short axis label for "YYYY-MM", "YYYY-MM-DD" or "YYYY-MM-DD HH:MM" strings; others pass through. */
export function shortLabel(label: string): string {
  let m = /^(\d{4})-(\d{2})$/.exec(label);
  if (m) return `${MONTHS[+m[2] - 1]} ${m[1]}`;
  m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(label);
  if (m) return `${MONTHS[+m[2] - 1]} ${+m[3]}, ${m[1]}`;
  m = /^\d{4}-(\d{2})-(\d{2})[ T](\d{2}):(\d{2})/.exec(label);
  if (m) return `${MONTHS[+m[1] - 1]} ${+m[2]} ${m[3]}:00`;
  return label;
}

/** "Nice" tick values covering [lo, hi]. */
export function niceTicks(lo: number, hi: number, count = 5): number[] {
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return [];
  if (lo === hi) return [lo];
  const raw = (hi - lo) / count;
  const pow = 10 ** Math.floor(Math.log10(raw));
  // Finest round step that keeps the tick count within 1.5x of the request.
  const step = [1, 2, 2.5, 5, 10].map((s) => s * pow).find((s) => (hi - lo) / s <= count * 1.5) ?? 10 * pow;
  const out: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(+v.toPrecision(12));
  return out;
}
