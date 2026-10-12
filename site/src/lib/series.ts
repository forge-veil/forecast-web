/** Pulls every number out of free-form text. Separators: commas, semicolons, whitespace. */
export function parseNumbers(text: string): number[] {
  return text
    .split(/[\s,;]+/)
    .filter((t) => t !== "")
    .map(Number)
    .filter(Number.isFinite);
}

export interface ParsedCsv {
  columns: string[];
  rows: (number | null)[][];
  /** The data rows as text, for label columns such as dates. */
  raw: string[][];
}

const unquote = (s: string) => s.trim().replace(/^"(.*)"$/, "$1").trim();

/** Parses delimited text. Detects tab, semicolon or comma, and a header row of non-numeric cells. */
export function parseCsv(text: string): ParsedCsv {
  const lines = text.split(/\r?\n/).filter((l) => l.trim() !== "");
  if (lines.length === 0) return { columns: [], rows: [], raw: [] };
  const first = lines[0];
  const delim = first.includes("\t") ? "\t" : first.includes(";") ? ";" : ",";
  const cells = lines.map((l) => l.split(delim).map(unquote));
  const width = Math.max(...cells.map((c) => c.length));
  const isNum = (s: string) => s !== "" && Number.isFinite(Number(s));
  const hasHeader = cells[0].some((c) => c !== "" && !isNum(c)) && cells.length > 1;
  const columns = Array.from({ length: width }, (_, i) =>
    hasHeader && cells[0][i] ? cells[0][i] : `Column ${i + 1}`,
  );
  const raw = cells.slice(hasHeader ? 1 : 0).map((r) => Array.from({ length: width }, (_, i) => r[i] ?? ""));
  const rows = raw.map((r) => r.map((c) => (isNum(c) ? Number(c) : null)));
  return { columns, rows, raw };
}

/** Returns one column. Defaults to the last column that contains any numbers. */
export function pickColumn(p: ParsedCsv, index?: number): { index: number; values: (number | null)[] } {
  let i = index;
  if (i === undefined) {
    i = -1;
    for (let c = p.columns.length - 1; c >= 0; c--) {
      if (p.rows.some((r) => r[c] !== null)) {
        i = c;
        break;
      }
    }
  }
  return { index: i, values: i < 0 ? [] : p.rows.map((r) => r[i] ?? null) };
}

/** Splits off the last n points as a test set, always keeping at least two for training. */
export function holdout<T>(series: T[], n: number): { train: T[]; test: T[] } {
  const k = Math.max(0, Math.min(Math.floor(n), series.length - 2));
  return { train: series.slice(0, series.length - k), test: series.slice(series.length - k) };
}

/** Turns pointer samples into n evenly spaced values by linear interpolation over x. */
export function resample(points: { x: number; y: number }[], n: number): number[] {
  const byX = new Map<number, number>();
  for (const p of points) byX.set(p.x, p.y); // later samples win for the same x
  const pts = [...byX].map(([x, y]) => ({ x, y })).sort((a, b) => a.x - b.x);
  if (pts.length < 2 || n < 2) return [];
  const x0 = pts[0].x;
  const x1 = pts[pts.length - 1].x;
  const out: number[] = [];
  let j = 0;
  for (let i = 0; i < n; i++) {
    const x = x0 + ((x1 - x0) * i) / (n - 1);
    while (j < pts.length - 2 && pts[j + 1].x < x) j++;
    const a = pts[j];
    const b = pts[j + 1];
    const t = b.x === a.x ? 0 : (x - a.x) / (b.x - a.x);
    out.push(+(a.y + (b.y - a.y) * Math.min(1, Math.max(0, t))).toFixed(10));
  }
  return out;
}

/** Continues "YYYY-MM" or "YYYY-MM-DD" labels n steps past the last one. Unknown formats give []. */
export function extendLabels(labels: string[], n: number, kind: "monthly" | "daily" | "weekdays"): string[] {
  const last = labels[labels.length - 1] ?? "";
  if (kind === "monthly") {
    const m = /^(\d{4})-(\d{2})$/.exec(last);
    if (!m) return [];
    let y = +m[1];
    let mo = +m[2];
    return Array.from({ length: n }, () => {
      mo++;
      if (mo > 12) (mo = 1), y++;
      return `${y}-${String(mo).padStart(2, "0")}`;
    });
  }
  const d = /^(\d{4})-(\d{2})-(\d{2})$/.exec(last);
  if (!d) return [];
  const t = new Date(Date.UTC(+d[1], +d[2] - 1, +d[3]));
  const out: string[] = [];
  while (out.length < n) {
    t.setUTCDate(t.getUTCDate() + 1);
    const wd = t.getUTCDay();
    if (kind === "weekdays" && (wd === 0 || wd === 6)) continue;
    out.push(t.toISOString().slice(0, 10));
  }
  return out;
}
