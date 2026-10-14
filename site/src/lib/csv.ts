export type TableCell = number | string | null;

export interface Table {
  columns: string[];
  rows: TableCell[][];
}

const MISSING = new Set(["", "NA", "N/A", "NaN", "nan", "null", "NULL", "?", "-"]);

/** Splits one CSV record, honoring double quotes ("" is a literal quote). */
function splitLine(line: string, delim: string): string[] {
  const out: string[] = [];
  let cur = "";
  let quoted = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (quoted) {
      if (ch === '"' && line[i + 1] === '"') {
        cur += '"';
        i++;
      } else if (ch === '"') quoted = false;
      else cur += ch;
    } else if (ch === '"') quoted = true;
    else if (ch === delim) {
      out.push(cur);
      cur = "";
    } else cur += ch;
  }
  out.push(cur);
  return out.map((s) => s.trim());
}

/**
 * Parses CSV/TSV text with a header row. Numeric-looking cells become numbers (so
 * "10" and "9" stay numeric rather than becoming text categories), common missing
 * markers become null, short rows are padded with null.
 */
export function parseTable(text: string): Table {
  const lines = text.split(/\r?\n/).filter((l) => l.trim() !== "");
  if (lines.length === 0) return { columns: [], rows: [] };
  const delim = lines[0].includes("\t") ? "\t" : lines[0].includes(";") ? ";" : ",";
  const columns = splitLine(lines[0], delim).map((c, i) => c || `Column ${i + 1}`);
  const rows = lines.slice(1).map((l) => {
    const cells = splitLine(l, delim);
    return columns.map((_, i) => {
      const c = cells[i] ?? "";
      if (MISSING.has(c)) return null;
      const n = Number(c);
      return Number.isFinite(n) ? n : c;
    });
  });
  return { columns, rows };
}
