/** One table cell as passed by callers. */
export type Cell = number | string | boolean | null | undefined;

const isMissing = (v: Cell) => v === null || v === undefined || (typeof v === "number" && !Number.isFinite(v));

/** Category key: strings as-is, booleans as Python prints them, numbers in mixed columns as text. */
const key = (v: Cell): string => (typeof v === "boolean" ? (v ? "True" : "False") : String(v));

/** Code-point order, as Python sorts str. */
const byCodePoint = (a: string, b: string) => (a < b ? -1 : a > b ? 1 : 0);

export interface Encoder {
  /** Per input column: true when it is encoded as categories. */
  categorical: boolean[];
  /** Columns out (categorical first, then numeric, each in input order), one Float64Array per column. */
  transform(X: Cell[][]): Float64Array[];
}

/**
 * Mirrors tabicl's TransformToNumerical on a pandas DataFrame: string and
 * boolean columns become ordinal codes (sorted categories, unknown or missing
 * -> -1) and move to the front; numeric columns are mean-imputed (an
 * all-missing column becomes 0). A column mixing numbers and strings is
 * treated as text, where upstream would raise.
 */
export function fitEncoder(X: Cell[][]): Encoder {
  const width = X[0]?.length ?? 0;
  const categorical: boolean[] = [];
  const categories: Map<string, number>[] = [];
  const means: number[] = [];
  for (let c = 0; c < width; c++) {
    const values = X.map((r) => r[c]).filter((v) => !isMissing(v));
    const cat = values.some((v) => typeof v === "string" || typeof v === "boolean");
    categorical.push(cat);
    if (cat) {
      const sorted = [...new Set(values.map(key))].sort(byCodePoint);
      categories[c] = new Map(sorted.map((k, i) => [k, i]));
    } else {
      means[c] = values.length ? (values as number[]).reduce((a, b) => a + b, 0) / values.length : 0;
    }
  }
  const order = [...categorical.keys()].filter((c) => categorical[c]).concat([...categorical.keys()].filter((c) => !categorical[c]));
  return {
    categorical,
    transform(rows) {
      return order.map((c) => {
        const out = new Float64Array(rows.length);
        rows.forEach((r, i) => {
          const v = r[c];
          if (categorical[c]) out[i] = isMissing(v) ? -1 : (categories[c].get(key(v)) ?? -1);
          else out[i] = isMissing(v) || typeof v !== "number" ? means[c] : v;
        });
        return out;
      });
    },
  };
}

/** sklearn LabelEncoder: sorted unique labels -> 0..C-1. Mixed numbers and strings are compared as text. */
export function encodeLabels(y: (number | string)[]): { codes: number[]; classes: (number | string)[] } {
  const text = y.some((v) => typeof v === "string");
  const norm = text ? y.map(String) : y;
  const classes = [...new Set(norm)].sort((a, b) => (text ? byCodePoint(a as string, b as string) : (a as number) - (b as number)));
  const index = new Map(classes.map((c, i) => [c, i]));
  return { codes: norm.map((v) => index.get(v)!), classes };
}
