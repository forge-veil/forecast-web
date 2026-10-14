import { fitPipeline, fitUniqueFilter, standardScale } from "src/tabular/preprocess";
import { yeoJohnsonLambda } from "src/tabular/yeojohnson";
import { fixture } from "./fixtures";

type Pipe = {
  scaler_mean: number[];
  scaler_scale: number[];
  lower: number[];
  upper: number[];
  train: number[][];
  test: number[][];
  lambdas?: number[];
  power_mean?: number[];
  power_scale?: number[];
};
const pre = fixture<{ tables: { name: string; X_train: number[][]; X_test: number[][]; keep: number[]; pipelines: Record<"none" | "power", Pipe> }[] }>(
  "tabicl/preprocess.json",
);

const columns = (rows: number[][], keep: number[]) => keep.map((c) => Float64Array.from(rows.map((r) => r[c])));
const rowsOf = (cols: Float64Array[]) => (cols[0] ? Array.from(cols[0], (_, i) => cols.map((c) => c[i])) : []);

function close(actual: number[][] | number[], expected: number[][] | number[], tol: number) {
  const a = (actual as unknown[]).flat() as number[];
  const e = (expected as unknown[]).flat() as number[];
  expect(a.length).toBe(e.length);
  a.forEach((v, i) => expect(Math.abs(v - e[i])).toBeLessThanOrEqual(tol * Math.max(1, Math.abs(e[i]))));
}

describe.each(pre.tables)("TabICL preprocessing: $name", (t) => {
  it("unique filter keeps the same columns", () => {
    expect(fitUniqueFilter(columns(t.X_train, t.X_train[0].map((_, i) => i)))).toEqual(t.keep);
  });

  it("Yeo-Johnson lambdas match scipy", () => {
    const scaled = standardScale(columns(t.X_train, t.keep)).train;
    const lambdas = scaled.map((c) => yeoJohnsonLambda(c));
    close(lambdas, t.pipelines.power.lambdas!, 1e-6);
  });

  // "power" stops within scipy's optimizer tolerance (xtol 1.48e-8) of the reference lambda,
  // so its outputs agree to about 3e-8; "none" has no optimizer and matches to 1e-9.
  it.each([
    ["none", 1e-9],
    ["power", 1e-7],
  ] as const)("%s pipeline matches upstream on train and test rows", (norm, tol) => {
    const p = fitPipeline(columns(t.X_train, t.keep), norm);
    const e = t.pipelines[norm];
    close(rowsOf(p.trainTransformed), e.train, tol);
    close(rowsOf(p.transform(columns(t.X_test, t.keep))), e.test, tol);
  });
});
