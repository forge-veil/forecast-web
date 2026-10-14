import { combineClassification } from "src/tabular/combine";
import { ALPHAS999, quantileStats } from "src/tabular/quantiles";
import { ensembleConfigs } from "src/tabular/ensemble";
import { fixture } from "./fixtures";

const q = fixture<{ alphas999: number[]; alphas: number[]; cases: { raw: number[]; mean: number; median: number; icdf: number[] }[] }>("tabicl/quantile.json");
const clf = fixture<{ tables: { name: string; X_train: unknown[][]; classes: number[]; expected: { proba: number[][] }; member_outputs?: number[][][] }[] }>(
  "tabicl-classifier-v2/reference.json",
);

describe("TabICL regression quantiles", () => {
  it("alpha grid equals torch.linspace(0, 1, 1001, float32)[1:-1]", () => {
    expect(Array.from(ALPHAS999)).toEqual(q.alphas999);
  });

  it.each(q.cases.map((c, i) => [i, c] as const))("case %i: mean, median and icdf match upstream", (_i, c) => {
    const s = quantileStats(Float32Array.from(c.raw), q.alphas);
    expect(s.mean).toBeCloseTo(c.mean, 5);
    expect(Math.abs(s.median - c.median)).toBeLessThan(1e-5 * Math.max(1, Math.abs(c.median)));
    s.quantiles.forEach((v, k) => expect(Math.abs(v - c.icdf[k])).toBeLessThan(1e-5 * Math.max(1, Math.abs(c.icdf[k]))));
  });
});

describe("TabICL classification combine", () => {
  it("reproduces predict_proba from the raw member logits", () => {
    const t = clf.tables.find((x) => x.name === "numeric_3class")!;
    const members = ensembleConfigs({ nFeatures: t.X_train[0].length, nClasses: t.classes.length, nEstimators: 8, seed: 42, normMethods: ["none", "power"] });
    const outs = t.member_outputs!;
    const nTest = outs[0].length;
    const C = t.classes.length;
    const proba = combineClassification(
      members.map((m, i) => ({ logits: Float32Array.from(outs[i].flat()), stride: C, cls: m.cls! })),
      nTest,
      C,
    );
    proba.forEach((row, r) => row.forEach((p, k) => expect(Math.abs(p - t.expected.proba[r][k])).toBeLessThan(1e-6)));
  });
});
