import { classShuffles, ensembleConfigs, featureShuffles } from "src/tabular/ensemble";
import { fixture } from "./fixtures";

const latin = fixture<{ cases: { n: number; squares: number[][] }[] }>("tabicl/latin.json");
const ens = fixture<{
  cases: { n_features: number; n_classes: number | null; n_estimators: number; classification: boolean; configs: unknown[] }[];
}>("tabicl/ensemble.json");

describe("TabICL ensemble generation", () => {
  it.each(latin.cases)("Latin-square feature shuffles for n=$n match upstream", ({ n, squares }) => {
    expect(featureShuffles(n, 8, 42)).toEqual(squares);
  });

  it.each(ens.cases)("ensemble configs F=$n_features C=$n_classes k=$n_estimators", (c) => {
    expect(
      ensembleConfigs({
        nFeatures: c.n_features,
        nClasses: c.classification ? c.n_classes! : undefined,
        nEstimators: c.n_estimators,
        seed: 42,
        normMethods: ["none", "power"],
      }),
    ).toEqual(c.configs);
  });

  it("uses only identity shuffles for a single member", () => {
    expect(featureShuffles(5, 1, 42)).toEqual([[0, 1, 2, 3, 4]]);
    expect(classShuffles(3, 1)).toEqual([[0, 1, 2]]);
    expect(ensembleConfigs({ nFeatures: 3, nClasses: 2, nEstimators: 1, seed: 42, normMethods: ["none", "power"] })).toEqual([
      { norm: "none", feat: [0, 1, 2], cls: [0, 1] },
    ]);
  });

  it("caps members at the number of distinct configurations", () => {
    expect(ensembleConfigs({ nFeatures: 1, nClasses: 2, nEstimators: 8, seed: 42, normMethods: ["none", "power"] })).toHaveLength(4);
  });
});
