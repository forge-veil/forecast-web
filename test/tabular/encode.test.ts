import { encodeLabels, fitEncoder } from "src/tabular/encode";

const cols = (c: Float64Array[]) => c.map((x) => Array.from(x));

describe("TabICL encoder (DataFrame path of TransformToNumerical)", () => {
  it("puts categorical columns first, codes sorted categories, mean-imputes numbers", () => {
    const X = [
      [1.5, "b", true],
      [2.5, "a", false],
      [null, "b", true],
    ];
    const enc = fitEncoder(X);
    expect(enc.categorical).toEqual([false, true, true]);
    expect(cols(enc.transform(X))).toEqual([
      [1, 0, 1],
      [1, 0, 1],
      [1.5, 2.5, 2],
    ]);
  });

  it("codes unseen and missing categories as -1 and imputes test gaps with the training mean", () => {
    const enc = fitEncoder([
      [1.5, "b", true],
      [2.5, "a", false],
      [null, "b", true],
    ]);
    expect(
      cols(
        enc.transform([
          [null, "z", null],
          [5, null, false],
        ]),
      ),
    ).toEqual([
      [-1, -1],
      [-1, 0],
      [2, 5],
    ]);
  });

  it("imputes an all-missing numeric column with 0", () => {
    expect(cols(fitEncoder([[null], [undefined]]).transform([[null], [3]]))).toEqual([[0, 3]]);
  });

  it("treats a column mixing numbers and strings as text (upstream raises here)", () => {
    const enc = fitEncoder([["12"], [7], ["n/a"]]);
    expect(enc.categorical).toEqual([true]);
    expect(cols(enc.transform([["12"], [7], ["n/a"]]))).toEqual([[0, 1, 2]]); // "12" < "7" < "n/a"
  });

  it("treats non-finite numbers as missing", () => {
    expect(cols(fitEncoder([[1], [3], [Infinity]]).transform([[NaN]]))).toEqual([[2]]);
  });

  it("encodes labels in sorted order (numbers numerically, strings by code point)", () => {
    expect(encodeLabels(["no", "yes", "no"])).toEqual({ codes: [0, 1, 0], classes: ["no", "yes"] });
    expect(encodeLabels([10, 2, 2])).toEqual({ codes: [1, 0, 0], classes: [2, 10] });
  });
});
