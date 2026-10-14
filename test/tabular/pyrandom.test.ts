import { PyRandom } from "src/tabular/pyrandom";
import { fixture } from "./fixtures";

const rng = fixture<{
  random: number[];
  getrandbits32: number[];
  randbelow: { n: number; values: number[] }[];
  shuffle: { input: number[]; output: number[] }[];
  choice: { input: number[]; outputs: number[] };
}>("tabicl/rng.json");

describe("PyRandom matches CPython random.Random(42)", () => {
  it("random()", () => {
    const r = new PyRandom(42);
    expect(rng.random.map(() => r.random())).toEqual(rng.random);
  });

  it("getrandbits(32)", () => {
    const r = new PyRandom(42);
    expect(rng.getrandbits32.map(() => r.getrandbits(32))).toEqual(rng.getrandbits32);
  });

  it.each(rng.randbelow)("randbelow($n)", ({ n, values }) => {
    const r = new PyRandom(42);
    expect(values.map(() => r.randbelow(n))).toEqual(values);
  });

  it.each(rng.shuffle)("shuffle of $input.length items", ({ input, output }) => {
    const r = new PyRandom(42);
    const a = [...input];
    r.shuffle(a);
    expect(a).toEqual(output);
  });

  it("choice", () => {
    const r = new PyRandom(42);
    expect(rng.choice.outputs.map(() => r.choice(rng.choice.input))).toEqual(rng.choice.outputs);
  });
});
