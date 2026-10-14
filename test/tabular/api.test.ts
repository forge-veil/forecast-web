import type { Feed } from "src/models/adapter";
import { createTabular, LIMITS } from "src/tabular/tabular";
import { TABULAR_MODELS } from "src/tabular/models";

type Call = { X: Feed; y_train: Feed };

/** Fake graph: logits favor the class whose code equals round(first feature of the test row) mod 10. */
function fakeGraph(kind: "classification" | "regression") {
  const calls: Call[] = [];
  const graph = {
    async run(feeds: Record<string, Feed>): Promise<Record<string, Float32Array>> {
      const { X, y_train } = feeds as unknown as Call;
      calls.push({ X: { data: Float32Array.from(X.data), dims: X.dims }, y_train: { data: Float32Array.from(y_train.data), dims: y_train.dims } });
      const [b, t, h] = X.dims;
      const n = y_train.dims[1];
      const m = t - n;
      if (kind === "classification") {
        const out = new Float32Array(b * m * 10);
        for (let i = 0; i < b * m; i++) out[i * 10] = 5; // every member votes for its class code 0
        return { logits: out };
      }
      const out = new Float32Array(b * m * 999);
      for (let i = 0; i < b * m; i++) for (let q = 0; q < 999; q++) out[i * 999 + q] = (q - 499) / 100;
      return { quantiles: out };
      void h;
    },
    async dispose() {},
  };
  return { graph, calls };
}

const rows = (n: number, f: number) => Array.from({ length: n }, (_, i) => Array.from({ length: f }, (_, j) => Math.sin(i * 7 + j)));

describe("forecast-web/tabular API", () => {
  it("rejects more than 10 classes before running the graph", async () => {
    const { graph, calls } = fakeGraph("classification");
    const t = createTabular(graph, TABULAR_MODELS["tabicl-classifier-v2"]);
    expect(() => t.fit(rows(22, 2), Array.from({ length: 22 }, (_, i) => i % 11))).toThrow(expect.objectContaining({ code: "TOO_MANY_CLASSES" }));
    expect(calls).toHaveLength(0);
  });

  it("rejects tables above the size cap", () => {
    const { graph } = fakeGraph("classification");
    const t = createTabular(graph, TABULAR_MODELS["tabicl-classifier-v2"]);
    expect(() => t.fit(rows(LIMITS.maxRows + 1, 2), Array(LIMITS.maxRows + 1).fill(0))).toThrow(expect.objectContaining({ code: "TOO_LARGE" }));
    expect(() => t.fit(rows(10, LIMITS.maxFeatures + 1), Array(10).fill(0))).toThrow(expect.objectContaining({ code: "TOO_LARGE" }));
    const n = Math.floor(LIMITS.maxCells / 20) + 1; // rows x 20 features just over the cell cap
    expect(() => t.fit(rows(n, 20), Array(n).fill(0))).toThrow(expect.objectContaining({ code: "TOO_LARGE" }));
  });

  it.each([
    ["empty X", [] as unknown[][], [] as number[]],
    ["ragged rows", [[1, 2], [3]], [0, 1]],
    ["label count mismatch", [[1], [2]], [0]],
  ])("rejects %s as INVALID_INPUT", (_name, X, y) => {
    const { graph } = fakeGraph("classification");
    const t = createTabular(graph, TABULAR_MODELS["tabicl-classifier-v2"]);
    expect(() => t.fit(X as never, y)).toThrow(expect.objectContaining({ code: "INVALID_INPUT" }));
  });

  it("returns probability 1 for a single training class without running the graph", async () => {
    const { graph, calls } = fakeGraph("classification");
    const fitted = createTabular(graph, TABULAR_MODELS["tabicl-classifier-v2"]).fit(rows(5, 2), ["a", "a", "a", "a", "a"]);
    expect(await fitted.predictProba(rows(2, 2))).toEqual([[1], [1]]);
    expect(await fitted.predict(rows(2, 2))).toEqual(["a", "a"]);
    expect(calls).toHaveLength(0);
  });

  it("runs one graph call per norm group with class-shifted labels, and maps argmax back to labels", async () => {
    const { graph, calls } = fakeGraph("classification");
    const fitted = createTabular(graph, TABULAR_MODELS["tabicl-classifier-v2"]).fit(rows(30, 3), Array.from({ length: 30 }, (_, i) => ["x", "y", "z"][i % 3]));
    const labels = await fitted.predict(rows(4, 3));
    expect(calls).toHaveLength(2); // "power" group, then "none"
    expect(calls[0].X.dims).toEqual([4, 34, 3]);
    expect(calls[0].y_train.dims).toEqual([4, 30]);
    // every member votes for its own class code 0; after undoing shifts the votes spread, but argmax is a valid label
    labels.forEach((l) => expect(["x", "y", "z"]).toContain(l));
    const distinct = new Set(
      calls.flatMap((c) => Array.from({ length: c.y_train.dims[0] }, (_, b) => Array.from(c.y_train.data.subarray(b * 30, b * 30 + 3)).join(","))),
    );
    expect(distinct.size).toBeGreaterThan(1); // members see differently shifted labels
  });

  it("regression returns de-normalized mean, median and quantiles per test row", async () => {
    const { graph } = fakeGraph("regression");
    const y = Array.from({ length: 30 }, (_, i) => 100 + i);
    const fitted = createTabular(graph, TABULAR_MODELS["tabicl-regressor-v2"]).fit(rows(30, 2), y);
    const mean = (await fitted.predict(rows(3, 2), { output: "mean" })) as number[];
    const q = (await fitted.predict(rows(3, 2), { output: "quantiles", alphas: [0.1, 0.9] })) as number[][];
    const yMean = y.reduce((a, b) => a + b, 0) / y.length;
    mean.forEach((v) => expect(v).toBeCloseTo(yMean, 3)); // symmetric fake quantiles around 0 -> y mean
    q.forEach(([lo, hi]) => expect(lo).toBeLessThan(hi));
  });

  it("splits large prediction batches so each graph call stays within the cell budget", async () => {
    const { graph, calls } = fakeGraph("classification");
    const fitted = createTabular(graph, TABULAR_MODELS["tabicl-classifier-v2"]).fit(rows(200, 50), Array.from({ length: 200 }, (_, i) => i % 2));
    const proba = await fitted.predictProba(rows(2000, 50));
    expect(proba).toHaveLength(2000);
    for (const c of calls) expect(c.X.dims[1] * c.X.dims[2]).toBeLessThanOrEqual(2 * LIMITS.maxCells);
    expect(calls.length).toBeGreaterThan(2);
  });

  it("rejects test rows with the wrong number of columns", async () => {
    const { graph } = fakeGraph("classification");
    const fitted = createTabular(graph, TABULAR_MODELS["tabicl-classifier-v2"]).fit(rows(10, 2), Array.from({ length: 10 }, (_, i) => i % 2));
    await expect(fitted.predictProba([[1]])).rejects.toMatchObject({ code: "INVALID_INPUT" });
    await expect(fitted.predict([1, 2] as never)).rejects.toMatchObject({ code: "INVALID_INPUT" });
  });

  it.each([0, -1, 2.5, NaN])("rejects nEstimators %s", (k) => {
    const { graph } = fakeGraph("classification");
    expect(() => createTabular(graph, TABULAR_MODELS["tabicl-classifier-v2"], { nEstimators: k })).toThrow(expect.objectContaining({ code: "INVALID_INPUT" }));
  });

  it("rejects missing class labels", () => {
    const { graph } = fakeGraph("classification");
    const t = createTabular(graph, TABULAR_MODELS["tabicl-classifier-v2"]);
    expect(() => t.fit(rows(4, 2), [1, null, 2, 1] as never)).toThrow(expect.objectContaining({ code: "INVALID_INPUT" }));
    expect(() => t.fit(rows(4, 2), [1, NaN, 2, 1])).toThrow(expect.objectContaining({ code: "INVALID_INPUT" }));
  });
});

