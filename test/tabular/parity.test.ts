/**
 * End-to-end parity with tabicl 2.2.0 (TabICLClassifier.predict_proba / predict,
 * TabICLRegressor.predict) through the real TS pipeline and the exported graphs.
 * Needs build/tabicl-<task>-v2/onnx (tools/models/tabicl scripts); skipped otherwise.
 */
import { copyFileSync, existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import * as ort from "onnxruntime-node";
import { expandHalfToFloat } from "src/core/half";
import type { RunGraph } from "src/models/adapter";
import { STANDARD_FILES } from "src/models/files";
import { TABULAR_MODELS, type TabularModelId } from "src/tabular/models";
import { createTabular } from "src/tabular/tabular";
import type { ResolvedDtype } from "src/types";
import { fixture } from "./fixtures";

type Table = {
  name: string;
  X_train: never[][];
  y_train: (number | string)[];
  X_test: never[][];
  classes?: (number | string)[];
  expected: { proba?: number[][]; labels?: (number | string)[]; mean?: number[]; median?: number[]; quantiles?: number[][] };
};

// Measured worst on the fixture suite (macOS arm64): fp32 9.7e-6 (classifier) and 3.1e-6
// (regressor); fp16 weights 1.04e-3 and 7.1e-4. Bounds leave room for CI's x86 kernels.
const TOL: Record<ResolvedDtype, number> = { fp32: 2e-5, fp16: 2e-3, q8: 0.05 };

function stage(dir: string, file: string, half: { file: string; location: string }) {
  const tmp = mkdtempSync(join(tmpdir(), "tabicl-parity-"));
  copyFileSync(dir + file, join(tmp, file));
  writeFileSync(join(tmp, half.location), expandHalfToFloat(readFileSync(dir + half.file)));
  return { path: join(tmp, file), cleanup: () => rmSync(tmp, { recursive: true, force: true }) };
}

const std = (y: number[]) => {
  const m = y.reduce((a, b) => a + b, 0) / y.length;
  return Math.sqrt(y.reduce((a, b) => a + (b - m) ** 2, 0) / y.length) || 1;
};

for (const id of ["tabicl-classifier-v2", "tabicl-regressor-v2"] as TabularModelId[]) {
  const info = TABULAR_MODELS[id];
  const dir = fileURLToPath(new URL(`../../build/${id}/onnx/`, import.meta.url));
  const fixturePath = fileURLToPath(new URL(`../fixtures/${id}/reference.json`, import.meta.url));
  const ref = existsSync(fixturePath) ? fixture<{ tables: Table[] }>(`${id}/reference.json`) : null;

  for (const dtype of info.dtypes) {
    const spec = STANDARD_FILES[dtype];
    const graphFile = spec.graph.replace("onnx/", "");
    describe.skipIf(!ref || !existsSync(dir + graphFile))(`parity: ${id} ${dtype}`, () => {
      let session: ort.InferenceSession;
      let cleanup = () => {};
      const rows: { table: string; worst: number }[] = [];

      beforeAll(async () => {
        let path = dir + graphFile;
        if (spec.half) ({ path, cleanup } = stage(dir, graphFile, { file: spec.half.file.replace("onnx/", ""), location: spec.half.location }));
        session = await ort.InferenceSession.create(path, { executionProviders: ["cpu"], logSeverityLevel: 3 });
      });

      afterAll(async () => {
        await session?.release();
        cleanup();
        console.log(`\n${id} ${dtype}: worst error per table (probability, or fraction of target std)`);
        console.table(rows);
      });

      const run: RunGraph = async (feeds) => {
        const tensors: Record<string, ort.Tensor> = {};
        for (const [k, f] of Object.entries(feeds)) tensors[k] = new ort.Tensor("float32", f.data, f.dims);
        const out = await session.run(tensors);
        return Object.fromEntries(Object.entries(out).map(([k, t]) => [k, t.data as Float32Array]));
      };

      for (const t of ref?.tables ?? []) {
        it(`${t.name}`, async () => {
          const model = createTabular({ run, dispose: async () => {}, backend: "wasm", dtype }, info);
          const fitted = model.fit(t.X_train, t.y_train);
          if (info.task === "classification") {
            const proba = await fitted.predictProba(t.X_test);
            let worst = 0;
            proba.forEach((row, r) => row.forEach((p, k) => (worst = Math.max(worst, Math.abs(p - t.expected.proba![r][k])))));
            rows.push({ table: t.name, worst: +worst.toExponential(2) });
            expect(worst).toBeLessThan(TOL[dtype]);
            if (dtype === "fp32") expect(await fitted.predict(t.X_test)).toEqual(t.expected.labels);
          } else {
            const s = std(t.y_train as number[]);
            const mean = (await fitted.predict(t.X_test, { output: "mean" })) as number[];
            const median = (await fitted.predict(t.X_test, { output: "median" })) as number[];
            const quant = (await fitted.predict(t.X_test, { output: "quantiles" })) as number[][];
            let worst = 0;
            mean.forEach((v, r) => (worst = Math.max(worst, Math.abs(v - t.expected.mean![r]) / s)));
            median.forEach((v, r) => (worst = Math.max(worst, Math.abs(v - t.expected.median![r]) / s)));
            quant.forEach((row, r) => row.forEach((v, k) => (worst = Math.max(worst, Math.abs(v - t.expected.quantiles![r][k]) / s))));
            rows.push({ table: t.name, worst: +worst.toExponential(2) });
            expect(worst).toBeLessThan(TOL[dtype]);
          }
        });
      }
    });
  }
}
