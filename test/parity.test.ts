/**
 * Parity with the official Python forecast. Needs the exported graphs in
 * build/hub/onnx (run tools/export_onnx.py and tools/quantize.py); skipped otherwise.
 */
import { existsSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import * as ort from "onnxruntime-node";
import { createForecaster } from "src/forecaster";
import type { Forecast } from "src/types";

const DIR = fileURLToPath(new URL("../build/hub/onnx/", import.meta.url));
const ref = JSON.parse(readFileSync(new URL("./fixtures/reference.json", import.meta.url), "utf8")) as {
  horizon: number;
  cases: { name: string; input: (number | null)[]; output: number[][] }[];
};

const KEYS = ["mean", "q10", "q20", "q30", "q40", "q50", "q60", "q70", "q80", "q90"] as const;

/** Std of the last 512 finite inputs, the natural unit for forecast error. */
function scale(input: (number | null)[]): number {
  const v = input.filter((x): x is number => x !== null).slice(-512);
  const mu = v.reduce((a, b) => a + b, 0) / v.length;
  const sd = Math.sqrt(v.reduce((a, b) => a + (b - mu) ** 2, 0) / v.length);
  return sd > 1e-6 ? sd : 1;
}

/** Worst and mean absolute error across all 10 channels and every step. */
function errors(f: Forecast, expected: number[][]): { worst: number; mean: number } {
  let worst = 0;
  let sum = 0;
  for (let t = 0; t < expected.length; t++)
    KEYS.forEach((k, q) => {
      const d = Math.abs(f[k][t] - expected[t][q]);
      worst = Math.max(worst, d);
      sum += d;
    });
  return { worst, mean: sum / (expected.length * KEYS.length) };
}

// Tolerances are fractions of each series' std. fp32 must match tightly; the
// compressed variants were measured at mean < 0.8% and worst < 3.5% of std.
const variants = [
  { file: "model.onnx", worst: 1e-3, mean: 1e-4 },
  { file: "model_fp16.onnx", worst: 0.05, mean: 0.01 },
  { file: "model_quantized.onnx", worst: 0.05, mean: 0.01 },
];

for (const { file, worst, mean } of variants) {
  const path = DIR + file;
  describe.skipIf(!existsSync(path))(`parity: ${file}`, () => {
    let session: ort.InferenceSession;
    let results: Forecast[];

    beforeAll(async () => {
      session = await ort.InferenceSession.create(path, { executionProviders: ["cpu"], logSeverityLevel: 3 });
      const forecast = createForecaster(async (inputs, masks, batch) => {
        const out = await session.run({
          inputs: new ort.Tensor("float32", inputs, [batch, 512]),
          masks: new ort.Tensor("float32", masks, [batch, 512]),
        });
        return { point: out.point.data as Float32Array, qspread: out.qspread.data as Float32Array };
      });
      results = await forecast(
        ref.cases.map((c) => c.input),
        { horizon: ref.horizon },
      );
    });

    afterAll(async () => {
      await session?.release();
      if (!results) return;
      const rows = ref.cases.map((c, i) => {
        const e = errors(results[i], c.output);
        const s = scale(c.input);
        return { series: c.name, worst: +(e.worst / s).toExponential(2), mean: +(e.mean / s).toExponential(2) };
      });
      console.log(`\n${file} error as a fraction of series std`);
      console.table(rows);
    });

    it.each(ref.cases.map((c, i) => [c.name, i] as const))(
      `%s: worst < ${worst}, mean < ${mean} of std`,
      (_name, i) => {
        const c = ref.cases[i];
        const e = errors(results[i], c.output);
        const s = scale(c.input);
        expect(e.worst / s).toBeLessThan(worst);
        expect(e.mean / s).toBeLessThan(mean);
      },
    );
  });
}
