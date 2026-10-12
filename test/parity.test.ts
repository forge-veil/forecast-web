/**
 * Parity with the official Python forecast. Needs the exported graphs in
 * build/hub/onnx (run tools/export_onnx.py and tools/quantize.py); skipped otherwise.
 */
import { copyFileSync, existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import * as ort from "onnxruntime-node";
import { createForecaster } from "src/forecaster";
import { expandHalfToFloat } from "src/half";
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
const variants: { file: string; half?: { file: string; location: string }; worst: number; mean: number }[] = [
  { file: "model.onnx", worst: 1e-3, mean: 1e-4 },
  {
    file: "model_fp16w.onnx",
    half: { file: "model_fp16w.f16", location: "model_fp16w.onnx_data" },
    worst: 0.01,
    mean: 0.001,
  },
  { file: "model_quantized.onnx", worst: 0.05, mean: 0.01 },
];

/** For the fp16 pair: expand the weights with the browser code path into a temp dir next to the graph. */
function stage(file: string, half: { file: string; location: string }): { path: string; cleanup: () => void } {
  const dir = mkdtempSync(join(tmpdir(), "timesfm-parity-"));
  copyFileSync(DIR + file, join(dir, file));
  writeFileSync(join(dir, half.location), expandHalfToFloat(readFileSync(DIR + half.file)));
  return { path: join(dir, file), cleanup: () => rmSync(dir, { recursive: true, force: true }) };
}

for (const { file, half, worst, mean } of variants) {
  const path = DIR + file;
  describe.skipIf(!existsSync(path))(`parity: ${file}`, () => {
    let session: ort.InferenceSession;
    let results: Forecast[];
    let cleanup = () => {};

    beforeAll(async () => {
      let modelPath = path;
      if (half) ({ path: modelPath, cleanup } = stage(file, half));
      session = await ort.InferenceSession.create(modelPath, { executionProviders: ["cpu"], logSeverityLevel: 3 });
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
      cleanup();
      if (!results) return;
      const rows = ref.cases.map((c, i) => {
        const e = errors(results[i], c.output);
        const s = scale(c.input);
        return { series: c.name, worst: +(e.worst / s).toExponential(2), mean: +(e.mean / s).toExponential(2) };
      });
      console.log(`\n${file} error as a fraction of series std`);
      console.table(rows);
    });

    it("a short horizon returns the same leading steps as horizon 128", async () => {
      const forecast = createForecaster(async (inputs, masks, batch) => {
        const out = await session.run({
          inputs: new ort.Tensor("float32", inputs, [batch, 512]),
          masks: new ort.Tensor("float32", masks, [batch, 512]),
        });
        return { point: out.point.data as Float32Array, qspread: out.qspread.data as Float32Array };
      });
      const short = await forecast(ref.cases.map((c) => c.input), { horizon: 24 });
      short.forEach((f, i) => {
        expect(f.median).toHaveLength(24);
        expect(f.median).toEqual(results[i].median.slice(0, 24));
        expect(f.q90).toEqual(results[i].q90.slice(0, 24));
      });
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
