/**
 * Parity with ChronosBoltPipeline.predict_quantiles for every size and shipped
 * format. Needs build/chronos-bolt-<size>/onnx (tools/models/chronos-bolt
 * scripts) and test/fixtures/chronos-bolt-<size>; skipped otherwise.
 */
import { copyFileSync, existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import * as ort from "onnxruntime-node";
import { expandHalfToFloat } from "src/core/half";
import type { ForecastFn, RunGraph } from "src/models/adapter";
import { STANDARD_FILES } from "src/models/files";
import { MODELS, getAdapter } from "src/models/index";
import type { Forecast, ModelId, ResolvedDtype } from "src/types";

const KEYS = ["q10", "q20", "q30", "q40", "q50", "q60", "q70", "q80", "q90"] as const;
const TOL: Record<ResolvedDtype, { worst: number; mean: number }> = {
  fp32: { worst: 1e-3, mean: 1e-4 },
  fp16: { worst: 0.01, mean: 0.001 },
  q8: { worst: 0.05, mean: 0.01 },
};
/**
 * Horizons past one pass (64) feed each format's own quantiles back into nine
 * new contexts, so compression error compounds. fp32 stays strict there, which
 * proves the rollout itself; compressed formats get a looser, separate bound,
 * and the model cards report both horizons.
 */
const ROLLOUT_TOL: Record<ResolvedDtype, { worst: number; mean: number }> = {
  fp32: TOL.fp32,
  fp16: { worst: 0.03, mean: 0.003 },
  // Observed worst 9-14% at 200 steps; 0.2 leaves room for a CI rebuild, matching "under 20%" in the cards.
  q8: { worst: 0.2, mean: 0.03 },
};
const tol = (dtype: ResolvedDtype, h: number) => (h > 64 ? ROLLOUT_TOL : TOL)[dtype];
const SIZES = ["tiny", "mini", "small", "base"] as const;

type Ref = { horizons: number[]; cases: { name: string; input: (number | null)[]; outputs: Record<string, number[][]> }[] };

/** Population std of the last 2048 finite inputs, the natural unit for forecast error. */
function scale(input: (number | null)[]): number {
  const v = input.filter((x): x is number => x !== null).slice(-2048);
  if (v.length === 0) return 1;
  const mu = v.reduce((a, b) => a + b, 0) / v.length;
  const sd = Math.sqrt(v.reduce((a, b) => a + (b - mu) ** 2, 0) / v.length);
  return sd > 1e-6 ? sd : 1;
}

/** Worst and mean absolute error across all nine quantiles and every step. */
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

/** For the fp16 pair: expand the weights with the browser code path into a temp dir next to the graph. */
function stage(dir: string, file: string, half: { file: string; location: string }) {
  const tmp = mkdtempSync(join(tmpdir(), "chronos-parity-"));
  copyFileSync(dir + file, join(tmp, file));
  writeFileSync(join(tmp, half.location), expandHalfToFloat(readFileSync(dir + half.file)));
  return { path: join(tmp, file), cleanup: () => rmSync(tmp, { recursive: true, force: true }) };
}

for (const size of SIZES) {
  const id = `chronos-bolt-${size}` as ModelId;
  const dir = fileURLToPath(new URL(`../../../build/${id}/onnx/`, import.meta.url));
  const fixture = new URL(`../../fixtures/${id}/reference.json`, import.meta.url);
  const ref: Ref | null = existsSync(fixture) ? JSON.parse(readFileSync(fixture, "utf8")) : null;

  for (const dtype of MODELS[id].dtypes) {
    const spec = STANDARD_FILES[dtype];
    const graph = spec.graph.replace("onnx/", "");
    describe.skipIf(!ref || !existsSync(dir + graph))(`parity: ${id} ${dtype}`, () => {
      let session: ort.InferenceSession;
      let cleanup = () => {};
      let forecast: ForecastFn;
      const results: Record<number, Forecast[]> = {};

      beforeAll(async () => {
        let path = dir + graph;
        if (spec.half)
          ({ path, cleanup } = stage(dir, graph, { file: spec.half.file.replace("onnx/", ""), location: spec.half.location }));
        session = await ort.InferenceSession.create(path, { executionProviders: ["cpu"], logSeverityLevel: 3 });
        const run: RunGraph = async (feeds) => {
          const tensors: Record<string, ort.Tensor> = {};
          for (const [k, f] of Object.entries(feeds)) tensors[k] = new ort.Tensor("float32", f.data, f.dims);
          const out = await session.run(tensors);
          return Object.fromEntries(Object.entries(out).map(([k, t]) => [k, t.data as Float32Array]));
        };
        forecast = getAdapter(id).forecaster(run, { model_type: "chronos-bolt" });
        for (const h of ref!.horizons) results[h] = await forecast(ref!.cases.map((c) => c.input), { horizon: h });
      });

      afterAll(async () => {
        await session?.release();
        cleanup();
        if (!results[ref!.horizons[0]]) return;
        const rows = ref!.cases.flatMap((c, i) =>
          ref!.horizons.map((h) => {
            const e = errors(results[h][i], c.outputs[String(h)]);
            const s = scale(c.input);
            return { series: c.name, h, worst: +(e.worst / s).toExponential(2), mean: +(e.mean / s).toExponential(2) };
          }),
        );
        console.log(`\n${id} ${dtype} error as a fraction of series std`);
        console.table(rows);
      });

      it("keeps an all-missing series finite", async () => {
        const [f] = await forecast([[null, null, null]]);
        expect(f.median.every(Number.isFinite)).toBe(true);
      });

      for (const h of ref?.horizons ?? [])
        it.each(ref!.cases.map((c, i) => [c.name, i] as const))(
          `h=${h} %s: worst < ${tol(dtype, h).worst}, mean < ${tol(dtype, h).mean} of std`,
          (_name, i) => {
            const c = ref!.cases[i];
            const e = errors(results[h][i], c.outputs[String(h)]);
            const s = scale(c.input);
            expect(e.worst / s).toBeLessThan(tol(dtype, h).worst);
            expect(e.mean / s).toBeLessThan(tol(dtype, h).mean);
          },
        );
    });
  }
}
