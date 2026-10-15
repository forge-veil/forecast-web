/**
 * Tunes and checks each model's anomaly thresholds on a seeded synthetic suite,
 * and checks calibrated coverage on heavy-tailed noise. Runs the real models
 * (fp16, the WebGPU default) through withAnalysis with onnxruntime-node. Needs
 * build/<id>/onnx; skipped otherwise. Slow: run one model at a time with
 * `npx vitest run test/anomaly/integration.test.ts -t <id> --silent=false`.
 */
import { copyFileSync, existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import * as ort from "onnxruntime-node";
import { withAnalysis } from "src/anomaly/index";
import { expandHalfToFloat } from "src/core/half";
import type { RunGraph } from "src/models/adapter";
import { STANDARD_FILES } from "src/models/files";
import { MODELS, getAdapter } from "src/models/index";
import type { AnomalyThresholds, Forecaster, ModelId } from "src/types";

const N = 256;
const GRID = Array.from({ length: 11 }, (_, i) => 2.5 + 0.25 * i);
const SPIKES = [
  { index: 110, sign: 1 },
  { index: 160, sign: -1 },
  { index: 215, sign: 1 },
];
const SHIFT = 240;

function rng(seed: number) {
  let a = seed >>> 0;
  const uniform = () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const normal = () => Math.sqrt(-2 * Math.log(1 - uniform())) * Math.cos(2 * Math.PI * uniform());
  const student = (df: number) => {
    let chi = 0;
    for (let i = 0; i < df; i++) chi += normal() ** 2;
    return normal() / Math.sqrt(chi / df);
  };
  return { normal, student };
}

type Clean = { name: string; values: number[]; sd: number; up: boolean; walk?: boolean };

/** Backtest block length detectAnomalies uses at this length, and a point's steps ahead of its cut. */
const H = Math.ceil((N - 64) / 128);
const stepOf = (i: number) => H - ((N - i - 1) % H);

/** Seasonal, trend, random walk, counts and heavier-tailed noise; three seeds each. */
function cleanSuite(): Clean[] {
  const out: Clean[] = [];
  for (let seed = 1; seed <= 3; seed++) {
    const r = (k: number) => rng(seed * 100 + k);
    const g1 = r(1), g2 = r(2), g3 = r(3), g4 = r(4), g5 = r(5);
    const t = Array.from({ length: N }, (_, i) => i);
    out.push({ name: `seasonal-${seed}`, sd: 1, up: false, values: t.map((i) => 10 + 3 * Math.sin((2 * Math.PI * i) / 24) + g1.normal()) });
    out.push({ name: `trend-${seed}`, sd: 1, up: false, values: t.map((i) => 5 + 0.05 * i + 3 * Math.sin((2 * Math.PI * i) / 12) + g2.normal()) });
    let w = 50;
    out.push({ name: `walk-${seed}`, sd: 1, up: false, walk: true, values: t.map(() => (w += g3.normal())) });
    out.push({ name: `counts-${seed}`, sd: Math.sqrt(20), up: true, values: t.map((i) => Math.max(0, Math.round(20 + 8 * Math.sin((2 * Math.PI * i) / 7) + Math.sqrt(20) * g4.normal()))) });
    out.push({ name: `t5-${seed}`, sd: Math.sqrt(5 / 3), up: false, values: t.map((i) => 10 + 3 * Math.sin((2 * Math.PI * i) / 24) + g5.student(5)) });
  }
  return out;
}

/**
 * Three spikes and a level shift, each 6 sd of what an ideal forecaster could not
 * know at that point (a random walk's spread grows as sqrt(step)); counts only spike up.
 */
function plant(c: Clean): { values: number[]; expected: number[] } {
  const values = c.values.slice();
  for (const s of SPIKES) values[s.index] += (c.up ? 1 : s.sign) * 6 * c.sd * (c.walk ? Math.sqrt(stepOf(s.index)) : 1);
  for (let i = SHIFT; i < N; i++) values[i] += 6 * c.sd;
  return { values, expected: [...SPIKES.map((s) => s.index), SHIFT] };
}

/** The spec's rule: medium is the smallest grid value with at most 1% false flags on clean points. */
function tune(scores: number[]): AnomalyThresholds & { rate: number } {
  const rate = (t: number) => scores.filter((s) => s > t).length / scores.length;
  const medium = GRID.find((t) => rate(t) <= 0.01) ?? GRID[GRID.length - 1];
  return { low: medium + 1, medium, high: Math.max(2, medium - 0.5), rate: rate(medium) };
}

async function localModel(id: ModelId, dir: string): Promise<{ model: Forecaster; release: () => Promise<void> }> {
  const spec = STANDARD_FILES.fp16;
  const graph = spec.graph.replace("onnx/", "");
  const tmp = mkdtempSync(join(tmpdir(), "anomaly-int-"));
  copyFileSync(dir + graph, join(tmp, graph));
  writeFileSync(join(tmp, spec.half!.location), expandHalfToFloat(readFileSync(dir + spec.half!.file.replace("onnx/", ""))));
  const session = await ort.InferenceSession.create(join(tmp, graph), { executionProviders: ["cpu"], logSeverityLevel: 3 });
  const run: RunGraph = async (feeds) => {
    const tensors: Record<string, ort.Tensor> = {};
    for (const [k, f] of Object.entries(feeds)) tensors[k] = new ort.Tensor("float32", f.data, f.dims);
    const out = await session.run(tensors);
    return Object.fromEntries(Object.entries(out).map(([k, t]) => [k, t.data as Float32Array]));
  };
  const adapter = getAdapter(id);
  const forecast = adapter.forecaster(run, { model_type: id.startsWith("chronos") ? "chronos-bolt" : "timesfm" });
  const model = withAnalysis({ model: id, info: adapter.info, backend: "wasm", dtype: "fp16", config: { model_type: "test" }, forecast, dispose: async () => {} });
  return { model, release: async () => (await session.release(), rmSync(tmp, { recursive: true, force: true })) };
}

for (const id of Object.keys(MODELS) as ModelId[]) {
  const dir = fileURLToPath(new URL(`../../build/${id === "timesfm-2.5" ? "timesfm" : id}/onnx/`, import.meta.url));
  describe.skipIf(!existsSync(dir + STANDARD_FILES.fp16.graph.replace("onnx/", "")))(`anomalies: ${id}`, () => {
    let model: Forecaster;
    let release = async () => {};
    beforeAll(async () => ({ model, release } = await localModel(id, dir)), 600_000);
    afterAll(() => release());

    it("thresholds match the tuning rule and catch the planted anomalies", async () => {
      const suite = cleanSuite();
      const clean: number[] = [];
      for (const c of suite) clean.push(...(await model.detectAnomalies(c.values, { sensitivity: 1e9 })).points.map((p) => p.score));
      const tuned = tune(clean);
      const missed: string[] = [];
      let planted = 0;
      for (const c of suite) {
        const { values, expected } = plant(c);
        const res = await model.detectAnomalies(values, { sensitivity: tuned.medium });
        const flagged = new Set(res.anomalies.map((a) => a.index));
        planted += expected.length;
        for (const i of expected) if (!flagged.has(i)) missed.push(`${c.name}@${i} (score ${res.points.find((p) => p.index === i)?.score.toFixed(2)})`);
      }
      const caught = 1 - missed.length / planted;
      console.log(`${id}: tuned ${JSON.stringify(tuned)} over ${clean.length} clean points; caught ${planted - missed.length}/${planted}; missed ${missed.length ? missed.join(", ") : "none"}`);
      expect({ low: tuned.low, medium: tuned.medium, high: tuned.high }).toEqual(MODELS[id].anomalyThresholds);
      expect(tuned.rate).toBeLessThanOrEqual(0.01);
      // The spec's bar is every planted anomaly for TimesFM. Chronos-Bolt smooths random walks more, and
      // misses a few walk anomalies whose own noise cancelled part of the spike (measured 95 to 100%).
      if (id === "timesfm-2.5") expect(missed).toEqual([]);
      else expect(caught).toBeGreaterThanOrEqual(0.9);
    }, 3_600_000);

    it("calibrated 80% bands cover 70 to 90% of held-out heavy-tailed values", async () => {
      let raw = 0;
      let cal = 0;
      let total = 0;
      const scales: number[] = [];
      for (let seed = 1; seed <= 8; seed++) {
        const g = rng(9000 + seed);
        const v = Array.from({ length: 320 }, (_, i) => 10 + 3 * Math.sin((2 * Math.PI * i) / 24) + g.student(3));
        const history = v.slice(0, 256);
        const [r] = await model.forecast([history], { horizon: 64 });
        const [c] = await model.forecast([history], { horizon: 64, calibrate: true });
        scales.push(c.calibration!.scale);
        for (let s = 0; s < 64; s++) {
          const y = v[256 + s];
          raw += +(y >= r.q10[s] && y <= r.q90[s]);
          cal += +(y >= c.q10[s] && y <= c.q90[s]);
          total++;
        }
      }
      console.log(`${id}: held-out coverage raw ${(raw / total).toFixed(3)}, calibrated ${(cal / total).toFixed(3)}; scales ${scales.map((s) => s.toFixed(2)).join(" ")}`);
      expect(cal / total).toBeGreaterThanOrEqual(0.7);
      expect(cal / total).toBeLessThanOrEqual(0.9);
    }, 3_600_000);
  });
}
