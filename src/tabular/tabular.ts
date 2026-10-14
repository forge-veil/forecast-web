import type { Feed, RunGraph } from "src/models/adapter";
import { combineClassification } from "src/tabular/combine";
import { encodeLabels, fitEncoder, type Cell } from "src/tabular/encode";
import { ensembleConfigs, type MemberConfig, type Norm } from "src/tabular/ensemble";
import type { TabularInfo } from "src/tabular/models";
import { fitPipeline, fitUniqueFilter, type Pipeline } from "src/tabular/preprocess";
import { quantileStats } from "src/tabular/quantiles";
import { ForecastError } from "src/types";

/**
 * Largest tables the browser handles comfortably; above these, fit throws TOO_LARGE.
 * Cost grows with training rows x features: measured on onnxruntime-node CPU with 8
 * members and 100 test rows, 500x10 took 2.5 s, 1000x10 5.4 s, 2000x20 67 s, 1000x50 91 s.
 * WebGPU is faster; fewer ensemble members (nEstimators) cut time proportionally.
 */
export const LIMITS = { maxRows: 5000, maxFeatures: 100, maxCells: 20_000 };

const NORMS: Norm[] = ["none", "power"];
const SEED = 42;
const f32 = Math.fround;

export interface TabularOptions {
  /** Ensemble members. Defaults to 8, as upstream; 1 is fastest. */
  nEstimators?: number;
}

export interface RegressionOptions {
  output?: "mean" | "median" | "quantiles";
  /** Quantile levels for output "quantiles". Defaults to 0.1 through 0.9. */
  alphas?: number[];
}

export interface FittedTabular {
  /** Class labels in output order (classification). */
  readonly classes: (number | string)[];
  predictProba(X: Cell[][]): Promise<number[][]>;
  predict(X: Cell[][], opts?: RegressionOptions): Promise<(number | string)[] | number[] | number[][]>;
}

export interface Tabular {
  readonly info: TabularInfo;
  fit(X: Cell[][], y: (number | string)[]): FittedTabular;
  dispose(): Promise<void>;
}

function validate(X: Cell[][], y: unknown[]) {
  if (!Array.isArray(X) || X.length === 0 || !Array.isArray(X[0]) || X[0].length === 0)
    throw new ForecastError("INVALID_INPUT", "X must be a non-empty array of rows");
  const width = X[0].length;
  if (X.some((r) => !Array.isArray(r) || r.length !== width)) throw new ForecastError("INVALID_INPUT", "every row of X needs the same number of columns");
  if (!Array.isArray(y) || y.length !== X.length) throw new ForecastError("INVALID_INPUT", `y has ${y?.length} labels for ${X.length} rows`);
  if (X.length > LIMITS.maxRows || width > LIMITS.maxFeatures || X.length * width > LIMITS.maxCells)
    throw new ForecastError(
      "TOO_LARGE",
      `table is ${X.length} rows x ${width} columns; the browser limits are ${LIMITS.maxRows} rows, ${LIMITS.maxFeatures} columns and ${LIMITS.maxCells} cells`,
    );
}

function validateTest(X: Cell[][], width: number) {
  if (!Array.isArray(X) || X.some((r) => !Array.isArray(r) || r.length !== width))
    throw new ForecastError("INVALID_INPUT", `every test row needs ${width} columns, like the training rows`);
}

/** float32 StandardScaler on y, as TabICLRegressor does: mean and variance in float64 over float32 values. */
function scaleTarget(y: number[]) {
  const y32 = Float32Array.from(y);
  let mean = 0;
  for (const v of y32) mean += v;
  mean /= y32.length;
  let sq = 0;
  let corr = 0;
  for (const v of y32) {
    sq += (v - mean) ** 2;
    corr += v - mean;
  }
  const variance = (sq - (corr * corr) / y32.length) / y32.length;
  const eps = 2.220446049250313e-16;
  const constant = variance <= y32.length * eps * variance + (y32.length * mean * eps) ** 2;
  const scale = constant ? 1 : Math.sqrt(variance);
  return { mean, scale, scaled: Float32Array.from(y32, (v) => f32(f32(v - mean) / scale)) };
}

/** Builds the tabular API around a loaded graph. Exported for tests; use loadTabular. */
export function createTabular(graph: { run: RunGraph; dispose(): Promise<void> }, info: TabularInfo, opts: TabularOptions = {}): Tabular {
  const nEstimators = opts.nEstimators ?? 8;
  if (!Number.isInteger(nEstimators) || nEstimators < 1)
    throw new ForecastError("INVALID_INPUT", `nEstimators must be a positive integer, got ${nEstimators}`);
  const classification = info.task === "classification";

  return {
    info,
    dispose: () => graph.dispose(),
    fit(X, y) {
      validate(X, y);
      let codes: number[] = [];
      let classes: (number | string)[] = [];
      let target: ReturnType<typeof scaleTarget> | null = null;
      if (classification) {
        if (y.some((v) => (typeof v !== "number" && typeof v !== "string") || (typeof v === "number" && !Number.isFinite(v))))
          throw new ForecastError("INVALID_INPUT", "class labels must be numbers or strings, with no missing values");
        ({ codes, classes } = encodeLabels(y));
        if (classes.length > info.maxClasses)
          throw new ForecastError("TOO_MANY_CLASSES", `${classes.length} classes; ${info.name} supports at most ${info.maxClasses}`);
      } else {
        if (y.some((v) => typeof v !== "number" || !Number.isFinite(v))) throw new ForecastError("INVALID_INPUT", "regression targets must be finite numbers");
        target = scaleTarget(y as number[]);
      }
      const encoder = fitEncoder(X);
      const all = encoder.transform(X);
      const keep = fitUniqueFilter(all);
      const train = keep.map((c) => all[c]);
      const members = ensembleConfigs({
        nFeatures: keep.length,
        nClasses: classification ? classes.length : undefined,
        nEstimators,
        seed: SEED,
        normMethods: NORMS,
      });
      const pipelines = new Map<Norm, Pipeline>();
      for (const m of members) if (!pipelines.has(m.norm)) pipelines.set(m.norm, fitPipeline(train, m.norm));
      const n = X.length;
      const h = keep.length;
      const width = X[0].length;
      // Test rows never influence each other, so predicting in chunks gives identical results
      // while keeping each graph call within twice the training cell budget.
      const chunkRows = Math.max(1, Math.floor((2 * LIMITS.maxCells) / h) - n);
      async function chunked<T>(Xtest: Cell[][], fn: (part: Cell[][]) => Promise<T[]>): Promise<T[]> {
        validateTest(Xtest, width);
        const out: T[] = [];
        for (let i = 0; i < Xtest.length; i += chunkRows) out.push(...(await fn(Xtest.slice(i, i + chunkRows))));
        return out;
      }

      /** One graph call per norm group, members in upstream order; returns raw outputs per group. */
      const runAll = async (Xtest: Cell[][]) => {
        const test = encoder.transform(Xtest);
        const testKept = keep.map((c) => test[c]);
        const m = Xtest.length;
        const results: { members: MemberConfig[]; out: Float32Array }[] = [];
        for (const norm of [...new Set(members.map((x) => x.norm))]) {
          const group = members.filter((x) => x.norm === norm);
          const p = pipelines.get(norm)!;
          const trainT = p.trainTransformed;
          const testT = p.transform(testKept);
          const t = n + m;
          const Xd = new Float32Array(group.length * t * h);
          const yd = new Float32Array(group.length * n);
          group.forEach((mem, b) => {
            for (let r = 0; r < t; r++)
              for (let j = 0; j < h; j++) Xd[(b * t + r) * h + j] = r < n ? trainT[mem.feat[j]][r] : testT[mem.feat[j]][r - n];
            for (let r = 0; r < n; r++) yd[b * n + r] = classification ? mem.cls![codes[r]] : target!.scaled[r];
          });
          const feeds: Record<string, Feed> = { X: { data: Xd, dims: [group.length, t, h] }, y_train: { data: yd, dims: [group.length, n] } };
          const outputs = await graph.run(feeds);
          results.push({ members: group, out: outputs[classification ? "logits" : "quantiles"] });
        }
        return { results, m };
      };

      const predictProba = (Xtest: Cell[][]): Promise<number[][]> =>
        chunked(Xtest, async (part) => {
          if (!classification) throw new ForecastError("INVALID_INPUT", "predictProba is for classification");
          if (classes.length === 1) return part.map(() => [1]);
          const { results, m } = await runAll(part);
          const logits = results.flatMap(({ members: group, out }) =>
            group.map((mem, b) => ({ logits: out.subarray(b * m * 10, (b + 1) * m * 10), stride: 10, cls: mem.cls! })),
          );
          return combineClassification(logits, m, classes.length);
        });

      return {
        classes,
        predictProba,
        async predict(Xtest, popts = {}) {
          if (classification) {
            const proba = await predictProba(Xtest);
            return proba.map((p) => classes[p.indexOf(Math.max(...p))]);
          }
          const output = popts.output ?? "mean";
          const alphas = popts.alphas ?? [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
          const values = await chunked<number | number[]>(Xtest, async (part) => {
            const { results, m } = await runAll(part);
            const { mean, scale } = target!;
            const back = (v: number) => v * scale + mean;
            // Per member: de-normalize each statistic, then average members, as upstream.
            const perMember = results.flatMap(({ members: group, out }) =>
              group.map((_, b) =>
                Array.from({ length: m }, (_, r) => {
                  const s = quantileStats(out.subarray((b * m + r) * 999, (b * m + r + 1) * 999), output === "quantiles" ? alphas : []);
                  return { mean: back(s.mean), median: back(s.median), quantiles: s.quantiles.map(back) };
                }),
              ),
            );
            const k = perMember.length;
            if (output === "quantiles")
              return Array.from({ length: m }, (_, r) => alphas.map((_, a) => perMember.reduce((acc, mem) => acc + mem[r].quantiles[a], 0) / k));
            return Array.from({ length: m }, (_, r) => perMember.reduce((acc, mem) => acc + mem[r][output], 0) / k);
          });
          return values as number[] | number[][];
        },
      };
    },
  };
}
