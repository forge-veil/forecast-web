import { validateHorizon } from "src/core/validate";
import { BLOCK, CONTEXT, MAX_HORIZON, NQ, QUANTILES } from "src/models/chronos-bolt/constants";
import { quantileLinear } from "src/models/chronos-bolt/postprocess";
import { lastContext, packBatch } from "src/models/chronos-bolt/preprocess";
import { ForecastError, type Forecast, type ForecastOptions, type Series } from "src/types";

/** Runs the graph on [batch, CONTEXT] context and mask; returns normalized [batch, 9, 64] quantiles. */
export type ChronosRun = (context: Float32Array, mask: Float32Array, batch: number) => Promise<Float32Array>;

/** Block of 9 quantile rows of 64 steps each, in the series' own units. */
type Block = Float64Array[];

async function pass(run: ChronosRun, rows: Float64Array[]): Promise<Block[]> {
  const { context, mask, stats } = packBatch(rows);
  const out = await run(context, mask, rows.length);
  return stats.map(({ loc, scale }, r) =>
    QUANTILES.map((_, q) => {
      const row = new Float64Array(BLOCK);
      const o = (r * NQ + q) * BLOCK;
      for (let t = 0; t < BLOCK; t++) row[t] = out[o + t] * scale + loc;
      return row;
    }),
  );
}

/** Appends `tail` and keeps the last CONTEXT points. */
function extend(x: Float64Array, tail: Float64Array): Float64Array {
  const n = Math.min(CONTEXT, x.length + tail.length);
  const out = new Float64Array(n);
  const keep = n - tail.length;
  out.set(x.subarray(x.length - keep), 0);
  out.set(tail, keep);
  return out;
}

/** 81 values per step (9 paths x 9 quantiles) -> 9 quantiles, as ChronosBoltPipeline.predict does. */
function reduce(paths: Block[]): Block {
  const res = QUANTILES.map(() => new Float64Array(BLOCK));
  const samples = new Float64Array(NQ * NQ);
  for (let t = 0; t < BLOCK; t++) {
    for (let p = 0; p < NQ; p++) for (let q = 0; q < NQ; q++) samples[p * NQ + q] = paths[p][q][t];
    samples.sort();
    QUANTILES.forEach((level, q) => (res[q][t] = quantileLinear(samples, level)));
  }
  return res;
}

function toForecast(blocks: Block[], horizon: number): Forecast {
  const row = (q: number) => blocks.flatMap((b) => Array.from(b[q])).slice(0, horizon);
  const [q10, q20, q30, q40, q50, q60, q70, q80, q90] = QUANTILES.map((_, q) => row(q));
  return { horizon, median: q50, q10, q20, q30, q40, q50, q60, q70, q80, q90 };
}

/**
 * Chronos-Bolt forecast with the official long-horizon heuristic: after the
 * first 64 steps, each quantile row extends its own copy of the context, all
 * 9 copies are forecast together, and the 81 values per step are reduced to 9.
 */
export function createForecaster(run: ChronosRun) {
  return async function forecast(series: Series[], opts: ForecastOptions = {}): Promise<Forecast[]> {
    const horizon = opts.horizon ?? BLOCK;
    validateHorizon(horizon, MAX_HORIZON);
    if (!series || series.length === 0) throw new ForecastError("EMPTY_INPUT", "pass at least one series");
    const contexts = series.map(lastContext);
    const first = await pass(run, contexts);
    const blocks: Block[][] = first.map((b) => [b]);
    let paths = contexts.map((x, s) => first[s].map((row) => extend(x, row)));
    for (let done = BLOCK; done < horizon; done += BLOCK) {
      const out = await pass(run, paths.flat());
      const reduced = series.map((_, s) => reduce(out.slice(s * NQ, (s + 1) * NQ)));
      reduced.forEach((b, s) => blocks[s].push(b));
      if (done + BLOCK < horizon) paths = paths.map((ps, s) => ps.map((x, q) => extend(x, reduced[s][q])));
    }
    return blocks.map((b) => toForecast(b, horizon));
  };
}
