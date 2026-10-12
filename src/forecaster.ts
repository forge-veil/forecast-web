import { postprocess } from "src/postprocess";
import { prepareBatch } from "src/preprocess";
import {
  CONTEXT,
  MAX_HORIZON,
  TimesFMError,
  type Forecast,
  type ForecastOptions,
  type Series,
} from "src/types";

/** Runs the graph on a [batch, context] input and returns flat [batch, 128, 10] outputs. */
export type RunGraph = (
  inputs: Float32Array,
  masks: Float32Array,
  batch: number,
) => Promise<{ point: Float32Array; qspread: Float32Array }>;

export function validateHorizon(horizon: number): void {
  if (!Number.isInteger(horizon) || horizon < 1)
    throw new TimesFMError("INVALID_HORIZON", `horizon must be a positive integer, got ${horizon}`);
  if (horizon > MAX_HORIZON)
    throw new TimesFMError("HORIZON_TOO_LONG", `horizon ${horizon} exceeds the model limit of ${MAX_HORIZON}`);
}

/** Wires preprocessing, a graph runner, and postprocessing into one forecast call. */
export function createForecaster(run: RunGraph, context = CONTEXT) {
  return async function forecast(series: Series[], opts: ForecastOptions = {}): Promise<Forecast[]> {
    const horizon = opts.horizon ?? MAX_HORIZON;
    validateHorizon(horizon);
    if (!series || series.length === 0) throw new TimesFMError("EMPTY_INPUT", "pass at least one series");
    const { inputs, masks, stats } = prepareBatch(series, context);
    const { point, qspread } = await run(inputs, masks, 2 * series.length);
    return postprocess(point, qspread, stats, horizon);
  };
}
