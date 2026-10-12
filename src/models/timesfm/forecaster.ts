import { validateHorizon } from "src/core/validate";
import { CONTEXT, MAX_HORIZON } from "src/models/timesfm/constants";
import { postprocess } from "src/models/timesfm/postprocess";
import { prepareBatch } from "src/models/timesfm/preprocess";
import { ForecastError, type Forecast, type ForecastOptions, type Series } from "src/types";

/** Runs the graph on a [batch, context] input and returns flat [batch, 128, 10] outputs. */
export type TimesFMRun = (
  inputs: Float32Array,
  masks: Float32Array,
  batch: number,
) => Promise<{ point: Float32Array; qspread: Float32Array }>;

/** Wires preprocessing, a graph runner, and postprocessing into one forecast call. */
export function createForecaster(run: TimesFMRun, context = CONTEXT) {
  return async function forecast(series: Series[], opts: ForecastOptions = {}): Promise<Forecast[]> {
    const horizon = opts.horizon ?? MAX_HORIZON;
    validateHorizon(horizon, MAX_HORIZON);
    if (!series || series.length === 0) throw new ForecastError("EMPTY_INPUT", "pass at least one series");
    const { inputs, masks, stats } = prepareBatch(series, context);
    const { point, qspread } = await run(inputs, masks, 2 * series.length);
    return postprocess(point, qspread, stats, horizon);
  };
}
