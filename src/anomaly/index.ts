import type { ForecastFn } from "src/anomaly/backtest";
import { calibrateForecasts } from "src/anomaly/calibrate";
import { detectAnomalies } from "src/anomaly/detect";
import { ForecastError, type Forecaster } from "src/types";

type BaseForecaster = Omit<Forecaster, "detectAnomalies">;

/** Adds `detectAnomalies` and `forecast(..., { calibrate: true })` on top of a model's plain forecast. */
export function withAnalysis(base: BaseForecaster): Forecaster {
  let disposed = false;
  // Checked before any input validation, so a disposed model always says so.
  const live = () => {
    if (disposed) throw new ForecastError("DISPOSED", "this model has been disposed");
  };
  const run: ForecastFn = (series, { horizon }) => base.forecast(series, { horizon });
  return {
    get model() {
      return base.model;
    },
    get info() {
      return base.info;
    },
    get backend() {
      return base.backend;
    },
    get dtype() {
      return base.dtype;
    },
    async forecast(series, opts) {
      live();
      if (!opts?.calibrate) return base.forecast(series, opts);
      const { calibrate: _, ...rest } = opts;
      const forecasts = await base.forecast(series, rest);
      return calibrateForecasts(series, forecasts, run, base.info, rest.horizon ?? base.info.horizon);
    },
    async detectAnomalies(series, opts) {
      live();
      return detectAnomalies(series, run, base.info, opts);
    },
    dispose() {
      disposed = true;
      return base.dispose();
    },
  };
}
