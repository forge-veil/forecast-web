import type { ForecastFn } from "src/anomaly/backtest";
import { calibrateForecasts } from "src/anomaly/calibrate";
import { detectAnomalies } from "src/anomaly/detect";
import type { Forecaster } from "src/types";

type BaseForecaster = Omit<Forecaster, "detectAnomalies">;

/** Adds `detectAnomalies` and `forecast(..., { calibrate: true })` on top of a model's plain forecast. */
export function withAnalysis(base: BaseForecaster): Forecaster {
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
    get config() {
      return base.config;
    },
    async forecast(series, opts) {
      if (!opts?.calibrate) return base.forecast(series, opts);
      const { calibrate: _, ...rest } = opts;
      const forecasts = await base.forecast(series, rest);
      return calibrateForecasts(series, forecasts, run, base.info, rest.horizon ?? base.info.horizon);
    },
    detectAnomalies: (series, opts) => detectAnomalies(series, run, base.info, opts),
    dispose: () => base.dispose(),
  };
}
