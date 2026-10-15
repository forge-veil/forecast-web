import type { ModelAdapter } from "src/models/adapter";
import { BLOCK, CONTEXT, MAX_HORIZON, NQ, PATCH } from "src/models/chronos-bolt/constants";
import { createForecaster } from "src/models/chronos-bolt/forecaster";
import { STANDARD_FILES } from "src/models/files";
import { ForecastError, type ModelId, type ModelInfo, type ResolvedDtype } from "src/types";

export type ChronosSize = "tiny" | "mini" | "small" | "base";

export function chronosBolt(
  size: ChronosSize,
  sizes: ModelInfo["sizes"],
  /** Tuned per size by test/anomaly/integration.test.ts. */
  anomalyThresholds: ModelInfo["anomalyThresholds"] = { low: 4, medium: 3, high: 2.5 },
  dtypes: ResolvedDtype[] = ["fp16", "q8", "fp32"],
): ModelAdapter {
  const id = `chronos-bolt-${size}` as ModelId;
  return {
    info: {
      id,
      name: `Chronos-Bolt ${size[0].toUpperCase()}${size.slice(1)}`,
      source: `https://huggingface.co/amazon/chronos-bolt-${size}`,
      license: "Apache-2.0",
      maxContext: CONTEXT,
      maxHorizon: MAX_HORIZON,
      horizon: BLOCK,
      dtypes,
      // fp16 on WASM too: q8 matches Python for one 64-step pass, but past that each pass
      // feeds its own quantiles back and the drift compounds (up to ~14% of std at 200 steps).
      defaults: { webgpu: "fp16", wasm: "fp16" },
      sizes,
      anomalyThresholds,
    },
    weights: `bakshinghampalace/chronos-bolt-${size}-onnx`,
    files: (dtype) => STANDARD_FILES[dtype],
    checkConfig(config) {
      const q = config.quantiles;
      if (
        config.context !== CONTEXT ||
        config.prediction_length !== BLOCK ||
        config.patch !== PATCH ||
        !Array.isArray(q) ||
        q.length !== NQ
      )
        throw new ForecastError(
          "LOAD_FAILED",
          `unsupported Chronos-Bolt config: context ${config.context}, prediction_length ${config.prediction_length}, patch ${config.patch} (expected ${CONTEXT}, ${BLOCK}, ${PATCH} and ${NQ} quantiles)`,
        );
    },
    forecaster(run) {
      return createForecaster(async (context, mask, batch) => {
        const out = await run({
          context: { data: context, dims: [batch, CONTEXT] },
          mask: { data: mask, dims: [batch, CONTEXT] },
        });
        return out.quantiles;
      });
    },
  };
}
