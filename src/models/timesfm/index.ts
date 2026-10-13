import type { ModelAdapter } from "src/models/adapter";
import { STANDARD_FILES } from "src/models/files";
import { CONTEXT, MAX_HORIZON } from "src/models/timesfm/constants";
import { createForecaster } from "src/models/timesfm/forecaster";
import { ForecastError } from "src/types";

export const timesfm: ModelAdapter = {
  info: {
    id: "timesfm-2.5",
    name: "TimesFM 2.5",
    source: "https://huggingface.co/google/timesfm-2.5-200m-pytorch",
    license: "Apache-2.0",
    maxContext: CONTEXT,
    maxHorizon: MAX_HORIZON,
    horizon: MAX_HORIZON,
    dtypes: ["fp16", "q8", "fp32"],
    defaults: { webgpu: "fp16", wasm: "q8" },
    sizes: { fp16: 463_368_499, q8: 261_301_673, fp32: 925_925_778 },
  },
  weights: "bakshinghampalace/timesfm-2.5-200m-onnx",
  files: (dtype) => STANDARD_FILES[dtype],
  checkConfig(config) {
    if (config.context !== CONTEXT || config.horizon !== MAX_HORIZON)
      throw new ForecastError(
        "LOAD_FAILED",
        `unsupported model shape: context ${config.context}, horizon ${config.horizon} (expected ${CONTEXT} and ${MAX_HORIZON})`,
      );
  },
  forecaster(run) {
    return createForecaster(async (inputs, masks, batch) => {
      const out = await run({
        inputs: { data: inputs, dims: [batch, CONTEXT] },
        masks: { data: masks, dims: [batch, CONTEXT] },
      });
      return { point: out.point, qspread: out.qspread };
    }, CONTEXT);
  },
};
