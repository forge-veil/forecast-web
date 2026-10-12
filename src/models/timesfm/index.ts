import type { FileSpec, ModelAdapter } from "src/models/adapter";
import { CONTEXT, MAX_HORIZON } from "src/models/timesfm/constants";
import { createForecaster } from "src/models/timesfm/forecaster";
import { ForecastError, type ResolvedDtype } from "src/types";

/**
 * fp32 and q8 use Transformers.js file names. fp16 is a compact pair: an fp32
 * graph whose weights are external data, shipped as fp16 and expanded on load.
 */
const FILES: Record<ResolvedDtype, FileSpec> = {
  fp32: { graph: "onnx/model.onnx" },
  fp16: { graph: "onnx/model_fp16w.onnx", half: { file: "onnx/model_fp16w.f16", location: "model_fp16w.onnx_data" } },
  q8: { graph: "onnx/model_quantized.onnx" },
};

export const timesfm: ModelAdapter = {
  info: {
    id: "timesfm-2.5",
    name: "TimesFM 2.5",
    source: "https://huggingface.co/google/timesfm-2.5-200m-pytorch",
    license: "Apache-2.0",
    maxContext: CONTEXT,
    maxHorizon: MAX_HORIZON,
    dtypes: ["fp16", "q8", "fp32"],
    defaults: { webgpu: "fp16", wasm: "q8" },
    sizes: { fp16: 463_368_499, q8: 261_301_673, fp32: 925_925_778 },
  },
  weights: "bakshinghampalace/timesfm-2.5-200m-onnx",
  files: (dtype) => FILES[dtype],
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
