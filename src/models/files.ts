import type { FileSpec } from "src/models/adapter";
import type { ResolvedDtype } from "src/types";

/**
 * fp32 and q8 use Transformers.js file names. fp16 is a compact pair: an fp32
 * graph whose weights are external data, shipped as fp16 and expanded on load.
 */
export const STANDARD_FILES: Record<ResolvedDtype, FileSpec> = {
  fp32: { graph: "onnx/model.onnx" },
  fp16: { graph: "onnx/model_fp16w.onnx", half: { file: "onnx/model_fp16w.f16", location: "model_fp16w.onnx_data" } },
  q8: { graph: "onnx/model_quantized.onnx" },
};
