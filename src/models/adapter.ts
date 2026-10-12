import type { Forecast, ForecastOptions, ModelConfig, ModelInfo, ResolvedDtype, Series } from "src/types";

export interface Feed {
  data: Float32Array;
  dims: number[];
}

/** Runs the ONNX graph on named float32 inputs and returns every output by name. */
export type RunGraph = (feeds: Record<string, Feed>) => Promise<Record<string, Float32Array>>;

export type ForecastFn = (series: Series[], opts?: ForecastOptions) => Promise<Forecast[]>;

/** Weight files for one dtype, relative to the weights base URL. */
export interface FileSpec {
  graph: string;
  /** fp16 weights the loader expands and passes to onnxruntime as external data named `location`. */
  half?: { file: string; location: string };
}

/** Everything model-specific. Shared code never branches on the model. */
export interface ModelAdapter {
  info: ModelInfo;
  /** Default weights: a Hugging Face repo id. */
  weights: string;
  files(dtype: ResolvedDtype): FileSpec;
  /** Throws LOAD_FAILED when config.json describes a graph this adapter cannot drive. */
  checkConfig(config: ModelConfig): void;
  forecaster(run: RunGraph, config: ModelConfig): ForecastFn;
}
