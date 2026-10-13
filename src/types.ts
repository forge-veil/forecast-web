/** A time series, oldest first. NaN, null and undefined mark missing values. */
export type Series = ArrayLike<number | null | undefined>;

/** Registry key of a forecasting model. See MODELS. */
export type ModelId = "timesfm-2.5" | "chronos-bolt-tiny" | "chronos-bolt-mini" | "chronos-bolt-small" | "chronos-bolt-base";

/**
 * Weight format. "auto" picks the model's default for the backend
 * (TimesFM: fp16 on WebGPU, q8 on WASM). q8 runs on WASM only.
 */
export type Dtype = "auto" | "q8" | "fp16" | "fp32";
export type ResolvedDtype = Exclude<Dtype, "auto">;
export type Device = "auto" | "webgpu" | "wasm";
export type Backend = "webgpu" | "wasm";

/** What a model supports, available without loading it. */
export interface ModelInfo {
  id: ModelId;
  name: string;
  /** Upstream model page. */
  source: string;
  license: string;
  /** Most recent points the model reads; older history is ignored. */
  maxContext: number;
  maxHorizon: number;
  /** Steps one forward pass produces; the default horizon. Longer horizons up to maxHorizon take more passes. */
  horizon: number;
  dtypes: ResolvedDtype[];
  /** The dtype "auto" picks on each backend. */
  defaults: Record<Backend, ResolvedDtype>;
  /** Download size in bytes per dtype. */
  sizes: Partial<Record<ResolvedDtype, number>>;
}

export interface Forecast {
  horizon: number;
  /** Mean head output, for models that have one (TimesFM does). */
  mean?: number[];
  /** The median (q50), the point forecast. */
  median: number[];
  q10: number[];
  q20: number[];
  q30: number[];
  q40: number[];
  q50: number[];
  q60: number[];
  q70: number[];
  q80: number[];
  q90: number[];
}

export interface ForecastOptions {
  /** Steps to forecast, 1 to the model's maxHorizon. Defaults to the model's horizon. */
  horizon?: number;
}

/** The weights repo's config.json. Fields beyond these are model-specific. */
export interface ModelConfig {
  model_type: string;
  /** Size and SHA-256 per weight file, relative to the weights base URL. */
  files?: Record<string, { bytes: number; sha256: string }>;
  [key: string]: unknown;
}

export interface Progress {
  loaded: number;
  total: number;
  /** True when every byte so far came from the browser cache. */
  cached: boolean;
}

export interface LoadOptions {
  /** Which model to run. Defaults to "timesfm-2.5". */
  model?: ModelId;
  /** Hugging Face repo id, or a base URL that contains config.json and onnx/. Defaults to the model's published weights. */
  weights?: string;
  /** Hugging Face revision. Ignored for base URLs. Defaults to "main". */
  revision?: string;
  dtype?: Dtype;
  device?: Device;
  /** Run inference in a Web Worker. Defaults to true where Worker exists. */
  worker?: boolean;
  /** Store downloads in the Cache API. Defaults to true. */
  cache?: boolean;
  /** Where onnxruntime-web fetches its .wasm files. Defaults to jsDelivr for the installed version. */
  wasmPaths?: string;
  onProgress?: (p: Progress) => void;
  signal?: AbortSignal;
}

export interface Forecaster {
  readonly model: ModelId;
  readonly info: ModelInfo;
  readonly backend: Backend;
  readonly dtype: ResolvedDtype;
  readonly config: ModelConfig;
  forecast(series: Series[], opts?: ForecastOptions): Promise<Forecast[]>;
  dispose(): Promise<void>;
}

export type ErrorCode =
  | "HORIZON_TOO_LONG"
  | "INVALID_HORIZON"
  | "EMPTY_INPUT"
  | "UNKNOWN_MODEL"
  | "LOAD_FAILED"
  | "ABORTED"
  | "DISPOSED";

export class ForecastError extends Error {
  readonly code: ErrorCode;
  constructor(code: ErrorCode, message: string, options?: { cause?: unknown }) {
    super(message, options);
    this.name = "ForecastError";
    this.code = code;
  }
}
