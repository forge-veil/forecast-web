/** Fixed shape of the exported graph. */
export const CONTEXT = 512;
export const MAX_HORIZON = 128;
/** Output channels: 0 = mean head, 1..9 = q10..q90. */
export const CHANNELS = 10;
export const QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] as const;

/** A time series, oldest first. NaN, null and undefined mark missing values. */
export type Series = ArrayLike<number | null | undefined>;

export type Dtype = "q8" | "fp16" | "fp32";
export type Device = "auto" | "webgpu" | "wasm";
export type Backend = "webgpu" | "wasm";

export interface SeriesStats {
  mu: number;
  sigma: number;
  isPositive: boolean;
}

export interface Forecast {
  horizon: number;
  /** Output of the mean head (channel 0). */
  mean: number[];
  /** The median (q50). This is the official TimesFM point forecast. */
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
  /** Steps to forecast, 1 to 128. Defaults to 128. */
  horizon?: number;
}

export interface ModelConfig {
  model_type: string;
  context: number;
  horizon: number;
  patch: number;
  quantiles: number[];
}

export interface Progress {
  loaded: number;
  total: number;
  /** True when every byte so far came from the browser cache. */
  cached: boolean;
}

export interface LoadOptions {
  /** Hugging Face repo id, or a base URL that contains config.json and onnx/. */
  model?: string;
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

export interface TimesFM {
  readonly backend: Backend;
  readonly dtype: Dtype;
  readonly config: ModelConfig;
  forecast(series: Series[], opts?: ForecastOptions): Promise<Forecast[]>;
  dispose(): Promise<void>;
}

export type ErrorCode =
  | "HORIZON_TOO_LONG"
  | "INVALID_HORIZON"
  | "EMPTY_INPUT"
  | "LOAD_FAILED"
  | "ABORTED"
  | "DISPOSED";

export class TimesFMError extends Error {
  readonly code: ErrorCode;
  constructor(code: ErrorCode, message: string, options?: { cause?: unknown }) {
    super(message, options);
    this.name = "TimesFMError";
    this.code = code;
  }
}
