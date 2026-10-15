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
  /** Anomaly score thresholds per sensitivity, tuned for this model. */
  anomalyThresholds: AnomalyThresholds;
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
  /** Present when the forecast was made with `calibrate: true`. */
  calibration?: Calibration;
}

/** How a forecast's bands were rescaled from a backtest over the series' own history. */
export interface Calibration {
  /** Factor applied to every quantile's distance from the median (>1 widened, <1 narrowed). */
  scale: number;
  /** Share of backtested values inside the raw 80% band (q10 to q90); null when too few were scored. */
  coverage: number | null;
  /** Backtested values the scale was computed from. */
  points: number;
}

export interface ForecastOptions {
  /** Steps to forecast, 1 to the model's maxHorizon. Defaults to the model's horizon. */
  horizon?: number;
  /**
   * Rescale the bands when, over a backtest of the series, the raw 80% range held less than 70% or more
   * than 90% of past values. Costs one extra batched forecast call per series.
   */
  calibrate?: boolean;
}

export type SensitivityLevel = "low" | "medium" | "high";

/** Score above which a point is flagged, keyed by sensitivity level: higher sensitivity uses a lower threshold. */
export interface AnomalyThresholds {
  low: number;
  medium: number;
  high: number;
}

/** A preset level from the model's `anomalyThresholds` (default "medium"), or a score threshold. */
export type Sensitivity = SensitivityLevel | number;

export interface AnomalyOptions {
  sensitivity?: Sensitivity;
}

/** One history point, compared with what the model expected from the points before it. */
export interface AnomalyPoint {
  index: number;
  value: number;
  /** The model's median forecast for this point. */
  expected: number;
  /** Calibrated 80% range around `expected`. */
  lower: number;
  upper: number;
  /** Distance from `expected` in calibrated standard deviations. */
  score: number;
  direction: "high" | "low";
  anomaly: boolean;
}

export interface AnomalyResult {
  /** Every point that was checked, oldest first. */
  points: AnomalyPoint[];
  /** The flagged points, oldest first. */
  anomalies: AnomalyPoint[];
  /** Calibration scale applied to the model's bands for this series. */
  scale: number;
  /** Score threshold used. */
  threshold: number;
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
  /** Flags points the model could not have predicted from the history before them. One batched forecast call. */
  detectAnomalies(series: Series, opts?: AnomalyOptions): Promise<AnomalyResult>;
  dispose(): Promise<void>;
}

export type ErrorCode =
  | "HORIZON_TOO_LONG"
  | "INVALID_HORIZON"
  | "EMPTY_INPUT"
  | "UNKNOWN_MODEL"
  | "TOO_MANY_CLASSES"
  | "TOO_LARGE"
  | "INVALID_INPUT"
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
