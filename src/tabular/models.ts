import type { FileSpec } from "src/models/adapter";
import { STANDARD_FILES } from "src/models/files";
import { ForecastError, type Backend, type ModelConfig, type ResolvedDtype } from "src/types";

/** Registry key of a tabular model. See `TABULAR_MODELS`. */
export type TabularModelId = "tabicl-classifier-v2" | "tabicl-regressor-v2";
/** What a tabular model predicts: a class or a number. */
export type TabularTask = "classification" | "regression";

/** What a tabular model supports, available without loading it. */
export interface TabularInfo {
  readonly id: TabularModelId;
  readonly name: string;
  readonly task: TabularTask;
  readonly source: string;
  readonly license: string;
  /** Most classes supported (classification). */
  readonly maxClasses: number;
  readonly dtypes: readonly ResolvedDtype[];
  readonly defaults: Readonly<Record<Backend, ResolvedDtype>>;
  /** Download size in bytes per dtype. */
  readonly sizes: Readonly<Partial<Record<ResolvedDtype, number>>>;
}

/** Everything the core needs to load a graph that is not a forecaster. */
export interface GraphSpec {
  info: Pick<TabularInfo, "name" | "dtypes" | "defaults">;
  weights: string;
  files(dtype: ResolvedDtype): FileSpec;
  checkConfig(config: ModelConfig): void;
}

const spec = (info: TabularInfo, weights: string): GraphSpec & { info: TabularInfo } => ({
  info: Object.freeze({ ...info, dtypes: Object.freeze([...info.dtypes]) as ResolvedDtype[], defaults: Object.freeze({ ...info.defaults }), sizes: Object.freeze({ ...info.sizes }) }),
  weights,
  files: (dtype) => STANDARD_FILES[dtype],
  checkConfig(config) {
    if (config.model_type !== "tabicl" || config.task !== info.task)
      throw new ForecastError("LOAD_FAILED", `expected a TabICL ${info.task} graph, got ${String(config.model_type)} ${String(config.task)}`);
  },
});

// Download sizes in bytes, from each weights repo's config.json. q8 is not shipped: for this
// network it is larger than fp16 weights (many small tensors stay fp32).
const TABULAR: Record<TabularModelId, GraphSpec & { info: TabularInfo }> = {
  "tabicl-classifier-v2": spec(
    {
      id: "tabicl-classifier-v2",
      name: "TabICL v2 classifier",
      task: "classification",
      source: "https://huggingface.co/jingang/TabICL",
      license: "BSD-3-Clause",
      maxClasses: 10,
      dtypes: ["fp16", "fp32"],
      defaults: { webgpu: "fp16", wasm: "fp16" },
      sizes: { fp16: 56_482_327, fp32: 111_449_927 },
    },
    "bakshinghampalace/tabicl-classifier-v2-onnx",
  ),
  "tabicl-regressor-v2": spec(
    {
      id: "tabicl-regressor-v2",
      name: "TabICL v2 regressor",
      task: "regression",
      source: "https://huggingface.co/jingang/TabICL",
      license: "BSD-3-Clause",
      maxClasses: 0,
      dtypes: ["fp16", "fp32"],
      defaults: { webgpu: "fp16", wasm: "fp16" },
      sizes: { fp16: 58_509_159, fp32: 115_489_568 },
    },
    "bakshinghampalace/tabicl-regressor-v2-onnx",
  ),
};

/** Every tabular model's info, without loading it. Read-only. */
export const TABULAR_MODELS = Object.freeze(
  Object.fromEntries(Object.entries(TABULAR).map(([id, s]) => [id, s.info])),
) as Readonly<Record<TabularModelId, TabularInfo>>;

export const isTabularId = (id: string): id is TabularModelId => Object.hasOwn(TABULAR, id);

export function getTabularSpec(id: TabularModelId): GraphSpec & { info: TabularInfo } {
  return TABULAR[id];
}

export const tabularModelFor = (task: TabularTask): TabularModelId => (task === "classification" ? "tabicl-classifier-v2" : "tabicl-regressor-v2");
