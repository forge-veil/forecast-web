import type { ModelAdapter } from "src/models/adapter";
import { timesfm } from "src/models/timesfm/index";
import { ForecastError, type ModelId, type ModelInfo } from "src/types";

export const DEFAULT_MODEL_ID: ModelId = "timesfm-2.5";

const ADAPTERS: Record<ModelId, ModelAdapter> = {
  "timesfm-2.5": timesfm,
};

/** Every available model's info, for pickers and download-size labels. Loads nothing. */
export const MODELS = Object.fromEntries(
  Object.entries(ADAPTERS).map(([id, a]) => [id, a.info]),
) as Record<ModelId, ModelInfo>;

export function getAdapter(id: string): ModelAdapter {
  if (typeof id === "string" && Object.hasOwn(ADAPTERS, id)) return ADAPTERS[id as ModelId];
  // Plain JS callers can pass anything; report it as a ForecastError, never a TypeError.
  const name = String(id);
  const hint = name.includes("/") ? '; pass Hugging Face repo ids and URLs as "weights"' : "";
  throw new ForecastError(
    "UNKNOWN_MODEL",
    `unknown model "${name}", expected one of: ${Object.keys(ADAPTERS).join(", ")}${hint}`,
  );
}
