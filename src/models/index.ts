import type { ModelAdapter } from "src/models/adapter";
import { timesfm } from "src/models/timesfm/index";
import { ForecastError, type ModelId, type ModelInfo } from "src/types";

export const DEFAULT_MODEL_ID: ModelId = "timesfm-2.5";

const ADAPTERS: Record<ModelId, ModelAdapter> = {
  "timesfm-2.5": timesfm,
};

/** Freezes an object and every object or array inside it. */
function deepFreeze<T>(value: T): T {
  if (value && typeof value === "object" && !Object.isFrozen(value)) {
    Object.freeze(value);
    for (const v of Object.values(value)) deepFreeze(v);
  }
  return value;
}

// The loader reads info at runtime, so callers must not be able to change it through MODELS.
for (const a of Object.values(ADAPTERS)) deepFreeze(a.info);

/** Every available model's info, for pickers and download-size labels. Loads nothing. Read-only. */
export const MODELS = Object.freeze(
  Object.fromEntries(Object.entries(ADAPTERS).map(([id, a]) => [id, a.info])),
) as Readonly<Record<ModelId, ModelInfo>>;

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
