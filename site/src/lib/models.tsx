import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import {
  loadForecaster,
  MODELS,
  type AnomalyResult,
  type Backend,
  type Forecast,
  type Forecaster,
  type ModelId,
  type ModelInfo,
  type Progress,
} from "forecast-web";

export const DEFAULT_MODEL: ModelId = "timesfm-2.5";
const KEY = "forecast-web-demo:model";

/** The only model that reliably fits in a phone's memory; larger ones crashed mobile browsers in testing. */
export const MOBILE_MODEL: ModelId = "chronos-bolt-tiny";

export function readChoice(stored: string | null, mobile = false): ModelId {
  if (stored && Object.hasOwn(MODELS, stored)) return stored as ModelId;
  return mobile ? MOBILE_MODEL : DEFAULT_MODEL;
}

/** Phones and tablets, if either the browser's mobile hint or the user agent says so (iPads can report a Mac). */
export function isMobile(nav: Navigator | undefined = typeof navigator !== "undefined" ? navigator : undefined): boolean {
  if (!nav) return false;
  if ((nav as Navigator & { userAgentData?: { mobile?: boolean } }).userAgentData?.mobile === true) return true;
  const ua = nav.userAgent ?? "";
  return /Android|iPhone|iPad|iPod|Mobile/i.test(ua) || (/Macintosh/.test(ua) && (nav.maxTouchPoints ?? 0) > 1);
}

/** Label for a model in the picker on mobile: Tiny is recommended, anything larger may crash. */
export function mobileNote(id: ModelId, mobile: boolean): { text: string; warn: boolean } | null {
  if (!mobile) return null;
  return id === MOBILE_MODEL ? { text: "Recommended on phones", warn: false } : { text: "May crash on phones", warn: true };
}

/** Picker order: the mobile model first on mobile, registry order otherwise. */
export const menuOrder = (ids: ModelId[], mobile: boolean): ModelId[] =>
  mobile ? [...ids.filter((id) => id === MOBILE_MODEL), ...ids.filter((id) => id !== MOBILE_MODEL)] : ids;

/** Bytes "auto" downloads on this machine. */
export function downloadBytes(info: ModelInfo, gpu: boolean): number | undefined {
  return info.sizes[info.defaults[gpu ? "webgpu" : "wasm"]];
}

export const hasGpu = () => typeof navigator !== "undefined" && "gpu" in navigator;

/** Identifies a forecast by model and series, so a model switch never reuses another model's result. */
export const forecastKey = (model: ModelId, v: number[]) => `${model}|${v.length}:${v[0]}:${v[v.length - 1]}`;

export function compareVerdict(a: { name: string; mae: number }, b: { name: string; mae: number }): string {
  const [win, lose] = a.mae <= b.mae ? [a, b] : [b, a];
  const gain = lose.mae > 0 ? 1 - win.mae / lose.mae : 0;
  if (gain <= 0.02) return `${a.name} and ${b.name}: same error`;
  return `${win.name}: ${Math.round(gain * 100)}% less error than ${lose.name}`;
}

export const MODEL_IDS = Object.keys(MODELS) as ModelId[];

/** Fixed color token per model, in registry order (validated as a categorical palette in styles.css). */
export const modelColor = (id: ModelId) => `var(--m${MODEL_IDS.indexOf(id) + 1})`;

/** Scored models first, lowest error first; unscored ones after, in registry order. */
export function rankByError<T extends { id: ModelId; mae?: number }>(rows: T[]): T[] {
  const scored = rows.filter((r) => r.mae !== undefined && Number.isFinite(r.mae)).sort((a, b) => a.mae! - b.mae!);
  return [...scored, ...rows.filter((r) => !scored.includes(r))];
}

/** The model that draws the band: the selected one, or the first loaded model while it is not ready. */
export function resolvePrimary<T extends string>(choice: T, loaded: T[]): T {
  return loaded.length === 0 || loaded.includes(choice) ? choice : loaded[0];
}

export type Status = "idle" | "loading" | "ready" | "error";
export interface Slot {
  status: Status;
  progress: Progress | null;
  backend: Backend | null;
  error: Error | null;
}
const IDLE: Slot = { status: "idle", progress: null, backend: null, error: null };

interface Models {
  choice: ModelId;
  setChoice(id: ModelId): void;
  slot(id: ModelId): Slot;
  load(id: ModelId): Promise<Forecaster>;
  forecast(id: ModelId, series: number[][], opts: { horizon: number; calibrate?: boolean }): Promise<Forecast[]>;
  detectAnomalies(id: ModelId, series: number[]): Promise<AnomalyResult>;
}

const Ctx = createContext<Models | null>(null);

const readStored = () => {
  try {
    return readChoice(localStorage.getItem(KEY), isMobile());
  } catch {
    return readChoice(null, isMobile());
  }
};

/** Holds one Forecaster per model id, created on first use. `weights` overrides TimesFM's weights. */
export function ModelsProvider({ weights, children }: { weights?: string; children: ReactNode }) {
  const [choice, setChoiceState] = useState<ModelId>(readStored);
  const [slots, setSlots] = useState<Partial<Record<ModelId, Slot>>>({});
  const loads = useRef(new Map<ModelId, Promise<Forecaster>>());
  const patch = (id: ModelId, s: Partial<Slot>) => setSlots((m) => ({ ...m, [id]: { ...(m[id] ?? IDLE), ...s } }));

  const load = useCallback(
    (id: ModelId) => {
      const existing = loads.current.get(id);
      if (existing) return existing;
      patch(id, { status: "loading", error: null });
      const p = loadForecaster({
        model: id,
        weights: id === DEFAULT_MODEL ? weights : undefined,
        onProgress: (progress) => patch(id, { progress }),
      });
      loads.current.set(id, p);
      p.then(
        (m) => patch(id, { status: "ready", backend: m.backend }),
        (e: unknown) => {
          loads.current.delete(id);
          patch(id, { status: "error", error: e instanceof Error ? e : new Error(String(e)) });
        },
      );
      return p;
    },
    [weights],
  );

  useEffect(() => {
    const all = loads.current;
    return () => all.forEach((p) => p.then((m) => m.dispose()).catch(() => {}));
  }, []);

  const value = useMemo<Models>(
    () => ({
      choice,
      setChoice(id) {
        setChoiceState(id);
        try {
          localStorage.setItem(KEY, id);
        } catch {
          /* private mode: the choice lasts for this visit */
        }
      },
      slot: (id) => slots[id] ?? IDLE,
      load,
      forecast: async (id, series, opts) => (await load(id)).forecast(series, opts),
      detectAnomalies: async (id, series) => (await load(id)).detectAnomalies(series),
    }),
    [choice, slots, load],
  );
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useModels(): Models {
  const v = useContext(Ctx);
  if (!v) throw new Error("useModels must be used inside <ModelsProvider>");
  return v;
}
