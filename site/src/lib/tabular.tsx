import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { loadTabular, TABULAR_MODELS, type Tabular } from "forecast-web/tabular";
import type { Progress, ResolvedDtype } from "forecast-web";
import type { Task } from "src/lib/tables";
import type { Slot } from "src/lib/models";

export type VariantId = "full" | "fast" | "fp32";

export interface Variant {
  id: VariantId;
  name: string;
  detail: string;
  nEstimators: number;
  dtype: ResolvedDtype;
}

/** TabICL configurations offered in the Tables tab. "full" is the upstream default. */
export const VARIANTS: Variant[] = [
  { id: "full", name: "TabICL", detail: "8 members", nEstimators: 8, dtype: "fp16" },
  { id: "fast", name: "TabICL fast", detail: "1 member, about 8x quicker", nEstimators: 1, dtype: "fp16" },
  { id: "fp32", name: "TabICL fp32", detail: "8 members, full precision", nEstimators: 8, dtype: "fp32" },
];
export const VARIANT_IDS = VARIANTS.map((v) => v.id);
export const variantOf = (id: VariantId) => VARIANTS.find((v) => v.id === id)!;

export const tabularInfo = (task: Task) => TABULAR_MODELS[task === "classification" ? "tabicl-classifier-v2" : "tabicl-regressor-v2"];
export const variantBytes = (id: VariantId, task: Task) => tabularInfo(task).sizes[variantOf(id).dtype];

const IDLE: Slot = { status: "idle", progress: null, backend: null, error: null };
const KEY = "forecast-web-demo:tabular-variant";
const SELECTED_KEY = "forecast-web-demo:tabular-selected";
const key = (v: VariantId, t: Task) => `${v}:${t}`;

interface TabularModels {
  /** Main variant (draws the summary and tables). */
  choice: VariantId;
  setChoice(id: VariantId): void;
  /** Variants the visitor picked; they apply to any target, loading weights per task on demand. */
  selected: VariantId[];
  select(id: VariantId): void;
  unselect(id: VariantId): void;
  slot(v: VariantId, t: Task): Slot;
  load(v: VariantId, t: Task): Promise<Tabular>;
}

const Ctx = createContext<TabularModels | null>(null);

const readSelected = (): VariantId[] => {
  try {
    const v = JSON.parse(localStorage.getItem(SELECTED_KEY) ?? "[]");
    return Array.isArray(v) ? v.filter((x): x is VariantId => VARIANT_IDS.includes(x)) : [];
  } catch {
    return [];
  }
};
const save = (k: string, v: string) => {
  try {
    localStorage.setItem(k, v);
  } catch {
    /* private mode: lasts for this visit */
  }
};

const readStored = (): VariantId => {
  try {
    const v = localStorage.getItem(KEY);
    return VARIANT_IDS.includes(v as VariantId) ? (v as VariantId) : "full";
  } catch {
    return "full";
  }
};

/** One TabICL model per (variant, task), loaded on first use. */
export function TabularProvider({ children }: { children: ReactNode }) {
  const [choice, setChoiceState] = useState<VariantId>(readStored);
  const [selected, setSelected] = useState<VariantId[]>(readSelected);
  const [slots, setSlots] = useState<Record<string, Slot>>({});
  const loads = useRef(new Map<string, Promise<Tabular>>());
  const patch = (k: string, s: Partial<Slot>) => setSlots((m) => ({ ...m, [k]: { ...(m[k] ?? IDLE), ...s } }));

  const load = useCallback((v: VariantId, t: Task) => {
    const k = key(v, t);
    const existing = loads.current.get(k);
    if (existing) return existing;
    patch(k, { status: "loading", error: null });
    const { nEstimators, dtype } = variantOf(v);
    const p = loadTabular({ task: t, nEstimators, dtype, onProgress: (progress: Progress) => patch(k, { progress }) });
    loads.current.set(k, p);
    p.then(
      () => patch(k, { status: "ready" }),
      (e: unknown) => {
        loads.current.delete(k);
        patch(k, { status: "error", error: e instanceof Error ? e : new Error(String(e)) });
      },
    );
    return p;
  }, []);

  useEffect(() => {
    const all = loads.current;
    return () => all.forEach((p) => p.then((m) => m.dispose()).catch(() => {}));
  }, []);

  const value = useMemo<TabularModels>(
    () => ({
      choice,
      setChoice(id) {
        setChoiceState(id);
        save(KEY, id);
      },
      selected,
      select(id) {
        setSelected((cur) => {
          if (cur.includes(id)) return cur;
          const next = VARIANT_IDS.filter((v) => v === id || cur.includes(v));
          save(SELECTED_KEY, JSON.stringify(next));
          return next;
        });
        if (selected.length === 0) {
          setChoiceState(id);
          save(KEY, id);
        }
      },
      unselect(id) {
        setSelected((cur) => {
          const next = cur.filter((v) => v !== id);
          save(SELECTED_KEY, JSON.stringify(next));
          return next;
        });
      },
      slot: (v, t) => slots[key(v, t)] ?? IDLE,
      load,
    }),
    [choice, selected, slots, load],
  );
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useTabular(): TabularModels {
  const v = useContext(Ctx);
  if (!v) throw new Error("useTabular must be used inside <TabularProvider>");
  return v;
}
