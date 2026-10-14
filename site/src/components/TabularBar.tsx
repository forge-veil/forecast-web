import { useEffect, useRef, useState } from "react";
import { formatMB } from "src/lib/format";
import { resolvePrimary } from "src/lib/models";
import { useTabular, VARIANT_IDS, variantBytes, variantOf, type VariantId } from "src/lib/tabular";
import type { Task } from "src/lib/tables";

export const variantColor = (id: VariantId) => `var(--m${VARIANT_IDS.indexOf(id) + 1})`;

/** Menu of TabICL variants not picked yet. Picking one starts its download when the target's task is known. */
export function AddVariant({ task }: { task: Task | null }) {
  const { selected, select, load } = useTabular();
  const [open, setOpen] = useState(false);
  const wrap = useRef<HTMLDivElement>(null);
  const available = VARIANT_IDS.filter((v) => !selected.includes(v));

  useEffect(() => {
    if (!open) return;
    const onDown = (e: PointerEvent) => !wrap.current?.contains(e.target as Node) && setOpen(false);
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    addEventListener("pointerdown", onDown);
    addEventListener("keydown", onKey);
    return () => {
      removeEventListener("pointerdown", onDown);
      removeEventListener("keydown", onKey);
    };
  }, [open]);

  if (available.length === 0) return null;
  const add = (v: VariantId) => {
    setOpen(false);
    select(v);
    if (task) void load(v, task).catch(() => {});
  };
  return (
    <div className="add-model" ref={wrap}>
      <button className={`chip add ${selected.length === 0 ? "cta" : ""}`} aria-haspopup="menu" aria-expanded={open} onClick={() => setOpen((o) => !o)}>
        + {selected.length === 0 ? "Choose a model" : "Add model"}
      </button>
      {open && (
        <div className="menu" role="menu">
          {available.map((v) => (
            <button key={v} role="menuitem" onClick={() => add(v)}>
              <i className="chip-dot" style={{ borderColor: variantColor(v) }} />
              <span className="menu-name">
                {variantOf(v).name}
                <small className="menu-detail">{variantOf(v).detail}</small>
              </span>
              <span className="menu-size num">{formatMB(variantBytes(v, task ?? "classification"))}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

/**
 * The picked variants, independent of the columns. Status reflects the current
 * task's weights: ready, downloading, or (not yet downloaded) its size, fetched on Predict.
 */
export function TabularBar({ task }: { task: Task | null }) {
  const { choice, setChoice, selected, unselect, slot, load } = useTabular();
  const main = resolvePrimary(choice, selected);
  return (
    <div className="modelbar">
      {selected.map((v) => {
        const s = task ? slot(v, task) : null;
        const pct = s?.progress?.total ? Math.round((100 * s.progress.loaded) / s.progress.total) : 0;
        const meta =
          !s ? "" : s.status === "loading" ? (pct ? `${pct}%` : "…") : s.status === "error" ? "Retry" : s.status === "idle" ? `↓ ${formatMB(variantBytes(v, task!))}` : "";
        return (
          <span key={v} className="chip-group">
            <button
              className={`chip ${s?.status ?? "idle"}`}
              aria-pressed={v === main}
              title={s?.status === "error" ? `${s.error?.message ?? "Failed to load"}. Click to retry.` : `${variantOf(v).detail}. Click to make it the main model.`}
              style={s?.status === "loading" ? { ["--pct" as string]: `${pct}%` } : undefined}
              onClick={() => (s?.status === "error" && task ? void load(v, task).catch(() => {}) : setChoice(v))}
            >
              <i className="chip-dot" style={{ borderColor: variantColor(v), background: s?.status === "ready" ? variantColor(v) : "transparent" }} />
              <span>{variantOf(v).name}</span>
              {meta && <span className="chip-meta num">{meta}</span>}
              {v === main && <span className="chip-tag">Main</span>}
            </button>
            <button className="chip-x" aria-label={`Remove ${variantOf(v).name}`} onClick={() => unselect(v)}>×</button>
          </span>
        );
      })}
      <AddVariant task={task} />
    </div>
  );
}
