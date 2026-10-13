import { useEffect, useRef, useState } from "react";
import { MODELS, type ModelId } from "forecast-web";
import { formatMB } from "src/lib/format";
import { downloadBytes, hasGpu, MODEL_IDS, modelColor, resolvePrimary, useModels } from "src/lib/models";

/** "+ Load a model" / "+ Add model" button with a menu of the models not loaded yet. */
export function AddModel() {
  const { choice, setChoice, slot, load } = useModels();
  const [open, setOpen] = useState(false);
  const wrap = useRef<HTMLDivElement>(null);
  const gpu = hasGpu();
  const first = MODEL_IDS.every((id) => slot(id).status === "idle");
  const available = MODEL_IDS.filter((id) => slot(id).status === "idle");

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

  const add = (id: ModelId) => {
    setOpen(false);
    // The first model a visitor adds becomes the main one; later ones join for comparison.
    const current = slot(choice).status;
    if (current === "idle" || current === "error") setChoice(id);
    void load(id).catch(() => {});
  };

  if (available.length === 0) return null;
  return (
    <div className="add-model" ref={wrap}>
      <button className={`chip add ${first ? "cta" : ""}`} aria-haspopup="menu" aria-expanded={open} onClick={() => setOpen((o) => !o)}>
        + {first ? "Load a model" : "Add model"}
      </button>
      {open && (
        <div className="menu" role="menu">
          {available.map((id) => (
            <button key={id} role="menuitem" onClick={() => add(id)}>
              <i className="chip-dot" style={{ borderColor: modelColor(id) }} />
              <span className="menu-name">{MODELS[id].name}</span>
              <span className="menu-size num">{formatMB(downloadBytes(MODELS[id], gpu))}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

/**
 * Models that are loading or loaded, as chips (click one to make it the main
 * forecast), followed by the add-model menu. Hidden until a model is added.
 */
export function ModelBar() {
  const { choice, setChoice, slot, load } = useModels();
  const active = MODEL_IDS.filter((id) => slot(id).status !== "idle");
  const loaded = MODEL_IDS.filter((id) => slot(id).status === "ready");
  const main = resolvePrimary(choice, loaded);
  if (active.length === 0) return null;
  return (
    <div className="modelbar">
      {active.map((id) => {
        const s = slot(id);
        const pct = s.progress?.total ? Math.round((100 * s.progress.loaded) / s.progress.total) : 0;
        const isMain = id === main && s.status === "ready";
        return (
          <button
            key={id}
            className={`chip ${s.status}`}
            aria-pressed={isMain}
            disabled={s.status === "loading"}
            title={s.status === "error" ? `${s.error?.message ?? "Failed to load"}. Click to retry.` : s.status === "ready" ? `Make ${MODELS[id].name} the main forecast` : undefined}
            style={s.status === "loading" ? { ["--pct" as string]: `${pct}%` } : undefined}
            onClick={() => (s.status === "error" ? void load(id).catch(() => {}) : setChoice(id))}
          >
            <i className="chip-dot" style={{ borderColor: modelColor(id), background: s.status === "ready" ? modelColor(id) : "transparent" }} />
            <span>{MODELS[id].name}</span>
            {s.status === "loading" && <span className="chip-meta num">{pct ? `${pct}%` : "…"}</span>}
            {s.status === "error" && <span className="chip-meta">Retry</span>}
            {isMain && <span className="chip-tag">Main</span>}
          </button>
        );
      })}
      <AddModel />
    </div>
  );
}
