import { useEffect, useMemo, useRef, useState, type PointerEvent } from "react";
import { parseCsv, parseNumbers, pickColumn, resample } from "src/lib/series";
import type { SeriesData } from "src/lib/sources";

export const SAMPLE = `month,orders
2024-01,120
2024-02,132
2024-03,151
2024-04,149
2024-05,170
2024-06,188
2024-07,201
2024-08,195
2024-09,178
2024-10,169
2024-11,210
2024-12,262
2025-01,131
2025-02,145
2025-03,166
2025-04,163
2025-05,187
2025-06,206
2025-07,220
2025-08,214
2025-09,195
2025-10,186
2025-11,231
2025-12,288`;

/** Parses pasted text: a CSV with a label column, or a plain list of numbers. */
export function parsePasted(text: string, column?: number) {
  const lines = text.trim().split(/\r?\n/);
  const multi = lines.length > 1 && /[,;\t]/.test(lines[0]) && lines.every((l) => l.split(/[,;\t]/).length > 1);
  if (!multi) {
    const v = parseNumbers(text);
    return { values: v, labels: v.map((_, i) => String(i + 1)), columns: [] as string[], picked: -1 };
  }
  const csv = parseCsv(text);
  const { index, values } = pickColumn(csv, column);
  const labelCol = index !== 0 ? 0 : -1;
  const kept = values.map((v, i) => ({ v, label: labelCol >= 0 ? csv.raw[i][labelCol] : "" })).filter((r) => r.v !== null);
  return { values: kept.map((r) => r.v as number), labels: kept.map((r, i) => r.label || String(i + 1)), columns: csv.columns, picked: index };
}

export function YourDataEditor({ onChange, onSeason }: { onChange: (d: SeriesData | null) => void; onSeason: (s?: number) => void }) {
  const [mode, setMode] = useState<"paste" | "draw">("paste");
  const [text, setText] = useState(SAMPLE);
  const [column, setColumn] = useState<number | undefined>(undefined);
  const [season, setSeason] = useState("12");
  const [drawn, setDrawn] = useState<number[]>([]);
  const parsed = useMemo(() => parsePasted(text, column), [text, column]);

  useEffect(() => {
    const values = mode === "paste" ? parsed.values : drawn;
    onChange(values.length >= 3 ? { values, labels: mode === "paste" ? parsed.labels : values.map((_, i) => String(i + 1)) } : null);
    onSeason(mode === "paste" && Number(season) >= 2 ? Math.round(Number(season)) : undefined);
  }, [mode, parsed, drawn, season, onChange, onSeason]);

  return (
    <div className="editor">
      <div className="seg" role="group" aria-label="Input method" style={{ marginBottom: 10 }}>
        <button aria-pressed={mode === "paste"} onClick={() => setMode("paste")}>Paste</button>
        <button aria-pressed={mode === "draw"} onClick={() => setMode("draw")}>Draw</button>
      </div>
      {mode === "paste" ? (
        <>
          <textarea value={text} onChange={(e) => setText(e.target.value)} aria-label="Paste numbers or CSV" spellCheck={false} />
          <div className="opts">
            {parsed.columns.length > 1 && (
              <label>
                Column{" "}
                <select value={parsed.picked} onChange={(e) => setColumn(Number(e.target.value))}>
                  {parsed.columns.map((c, i) => <option key={i} value={i}>{c}</option>)}
                </select>
              </label>
            )}
            <label>
              Season{" "}
              <input type="number" min={0} value={season} onChange={(e) => setSeason(e.target.value)} style={{ width: 64 }} />
            </label>
            <span>{parsed.values.length} values</span>
          </div>
        </>
      ) : (
        <DrawPad onChange={setDrawn} />
      )}
    </div>
  );
}

function DrawPad({ onChange }: { onChange: (v: number[]) => void }) {
  const ref = useRef<SVGSVGElement>(null);
  const [pts, setPts] = useState<{ x: number; y: number }[]>([]);
  const drawing = useRef(false);
  const at = (e: PointerEvent) => {
    const r = ref.current!.getBoundingClientRect();
    return { x: ((e.clientX - r.left) / r.width) * 1000, y: ((e.clientY - r.top) / r.height) * 200 };
  };
  const finish = () => {
    drawing.current = false;
    onChange(resample(pts, 160).map((v) => +(200 - v).toFixed(2)));
  };
  return (
    <>
      <svg
        ref={ref}
        className="draw"
        viewBox="0 0 1000 200"
        preserveAspectRatio="none"
        role="img"
        aria-label="Drawing area. Drag from left to right to draw a series."
        onPointerDown={(e) => {
          (e.target as Element).setPointerCapture(e.pointerId);
          drawing.current = true;
          setPts([at(e)]);
        }}
        onPointerMove={(e) => drawing.current && setPts((p) => [...p, at(e)])}
        onPointerUp={finish}
        onPointerCancel={finish}
      >
        <path d={pts.map((p, i) => `${i ? "L" : "M"}${p.x.toFixed(1)},${p.y.toFixed(1)}`).join("")} vectorEffect="non-scaling-stroke" />
      </svg>
      <div className="opts">
        <span>Drag left to right.</span>
        <button className="link-btn" onClick={() => (setPts([]), onChange([]))}>Clear</button>
      </div>
    </>
  );
}
