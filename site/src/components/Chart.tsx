import { useEffect, useId, useLayoutEffect, useMemo, useRef, useState, type KeyboardEvent, type PointerEvent } from "react";
import { formatNumber, niceTicks, shortLabel } from "src/lib/format";

export interface ChartProps {
  history: number[];
  historyLabels: string[];
  futureLabels?: string[];
  median?: number[];
  q10?: number[];
  q90?: number[];
  /** Held-out actuals (backtest). */
  actual?: number[];
  /** Simple reference forecast (backtest). */
  baseline?: { label: string; values: number[] };
  /** Another source's forecast (forecast mode). */
  comparison?: { label: string; values: number[] };
  /** Expected value and calibrated 80% range over the history (anomalies); NaN where a point was not checked. */
  historyBand?: { expected: number[]; lower: number[]; upper: number[]; score: number[] };
  /** History indices drawn as unusual points. */
  markers?: number[];
  /** Median lines of the other loaded models, drawn dashed in their own colors. */
  lines?: { label: string; values: number[]; color: string }[];
  mode: "forecast" | "backtest";
  /** Name of the model drawn with the band. */
  forecastLabel: string;
  /** Color of the banded model's line and band when several models are shown. */
  forecastColor?: string;
  /** Line color for forecast mode, a CSS color or var(). */
  color: string;
  unit: string;
  ariaLabel: string;
}

const M = { top: 26, right: 52, bottom: 24, left: 4 };

export function Chart(p: ChartProps) {
  const { history, median = [], q10 = [], q90 = [], actual, baseline, comparison, mode, lines = [], historyBand, markers = [] } = p;
  const wrapRef = useRef<HTMLDivElement>(null);
  const gid = useId().replace(/:/g, "");
  const [width, setWidth] = useState(0);
  const [hover, setHover] = useState<number | null>(null);
  /** Index range being dragged across, and the zoomed index range. */
  const [drag, setDrag] = useState<{ a: number; b: number } | null>(null);
  const [zoom, setZoom] = useState<{ v0: number; v1: number } | null>(null);

  // Measure before first paint so the chart never draws at a guessed width.
  useLayoutEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    setWidth(el.clientWidth);
    const ro = new ResizeObserver(([e]) => e.contentRect.width > 0 && setWidth(e.contentRect.width));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const W = Math.max(280, width);
  const Ht = W < 560 ? 260 : 340;
  const H = history.length;
  const F = Math.max(median.length, actual?.length ?? 0, ...lines.map((l) => l.values.length));
  const base = baseline?.values.slice(0, F);
  const compare = comparison?.values.slice(0, F);
  const n = H + F;
  const lastObs = history[H - 1];
  const backtest = mode === "backtest";
  // A different series or layout starts unzoomed.
  const shape = `${mode}|${n}|${history[0]}|${lastObs}|${F}`;
  useEffect(() => {
    setZoom(null);
    setDrag(null);
  }, [shape]);
  const v0 = zoom?.v0 ?? 0;
  const v1 = zoom?.v1 ?? Math.max(1, n - 1);
  const inView = (i: number) => i >= v0 && i <= v1;
  /** The value a range readout measures at index i: history, then actuals, then the median. */
  const valueAt = (i: number) => (i < H ? history[i] : (actual?.[i - H] ?? median[i - H]));
  const lineColor = backtest ? "var(--ink-2)" : p.color;
  const fcColor = p.forecastColor ?? (backtest ? "var(--forecast)" : p.color);

  const { y0, y1 } = useMemo(() => {
    // Only what is in view sets the scale, so a zoomed range fills the height.
    const future = (vals: number[] | undefined) => (vals ?? []).filter((_, k) => inView(H + k));
    const past = (vals: number[] | undefined) => (vals ?? []).filter((_, i) => inView(i));
    const all = [...past(history), ...future(q10), ...future(q90), ...future(median), ...future(actual), ...future(base), ...future(compare), ...lines.flatMap((l) => future(l.values)), ...past(historyBand?.lower), ...past(historyBand?.upper)].filter(Number.isFinite);
    let lo = Math.min(...all);
    let hi = Math.max(...all);
    if (!Number.isFinite(lo)) return { y0: 0, y1: 1 };
    if (lo === hi) {
      lo -= 1;
      hi += 1;
    }
    const pad = (hi - lo) * 0.08;
    return { y0: lo - pad, y1: hi + pad };
  }, [history, q10, q90, median, actual, base, compare, lines, historyBand, v0, v1]);

  const pw = W - M.left - M.right;
  const ph = Ht - M.top - M.bottom;
  const x = (i: number) => M.left + (n <= 1 ? 0 : ((i - v0) / (v1 - v0)) * pw);
  const y = (v: number) => M.top + (1 - (v - y0) / (y1 - y0)) * ph;
  const pts = (vals: number[], off: number) => {
    let d = "";
    let pen = false;
    vals.forEach((v, i) => {
      if (!Number.isFinite(v)) return void (pen = false);
      d += `${pen ? "L" : "M"}${x(i + off).toFixed(1)},${y(v).toFixed(1)}`;
      pen = true;
    });
    return d;
  };
  // Future lines start at the last observation so there is no gap at "now".
  const seam = (vals: number[]) => (H > 0 && vals.length ? [lastObs, ...vals] : vals);
  const off = H - 1;

  const area = H > 1 ? `${pts(history, 0)}L${x(H - 1).toFixed(1)},${M.top + ph}L${x(0).toFixed(1)},${M.top + ph}Z` : "";
  const band =
    q10.length && q90.length
      ? `${seam(q90).map((v, i) => `${i ? "L" : "M"}${x(i + off).toFixed(1)},${y(v).toFixed(1)}`).join("")}${seam(q10)
          .map((v, i) => [i, v] as const)
          .reverse()
          .map(([i, v]) => `L${x(i + off).toFixed(1)},${y(v).toFixed(1)}`)
          .join("")}Z`
      : "";

  // One closed shape per run of checked points, so gaps in the checks stay open.
  const historyBandPath = (() => {
    if (!historyBand) return "";
    let d = "";
    let run: number[] = [];
    const flush = () => {
      if (run.length > 1)
        d += run.map((i, j) => `${j ? "L" : "M"}${x(i).toFixed(1)},${y(historyBand.upper[i]).toFixed(1)}`).join("") +
          run.slice().reverse().map((i) => `L${x(i).toFixed(1)},${y(historyBand.lower[i]).toFixed(1)}`).join("") + "Z";
      run = [];
    };
    history.forEach((_, i) => (Number.isFinite(historyBand.upper[i]) ? run.push(i) : flush()));
    flush();
    return d;
  })();

  const yTicks = niceTicks(y0, y1, Ht < 300 ? 4 : 5);
  const xCount = Math.max(2, Math.min(6, Math.floor(pw / 130)));
  const xTicks = Array.from({ length: xCount }, (_, k) => Math.round(v0 + (k * (v1 - v0)) / (xCount - 1)));
  const labelAt = (i: number) => (i < H ? p.historyLabels[i] : p.futureLabels?.[i - H] ?? `+${i - H + 1}`);

  const pick = (clientX: number) => {
    const r = wrapRef.current!.getBoundingClientRect();
    return Math.max(v0, Math.min(v1, Math.round(v0 + ((clientX - r.left - M.left) / pw) * (v1 - v0))));
  };
  const onKey = (e: KeyboardEvent) => {
    if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
    e.preventDefault();
    setHover((h) => Math.max(v0, Math.min(v1, (h ?? v1) + (e.key === "ArrowLeft" ? -1 : 1))));
  };
  const onDown = (e: PointerEvent<SVGSVGElement>) => {
    if (e.button !== 0) return;
    e.currentTarget.setPointerCapture(e.pointerId);
    const i = pick(e.clientX);
    setDrag({ a: i, b: i });
  };
  const onUp = () => {
    if (drag && Math.abs(drag.b - drag.a) >= 2) setZoom({ v0: Math.min(drag.a, drag.b), v1: Math.max(drag.a, drag.b) });
    setDrag(null);
  };
  // Change between the two ends of a drag, as Apple's Stocks app shows it.
  const range = (() => {
    if (!drag || drag.a === drag.b) return null;
    const [a, b] = drag.a < drag.b ? [drag.a, drag.b] : [drag.b, drag.a];
    const va = valueAt(a);
    const vb = valueAt(b);
    if (!Number.isFinite(va) || !Number.isFinite(vb)) return null;
    const d = vb - va;
    const pct = va !== 0 ? ` (${d >= 0 ? "+" : "−"}${Math.abs((100 * d) / va).toFixed(1)}%)` : "";
    return { a, b, text: `${d >= 0 ? "+" : "−"}${formatNumber(Math.abs(d))}${pct}`, up: d >= 0 };
  })();

  const readout = (() => {
    if (hover === null) return null;
    const k = hover - H;
    const rows: [string, string, string][] = [];
    if (k < 0) {
      rows.push(["", formatNumber(history[hover]), lineColor]);
      if (historyBand && Number.isFinite(historyBand.expected[hover])) {
        rows.push(["Expected", `${formatNumber(historyBand.expected[hover])} (${formatNumber(historyBand.lower[hover])} to ${formatNumber(historyBand.upper[hover])})`, "var(--forecast)"]);
        rows.push(["Score", historyBand.score[hover].toFixed(1), markers.includes(hover) ? "var(--red)" : "var(--ink-3)"]);
      }
    }
    else {
      if (actual?.[k] !== undefined) rows.push(["Actual", formatNumber(actual[k]), "var(--ink)"]);
      if (median[k] !== undefined) rows.push([p.forecastLabel, `${formatNumber(median[k])} (${formatNumber(q10[k])} to ${formatNumber(q90[k])})`, fcColor]);
      if (base?.[k] !== undefined) rows.push(["Baseline", formatNumber(base[k]), "var(--ink-2)"]);
      if (compare?.[k] !== undefined && Number.isFinite(compare[k])) rows.push([comparison!.label, formatNumber(compare[k]), "var(--compare)"]);
      for (const l of lines) if (l.values[k] !== undefined) rows.push([l.label, formatNumber(l.values[k]), l.color]);
    }
    return { k, rows };
  })();
  const dotsAt = (i: number) => {
    const k = i - H;
    const out: [number, string][] = [];
    if (k < 0) out.push([history[i], lineColor]);
    else {
      if (actual?.[k] !== undefined) out.push([actual[k], "var(--ink)"]);
      if (median[k] !== undefined) out.push([median[k], fcColor]);
      for (const l of lines) if (l.values[k] !== undefined) out.push([l.values[k], l.color]);
    }
    return out;
  };

  return (
    <div className="chart" ref={wrapRef}>
      {width > 0 && (
        <svg
          viewBox={`0 0 ${W} ${Ht}`}
          role="img"
          aria-label={p.ariaLabel}
          tabIndex={0}
          onPointerDown={onDown}
          onPointerMove={(e: PointerEvent) => {
            const i = pick(e.clientX);
            setHover(i);
            if (drag) setDrag({ a: drag.a, b: i });
          }}
          onPointerUp={onUp}
          onPointerCancel={() => setDrag(null)}
          onPointerLeave={() => !drag && setHover(null)}
          onDoubleClick={() => setZoom(null)}
          onKeyDown={onKey}
          onBlur={() => setHover(null)}
        >
          <defs>
            <clipPath id={`c${gid}`}>
              <rect x={M.left} y={0} width={pw} height={Ht} />
            </clipPath>
            <linearGradient id={`g${gid}`} x1="0" x2="0" y1="0" y2="1">
              <stop offset="0" stopColor={lineColor} stopOpacity={backtest ? 0.1 : 0.24} />
              <stop offset="1" stopColor={lineColor} stopOpacity={0} />
            </linearGradient>
          </defs>
          {yTicks.map((t) => (
            <g key={t}>
              <line x1={M.left} x2={W - M.right} y1={y(t)} y2={y(t)} stroke="var(--grid)" />
              <text x={W - M.right + 8} y={y(t) + 4}>{formatNumber(t)}</text>
            </g>
          ))}
          {xTicks.map((i, k) => (
            <text key={k} x={x(i)} y={Ht - 6} textAnchor={k === 0 ? "start" : k === xTicks.length - 1 ? "end" : "middle"}>
              {shortLabel(labelAt(i) ?? "")}
            </text>
          ))}
          <g clipPath={`url(#c${gid})`}>
          {F > 0 && H > 0 && off <= v1 && (
            <>
              <rect x={x(off)} y={M.top} width={W - M.right - x(off)} height={ph} fill="var(--fill)" opacity={0.45} />
              <line x1={x(off)} x2={x(off)} y1={M.top - 4} y2={M.top + ph} stroke="var(--ink-3)" />
              <text className="zone" x={x(off) + 8} y={M.top - 10}>{backtest ? "Hidden" : "Forecast"}</text>
              <text className="zone" x={x(off) - 8} y={M.top - 10} textAnchor="end">{"History"}</text>
            </>
          )}
          {H > 0 && <line x1={M.left} x2={W - M.right} y1={y(lastObs)} y2={y(lastObs)} stroke="var(--ink-3)" strokeDasharray="1 3" />}
          {area && <path d={area} fill={`url(#g${gid})`} />}
          {band && <path d={band} fill={fcColor} fillOpacity={0.16} />}
          {historyBandPath && <path d={historyBandPath} fill="var(--forecast)" fillOpacity={0.16} />}
          <path d={pts(history, 0)} fill="none" stroke={lineColor} strokeWidth={2} strokeLinejoin="round" />
          {compare && <path d={pts(seam(compare), off)} fill="none" stroke="var(--compare)" strokeWidth={1.75} strokeDasharray="5 4" />}
          {base && <path d={pts(seam(base), off)} fill="none" stroke="var(--ink-2)" strokeWidth={1.5} strokeDasharray="1.5 3.5" strokeLinecap="round" />}
          {actual && <path d={pts(seam(actual), off)} fill="none" stroke="var(--ink)" strokeWidth={2} strokeLinejoin="round" />}
          {lines.map((l) => (
            <path key={l.label} d={pts(seam(l.values.slice(0, F)), off)} fill="none" stroke={l.color} strokeWidth={2} strokeDasharray="6 4" strokeLinejoin="round" />
          ))}
          {median.length > 0 && <path d={pts(seam(median), off)} fill="none" stroke={fcColor} strokeWidth={2} strokeDasharray="6 4" strokeLinejoin="round" />}
          {markers.filter((i) => i < H && Number.isFinite(history[i])).map((i) => (
            <circle key={`m${i}`} cx={x(i)} cy={y(history[i])} r={4.5} fill="var(--red)" stroke="var(--bg)" strokeWidth={2} />
          ))}
          {range && <rect x={x(range.a)} y={M.top} width={x(range.b) - x(range.a)} height={ph} fill="var(--ink)" opacity={0.07} />}
          </g>
          {hover !== null && (
            <>
              <line x1={x(hover)} x2={x(hover)} y1={M.top} y2={M.top + ph} stroke="var(--ink-2)" />
              {dotsAt(hover).filter(([v]) => Number.isFinite(v)).map(([v, c], j) => (
                <circle key={j} cx={x(hover)} cy={y(v)} r={4} fill={c} stroke="var(--bg)" strokeWidth={2} />
              ))}
            </>
          )}
        </svg>
      )}
      {zoom && (
        <button className="zoom-reset" onClick={() => setZoom(null)}>Reset zoom</button>
      )}
      {range && (
        <div className="readout range" style={{ left: Math.max(0, Math.min(W - 180, (x(range.a) + x(range.b)) / 2 - 90)) }}>
          <div className="ro-when">{shortLabel(labelAt(range.a) ?? "")} to {shortLabel(labelAt(range.b) ?? "")}</div>
          <div className={`ro-change ${range.up ? "up" : "down"}`}>{range.text}</div>
        </div>
      )}
      {readout && hover !== null && !range && (
        <div
          className="readout"
          // Anchor by the near edge so the card keeps its natural width on either side of the cursor.
          style={x(hover) < W / 2 ? { left: x(hover) + 14 } : { right: W - x(hover) + 14 }}
        >
          <div className="ro-when">
            {shortLabel(labelAt(hover) ?? "")}
            {p.unit && ` · ${p.unit}`}
          </div>
          {readout.rows.map(([name, v, c]) => (
            <div className="ro-row" key={name || "value"}>
              <i className="sw" style={{ borderColor: c }} />
              <span className="ro-name">{name || "Value"}</span>
              <span className="ro-value">{v}</span>
            </div>
          ))}
        </div>
      )}
      <div className="legend" aria-hidden="true">
        {backtest ? (
          <>
            <span><i className="sw" style={{ borderColor: "var(--ink-2)" }} />History</span>
            <span><i className="sw" style={{ borderColor: "var(--ink)" }} />Actual</span>
            {median.length > 0 && <span><i className="sw dash" style={{ borderColor: fcColor }} />{p.forecastLabel}</span>}
            {median.length > 0 && <span><i className="sw band" style={{ background: fcColor, opacity: 0.3 }} />80% range</span>}
            {lines.map((l) => <span key={l.label}><i className="sw dash" style={{ borderColor: l.color }} />{l.label}</span>)}
            {compare && <span><i className="sw dash" style={{ borderColor: "var(--compare)" }} />{comparison!.label}</span>}
            {base && <span><i className="sw dot" style={{ borderColor: "var(--ink-2)" }} />Baseline</span>}
          </>
        ) : (
          <>
            <span><i className="sw" style={{ borderColor: p.color }} />History</span>
            {historyBand && <span><i className="sw band" style={{ background: "var(--forecast)", opacity: 0.3 }} />Expected range</span>}
            {historyBand && <span><i className="sw pt" style={{ background: "var(--red)" }} />Unusual</span>}
            {median.length > 0 && <span><i className="sw dash" style={{ borderColor: fcColor }} />{p.forecastLabel}</span>}
            {median.length > 0 && <span><i className="sw band" style={{ background: fcColor, opacity: 0.3 }} />80% range</span>}
            {lines.map((l) => <span key={l.label}><i className="sw dash" style={{ borderColor: l.color }} />{l.label}</span>)}
            {compare && <span><i className="sw dash" style={{ borderColor: "var(--compare)" }} />{comparison!.label}</span>}
            {!historyBand && <span><i className="sw dot" style={{ borderColor: "var(--ink-3)" }} />Last</span>}
          </>
        )}
      </div>
    </div>
  );
}
