import { useId, useLayoutEffect, useMemo, useRef, useState, type KeyboardEvent, type PointerEvent } from "react";
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
  mode: "forecast" | "backtest";
  /** Line color for forecast mode, a CSS color or var(). */
  color: string;
  unit: string;
  ariaLabel: string;
}

const M = { top: 26, right: 52, bottom: 24, left: 4 };

export function Chart(p: ChartProps) {
  const { history, median = [], q10 = [], q90 = [], actual, baseline, comparison, mode } = p;
  const wrapRef = useRef<HTMLDivElement>(null);
  const gid = useId().replace(/:/g, "");
  const [width, setWidth] = useState(0);
  const [hover, setHover] = useState<number | null>(null);

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
  const F = Math.max(median.length, actual?.length ?? 0);
  const base = baseline?.values.slice(0, F);
  const compare = comparison?.values.slice(0, F);
  const n = H + F;
  const lastObs = history[H - 1];
  const backtest = mode === "backtest";
  const lineColor = backtest ? "var(--ink-2)" : p.color;
  const fcColor = backtest ? "var(--forecast)" : p.color;

  const { y0, y1 } = useMemo(() => {
    const all = [...history, ...q10, ...q90, ...median, ...(actual ?? []), ...(base ?? []), ...(compare ?? [])].filter(Number.isFinite);
    let lo = Math.min(...all);
    let hi = Math.max(...all);
    if (!Number.isFinite(lo)) return { y0: 0, y1: 1 };
    if (lo === hi) {
      lo -= 1;
      hi += 1;
    }
    const pad = (hi - lo) * 0.08;
    return { y0: lo - pad, y1: hi + pad };
  }, [history, q10, q90, median, actual, base, compare]);

  const pw = W - M.left - M.right;
  const ph = Ht - M.top - M.bottom;
  const x = (i: number) => M.left + (n <= 1 ? 0 : (i / (n - 1)) * pw);
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

  const yTicks = niceTicks(y0, y1, Ht < 300 ? 4 : 5);
  const xCount = Math.max(2, Math.min(6, Math.floor(pw / 130)));
  const xTicks = Array.from({ length: xCount }, (_, k) => Math.round((k * (n - 1)) / (xCount - 1)));
  const labelAt = (i: number) => (i < H ? p.historyLabels[i] : p.futureLabels?.[i - H] ?? `+${i - H + 1}`);

  const pick = (clientX: number) => {
    const r = wrapRef.current!.getBoundingClientRect();
    return Math.max(0, Math.min(n - 1, Math.round(((clientX - r.left - M.left) / pw) * (n - 1))));
  };
  const onKey = (e: KeyboardEvent) => {
    if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
    e.preventDefault();
    setHover((h) => Math.max(0, Math.min(n - 1, (h ?? n - 1) + (e.key === "ArrowLeft" ? -1 : 1))));
  };

  const readout = (() => {
    if (hover === null) return null;
    const k = hover - H;
    const rows: [string, string, string][] = [];
    if (k < 0) rows.push(["", formatNumber(history[hover]), lineColor]);
    else {
      if (actual?.[k] !== undefined) rows.push(["Actual", formatNumber(actual[k]), "var(--ink)"]);
      if (median[k] !== undefined) rows.push(["TimesFM", `${formatNumber(median[k])} (${formatNumber(q10[k])} to ${formatNumber(q90[k])})`, fcColor]);
      if (base?.[k] !== undefined) rows.push(["Baseline", formatNumber(base[k]), "var(--ink-2)"]);
      if (compare?.[k] !== undefined && Number.isFinite(compare[k])) rows.push([comparison!.label, formatNumber(compare[k]), "var(--compare)"]);
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
          onPointerMove={(e: PointerEvent) => setHover(pick(e.clientX))}
          onPointerLeave={() => setHover(null)}
          onKeyDown={onKey}
          onBlur={() => setHover(null)}
        >
          <defs>
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
          {F > 0 && H > 0 && (
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
          <path d={pts(history, 0)} fill="none" stroke={lineColor} strokeWidth={2} strokeLinejoin="round" />
          {compare && <path d={pts(seam(compare), off)} fill="none" stroke="var(--compare)" strokeWidth={1.75} strokeDasharray="5 4" />}
          {base && <path d={pts(seam(base), off)} fill="none" stroke="var(--ink-2)" strokeWidth={1.5} strokeDasharray="1.5 3.5" strokeLinecap="round" />}
          {actual && <path d={pts(seam(actual), off)} fill="none" stroke="var(--ink)" strokeWidth={2} strokeLinejoin="round" />}
          {median.length > 0 && <path d={pts(seam(median), off)} fill="none" stroke={fcColor} strokeWidth={2} strokeDasharray="6 4" strokeLinejoin="round" />}
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
      {readout && hover !== null && (
        <div className="readout" style={{ left: Math.min(Math.max(x(hover), 90), W - 120) }}>
          <span className="when">{shortLabel(labelAt(hover) ?? "")}</span>
          {readout.rows.map(([name, v, c]) => (
            <span key={name}> · {name && <span style={{ color: c }}>{name} </span>}<b>{v}</b></span>
          ))}
          {p.unit && <span className="when"> {p.unit}</span>}
        </div>
      )}
      <div className="legend" aria-hidden="true">
        {backtest ? (
          <>
            <span><i className="sw" style={{ borderColor: "var(--ink-2)" }} />History</span>
            <span><i className="sw" style={{ borderColor: "var(--ink)" }} />Actual</span>
            {median.length > 0 && <span><i className="sw dash" style={{ borderColor: "var(--forecast)" }} />TimesFM</span>}
            {median.length > 0 && <span><i className="sw band" style={{ background: "var(--forecast)", opacity: 0.3 }} />80% range</span>}
            {base && <span><i className="sw dot" style={{ borderColor: "var(--ink-2)" }} />Baseline</span>}
          </>
        ) : (
          <>
            <span><i className="sw" style={{ borderColor: p.color }} />History</span>
            {median.length > 0 && <span><i className="sw dash" style={{ borderColor: p.color }} />TimesFM</span>}
            {median.length > 0 && <span><i className="sw band" style={{ background: p.color, opacity: 0.3 }} />80% range</span>}
            {compare && <span><i className="sw dash" style={{ borderColor: "var(--compare)" }} />{comparison!.label}</span>}
            <span><i className="sw dot" style={{ borderColor: "var(--ink-3)" }} />Last</span>
          </>
        )}
      </div>
    </div>
  );
}
