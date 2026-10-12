/** Watchlist sparkline: recent history, then the forecast in a lighter dashed stroke. */
export function Sparkline({ history, forecast, dir }: { history: number[]; forecast?: number[]; dir: "up" | "down" | "flat" }) {
  const w = 64;
  const h = 30;
  const all = [...history, ...(forecast ?? [])].filter(Number.isFinite);
  if (all.length < 2) return <svg className="spark" aria-hidden="true" />;
  const lo = Math.min(...all);
  const hi = Math.max(...all);
  const n = history.length + (forecast?.length ?? 0);
  const x = (i: number) => (i / (n - 1)) * w;
  const y = (v: number) => (hi === lo ? h / 2 : 2 + (1 - (v - lo) / (hi - lo)) * (h - 4));
  const path = (vals: number[], off: number) => vals.map((v, i) => `${i ? "L" : "M"}${x(i + off).toFixed(1)},${y(v).toFixed(1)}`).join("");
  const color = dir === "up" ? "var(--green)" : dir === "down" ? "var(--red)" : "var(--ink-3)";
  const last = history.length - 1;
  return (
    <svg className="spark" viewBox={`0 0 ${w} ${h}`} aria-hidden="true">
      <line x1={0} x2={w} y1={y(history[last])} y2={y(history[last])} stroke="var(--ink-3)" strokeWidth={1} strokeDasharray="1 2" />
      <path d={path(history, 0)} fill="none" stroke={color} strokeWidth={1.5} strokeLinejoin="round" />
      {forecast && forecast.length > 0 && (
        <path d={path([history[last], ...forecast], last)} fill="none" stroke={color} strokeWidth={1.5} strokeDasharray="2 2" opacity={0.75} />
      )}
    </svg>
  );
}
