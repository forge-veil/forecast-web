import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import type { Forecast } from "timesfm-web";
import { useTimesFM } from "timesfm-web/react";
import { Chart } from "src/components/Chart";
import { formatNumber, formatPercent, shortLabel } from "src/lib/format";
import { changeText, type Item } from "src/lib/items";
import { scoreBacktest } from "src/lib/metrics";
import { holdout } from "src/lib/series";
import type { Entry } from "src/lib/useWatchlist";

const singular = (s: string) => s.replace(/s$/, "");

function baselineLabel(season: number | undefined, stepName: string, seasonal: boolean): string {
  if (!seasonal || !season) return "Repeating the last value";
  if (season === 12 && stepName === "months") return "Same month last year";
  if (season === 24 && stepName === "hours") return "Same hour yesterday";
  if (season === 7 && stepName === "days") return "Same day last week";
  if (season === 132 && stepName === "months") return "Last 11-year solar cycle";
  return `Value ${season} ${stepName} earlier`;
}

const slice = (f: Forecast, h: number): Forecast => ({
  ...f,
  horizon: h,
  mean: f.mean.slice(0, h), median: f.median.slice(0, h), q10: f.q10.slice(0, h), q20: f.q20.slice(0, h),
  q30: f.q30.slice(0, h), q40: f.q40.slice(0, h), q50: f.q50.slice(0, h), q60: f.q60.slice(0, h),
  q70: f.q70.slice(0, h), q80: f.q80.slice(0, h), q90: f.q90.slice(0, h),
});

export function Detail(props: {
  item: Item;
  entry?: Entry;
  /** The shared 128-step forecast from the watchlist batch. */
  full?: Forecast;
  season?: number;
  onBack: () => void;
  children?: ReactNode;
}) {
  const { item, entry } = props;
  const { status, forecast, load, backend, progress } = useTimesFM();
  const [mode, setMode] = useState<"forecast" | "backtest">("forecast");
  const [rangeIdx, setRangeIdx] = useState(item.defaultRange);
  const [bt, setBt] = useState<{ key: string; f: Forecast } | null>(null);
  const [btError, setBtError] = useState<string | null>(null);
  const req = useRef(0);
  const season = props.season ?? item.season;
  const data = entry?.data;
  const values = data?.values ?? [];
  const range = item.ranges[rangeIdx];
  const steps = Math.min(range.steps, mode === "backtest" ? Math.max(1, values.length - 2) : 128);
  const backtest = mode === "backtest";

  const split = useMemo(() => {
    if (!data) return null;
    if (backtest) {
      const { train, test } = holdout(data.values, steps);
      return { context: train, test, labels: data.labels.slice(0, train.length), future: data.labels.slice(train.length) };
    }
    return { context: data.values, test: undefined, labels: data.labels, future: data.futureLabels?.slice(0, steps) };
  }, [data, backtest, steps]);

  const btKey = `${item.id}|${values.length}|${values[values.length - 1]}|${steps}`;
  useEffect(() => {
    if (!backtest || status !== "ready" || !split || split.context.length < 2) return;
    const id = ++req.current;
    setBtError(null);
    const t = setTimeout(() => {
      forecast([split.context], { horizon: split.test!.length }).then(
        ([f]) => id === req.current && setBt({ key: btKey, f }),
        (e: Error) => id === req.current && setBtError(e.message),
      );
    }, 120);
    return () => clearTimeout(t);
  }, [backtest, status, split, btKey, forecast]);

  if (!data) {
    return (
      <div className="detail-inner">
        <button className="back" onClick={props.onBack}>‹ Watchlist</button>
        <h2 className="d-name">{item.name}</h2>
        <div className="d-sub">{item.subtitle}</div>
        <div key="slot">{props.children}</div>
        <p className={entry?.error ? "error" : "empty"}>{entry?.error ? `Could not load this series: ${entry.error}` : item.id === "yours" ? "Enter at least 3 numbers." : "Loading…"}</p>
      </div>
    );
  }

  const f = backtest ? (bt?.key === btKey ? bt.f : undefined) : props.full ? slice(props.full, steps) : undefined;
  const last = values[values.length - 1];
  const end = f?.median[f.median.length - 1];
  const dir = end === undefined ? "flat" : end > last ? "up" : end < last ? "down" : "flat";
  const color = dir === "up" ? "var(--green)" : dir === "down" ? "var(--red)" : "var(--forecast)";
  const ctxUsed = Math.min(split!.context.length, 512);
  const seasonal = !!season && season >= 2 && split!.context.length >= season;
  const baseLabel = baselineLabel(season, item.stepName, seasonal);
  const score = backtest && f ? scoreBacktest({ train: split!.context, test: split!.test!, median: f.median, q10: f.q10, q90: f.q90, season }) : null;
  const unit = item.unit.length <= 6 ? item.unit : "";
  const u = (v: number | undefined) => (v === undefined ? "–" : `${formatNumber(v)}${unit ? ` ${unit}` : ""}`);
  const endLabel = split!.future?.[steps - 1];
  const period = (labels?: string[]) => (labels?.length ? `${shortLabel(labels[0])} to ${shortLabel(labels[labels.length - 1])}` : "");
  const visible = Math.min(split!.context.length, Math.max(steps * 3, 90), 360);
  const comparison = !backtest && data.comparison ? { label: data.comparison.label, values: data.comparison.values.slice(0, steps) } : undefined;
  const stepWord = (k: number) => `${k} ${k === 1 ? singular(item.stepName) : item.stepName}`;

  const explain = backtest
    ? `The last ${stepWord(steps)}${split!.future?.length ? ` (${period(split!.future)})` : ""} were hidden and forecast from the ${stepWord(ctxUsed)} before. Baseline: ${baseLabel.toLowerCase()}.`
    : `Next ${stepWord(steps)} of ${item.phrase}, forecast from the last ${stepWord(ctxUsed)}. Shaded: 80% range.`;

  const pct = progress?.total ? Math.round((100 * progress.loaded) / progress.total) : 0;
  let headline: ReactNode =
    status === "ready" ? (
      <div className="d-change flat">Forecasting…</div>
    ) : (
      <div className="unavailable" role="status">
        <div className="d-unavail">{status === "loading" ? "Forecast not available yet" : "Forecast not available"}</div>
        <div className="d-unavail-sub">
          {status === "loading" ? (
            <span className="num">Downloading{pct ? ` ${pct}%` : "…"}</span>
          ) : (
            <button className="pill-btn" onClick={() => void load().catch(() => {})}>{status === "error" ? "Try again" : "Load model"}</button>
          )}
        </div>
      </div>
    );
  if (f && !backtest) {
    headline = (
      <div className={`d-change ${dir}`}>
        {changeText(last, end!, item)} to {u(end)} {endLabel ? `by ${shortLabel(endLabel)}` : `in ${stepWord(steps)}`}
      </div>
    );
  } else if (score) {
    const s = score.skill;
    const verdict = !Number.isFinite(s) ? "No baseline" : Math.abs(s) <= 0.02 ? "Same error as baseline" : `${Math.round(Math.abs(s) * 100)}% ${s > 0 ? "less" : "more"} error than baseline`;
    headline = <div className={`d-change ${s > 0.02 ? "up" : s < -0.02 ? "down" : "flat"}`}>{verdict}</div>;
  }

  return (
    <div className="detail-inner">
      <button className="back" onClick={props.onBack}>‹ Watchlist</button>
      <h2 className="d-name">{item.name}</h2>
      <div className="d-sub">{item.subtitle}</div>
      <div className="d-price">
        {formatNumber(last)}
        {item.unit && <span className="unit">{item.unit}</span>}
      </div>
      {headline}
      <div className="d-caption">As of {shortLabel(data.labels[data.labels.length - 1])}</div>

      <div key="slot">{props.children}</div>

      <div className="controls">
        <div className="ranges" role="group" aria-label={backtest ? "How much to hide" : "Forecast horizon"}>
          <span className="lead">{backtest ? "Hide last" : "Forecast"}</span>
          {item.ranges.map((r, i) => (
            <button key={r.label} aria-pressed={i === rangeIdx} onClick={() => setRangeIdx(i)} disabled={backtest && r.steps > values.length - 2}>
              {r.label}
            </button>
          ))}
        </div>
        <div className="seg" role="group" aria-label="Mode">
          <button aria-pressed={!backtest} onClick={() => setMode("forecast")}>Forecast</button>
          <button aria-pressed={backtest} onClick={() => setMode("backtest")}>Backtest</button>
        </div>
      </div>

      <Chart
        history={split!.context.slice(-visible)}
        historyLabels={split!.labels.slice(-visible)}
        futureLabels={split!.future}
        median={f?.median}
        q10={f?.q10}
        q90={f?.q90}
        actual={split!.test}
        baseline={score ? { label: baseLabel, values: score.baselineValues } : undefined}
        comparison={comparison}
        mode={mode}
        color={color}
        unit={unit}
        ariaLabel={`${item.phrase}, ${visible} observations${f ? ` and a ${steps}-step forecast` : ""}`}
      />
      {btError && <p className="status error">{btError}</p>}

      <div className="stats">
        {backtest ? (
          <>
            <Stat k="TimesFM error" v={score ? u(score.mae) : "–"} />
            <Stat k="Baseline error" v={score ? u(score.baselineMae) : "–"} />
            <Stat k="vs. baseline" v={score && Number.isFinite(score.skill) ? `${score.skill > 0 ? "−" : "+"}${Math.round(Math.abs(score.skill) * 100)}%` : "–"} tone={score ? (score.skill > 0.02 ? "up" : score.skill < -0.02 ? "down" : undefined) : undefined} />
            <Stat k="In 80% range" v={score ? formatPercent(score.coverage) : "–"} />
            <Stat k="Hidden" v={stepWord(steps)} />
            <Stat k="Context" v={stepWord(ctxUsed)} />
          </>
        ) : (
          <>
            <Stat k="Last" v={u(last)} />
            <Stat k="Forecast" v={u(end)} />
            <Stat k="Change" v={end === undefined ? "–" : changeText(last, end, item)} tone={dir === "flat" ? undefined : dir} />
            <Stat k="Low" v={u(f?.q10[steps - 1])} />
            <Stat k="High" v={u(f?.q90[steps - 1])} />
            <Stat k="Horizon" v={stepWord(steps)} />
            <Stat k="Context" v={stepWord(ctxUsed)} />
            {comparison && <Stat k={comparison.label} v={u(comparison.values[steps - 1])} />}
            <Stat k="Backend" v={backend === "webgpu" ? "WebGPU" : backend === "wasm" ? "WebAssembly" : "–"} />
          </>
        )}
      </div>

      <section className="about">
        <h3>{backtest ? "Backtest" : "Forecast"}</h3>
        <p>{explain}</p>
        {item.note && <p>{item.note}</p>}
        {item.source && (
          <div className="src">
            Data: <a href={item.source.url} target="_blank" rel="noreferrer">{item.source.text}</a>
            {item.source.license ? ` (${item.source.license})` : ""}
          </div>
        )}
      </section>

      {f && (
        <section className="values">
          <h3>Values</h3>
          <div className="scroll">
            <table>
              <thead>
                <tr>
                  <th>When</th>
                  <th>Low</th>
                  <th>TimesFM</th>
                  <th>High</th>
                  {split!.test && <th>Actual</th>}
                  {score && <th>Baseline</th>}
                  {comparison && <th>{comparison.label}</th>}
                </tr>
              </thead>
              <tbody>
                {f.median.map((m, i) => (
                  <tr key={i}>
                    <td>{split!.future?.[i] ? shortLabel(split!.future[i]) : `+${i + 1}`}</td>
                    <td>{formatNumber(f.q10[i])}</td>
                    <td>{formatNumber(m)}</td>
                    <td>{formatNumber(f.q90[i])}</td>
                    {split!.test && <td>{formatNumber(split!.test[i])}</td>}
                    {score && <td>{formatNumber(score.baselineValues[i])}</td>}
                    {comparison && <td>{formatNumber(comparison.values[i])}</td>}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}
    </div>
  );
}

function Stat({ k, v, tone }: { k: string; v: string; tone?: "up" | "down" }) {
  return (
    <div className="stat">
      <span className="k">{k}</span>
      <span className={`v ${tone ?? ""}`}>{v}</span>
    </div>
  );
}
