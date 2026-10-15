import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { MODELS, type AnomalyResult, type Forecast, type ModelId, type SensitivityLevel } from "forecast-web";
import { Chart } from "src/components/Chart";
import { AddModel, ModelBar } from "src/components/ModelBar";
import { anomalySentence, calibrationSentence, flagged } from "src/lib/anomaly";
import { formatNumber, formatPercent, shortLabel } from "src/lib/format";
import { changeText, type Item } from "src/lib/items";
import { scoreBacktest } from "src/lib/metrics";
import { compareVerdict, MODEL_IDS, modelColor, rankByError, resolvePrimary, useModels } from "src/lib/models";
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
  mean: f.mean?.slice(0, h), median: f.median.slice(0, h), q10: f.q10.slice(0, h), q20: f.q20.slice(0, h),
  q30: f.q30.slice(0, h), q40: f.q40.slice(0, h), q50: f.q50.slice(0, h), q60: f.q60.slice(0, h),
  q70: f.q70.slice(0, h), q80: f.q80.slice(0, h), q90: f.q90.slice(0, h),
});

export function Detail(props: {
  item: Item;
  entry?: Entry;
  /** The selected model's shared 128-step forecast from the watchlist batch. */
  full?: Forecast;
  season?: number;
  onBack: () => void;
  children?: ReactNode;
}) {
  const { item, entry } = props;
  const models = useModels();
  const [mode, setMode] = useState<"forecast" | "backtest" | "anomalies">("forecast");
  const [level, setLevel] = useState<SensitivityLevel>("medium");
  const [calibrated, setCalibrated] = useState(false);
  const [checks, setChecks] = useState<Record<string, AnomalyResult>>({});
  const [rangeIdx, setRangeIdx] = useState(item.defaultRange);
  const [runs, setRuns] = useState<Record<string, Forecast>>({});
  const [runError, setRunError] = useState<string | null>(null);
  const req = useRef(0);
  const season = props.season ?? item.season;
  const data = entry?.data;
  const values = data?.values ?? [];
  const range = item.ranges[rangeIdx];
  const backtest = mode === "backtest";
  const anomalies = mode === "anomalies";
  const steps = Math.min(range.steps, backtest ? Math.max(1, values.length - 2) : 128);
  const statuses = MODEL_IDS.map((m) => models.slot(m).status).join();
  const loaded = MODEL_IDS.filter((m) => models.slot(m).status === "ready");
  const primary = resolvePrimary(models.choice, loaded);
  const { status, backend, progress } = models.slot(primary);
  // The watchlist batch already holds the selected model's forecast; every other run happens here.
  const fromWatchlist = mode === "forecast" && !calibrated && primary === models.choice && !!props.full;

  const split = useMemo(() => {
    if (!data) return null;
    if (backtest) {
      const { train, test } = holdout(data.values, steps);
      return { context: train, test, labels: data.labels.slice(0, train.length), future: data.labels.slice(train.length) };
    }
    return { context: data.values, test: undefined, labels: data.labels, future: data.futureLabels?.slice(0, steps) };
  }, [data, backtest, steps]);

  const runKey = `${mode}|${item.id}|${values.length}|${values[values.length - 1]}|${steps}`;
  // Only the banded model's forecast is calibrated; the other models' lines stay raw.
  const calOf = (m: ModelId) => mode === "forecast" && calibrated && m === primary;
  const keyOf = (m: ModelId) => `${m}|${runKey}${calOf(m) ? "|cal" : ""}`;
  useEffect(() => {
    if (!split || split.context.length < 2 || anomalies) return;
    const ids = loaded.filter((m) => !runs[keyOf(m)] && !(m === primary && fromWatchlist));
    if (ids.length === 0) return;
    const id = ++req.current;
    setRunError(null);
    const t = setTimeout(() => {
      for (const m of ids)
        models.forecast(m, [split.context], { horizon: backtest ? split.test!.length : steps, calibrate: calOf(m) }).then(
          ([f]) => id === req.current && setRuns((r) => ({ ...r, [keyOf(m)]: f })),
          (e: Error) => id === req.current && setRunError(e.message),
        );
    }, 120);
    return () => clearTimeout(t);
    // Keyed on the statuses string rather than models.slot, whose identity changes with every slot update.
  }, [statuses, split, runKey, primary, fromWatchlist, calibrated, models.forecast]);

  // Scores do not depend on the sensitivity, so one check per model and series serves all three levels.
  const checkKey = `${primary}|${item.id}|${values.length}|${values[values.length - 1]}`;
  const check = checks[checkKey];
  useEffect(() => {
    if (!anomalies || !data || !loaded.includes(primary) || checks[checkKey]) return;
    const id = ++req.current;
    setRunError(null);
    models.detectAnomalies(primary, data.values).then(
      (r) => setChecks((c) => ({ ...c, [checkKey]: r })),
      (e: Error) => id === req.current && setRunError(e.message),
    );
  }, [anomalies, statuses, checkKey, primary, data, models.detectAnomalies]);

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

  const forecastOf = (m: ModelId) => (m === primary && fromWatchlist ? slice(props.full!, steps) : runs[keyOf(m)]);
  const f = loaded.includes(primary) ? forecastOf(primary) : undefined;
  const others = loaded.filter((m) => m !== primary).map((id) => ({ id, f: forecastOf(id) }));
  const multi = others.length > 0;
  const lines = others.filter((o) => o.f).map((o) => ({ label: MODELS[o.id].name, values: o.f!.median, color: modelColor(o.id) }));
  const primaryName = MODELS[primary].name;
  const last = values[values.length - 1];
  const end = f?.median[f.median.length - 1];
  const dir = end === undefined ? "flat" : end > last ? "up" : end < last ? "down" : "flat";
  const color = dir === "up" ? "var(--green)" : dir === "down" ? "var(--red)" : "var(--forecast)";
  const ctxUsed = Math.min(split!.context.length, MODELS[primary].maxContext);
  const seasonal = !!season && season >= 2 && split!.context.length >= season;
  const baseLabel = baselineLabel(season, item.stepName, seasonal);
  const scoreOf = (fc?: Forecast) =>
    fc ? scoreBacktest({ train: split!.context, test: split!.test!, median: fc.median, q10: fc.q10, q90: fc.q90, season }) : null;
  const ranked = backtest
    ? rankByError(
        [primary, ...others.map((o) => o.id)].filter((m) => loaded.includes(m)).map((id) => {
          const sc = scoreOf(forecastOf(id));
          return { id, score: sc, mae: sc?.mae };
        }),
      )
    : [];
  const score = ranked.find((r) => r.id === primary)?.score ?? null;
  const unit = item.unit.length <= 6 ? item.unit : "";
  const u = (v: number | undefined) => (v === undefined ? "–" : `${formatNumber(v)}${unit ? ` ${unit}` : ""}`);
  const endLabel = split!.future?.[steps - 1];
  const period = (labels?: string[]) => (labels?.length ? `${shortLabel(labels[0])} to ${shortLabel(labels[labels.length - 1])}` : "");
  const visible = Math.min(split!.context.length, Math.max(steps * 3, 90), 360);
  const comparison = !backtest && data.comparison ? { label: data.comparison.label, values: data.comparison.values.slice(0, steps) } : undefined;
  const stepWord = (k: number) => `${k} ${k === 1 ? singular(item.stepName) : item.stepName}`;
  const who = multi ? "each loaded model" : primaryName;

  const threshold = MODELS[primary].anomalyThresholds[level];
  const unusual = check ? flagged(check.points, threshold) : [];
  const checkedFrom = check?.points[0]?.index ?? 0;
  const anomalyView = (() => {
    if (!anomalies || !check) return null;
    const n = values.length;
    const shown = Math.min(n, Math.max(120, n - checkedFrom + 16), 720);
    const start = n - shown;
    const band = { expected: Array<number>(shown).fill(NaN), lower: Array<number>(shown).fill(NaN), upper: Array<number>(shown).fill(NaN), score: Array<number>(shown).fill(NaN) };
    for (const p of check.points) {
      if (p.index < start) continue;
      const i = p.index - start;
      band.expected[i] = p.expected;
      band.lower[i] = p.lower;
      band.upper[i] = p.upper;
      band.score[i] = p.score;
    }
    return { start, shown, band, markers: unusual.map((p) => p.index - start).filter((i) => i >= 0) };
  })();

  const explain = anomalies
    ? `${primaryName} forecast each point${check ? ` from ${shortLabel(data.labels[checkedFrom] ?? "")} on` : ""} using only the history before it. Shaded: the range it expected. A point is unusual when it lands far outside that range, which is first rescaled to fit this series' typical misses. Higher sensitivity flags smaller surprises.`
    : backtest
    ? `The last ${stepWord(steps)}${split!.future?.length ? ` (${period(split!.future)})` : ""} were hidden and forecast by ${who} from the history before. Baseline: ${baseLabel.toLowerCase()}.`
    : `Next ${stepWord(steps)} of ${item.phrase}, forecast by ${who} from the last ${stepWord(ctxUsed)}. Shaded: ${primaryName}'s 80% range.`;

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
            <AddModel />
          )}
        </div>
      </div>
    );
  if (anomalies && status === "ready") {
    headline = (
      <div className={`d-change ${unusual.length ? "down" : "flat"}`}>
        {!check ? "Checking…" : unusual.length ? `${unusual.length} unusual ${unusual.length === 1 ? "point" : "points"} at ${level} sensitivity` : `Nothing unusual at ${level} sensitivity`}
      </div>
    );
  } else if (f && !backtest && !anomalies) {
    headline = (
      <div className={`d-change ${dir}`}>
        {changeText(last, end!, item)} to {u(end)} {endLabel ? `by ${shortLabel(endLabel)}` : `in ${stepWord(steps)}`}
      </div>
    );
  } else if (backtest && ranked[1]?.score) {
    headline = (
      <div className="d-change flat">
        {compareVerdict({ name: MODELS[ranked[0].id].name, mae: ranked[0].mae! }, { name: MODELS[ranked[1].id].name, mae: ranked[1].mae! })}
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

      <ModelBar />

      <div className="controls">
        {anomalies ? (
          <div className="ranges" role="group" aria-label="Sensitivity">
            <span className="lead">Sensitivity</span>
            {(["low", "medium", "high"] as const).map((l) => (
              <button key={l} aria-pressed={l === level} onClick={() => setLevel(l)} title={`Flag scores above ${MODELS[primary].anomalyThresholds[l]}`}>
                {l[0].toUpperCase() + l.slice(1)}
              </button>
            ))}
          </div>
        ) : (
          <div className="ranges" role="group" aria-label={backtest ? "How much to hide" : "Forecast horizon"}>
            <span className="lead">{backtest ? "Hide last" : "Forecast"}</span>
            {item.ranges.map((r, i) => (
              <button key={r.label} aria-pressed={i === rangeIdx} onClick={() => setRangeIdx(i)} disabled={backtest && r.steps > values.length - 2}>
                {r.label}
              </button>
            ))}
            {!backtest && (
              <label className="toggle" title="Rescale the 80% range when it held too few or too many past values">
                <input type="checkbox" checked={calibrated} onChange={(e) => setCalibrated(e.target.checked)} />
                Calibrated
              </label>
            )}
          </div>
        )}
        <div className="seg" role="group" aria-label="Mode">
          <button aria-pressed={mode === "forecast"} onClick={() => setMode("forecast")}>Forecast</button>
          <button aria-pressed={backtest} onClick={() => setMode("backtest")}>Backtest</button>
          <button aria-pressed={anomalies} onClick={() => setMode("anomalies")}>Anomalies</button>
        </div>
      </div>

      {anomalies ? (
        <Chart
          history={values.slice(anomalyView ? anomalyView.start : -visible)}
          historyLabels={data.labels.slice(anomalyView ? anomalyView.start : -visible)}
          historyBand={anomalyView?.band}
          markers={anomalyView?.markers}
          mode="forecast"
          forecastLabel={primaryName}
          color="var(--ink-2)"
          unit={unit}
          ariaLabel={`${item.phrase}, ${anomalyView?.shown ?? visible} observations${check ? `, ${unusual.length} flagged as unusual` : ""}`}
        />
      ) : (
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
        lines={lines}
        forecastLabel={primaryName}
        forecastColor={multi ? modelColor(primary) : undefined}
        mode={mode}
        color={color}
        unit={unit}
        ariaLabel={`${item.phrase}, ${visible} observations${f ? ` and a ${steps}-step forecast` : ""}`}
      />
      )}
      {mode === "forecast" && calibrated && f?.calibration && <p className="cal-note">{calibrationSentence(f.calibration)}.</p>}
      {runError && <p className="status error">{runError}</p>}

      <div className="stats">
        {anomalies ? (
          <>
            <Stat k="Unusual" v={check ? String(unusual.length) : "–"} tone={unusual.length ? "down" : undefined} />
            <Stat k="Checked" v={check ? stepWord(check.points.length) : "–"} />
            <Stat k="Flag above" v={`score ${threshold}`} />
            <Stat k="Range scale" v={check ? `${check.scale.toFixed(2)}x` : "–"} />
            <Stat k="Model" v={primaryName} />
          </>
        ) : backtest ? (
          <>
            <Stat k={`${primaryName} error`} v={score ? u(score.mae) : "–"} />
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

      {backtest && multi && (
        <section className="values ranking">
          <h3>Models</h3>
          <div className="scroll">
            <table>
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Error</th>
                  <th>vs. baseline</th>
                  <th>In 80% range</th>
                </tr>
              </thead>
              <tbody>
                {ranked.map((r) => (
                  <tr key={r.id}>
                    <td className="model-cell"><i className="sw" style={{ borderColor: modelColor(r.id) }} />{MODELS[r.id].name}</td>
                    {r.score ? (
                      <>
                        <td>{u(r.score.mae)}</td>
                        <td>{Number.isFinite(r.score.skill) ? `${r.score.skill > 0 ? "−" : "+"}${Math.round(Math.abs(r.score.skill) * 100)}%` : "–"}</td>
                        <td>{formatPercent(r.score.coverage)}</td>
                      </>
                    ) : (
                      <td colSpan={3}><span className="num">Forecasting…</span></td>
                    )}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {anomalies && check && (
        <section className="values">
          <h3>Unusual points</h3>
          <ul className="flags">
            {unusual.length === 0 && <li className="none">Nothing scored above {threshold}. Try High sensitivity to flag smaller surprises.</li>}
            {unusual.map((p) => (
              <li key={p.index}>
                <i className="sw pt" style={{ background: "var(--red)" }} />
                {anomalySentence(p, data.labels[p.index] ?? `#${p.index + 1}`, unit)}
              </li>
            ))}
          </ul>
        </section>
      )}

      <section className="about">
        <h3>{anomalies ? "Anomalies" : backtest ? "Backtest" : "Forecast"}</h3>
        <p>{explain}</p>
        {item.note && <p>{item.note}</p>}
        {item.source && (
          <div className="src">
            Data: <a href={item.source.url} target="_blank" rel="noreferrer">{item.source.text}</a>
            {item.source.license ? ` (${item.source.license})` : ""}
          </div>
        )}
      </section>

      {f && !anomalies && (
        <section className="values">
          <h3>Values</h3>
          <div className="scroll">
            <table>
              <thead>
                <tr>
                  <th>When</th>
                  <th>Low</th>
                  <th>{primaryName}</th>
                  <th>High</th>
                  {split!.test && <th>Actual</th>}
                  {score && <th>Baseline</th>}
                  {comparison && <th>{comparison.label}</th>}
                  {lines.map((l) => <th key={l.label}>{l.label}</th>)}
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
                    {lines.map((l) => <td key={l.label}>{formatNumber(l.values[i])}</td>)}
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
