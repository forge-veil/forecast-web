import { useEffect, useMemo, useRef, useState } from "react";
import { Backtest } from "src/components/Backtest";
import { PAGE_SIZE, Pager } from "src/components/Pager";
import { TabularBar, variantColor } from "src/components/TabularBar";
import { formatNumber, formatPercent } from "src/lib/format";
import { parseTable, type Table, type TableCell } from "src/lib/csv";
import { resolvePrimary } from "src/lib/models";
import { useTabular, VARIANT_IDS, variantOf, type VariantId } from "src/lib/tabular";
import { accuracy, columnKind, coverage, holdoutSplit, mae, pageOf, rankVariants, suggestTask, type Task } from "src/lib/tables";

export interface TableSample {
  id: string;
  name: string;
  subtitle: string;
  /** Plural noun for rows, used in sentences ("penguins"). */
  noun: string;
  group: "real-estate" | "classic" | "yours";
  file?: string;
  target?: string;
  source?: { text: string; url: string; license: string };
  /** Attribution statement a licence requires verbatim. */
  notice?: string;
}

export const TABLE_SAMPLES: TableSample[] = [
  {
    id: "table:penguins", group: "classic",
    name: "Penguins",
    subtitle: "Palmer Station, 344 penguins",
    noun: "penguins",
    file: "penguins.csv",
    target: "species",
    source: { text: "palmerpenguins (Horst, Hill and Gorman)", url: "https://allisonhorst.github.io/palmerpenguins/", license: "CC0" },
  },
  {
    id: "table:iris", group: "classic",
    name: "Iris",
    subtitle: "Fisher's 150 flowers",
    noun: "flowers",
    file: "iris.csv",
    target: "species",
    source: { text: "UCI Machine Learning Repository, Iris", url: "https://archive.ics.uci.edu/dataset/53/iris", license: "CC BY 4.0" },
  },
  {
    id: "table:wine", group: "classic",
    name: "Wine",
    subtitle: "178 Italian wines, 3 cultivars",
    noun: "wines",
    file: "wine.csv",
    source: { text: "UCI Machine Learning Repository, Wine", url: "https://archive.ics.uci.edu/dataset/109/wine", license: "CC BY 4.0" },
  },
  {
    id: "table:breast-cancer", group: "classic",
    name: "Breast cancer",
    subtitle: "569 biopsies, benign or malignant",
    noun: "biopsies",
    file: "breast-cancer.csv",
    source: { text: "UCI Machine Learning Repository, Breast Cancer Wisconsin (Diagnostic)", url: "https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic", license: "CC BY 4.0" },
  },
  {
    id: "table:auto-mpg", group: "classic",
    name: "Auto MPG",
    subtitle: "398 cars from the 1970s and 80s",
    noun: "cars",
    file: "auto-mpg.csv",
    source: { text: "UCI Machine Learning Repository, Auto MPG", url: "https://archive.ics.uci.edu/dataset/9/auto+mpg", license: "CC BY 4.0" },
  },
  {
    id: "table:wine-quality", group: "classic",
    name: "Red wine quality",
    subtitle: "1,599 Portuguese red wines",
    noun: "wines",
    file: "wine-quality-red.csv",
    source: { text: "UCI Machine Learning Repository, Wine Quality (Cortez et al.)", url: "https://archive.ics.uci.edu/dataset/186/wine+quality", license: "CC BY 4.0" },
  },
  {
    id: "table:nyc-homes", group: "real-estate",
    name: "NYC home sales",
    subtitle: "1,500 one- and two-family homes, 2026",
    noun: "home sales",
    file: "nyc-home-sales.csv",
    source: {
      text: "NYC Department of Finance, Citywide Rolling Calendar Sales (NYC Open Data)",
      url: "https://data.cityofnewyork.us/City-Government/NYC-Citywide-Rolling-Calendar-Sales/usep-8jbt",
      license: "NYC Open Data terms of use",
    },
  },
  {
    id: "table:london-prices", group: "real-estate",
    name: "London house prices",
    subtitle: "1,500 sales, Sep 2025 to Aug 2026",
    noun: "sales",
    file: "london-house-prices.csv",
    source: {
      text: "HM Land Registry Price Paid Data",
      url: "https://www.gov.uk/government/statistical-data-sets/price-paid-data-downloads",
      license: "Open Government Licence v3.0",
    },
    notice: "Contains HM Land Registry data © Crown copyright and database right 2026. This data is licensed under the Open Government Licence v3.0.",
  },
  {
    id: "table:taiwan-real-estate", group: "real-estate",
    name: "Taipei real estate",
    subtitle: "414 sales in Sindian District, New Taipei",
    noun: "sales",
    file: "taiwan-real-estate.csv",
    source: {
      text: "UCI Machine Learning Repository, Real Estate Valuation (Yeh and Hsu)",
      url: "https://archive.ics.uci.edu/dataset/477/real+estate+valuation+data+set",
      license: "CC BY 4.0",
    },
  },
  { id: "table:yours", group: "yours", name: "Your table", subtitle: "Paste a CSV", noun: "rows" },
];

const PASTE_EXAMPLE = `size_m2,bedrooms,district,price_k
54,1,center,310
82,2,center,455
120,3,suburb,420
65,2,suburb,260
95,3,center,520
140,4,suburb,480
48,1,suburb,190
77,2,center,
110,3,suburb,`;

interface Result {
  task: Task;
  testRows: number[];
  trainRows: number[];
  predicted: (string | number)[];
  confidence?: number[];
  bands?: number[][];
  unlabeled: { row: number; predicted: string | number; confidence?: number; band?: number[] }[];
  score: { accuracy?: number; mae?: number; coverage?: number };
  /** Wall time for the held-out predictions, in ms. */
  ms: number;
}

const SEED = 7;

/** One prediction run, with the settings it used so its results stay readable after edits. */
type Run = {
  key: string;
  results: Partial<Record<VariantId, Result>>;
  order: VariantId[];
  task: Task;
  targetIdx: number;
  targetName: string;
  featureIdx: number[];
};

export function Tables({ sample, onBack }: { sample: TableSample; onBack: () => void }) {
  const tabular = useTabular();
  const [text, setText] = useState(sample.file ? "" : PASTE_EXAMPLE);
  const [loadError, setLoadError] = useState<string | null>(null);
  // The target starts blank: the visitor picks what to predict.
  const [target, setTarget] = useState<string | undefined>(undefined);
  const [taskOverride, setTaskOverride] = useState<Task | null>(null);
  const [excluded, setExcluded] = useState<Set<string>>(new Set());
  const [run, setRun] = useState<Run | null>(null);
  const [busy, setBusy] = useState(false);
  const [dataPage, setDataPage] = useState(0);
  const [onlyPredicted, setOnlyPredicted] = useState(false);
  const [runError, setRunError] = useState<string | null>(null);
  const req = useRef(0);

  useEffect(() => {
    if (!sample.file) return;
    fetch(`${import.meta.env.BASE_URL}samples/${sample.file}`)
      .then((r) => (r.ok ? r.text() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then(setText, (e: Error) => setLoadError(`Could not load the sample: ${e.message}`));
  }, [sample.file]);

  // 1. Data
  const table: Table = useMemo(() => parseTable(text), [text]);

  // 2. Columns: the target, the task, and which features to use.
  const targetIdx = target === undefined ? -1 : table.columns.indexOf(target);
  const targetName = targetIdx >= 0 ? table.columns[targetIdx] : undefined;
  const targetValues = table.rows.map((r) => (targetIdx >= 0 ? r[targetIdx] : null));
  const task: Task | null = targetIdx < 0 ? null : (taskOverride ?? suggestTask(targetValues));
  const featureIdx = table.columns.map((_, i) => i).filter((i) => i !== targetIdx && !excluded.has(table.columns[i]));
  const kinds = table.columns.map((_, i) => columnKind(table.rows.map((r) => r[i])));
  const labeled = targetIdx < 0 ? [] : table.rows.map((_, i) => i).filter((i) => targetValues[i] !== null);
  const unlabeled = targetIdx < 0 ? [] : table.rows.map((_, i) => i).filter((i) => targetValues[i] === null);
  const split = holdoutSplit(labeled.length, 0.2, SEED);

  // 3. Models: picked once, independent of the columns; weights for the target's task load on Predict.
  const picked = tabular.selected;
  const main = resolvePrimary(tabular.choice, picked);

  // 4. Predict: only on request. The key records what the last run used, so later edits mark it out of date.
  const key = JSON.stringify([text.length, text.slice(0, 200), targetIdx, task, featureIdx, picked]);
  const stale = run !== null && run.key !== key;
  const canPredict = task !== null && picked.length > 0 && featureIdx.length > 0 && labeled.length >= 4;

  // Predict automatically whenever the target, features or models change. The previous results
  // stay on screen until the first new result arrives, so nothing blanks while it works.
  useEffect(() => {
    if (!canPredict || !task) return;
    const id = ++req.current;
    const thisKey = key;
    const order = [...picked];
    const snapshot = { order, task, targetIdx, targetName: targetName!, featureIdx: [...featureIdx] };
    setBusy(true);
    setRunError(null);
    const X = (idx: number[]) => idx.map((i) => featureIdx.map((c) => table.rows[i][c]));
    const y = (idx: number[]) => idx.map((i) => targetValues[i] as string | number);
    const trainRows = split.train.map((k) => labeled[k]);
    const testRows = split.test.map((k) => labeled[k]);
    const runVariant = async (variant: VariantId): Promise<Result> => {
      const model = await tabular.load(variant, task);
      const t0 = performance.now();
      const evalFit = model.fit(X(trainRows), y(trainRows));
      const fullFit = unlabeled.length ? model.fit(X(labeled), y(labeled)) : null;
      if (task === "classification") {
        const proba = await evalFit.predictProba(X(testRows));
        const predicted = proba.map((p) => evalFit.classes[p.indexOf(Math.max(...p))]);
        const ms = performance.now() - t0;
        const unl = fullFit ? await fullFit.predictProba(X(unlabeled)) : [];
        return {
          task,
          testRows,
          trainRows,
          predicted,
          confidence: proba.map((p) => Math.max(...p)),
          unlabeled: unl.map((p, k) => ({ row: unlabeled[k], predicted: fullFit!.classes[p.indexOf(Math.max(...p))], confidence: Math.max(...p) })),
          score: { accuracy: accuracy(testRows.map((i) => targetValues[i]), predicted) },
          ms,
        };
      }
      const predicted = (await evalFit.predict(X(testRows), { output: "median" })) as number[];
      const bands = (await evalFit.predict(X(testRows), { output: "quantiles", alphas: [0.1, 0.9] })) as number[][];
      const ms = performance.now() - t0;
      const unlMed = fullFit ? ((await fullFit.predict(X(unlabeled), { output: "median" })) as number[]) : [];
      const unlBands = fullFit ? ((await fullFit.predict(X(unlabeled), { output: "quantiles", alphas: [0.1, 0.9] })) as number[][]) : [];
      const actual = testRows.map((i) => targetValues[i] as number);
      return {
        task,
        testRows,
        trainRows,
        predicted,
        bands,
        unlabeled: unlMed.map((v, k) => ({ row: unlabeled[k], predicted: v, band: unlBands[k] })),
        score: { mae: mae(actual, predicted), coverage: coverage(actual, bands) },
        ms,
      };
    };
    (async () => {
      // One variant at a time, so timings are not skewed by sharing the CPU or GPU.
      for (const v of order) {
        const r = await runVariant(v);
        if (id !== req.current) return;
        setRun((cur) => (cur && cur.key === thisKey ? { ...cur, results: { ...cur.results, [v]: r } } : { key: thisKey, ...snapshot, results: { [v]: r } }));
      }
    })()
      .catch((e: Error) => id === req.current && setRunError(e.message))
      .finally(() => id === req.current && setBusy(false));
    // eslint-free note: keyed on `key`, which encodes the table, target, task, features and models.
  }, [key, canPredict]);

  const results = run?.results ?? {};
  const rt = run?.task ?? "classification";
  const rTargetName = run?.targetName ?? "";
  const rTargetValue = (row: number) => (run ? table.rows[row]?.[run.targetIdx] : null);
  const shownMain = run && results[main] ? main : (run?.order.find((v) => results[v]) ?? main);
  const result = results[shownMain] ?? null;
  const ranked = run
    ? rankVariants(
        run.order.map((id) => ({ id, result: results[id], score: rt === "classification" ? results[id]?.score.accuracy : results[id]?.score.mae })),
        rt,
      )
    : [];
  const shown = (run?.featureIdx ?? featureIdx).slice(0, 4);
  // Predictions keyed by table row: held-out rows (to check accuracy) and rows with a blank target.
  const predictions = new Map<number, { predicted: string | number; confidence?: number; band?: number[]; kind: "held-out" | "new" }>();
  if (result) {
    result.testRows.forEach((row, k) =>
      predictions.set(row, { predicted: result.predicted[k], confidence: result.confidence?.[k], band: result.bands?.[k], kind: "held-out" }),
    );
    result.unlabeled.forEach((u) => predictions.set(u.row, { predicted: u.predicted, confidence: u.confidence, band: u.band, kind: "new" }));
  }
  // Predicted rows first (in table order), then the rows used to fit; "Predicted" shows only the first group.
  const predictedRows = [...predictions.keys()].sort((a, b) => a - b);
  const visibleRows =
    onlyPredicted && predictions.size
      ? predictedRows
      : [...predictedRows, ...table.rows.map((_, i) => i).filter((i) => !predictions.has(i))];
  const pageRange = pageOf(visibleRows.length, dataPage, PAGE_SIZE);
  const fmt = (v: TableCell | undefined) => (v === null || v === undefined ? "–" : typeof v === "number" ? formatNumber(v) : v);
  const toggle = (name: string) =>
    setExcluded((cur) => {
      const next = new Set(cur);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      return next;
    });

  let summary: string;
  if (targetIdx < 0 && !run) summary = "Choose a column to predict";
  else if (picked.length === 0 && !run) summary = "Choose a model to predict";
  else if (!run) summary = busy ? "Predicting…" : "Not predicted yet";
  else if (busy && !result) summary = "Predicting…";
  else if (ranked.length > 1 && ranked[1].score !== undefined) {
    const [a, b] = ranked;
    summary =
      rt === "classification"
        ? `${variantOf(a.id).name}: ${formatPercent(a.score!)} correct, vs ${formatPercent(b.score!)} for ${variantOf(b.id).name}`
        : `${variantOf(a.id).name}: off by ${formatNumber(a.score!)}, vs ${formatNumber(b.score!)} for ${variantOf(b.id).name}`;
  } else if (result?.score.accuracy !== undefined) summary = `${formatPercent(result.score.accuracy)} correct on ${result.testRows.length} held-out rows`;
  else if (result?.score.mae !== undefined) summary = `Off by ${formatNumber(result.score.mae)} on average · ${formatPercent(result.score.coverage!)} inside the 80% range`;
  else summary = "Predicting…";

  return (
    <div className="detail-inner tables">
      <button className="back" onClick={onBack}>‹ Tables</button>
      <h2 className="d-name">{sample.name}</h2>
      <div className="d-sub">
        {table.rows.length} rows · {table.columns.length} columns{sample.source ? ` · ${sample.subtitle}` : ""}
      </div>
      <div className={`d-change flat${run ? "" : " muted"}`} aria-live="polite">
        {busy ? (
          <span className="busy-line">
            <span className="spinner" aria-hidden="true" />
            {run ? "Updating…" : `Predicting with ${variantOf(main).name}…`}
            {run && <span className="busy-prev">{summary}</span>}
          </span>
        ) : (
          summary
        )}
      </div>

      {!sample.file && (
        <textarea className="table-paste" value={text} onChange={(e) => setText(e.target.value)} aria-label="Paste CSV with a header row" spellCheck={false} />
      )}
      {loadError && <p className="status error">{loadError}</p>}

      <div className="settings">
        <div className="set-row">
          <span className="set-label">Predict</span>
          <div className="set-body">
            <label className="target-pick">
              <span className="sr-only">Column to predict</span>
              <select value={targetName ?? ""} onChange={(e) => (setTarget(e.target.value || undefined), setTaskOverride(null))}>
                <option value="">Choose a column</option>
                {table.columns.map((c) => <option key={c} value={c}>{c}</option>)}
              </select>
            </label>
            {task && (
              <div className="seg" role="group" aria-label="Kind of prediction">
                <button aria-pressed={task === "classification"} onClick={() => setTaskOverride("classification")}>Category</button>
                <button aria-pressed={task === "regression"} onClick={() => setTaskOverride("regression")}>Number</button>
              </div>
            )}
          </div>
        </div>
        {targetIdx >= 0 && (
          <div className="set-row">
            <span className="set-label">Using</span>
            <div className="set-body">
              <div className="feature-chips" aria-label="Columns used to predict">
                {table.columns.map((c, i) =>
                  i === targetIdx ? null : (
                    <button key={c} className="fchip" aria-pressed={!excluded.has(c)} title={kinds[i] === "number" ? "Number column" : "Category column"} onClick={() => toggle(c)}>
                      {c}
                    </button>
                  ),
                )}
              </div>
              <div className="set-meta">
                <span>
                  {featureIdx.length} of {table.columns.length - 1} columns
                </span>
                <button className="link-btn" onClick={() => setExcluded(new Set())}>All</button>
                <button className="link-btn" onClick={() => setExcluded(new Set(table.columns.filter((_, i) => i !== targetIdx)))}>None</button>
              </div>
            </div>
          </div>
        )}
        <div className="set-row">
          <span className="set-label">Model</span>
          <div className="set-body">
            <TabularBar task={task} />
          </div>
        </div>
      </div>
      {runError && <p className="status error">{runError}</p>}

      {result && run && (
        <div className={`fresh-wrap${busy ? " is-busy" : ""}`} key={`bt-${run.key}-${shownMain}`}>
          {busy && <div className="busy-bar" aria-hidden="true" />}
          <Backtest
          task={rt}
          targetName={rTargetName}
          noun={sample.noun}
          modelName={variantOf(shownMain).name}
          testRows={result.testRows}
          trainRows={result.trainRows}
          predicted={result.predicted}
          confidence={result.confidence}
          bands={result.bands}
          targetOf={(row) => rTargetValue(row) ?? null}
          onShowRows={() => (setOnlyPredicted(true), setDataPage(0))}
          />
        </div>
      )}

      {table.columns.length > 0 && (
        <section className={`values data-table${busy ? " is-busy" : ""}`}>
          {busy && <div className="busy-bar" aria-hidden="true" />}
          <div className="data-head">
            <h3>Data</h3>
            {predictions.size > 0 && (
              <div className="seg" role="group" aria-label="Rows">
                <button aria-pressed={!onlyPredicted} onClick={() => (setOnlyPredicted(false), setDataPage(0))}>All rows</button>
                <button aria-pressed={onlyPredicted} onClick={() => (setOnlyPredicted(true), setDataPage(0))}>Predicted · {predictions.size}</button>
              </div>
            )}
          </div>
          <div className="scroll">
            <table>
              <thead>
                <tr>
                  {table.columns.map((c, i) => (
                    <th key={c} className={i === targetIdx ? "is-target" : excluded.has(c) ? "is-off" : undefined}>{c}</th>
                  ))}
                  {(predictions.size > 0 || (busy && targetIdx >= 0)) && (
                    <>
                      <th className="is-pred">Predicted {rTargetName || targetName}</th>
                      <th className="is-pred">{(run ? rt : task) === "regression" ? "80% range" : "Confidence"}</th>
                    </>
                  )}
                </tr>
              </thead>
              <tbody>
                {visibleRows.slice(pageRange.start, pageRange.end).map((row) => {
                  const pr = predictions.get(row);
                  const actual = rTargetValue(row);
                  const miss =
                    pr?.kind === "held-out" &&
                    (rt === "classification" ? actual !== pr.predicted : !((actual as number) >= pr.band![0] && (actual as number) <= pr.band![1]));
                  return (
                    <tr key={row} className={miss ? "miss" : pr?.kind === "new" ? "new" : undefined}>
                      {table.rows[row].map((v, i) => (
                        <td key={i} className={i === targetIdx ? "is-target" : excluded.has(table.columns[i]) ? "is-off" : undefined}>{fmt(v)}</td>
                      ))}
                      {(predictions.size > 0 || (busy && targetIdx >= 0)) && (
                        <>
                          <td className={`is-pred${pr ? " fresh" : ""}`} key={`p-${run?.key}-${shownMain}-${row}`}>
                            {pr ? <b>{fmt(pr.predicted)}</b> : busy && predictions.size === 0 ? <span className="shimmer" /> : "–"}
                          </td>
                          <td className="is-pred">
                            {!pr ? (busy && predictions.size === 0 ? <span className="shimmer" /> : "–") : pr.confidence !== undefined ? formatPercent(pr.confidence) : `${formatNumber(pr.band![0])} to ${formatNumber(pr.band![1])}`}
                          </td>
                        </>
                      )}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <Pager total={visibleRows.length} page={pageRange.page} onPage={setDataPage} />
          {predictions.size > 0 && (
            <p className="data-legend">
              Predicted rows are held out (red when wrong{rt === "regression" ? ", outside the 80% range" : ""}) or had no {rTargetName} (highlighted). Other rows were used to fit.
            </p>
          )}
        </section>
      )}

      {run && (
        <div className="results">
          <div className="stats">
            <div className="stat"><span className="k">Model</span><span className="v">{variantOf(shownMain).name}</span></div>
            <div className="stat"><span className="k">Held out</span><span className="v">{result ? result.testRows.length : "–"}</span></div>
            <div className="stat"><span className="k">Features</span><span className="v">{run.featureIdx.length}</span></div>
            <div className="stat"><span className="k">Time</span><span className="v">{result ? `${(result.ms / 1000).toFixed(1)} s` : "–"}</span></div>
          </div>

          {ranked.length > 1 && (
            <section className="values ranking">
              <h3>Models</h3>
              <div className="scroll">
                <table>
                  <thead>
                    <tr>
                      <th>Model</th>
                      <th>{rt === "classification" ? "Accuracy" : "Average error"}</th>
                      {rt === "regression" && <th>In 80% range</th>}
                      <th>Time</th>
                    </tr>
                  </thead>
                  <tbody>
                    {ranked.map(({ id, result: r }) => (
                      <tr key={id}>
                        <td className="model-cell"><i className="sw" style={{ borderColor: variantColor(id) }} />{variantOf(id).name}</td>
                        {r ? (
                          <>
                            <td>{rt === "classification" ? formatPercent(r.score.accuracy!) : formatNumber(r.score.mae!)}</td>
                            {rt === "regression" && <td>{formatPercent(r.score.coverage!)}</td>}
                            <td>{(r.ms / 1000).toFixed(1)} s</td>
                          </>
                        ) : (
                          <td colSpan={rt === "regression" ? 3 : 2}><span className="busy-line"><span className="spinner" aria-hidden="true" />Predicting…</span></td>
                        )}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          )}

        </div>
      )}

      <section className="about">
        <h3>How this works</h3>
        <p>
          TabICL reads the labeled rows in context and predicts the rest in one pass, with no training. To show how well that works on this table, a fifth of the labeled rows are held back and predicted from the others. Everything runs in your browser; the table never leaves it.
        </p>
        {sample.source && (
          <div className="src">
            Data: <a href={sample.source.url} target="_blank" rel="noreferrer">{sample.source.text}</a> ({sample.source.license})
          </div>
        )}
        {sample.notice && <div className="src">{sample.notice}</div>}
      </section>
    </div>
  );
}
