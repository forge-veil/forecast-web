import { formatNumber, formatPercent } from "src/lib/format";
import type { TableCell } from "src/lib/csv";
import { majorityBaseline, meanBaselineError, perClass, type Task } from "src/lib/tables";

export interface BacktestProps {
  task: Task;
  targetName: string;
  /** Plural noun for rows, e.g. "penguins". */
  noun: string;
  modelName: string;
  testRows: number[];
  trainRows: number[];
  predicted: (string | number)[];
  confidence?: number[];
  bands?: number[][];
  targetOf: (row: number) => TableCell;
  onShowRows: () => void;
}

const fmt = (v: TableCell) => (v === null ? "–" : typeof v === "number" ? formatNumber(v) : v);

/** The held-out check as a sentence, a comparison with simple guessing, and one picture. */
export function Backtest(p: BacktestProps) {
  const actual = p.testRows.map(p.targetOf);
  const train = p.trainRows.map(p.targetOf);
  const n = p.testRows.length;
  if (p.task === "classification") {
    const right = actual.filter((a, i) => a === p.predicted[i]).length;
    const base = majorityBaseline(train, actual);
    const classes = perClass(actual, p.predicted);
    const misses = p.testRows.map((row, i) => ({ row, i })).filter(({ i }) => actual[i] !== p.predicted[i]);
    return (
      <section className="backtest">
        <h3>How accurate is it?</h3>
        <p className="bt-story">
          We hid the <b>{p.targetName}</b> of {n} {p.noun} and asked {p.modelName} to guess it from the other columns:{" "}
          <b>{right} right ({formatPercent(right / n)})</b>.
        </p>
        <p className="bt-baseline">
          Always guessing “{fmt(base.label)}”, the most common {p.targetName}, would get {formatPercent(base.accuracy)}.
        </p>
        <div className="bt-bars" role="list" aria-label={`Right guesses per ${p.targetName}`}>
          {classes.map((c) => (
            <div className="bt-bar" role="listitem" key={String(c.label)}>
              <span className="bt-label">{fmt(c.label)}</span>
              <span className="bt-track" aria-hidden="true">
                <span className="bt-fill" style={{ width: `${(100 * c.right) / c.total}%` }} />
              </span>
              <span className="bt-count num">
                {c.right} of {c.total}
              </span>
            </div>
          ))}
        </div>
        {misses.length > 0 && (
          <div className="bt-misses">
            <span className="bt-misses-title">Wrong guesses</span>
            <ul>
              {misses.slice(0, 5).map(({ row, i }) => (
                <li key={row}>
                  Row {row + 1}: guessed <b>{fmt(p.predicted[i])}</b>
                  {p.confidence ? ` (${formatPercent(p.confidence[i])} sure)` : ""}, actually <b>{fmt(actual[i])}</b>
                </li>
              ))}
            </ul>
            <button className="link-btn" onClick={p.onShowRows}>See all predicted rows in the table</button>
          </div>
        )}
      </section>
    );
  }

  const ys = actual as number[];
  const err = ys.reduce((acc, y, i) => acc + Math.abs(y - (p.predicted[i] as number)), 0) / Math.max(1, n);
  const inside = ys.filter((y, i) => y >= p.bands![i][0] && y <= p.bands![i][1]).length;
  const base = meanBaselineError(train as number[], ys);
  return (
    <section className="backtest">
      <h3>How accurate is it?</h3>
      <p className="bt-story">
        We hid the <b>{p.targetName}</b> of {n} {p.noun} and asked {p.modelName} to estimate it from the other columns: typically{" "}
        <b>off by {formatNumber(err)}</b>, and {inside} of {n} actual values ({formatPercent(inside / n)}) fell inside its 80% range.
      </p>
      <p className="bt-baseline">Always guessing the average {p.targetName} would be off by {formatNumber(base)}.</p>
      <Scatter actual={ys} predicted={p.predicted as number[]} bands={p.bands!} targetName={p.targetName} />
      <button className="link-btn" onClick={p.onShowRows}>See all predicted rows in the table</button>
    </section>
  );
}

/** Actual (x) against predicted (y), each with its 80% range; on the diagonal means a perfect guess. */
function Scatter({ actual, predicted, bands, targetName }: { actual: number[]; predicted: number[]; bands: number[][]; targetName: string }) {
  const W = 320;
  const H = 240;
  const m = { l: 44, r: 10, t: 10, b: 30 };
  const all = [...actual, ...bands.flat()].filter(Number.isFinite);
  let lo = Math.min(...all);
  let hi = Math.max(...all);
  if (lo === hi) (lo -= 1), (hi += 1);
  const pad = (hi - lo) * 0.05;
  lo -= pad;
  hi += pad;
  const x = (v: number) => m.l + ((v - lo) / (hi - lo)) * (W - m.l - m.r);
  const y = (v: number) => H - m.b - ((v - lo) / (hi - lo)) * (H - m.t - m.b);
  return (
    <figure className="bt-scatter">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Predicted against actual ${targetName}, with 80% ranges`}>
        <line x1={x(lo)} y1={y(lo)} x2={x(hi)} y2={y(hi)} stroke="var(--ink-3)" strokeDasharray="4 4" />
        {actual.map((a, i) => {
          const miss = a < bands[i][0] || a > bands[i][1];
          const c = miss ? "var(--red)" : "var(--blue)";
          return (
            <g key={i}>
              <line x1={x(a)} x2={x(a)} y1={y(bands[i][0])} y2={y(bands[i][1])} stroke={c} strokeOpacity={0.35} strokeWidth={2} />
              <circle cx={x(a)} cy={y(predicted[i])} r={3.5} fill={c} stroke="var(--bg)" strokeWidth={1} />
            </g>
          );
        })}
        <text x={(m.l + W - m.r) / 2} y={H - 6} textAnchor="middle">actual {targetName}</text>
        <text x={12} y={(m.t + H - m.b) / 2} textAnchor="middle" transform={`rotate(-90 12 ${(m.t + H - m.b) / 2})`}>predicted</text>
        <text x={m.l - 4} y={y(lo) - 2} textAnchor="end">{formatNumber(lo)}</text>
        <text x={m.l - 4} y={y(hi) + 10} textAnchor="end">{formatNumber(hi)}</text>
      </svg>
      <figcaption>Each dot is a held-out row; the dashed line is a perfect guess; red dots fell outside their 80% range.</figcaption>
    </figure>
  );
}
