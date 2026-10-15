---
title: Anomalies
group: Guides
---

# Anomalies and calibrated bands

Every forecaster can check a series against its own history. `detectAnomalies` backtests the model over the series: it cuts the history at regular points, forecasts the next steps from each cut in one batched call, and scores every point by how far it landed from the model's median, in standard deviations implied by the 80% band. The bands are first rescaled for this series (see `scale`), so a score of 3 means three of this series' typical forecast misses.

```ts
const result = await model.detectAnomalies(dailySignups, { sensitivity: "medium" });

result.anomalies; // [{ index: 212, value: 680, expected: 410, lower: 352, upper: 468, score: 6.0, direction: "high", anomaly: true }, ...]
result.points;    // every checked point, oldest first, for plotting the expected range
result.threshold; // the score above which a point is flagged
```

`sensitivity` is `"low"`, `"medium"` (default) or `"high"`, or a score threshold such as `3.5`. Each model has its own presets in `MODELS[id].anomalyThresholds`, tuned so that about 1% or fewer clean points are flagged at medium:

| Model | low | medium | high |
|---|---|---|---|
| TimesFM 2.5, Chronos-Bolt Mini, Small, Base | 4 | 3 | 2.5 |
| Chronos-Bolt Tiny | 4.25 | 3.25 | 2.75 |

On the tuning suite (seasonal, trend, random walk, count and heavy-tailed series with planted spikes and level shifts), TimesFM and Chronos-Bolt Small caught every planted anomaly at medium; Tiny, Mini and Base missed one to three on random walks. Checking starts after the first quarter of the series (16 to 64 points, counted from the first observed value), which is context only. At most the latest 8,192 points (Chronos-Bolt) or 16,384 points (TimesFM) after that are checked; `points[0].index` shows where checking started. A series needs at least 20 observed points from its first value (`INVALID_INPUT` otherwise).

## In practice

### Checking a metric from your monitoring system

Most monitoring APIs return timestamped points. Sort them oldest first, keep one value per interval (with `null` for intervals that have no data), and map flagged indices back to timestamps.

```ts
// [{ t: "2026-10-01T00:00:00Z", v: 212 }, { t: "2026-10-01T01:00:00Z", v: 198 }, ...]
const points: { t: string; v: number | null }[] = await (await fetch("/api/metrics/requests?step=1h")).json();
points.sort((a, b) => Date.parse(a.t) - Date.parse(b.t));

const { anomalies } = await model.detectAnomalies(points.map((p) => p.v));
for (const a of anomalies.reverse()) {
  const pct = Math.round((100 * (a.value - a.expected)) / a.expected);
  console.log(`${points[a.index].t}: ${a.value} requests, ${pct}% ${a.direction === "high" ? "above" : "below"} expected`);
}
```

### Alerting on the latest value

For a scheduled check, run on the recent history and look only at the newest point.

```ts
const result = await model.detectAnomalies(last30Days);
const latest = result.points[result.points.length - 1];
if (latest?.anomaly) notify(`Unusual value ${latest.value}; expected about ${latest.expected.toFixed(0)}`);
```

### Choosing a sensitivity

Start with the default, `"medium"`. If it misses things you care about, try `"high"`; if it flags too much, `"low"`. Scores do not depend on the sensitivity, so you can try thresholds on one result without running the model again:

```ts
const result = await model.detectAnomalies(series);
const strict = result.points.filter((p) => p.score > 4);
const loose = result.points.filter((p) => p.score > 2.5);
```

### Drawing the expected range

Every checked point carries the range the model expected, so you can shade it behind the actual values, the way the demo does.

```ts
const band = result.points.map((p) => ({ x: dates[p.index], low: p.lower, high: p.upper, actual: p.value }));
```

## Calibrated bands

`forecast(series, { calibrate: true })` is experimental: its behavior may change in a minor release. It uses the same backtest, one model pass ahead, to check the bands. If the model's 80% band held less than 70% or more than 90% of this series' past values, every quantile's distance from the median is rescaled so it would have held 80%; otherwise the bands are left alone. The median and mean are unchanged, and `forecast.calibration` reports `{ scale, coverage, points }`, where coverage is the raw band's share of past values (`null`, with `scale: 1`, when the history is too short to backtest). The dead band is deliberate: on heavy-tailed test series the models' raw bands already held 80 to 87% of new values, and rescaling them anyway brought that down to 72 to 76%.

Each costs one extra batched forecast call per series, runs on top of `forecast()` and works with the worker. In React, `useForecaster()` also returns `detectAnomalies`.
