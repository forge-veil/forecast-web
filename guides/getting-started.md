Time series forecasting and tabular prediction models that run in the browser. Give it a history, get back a forecast with nine quantile bands; give it a table, get back predicted classes or values with no training. Everything runs on the visitor's machine, on WebGPU when available and WASM otherwise.

## Install

```sh
npm install forecast-web
```

## Your first forecast

```ts
import { loadForecaster } from "forecast-web";

const model = await loadForecaster({
  model: "timesfm-2.5",
  onProgress: ({ loaded, total }) => console.log(`${Math.round((100 * loaded) / total)}%`),
});

const [forecast] = await model.forecast([dailySignups], { horizon: 30 });

forecast.median; // the point forecast, 30 numbers
forecast.q10;    // lower band
forecast.q90;    // upper band
```

The first load downloads the weights (for TimesFM, 249 to 441 MB, see below) and stores them with the browser Cache API, so later visits skip the download.

`MODELS` describes every model without loading it, which is enough to build a picker:

```ts
import { MODELS } from "forecast-web";

MODELS["timesfm-2.5"].maxHorizon; // 128
MODELS["timesfm-2.5"].sizes.q8;   // 261301673 bytes
```

## Shaping your data

A series is any array-like of numbers, **oldest first**, with one value per time step: hourly, daily, monthly, whatever your data uses, as long as the steps are evenly spaced. Every example below ends with something you can pass to `forecast`.

### A plain array

```ts
const visits = [120, 132, 101, 134, 90, 230, 210, 182, 191, 234];
const [forecast] = await model.forecast([visits], { horizon: 7 });
```

`forecast` takes a list of series and returns one forecast per series, which is why the array is wrapped in `[...]`.

### Gaps

Use `null`, `undefined` or `NaN` for a step with no value. Keep the step rather than dropping it, so the spacing stays even.

```ts
const temperatures = [14.2, 15.1, null, 16.0, NaN, 17.3, 16.8];
```

### Several series at once

Pass them together to forecast them in one batch, which is faster than separate calls. They can have different lengths.

```ts
const [north, south, west] = await model.forecast([salesNorth, salesSouth, salesWest], { horizon: 12 });
```

### From a CSV file

Pick the value column, sort by date, and keep the dates for labelling the forecast later.

```ts
// date,value
// 2026-01-01,412
// 2026-01-02,398
const text = await (await fetch("/data/signups.csv")).text();
const rows = text.trim().split("\n").slice(1).map((line) => {
  const [date, value] = line.split(",");
  return { date, value: value === "" ? null : Number(value) };
});
rows.sort((a, b) => a.date.localeCompare(b.date));

const series = rows.map((r) => r.value);
const dates = rows.map((r) => r.date);
```

### From JSON records

APIs often return newest first; reverse or sort before forecasting.

```ts
const records: { timestamp: string; price: number }[] = await (await fetch("/api/prices")).json();
const series = records
  .sort((a, b) => Date.parse(a.timestamp) - Date.parse(b.timestamp))
  .map((r) => r.price);
```

### From raw events

Models need regular steps, so roll individual events up into counts per period, and fill periods with no events (zero for counts, `null` for "unknown").

```ts
// events: [{ at: "2026-03-01T09:14:00Z" }, { at: "2026-03-01T17:02:00Z" }, ...]
const perDay = new Map<string, number>();
for (const e of events) {
  const day = e.at.slice(0, 10);
  perDay.set(day, (perDay.get(day) ?? 0) + 1);
}

const days: string[] = [];
for (let d = new Date("2026-03-01"); d <= new Date("2026-06-30"); d.setUTCDate(d.getUTCDate() + 1)) {
  days.push(d.toISOString().slice(0, 10));
}
const series = days.map((day) => perDay.get(day) ?? 0);
```

### Typed arrays

`Float64Array`, `Float32Array` and other array-likes work as they are, without copying into a regular array first.

```ts
const readings = new Float32Array(sensorBuffer);
const [forecast] = await model.forecast([readings], { horizon: 60 });
```

### Long histories

You can pass as much history as you have. Each model reads only its most recent points (512 for TimesFM, 2048 for Chronos-Bolt; see `MODELS[id].maxContext`) and ignores the rest.

### Putting dates on the forecast

Forecasts are arrays aligned to the steps after your last value. Extend your dates by the same step to label them.

```ts
const last = new Date(dates[dates.length - 1]);
const labelled = forecast.median.map((value, i) => {
  const day = new Date(last);
  day.setUTCDate(day.getUTCDate() + i + 1);
  return { date: day.toISOString().slice(0, 10), value, low: forecast.q10[i], high: forecast.q90[i] };
});
```

### How missing values are handled

For TimesFM, missing values are handled like the Python library: leading gaps are dropped, interior gaps are filled by linear interpolation, and trailing gaps hold the last value. Two edge cases differ on purpose. An empty or all-missing series gives a forecast of zeros, where Python raises an error. `Infinity` counts as missing, where Python keeps it and returns non-finite output. Chronos-Bolt masks missing values instead, exactly as upstream; an all-missing series gives the model's bands around 0, and `Infinity` also counts as missing.

## Next steps

- [Models](./models.md): which model to pick and what each supports.
- [Forecasting](./forecasting.md): options, horizons and errors.
- [Anomalies](./anomalies.md): check a series against its own history.
- [Tabular prediction](./tabular.md): classification and regression on tables.
- [React](./react.md): a provider and hook.
