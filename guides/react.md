---
title: React
group: Guides
---

# React

```tsx
import { useEffect, useState } from "react";
import { ForecastProvider, useForecaster } from "forecast-web/react";

function App() {
  return (
    <ForecastProvider model="timesfm-2.5">
      <Forecast series={data} />
    </ForecastProvider>
  );
}

function Forecast({ series }: { series: number[] }) {
  const { status, progress, forecast } = useForecaster();
  const [result, setResult] = useState<number[] | null>(null);

  useEffect(() => {
    forecast([series], { horizon: 24 }).then(([f]) => setResult(f.median));
  }, [series, forecast]);

  if (status === "loading" && progress) return <p>Loading {Math.round((100 * progress.loaded) / progress.total)}%</p>;
  if (status === "error") return <p>Could not load the model.</p>;
  return <Chart history={series} forecast={result} />;
}
```

The provider loads lazily, on the first `forecast()` or `load()` call. Pass `eager` to start the download on mount. It accepts the same options as `loadForecaster`, reads them when the load starts, and disposes the model on unmount. To switch models, remount it: `<ForecastProvider key={model} model={model}>`. `useForecaster()` returns `status` (`"idle" | "loading" | "ready" | "error"`), `progress`, `backend`, `error`, `load()`, `forecast()` and `detectAnomalies()`.

## Loading data, then forecasting

Real components usually fetch the data first. Keep the series in state, and forecast when it arrives.

```tsx
function SignupsForecast() {
  const { status, progress, error, forecast } = useForecaster();
  const [series, setSeries] = useState<number[] | null>(null);
  const [result, setResult] = useState<Forecast | null>(null);

  useEffect(() => {
    fetch("/api/signups?days=365")
      .then((r) => r.json())
      .then((rows: { day: string; count: number }[]) => setSeries(rows.map((r) => r.count)));
  }, []);

  useEffect(() => {
    if (series) forecast([series], { horizon: 30 }).then(([f]) => setResult(f));
  }, [series, forecast]);

  if (error) return <p>Could not load the model: {error.message}</p>;
  if (!series) return <p>Loading data…</p>;
  if (status === "loading") return <p>Downloading model {progress ? Math.round((100 * progress.loaded) / progress.total) : 0}%</p>;
  return <Chart history={series} median={result?.median} low={result?.q10} high={result?.q90} />;
}
```

`Forecast` is exported from `forecast-web/react`, along with the rest of the main API.

## Flagging anomalies in a component

```tsx
const { detectAnomalies } = useForecaster();
const [flags, setFlags] = useState<AnomalyPoint[]>([]);

useEffect(() => {
  detectAnomalies(series).then((r) => setFlags(r.anomalies));
}, [series, detectAnomalies]);
```
