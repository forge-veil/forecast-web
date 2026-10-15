# forecast-web

[![npm](https://img.shields.io/npm/v/forecast-web?color=1c1c1e&label=npm)](https://www.npmjs.com/package/forecast-web) [![CI](https://github.com/forge-veil/forecast-web/actions/workflows/ci.yml/badge.svg)](https://github.com/forge-veil/forecast-web/actions/workflows/ci.yml) [![License](https://img.shields.io/npm/l/forecast-web?color=1c1c1e)](LICENSE)

Time series forecasting, anomaly detection and tabular prediction that run entirely in the browser. Give it a history, get back a forecast with uncertainty bands; give it a table, get back predictions with no training. Inference runs on the visitor's machine, on WebGPU when available and WASM otherwise. No server, and no data leaves the page.

**[Documentation](https://vatsalbakshi.com/forecast-web/docs/)** · **[Live demo](https://vatsalbakshi.com/forecast-web/)** · **[Changelog](CHANGELOG.md)**

[![US retail sales forecast with TimesFM 2.5, with flagged anomalies](https://raw.githubusercontent.com/forge-veil/forecast-web/main/.github/assets/hero.gif)](https://vatsalbakshi.com/forecast-web/)

```sh
npm install forecast-web
```

## Forecast

```ts
import { loadForecaster } from "forecast-web";

const model = await loadForecaster({ model: "timesfm-2.5" });

const dailySignups = [412, 398, 455, 470, 433, 390, 401 /* ... oldest first */];
const [forecast] = await model.forecast([dailySignups], { horizon: 30 });

forecast.median; // 30 numbers: the point forecast
forecast.q10;    // lower edge of the 80% range
forecast.q90;    // upper edge
```

A series is any array of numbers, oldest first, one value per evenly spaced step; use `null` for gaps. [Shaping your data](https://vatsalbakshi.com/forecast-web/docs/#shaping-your-data) shows CSV files, API records, raw events and typed arrays.

## Find anomalies

```ts
const { anomalies } = await model.detectAnomalies(hourlyRequests);

for (const a of anomalies) console.log(`point ${a.index}: ${a.value}, expected about ${a.expected}`);
```

Each point is compared with what the model expected from the history before it, using thresholds tuned per model. [Anomalies guide](https://vatsalbakshi.com/forecast-web/docs/documents/Anomalies.html)

## Predict from a table

```ts
import { loadTabular } from "forecast-web/tabular";

const clf = await loadTabular({ task: "classification" });
const fitted = clf.fit(trainRows, trainLabels); // rows of numbers, text, booleans or null
const labels = await fitted.predict(newRows);
```

[TabICLv2](https://github.com/soda-inria/tabicl) reads the training rows in context, so `fit` is instant. Regression works the same way with `task: "regression"`. [Tabular guide](https://vatsalbakshi.com/forecast-web/docs/documents/Tabular_prediction.html)

## React

```tsx
import { ForecastProvider, useForecaster } from "forecast-web/react";

<ForecastProvider model="chronos-bolt-small">
  <SignupsChart />
</ForecastProvider>;

function SignupsChart() {
  const { status, forecast } = useForecaster(); // loads the model on first use
  // ...
}
```

[React guide](https://vatsalbakshi.com/forecast-web/docs/documents/React.html)

## Models

| Model | id | Download (default) | Notes |
|---|---|---|---|
| [TimesFM 2.5](https://github.com/google-research/timesfm), Google Research | `"timesfm-2.5"` | 463 MB WebGPU, 261 MB WASM | Default. Reads 512 points, forecasts up to 128 steps |
| [Chronos-Bolt](https://github.com/amazon-science/chronos-forecasting) Tiny, Mini, Small, Base, Amazon | `"chronos-bolt-tiny"` ... `"chronos-bolt-base"` | 18, 43, 96, 411 MB | Reads 2048 points, forecasts up to 512 steps |
| [TabICLv2](https://github.com/soda-inria/tabicl), Inria | `forecast-web/tabular` | 56 to 59 MB | Up to 10 classes, 5,000 rows, 100 features |

Every model is checked against its official Python library, and weights are downloaded once, verified against hashes in the package, and cached. [Models guide](https://vatsalbakshi.com/forecast-web/docs/documents/Models.html)

## Learn more

- [Getting started](https://vatsalbakshi.com/forecast-web/docs/): install, a first forecast, and shaping real data.
- [Forecasting](https://vatsalbakshi.com/forecast-web/docs/documents/Forecasting.html): options, horizons, caching and errors.
- [Deployment](https://vatsalbakshi.com/forecast-web/docs/documents/Deployment.html): bundlers, self-hosting the weights, browser support.
- [Troubleshooting](https://vatsalbakshi.com/forecast-web/docs/documents/Troubleshooting.html): backends, slow loads, memory, CORS and common errors.
- [Versioning](https://vatsalbakshi.com/forecast-web/docs/documents/Versioning.html): what may change in minor and major releases.
- [API reference](https://vatsalbakshi.com/forecast-web/docs/modules.html), and [every published version](https://www.jsdocs.io/package/forecast-web) on jsDocs.io.

Runs in desktop Chrome, Edge and Safari: CI tests every change in Chromium and WebKit, and Firefox and phones are not tested yet. Node support is planned for a minor release.

## License

Apache-2.0. TimesFM is by Google Research and Chronos by Amazon, both Apache-2.0. TabICL is by the Soda team at Inria under BSD-3-Clause. See [NOTICE](NOTICE). Contributions welcome; see [CONTRIBUTING.md](CONTRIBUTING.md).
