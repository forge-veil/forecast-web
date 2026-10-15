---
title: Forecasting
group: Guides
---

# Forecasting

## Options

| Option | Default | Meaning |
|---|---|---|
| `model` | `"timesfm-2.5"` | Which model to run. See `MODELS`. |
| `weights` | the model's published repo | Hugging Face repo id, or a URL to a folder with the same layout for self-hosting. |
| `revision` | the commit this release was verified against | Hugging Face revision. A custom `weights` repo defaults to `"main"`. |
| `dtype` | `"auto"` | Weight format: `"auto"`, `"q8"`, `"fp16"` or `"fp32"`. |
| `device` | `"auto"` | `"auto"`, `"webgpu"` or `"wasm"`. |
| `worker` | `true` | Run inference in a Web Worker so the page stays responsive. |
| `cache` | `true` | Keep downloads in the Cache API. |
| `wasmPaths` | jsDelivr | Where onnxruntime-web fetches its `.wasm` files. |
| `onProgress` | none | Called with `{ loaded, total, cached }` during the download. |
| `signal` | none | An `AbortSignal` that cancels the load. |

`forecast(series, { horizon, calibrate })` accepts a horizon from 1 to the model's `maxHorizon`, defaults to the model's `horizon` (128 for TimesFM, 64 for Chronos-Bolt), and returns, per series, `median` and `q10` through `q90`. The median is the point forecast. TimesFM also returns `mean`, its separate mean head. `calibrate: true` rescales the bands from a backtest; see [Anomalies and calibrated bands](./anomalies.md).

Calls on one model may overlap: they queue and each resolves with its own result. `dispose()` frees the model; every later call rejects with `DISPOSED`. `clearCache()` deletes downloaded model files, for one model or all of them:

```ts
import { clearCache } from "forecast-web";

await clearCache("timesfm-2.5"); // that model's published weights, every format
await clearCache();              // everything, including self-hosted weights
```

Both resolve `true` if anything was deleted.

## Errors

Every error is a `ForecastError` with a `code`. New codes may be added in minor releases, so handle unknown codes with a fallback.

| Code | When |
|---|---|
| `UNKNOWN_MODEL` | `model` is not in `MODELS`. |
| `LOAD_FAILED` | Weights could not be downloaded, failed their SHA-256 check or do not match the model, or the backend could not start. |
| `ABORTED` | The load's `signal` fired. |
| `DISPOSED` | The model was disposed. Checked before anything else. |
| `EMPTY_INPUT` | No series were passed to `forecast`. |
| `INVALID_HORIZON` | The horizon is not a positive integer. |
| `HORIZON_TOO_LONG` | The horizon is above the model's `maxHorizon`. |
| `INVALID_INPUT` | Malformed input (including an empty table), an unknown sensitivity, or a series too short to check for anomalies. |
| `TOO_MANY_CLASSES` | A classification table has more than 10 classes. |
| `TOO_LARGE` | A table is over the size limits in `LIMITS`. |
