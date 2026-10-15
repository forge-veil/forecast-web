---
title: Changelog
group: Guides
---

# Changelog

## 1.0.0

The API is now stable; see Versioning in the README.

- **Breaking:** each model's weights, forecasting and tabular, download at the commit this release was verified against instead of `main`. Pass `revision: "main"` for the old behavior. Custom `weights` repos still default to `main`.
- **Breaking:** `Forecaster.config`, the `ModelConfig` type and `cleanSeries` are no longer exported.
- **Breaking:** `ModelInfo` and `TabularInfo` are read-only types, matching the frozen objects in `MODELS` and `TABULAR_MODELS`.
- **Breaking:** `forecast-web/react` renames `Status` to `ForecasterStatus`, and now re-exports the whole main API.
- **Breaking:** `Tabular.backend` and `Tabular.dtype` are always present.
- Published weights are checked against sizes and SHA-256 hashes built into the package, so a release loads exactly the files it was tested with; other sources are checked against their `config.json`. A mismatch fails with `LOAD_FAILED`.
- Cached weights are keyed by content, so later revisions with unchanged files reuse them. Upgrading to 1.0 downloads each model once more; call `clearCache()` to free copies stored by 0.x.
- Added `clearCache(model?)` to delete downloaded model files for one model or all of them, from every entry point.
- A disposed model now rejects every call with `DISPOSED` before validating input.
- `calibrate` and `Forecast.calibration` are marked experimental.
- Browsers only for now; Node support is planned for a minor release. CI runs the built package in Chromium and WebKit.

## 0.5.0 (2026-10-14)

- Added `detectAnomalies(series, { sensitivity })` on every forecaster and in `useForecaster()`, with per-model thresholds in `MODELS[id].anomalyThresholds`.
- Added `forecast(series, { calibrate: true })` and `Forecast.calibration`.

## 0.4.1 (2026-10-14)

- Tabular: frozen `LIMITS`, a guard after `dispose()`, `backend` and `dtype` on the loaded model, and results typed by task.

## 0.4.0 (2026-10-14)

- Added `forecast-web/tabular`: TabICLv2 classification and regression with `loadTabular`.

## 0.3.0 (2026-10-12)

- Added Chronos-Bolt Tiny, Mini, Small and Base. fp16 is the default on both backends.

## 0.2.0 (2026-10-12)

- Renamed from `timesfm-web` to `forecast-web`, with a model registry (`MODELS`, `model` option).

## 0.1.0 (2026-10-11)

- First release as `timesfm-web`: TimesFM 2.5 in the browser on WebGPU and WASM.
