---
license: apache-2.0
library_name: onnx
pipeline_tag: time-series-forecasting
base_model: amazon/chronos-bolt-small
base_model_relation: quantized
tags:
  - chronos
  - chronos-bolt
  - time-series
  - forecasting
  - zero-shot
  - onnx
  - onnxruntime-web
  - webgpu
  - wasm
  - browser
  - javascript
---

# Chronos-Bolt Small for the browser

[![npm](https://img.shields.io/npm/v/forecast-web?label=npm%20forecast-web)](https://www.npmjs.com/package/forecast-web)
[![Live demo](https://img.shields.io/badge/demo-live-0a84ff)](https://vatsalbakshi.com/forecast-web/)
[![GitHub](https://img.shields.io/badge/GitHub-forge--veil%2Fforecast--web-181717?logo=github)](https://github.com/forge-veil/forecast-web)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](#license)

ONNX exports of Amazon's [Chronos-Bolt Small (48M)](https://huggingface.co/amazon/chronos-bolt-small), built to run time series forecasting entirely in the browser on WebGPU or WebAssembly. They are the `chronos-bolt-small` model of the [`forecast-web`](https://www.npmjs.com/package/forecast-web) npm package.

Give it up to 2048 points of history and get back a forecast with nine quantile bands, 64 steps per pass and up to 512 steps with the official multi-pass rollout. No server, no training, no API key.

## Quick start

```sh
npm install forecast-web
```

```ts
import { loadForecaster } from "forecast-web";

const model = await loadForecaster({ model: "chronos-bolt-small" });
const [forecast] = await model.forecast([dailySignups], { horizon: 30 });

forecast.median; // point forecast
forecast.q10;    // lower band
forecast.q90;    // upper band
```

## Files

| Format | Files | Runs on | `dtype` |
|---|---|---|---|
| fp16 weights, fp32 compute | `onnx/model_fp16w.onnx` + `onnx/model_fp16w.f16` | WebGPU, WASM | `"fp16"` (default on WebGPU and WASM) |
| 8-bit weight-only | `onnx/model_quantized.onnx` | WASM | `"q8"` (opt-in; accurate up to 64 steps) |
| fp32 | `onnx/model.onnx` | WebGPU, WASM | `"fp32"` |

`config.json` lists every file's size and SHA-256 plus the pinned upstream revision. The fp16 pair is an fp32 graph whose weights ship as fp16 and are expanded in the browser: a standard fp16 conversion accumulates in half precision on WebGPU.

## The graph

One forward pass for a fixed 2048-point context.

| Name | Type | Shape | Meaning |
|---|---|---|---|
| `context` | float32 | `[batch, 2048]` | normalized values, 0 where missing or padded, left-padded |
| `mask` | float32 | `[batch, 2048]` | 1 = observed |
| `quantiles` | float32 | `[batch, 9, 64]` | normalized q10..q90 |

Outside the graph, as in `chronos_bolt.py`: keep the last 2048 points; `loc` = mean and `scale` = population standard deviation of the observed values (0 and 1 if none are observed, `scale` 1e-5 if 0); normalize and zero the missing values; after the pass, multiply by `scale` and add `loc`. For horizons above 64, append each of the 9 quantile rows to its own copy of the context, run all 9, and reduce the 81 values per step to 9 with linear-interpolation quantiles, repeating per 64 steps. The reference implementation is [`src/models/chronos-bolt`](https://github.com/forge-veil/forecast-web/tree/main/src/models/chronos-bolt).

## Accuracy

Agreement with the official `ChronosBoltPipeline.predict_quantiles()` (chronos-forecasting 2.3.2) on twelve test series (seasonal, trending, very short, gappy, constant, negative, tiny-scale, longer than 2048), across all nine quantiles. Errors are fractions of each series' standard deviation. 64 steps is one pass; 200 steps uses the four-pass rollout, where each format's own quantiles are fed back, so compression error compounds.

| Format | Mean, 64 steps | Worst, 64 steps | Mean, 200 steps | Worst, 200 steps |
|---|---|---|---|---|
| fp32 | under 0.00010% | under 0.0007% | under 0.0003% | under 0.004% |
| fp16 weights | under 0.02% | under 0.10% | under 0.07% | under 2% |
| 8-bit | under 0.7% | under 4% | under 3% | under 20% |

fp16 is the default on both backends. q8 matches closely for one 64-step pass, but past that each pass feeds its own quantiles back and the difference compounds, so use it for short horizons when download size matters.

The parity suite runs in CI before every upload; nothing is published if it fails.

## How these files were made

Built and verified by the [`release-weights`](https://github.com/forge-veil/forecast-web/blob/main/.github/workflows/release-weights.yml) GitHub Actions workflow: download the checkpoint at a pinned revision, export the graph (opset 17), build the 8-bit and fp16-weight variants, generate reference forecasts with `chronos-forecasting`, run the parity suite against every format, and upload only if every check passes.

## Limitations

- Context is the most recent 2048 points; horizon at most 512 steps (quality may degrade past 64, as upstream notes).
- Univariate only. No covariates (that is Chronos-2).

## Citation

```bibtex
@article{ansari2024chronos,
  title   = {Chronos: Learning the Language of Time Series},
  author  = {Ansari, Abdul Fatir and Stella, Lorenzo and Turkmen, Caner and Zhang, Xiyuan and Mercado, Pedro and Shen, Huibin and Shchur, Oleksandr and Rangapuram, Syama Sundar and Pineda Arango, Sebastian and Kapoor, Shubham and Zschiegner, Jasper and Maddix, Danielle C. and Mahoney, Michael W. and Torkkola, Kari and Gordon Wilson, Andrew and Bohlke-Schneider, Michael and Wang, Yuyang},
  journal = {Transactions on Machine Learning Research},
  year    = {2024}
}
```

## License

Apache-2.0, the same license as the original Chronos-Bolt weights. Chronos is developed by Amazon. These conversions and the `forecast-web` library are independent and not affiliated with Amazon.
