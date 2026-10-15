---
title: Models
group: Guides
---

# Models

## Which model should I use?

| If you need | Use |
|---|---|
| Phones and tablets | Chronos-Bolt Tiny (18 MB); larger models crashed mobile browsers in testing |
| A small download or many quick forecasts | Chronos-Bolt Tiny (18 MB) or Mini (43 MB) |
| A balance of download size and quality | Chronos-Bolt Small (96 MB) |
| The most capable forecasts, with room for a larger download | TimesFM 2.5 (default; 261 to 463 MB) or Chronos-Bolt Base (411 MB) |
| More than 512 points of history to matter | Chronos-Bolt, which reads up to 2048 |
| Horizons beyond 128 steps | Chronos-Bolt, up to 512 |
| A mean forecast as well as the median | TimesFM 2.5 |
| Predictions from a table rather than a time series | [`forecast-web/tabular`](./tabular.md) |

Larger models are generally more accurate and slower; how much that matters depends on your data. To compare on your own series, load several models in the [demo](https://vatsalbakshi.com/forecast-web/) and use Backtest: it hides the most recent values and scores each model against them.

## All models

| Model | id | Context | Horizon | Weights |
|---|---|---|---|---|
| [TimesFM 2.5](https://github.com/google-research/timesfm) by Google Research | `"timesfm-2.5"` (default) | 512 | 128 | [bakshinghampalace/timesfm-2.5-200m-onnx](https://huggingface.co/bakshinghampalace/timesfm-2.5-200m-onnx) |
| [Chronos-Bolt](https://github.com/amazon-science/chronos-forecasting) Tiny (9M) by Amazon | `"chronos-bolt-tiny"` | 2048 | 64 per pass, up to 512 | [bakshinghampalace/chronos-bolt-tiny-onnx](https://huggingface.co/bakshinghampalace/chronos-bolt-tiny-onnx) |
| [Chronos-Bolt](https://github.com/amazon-science/chronos-forecasting) Mini (21M) by Amazon | `"chronos-bolt-mini"` | 2048 | 64 per pass, up to 512 | [bakshinghampalace/chronos-bolt-mini-onnx](https://huggingface.co/bakshinghampalace/chronos-bolt-mini-onnx) |
| [Chronos-Bolt](https://github.com/amazon-science/chronos-forecasting) Small (48M) by Amazon | `"chronos-bolt-small"` | 2048 | 64 per pass, up to 512 | [bakshinghampalace/chronos-bolt-small-onnx](https://huggingface.co/bakshinghampalace/chronos-bolt-small-onnx) |
| [Chronos-Bolt](https://github.com/amazon-science/chronos-forecasting) Base (205M) by Amazon | `"chronos-bolt-base"` | 2048 | 64 per pass, up to 512 | [bakshinghampalace/chronos-bolt-base-onnx](https://huggingface.co/bakshinghampalace/chronos-bolt-base-onnx) |

Every model's forecasts are checked against its official Python library. For TimesFM, the fp32 model agrees with `TimesFM_2p5_200M_torch.forecast()` to within 0.002% of each series' standard deviation, and an automated parity test enforces agreement within 0.1%. Every Chronos-Bolt size matches `ChronosBoltPipeline.predict_quantiles()`; each model card publishes the measured error per format at 64 and 200 steps.

## Chronos-Bolt

Chronos-Bolt reads up to 2048 points and predicts 64 steps per pass, which is its default horizon. Longer horizons, up to 512, use the official heuristic: each of the nine quantile paths is fed back and forecast again, and the 81 results per step are reduced to nine quantiles. Every pass after the first runs nine times the batch, so a 512-step forecast costs 8 sequential passes. Missing values are masked, not interpolated, exactly as upstream. Chronos-Bolt has no mean head, so `mean` is absent.

Past 64 steps each pass feeds its own quantiles back, so compression error compounds: at 200 steps fp32 still matches Python within 0.004% of the series' standard deviation and fp16 (the default on both backends) within 2%, while q8 can differ by up to about 15%. q8 is opt-in (`dtype: "q8"`): it saves download size and matches Python closely for horizons up to 64.

| Size | fp16 download (default) | q8 download (opt-in) |
|---|---|---|
| Tiny | 18 MB | 10 MB |
| Mini | 43 MB | 24 MB |
| Small | 96 MB | 54 MB |
| Base | 411 MB | 232 MB |

## TimesFM weights and backend

`"auto"` picks fp16 on WebGPU and q8 on WASM. q8 cannot run on WebGPU: onnxruntime-web's WebGPU backend only supports 2-bit and 4-bit packed weights, and 4-bit is too lossy for this model.

| dtype | Download | Runs on | Accuracy vs Python (mean error / std) |
|---|---|---|---|
| `fp16` | 441 MB | WebGPU, WASM | under 0.05% |
| `q8` | 249 MB | WASM | under 0.8% |
| `fp32` | 883 MB | WebGPU, WASM | under 0.0005% |

Measured median forecast times for one series on an Apple M2 with 8 GB of memory in Chrome 154:

| Mode | Median forecast |
|---|---|
| WebGPU, fp16 | 55 to 86 ms |
| WASM, fp16 | 351 ms |
| WASM, q8 | 864 ms |

The first forecast on WebGPU also compiles shaders, which took between 1 and 15 seconds on that machine depending on load. Treat all of these numbers as rough. They varied several-fold with other work on the machine.

## TimesFM limits

- Loading briefly needs a few times the file size in memory, so fp32 TimesFM (926 MB) may not load on machines with little free memory. The default formats are about half that size.

- Context is the most recent 512 points. Longer series are truncated, as in the Python library with `max_context=512`.
- Horizon is at most 128 steps.
- TimesFM 3.0 is not supported. Its weights are licensed for non-commercial use only.
- No covariates. TimesFM 2.5 supports them in Python through a separate regression step, which this package does not port.
- No multivariate input. That is a TimesFM 3.0 feature.
