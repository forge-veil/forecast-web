---
license: apache-2.0
library_name: onnx
pipeline_tag: time-series-forecasting
base_model: google/timesfm-2.5-200m-pytorch
base_model_relation: quantized
tags:
  - timesfm
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

# TimesFM 2.5 200M for the browser

[![npm](https://img.shields.io/npm/v/timesfm-web?label=npm%20timesfm-web)](https://www.npmjs.com/package/timesfm-web)
[![Live demo](https://img.shields.io/badge/demo-live-0a84ff)](https://vatsalbakshi.com/timesfm-web/)
[![GitHub](https://img.shields.io/badge/GitHub-forge--veil%2Ftimesfm--web-181717?logo=github)](https://github.com/forge-veil/timesfm-web)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](#license)

ONNX exports of Google Research's [TimesFM 2.5 (200M)](https://huggingface.co/google/timesfm-2.5-200m-pytorch), built to run time series forecasting entirely in the browser on WebGPU or WebAssembly. They are the default weights of the [`timesfm-web`](https://www.npmjs.com/package/timesfm-web) npm package.

Give it a history of up to 512 points and get back a forecast up to 128 steps ahead, with a median and nine quantile bands. No server, no training, no API key.

**[Try the live demo](https://vatsalbakshi.com/timesfm-web/)**: stocks, live weather, Wikipedia traffic and classic seasonal data, forecast and backtested in your browser.

## Quick start

```sh
npm install timesfm-web
```

```ts
import { loadTimesFM } from "timesfm-web";

const model = await loadTimesFM(); // downloads these weights once, then caches them
const [forecast] = await model.forecast([dailySignups], { horizon: 30 });

forecast.median; // point forecast, 30 values
forecast.q10;    // lower band
forecast.q90;    // upper band
```

React:

```tsx
import { TimesFMProvider, useTimesFM } from "timesfm-web/react";

<TimesFMProvider>
  <App />
</TimesFMProvider>;

const { status, progress, forecast } = useTimesFM();
```

The library picks the format and backend automatically, runs inference in a Web Worker, handles missing values, and reproduces the official forecast post-processing. See the [package README](https://github.com/forge-veil/timesfm-web#readme) for all options.

## Files

| Format | Files | Size | Runs on | `dtype` |
|---|---|---|---|---|
| fp16 weights, fp32 compute | `onnx/model_fp16w.onnx` + `onnx/model_fp16w.f16` | 441 MB | WebGPU, WASM | `"fp16"` (default on WebGPU) |
| 8-bit weight-only | `onnx/model_quantized.onnx` | 249 MB | WASM | `"q8"` (default on WASM) |
| fp32 | `onnx/model.onnx` | 883 MB | WebGPU, WASM | `"fp32"` |

`config.json` lists every file's size and SHA-256, plus the pinned source revision. The library keys its browser cache by those hashes, so updated weights never mix with stale cached copies.

### Why a custom fp16 format

A standard fp16 ONNX conversion gives wrong results on WebGPU, where reductions and attention scores accumulate in half precision. Casting fp16 weights to fp32 inside the graph fixes accuracy, but onnxruntime-web does not fold those casts on WebGPU, so every forecast pays for them.

`model_fp16w` avoids both problems. The graph is fp32 and reads its large weights as external data named `model_fp16w.onnx_data`. The `.f16` file holds those weights as fp16 at exactly half the graph's offsets. Expand it to fp32 in one pass and pass the result to onnxruntime as external data. `timesfm-web` does this automatically. The result runs at fp32 speed from a half-size download.

### Why 8-bit is WASM only

onnxruntime-web's WebGPU `MatMulNBits` kernel supports 2-bit and 4-bit weights only. A 4-bit build was measured and rejected: mean error of 3 to 7% of each series' standard deviation, with worst cases above 50%.

## Accuracy

Agreement with the official `TimesFM_2p5_200M_torch.forecast()` on eleven test series (seasonal, trending, short, gappy, constant, negative, tiny-scale, long), across all ten output channels and 128 steps. Errors are fractions of each series' standard deviation.

| Format | Mean error | Worst error |
|---|---|---|
| fp32 | under 0.0005% | under 0.002% |
| fp16 weights | under 0.05% | under 0.2% |
| 8-bit | under 0.8% | under 3.5% |

The same checks pass in headless Chrome on both WebGPU and WASM. The parity suite runs in CI before every upload; nothing is published if it fails.

## Speed

Median time for one forecast in Chrome on an Apple M2 (8 GB). The graph always computes all 128 steps, so the horizon does not change the cost.

| Backend | Format | Median forecast |
|---|---|---|
| WebGPU | fp16 weights | 55 to 86 ms |
| WASM | fp16 weights | about 350 ms |
| WASM | 8-bit | about 860 ms |

The first WebGPU forecast also compiles shaders, which takes one to several seconds. To forecast many series, pass them in one call: they run as a single batch.

## The graph

Each file is one prefill pass of the model for a fixed 512-point context, which covers horizons up to 128 with no autoregressive steps: running per-patch statistics, reversible instance normalization, the 20-layer transformer, and de-normalization of the last patch.

| Name | Type | Shape | Meaning |
|---|---|---|---|
| `inputs` | float32 | `[batch, 512]` | normalized context, left-padded |
| `masks` | float32 | `[batch, 512]` | 1 marks padding |
| `point` | float32 | `[batch, 128, 10]` | channel 0 is the mean head, 1 to 9 are q10 to q90 |
| `qspread` | float32 | `[batch, 128, 10]` | the continuous quantile head |

Forecast-level processing is not in the graph. To match the official library outside `timesfm-web`, reproduce `ForecastConfig(max_context=512, max_horizon=128, normalize_inputs=True, use_continuous_quantile_head=True, force_flip_invariance=True, infer_is_positive=True, fix_quantile_crossing=True)`:

1. Strip leading NaNs, linearly interpolate interior gaps, keep the last 512 points, left-pad with zeros and mark the padding.
2. Normalize by the mean and unbiased standard deviation of the padded window.
3. Run the graph on the series and on its negation, then average them with the quantile flip (channel q pairs with 10 minus q).
4. Rebuild quantiles 1 to 9 from the continuous head anchored on the median, fix quantile crossing, de-normalize, and clamp at zero for non-negative inputs.

The reference implementation is [`src/preprocess.ts` and `src/postprocess.ts`](https://github.com/forge-veil/timesfm-web/tree/main/src) in the library.

### Running the raw graph

```python
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download

path = hf_hub_download("bakshinghampalace/timesfm-2.5-200m-onnx", "onnx/model.onnx")
session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])

inputs = np.zeros((1, 512), dtype=np.float32)  # apply steps 1 and 2 above first
masks = np.zeros((1, 512), dtype=np.float32)
point, qspread = session.run(None, {"inputs": inputs, "masks": masks})
```

## How these files were made

Built and verified by the [`release-weights`](https://github.com/forge-veil/timesfm-web/blob/main/.github/workflows/release-weights.yml) GitHub Actions workflow:

1. Download Google's checkpoint at a pinned revision.
2. Export the prefill graph to ONNX (opset 17). Per-patch statistics are joined pairwise, because a single 16-input Concat exceeds WebGPU's per-shader buffer limit and produces wrong results there.
3. Build the 8-bit (`MatMulNBits`, block 32, symmetric) and fp16-weight variants.
4. Generate reference forecasts with the official Python library and run the parity suite against every format.
5. Upload, only if every check passes.

The export approach builds on [hwkim3330/timesfm-web](https://github.com/hwkim3330/timesfm-web).

## Limitations

- Context is the most recent 512 points and the horizon is at most 128 steps. The base model supports longer context; these exports do not.
- Univariate only. No covariates.
- TimesFM 3.0 is not included: its weights are licensed for non-commercial use only.
- Forecasts are statistical extrapolations. They are not financial advice and say nothing about events the history does not contain.

## Citation

```bibtex
@inproceedings{das2024decoder,
  title     = {A decoder-only foundation model for time-series forecasting},
  author    = {Das, Abhimanyu and Kong, Weihao and Sen, Rajat and Zhou, Yichen},
  booktitle = {Forty-first International Conference on Machine Learning},
  year      = {2024}
}
```

## License

Apache-2.0, the same license as the original TimesFM 2.5 weights. TimesFM is developed by Google Research. These conversions and the `timesfm-web` library are independent and not affiliated with Google.
