---
license: apache-2.0
library_name: onnx
pipeline_tag: time-series-forecasting
base_model: google/timesfm-2.5-200m-pytorch
tags:
  - timesfm
  - time-series
  - forecasting
  - onnx
  - webgpu
---

# TimesFM 2.5 200M, ONNX for the browser

ONNX exports of [google/timesfm-2.5-200m-pytorch](https://huggingface.co/google/timesfm-2.5-200m-pytorch) for use with [timesfm-web](https://www.npmjs.com/package/timesfm-web), which runs them in the browser on WebGPU or WASM.

## Files

| File | Format | Size | Runs on |
|---|---|---|---|
| `onnx/model.onnx` | fp32 | 883 MB | WebGPU, WASM |
| `onnx/model_fp16w.onnx` + `onnx/model_fp16w.f16` | fp32 graph, fp16 weights | 441 MB | WebGPU, WASM |
| `onnx/model_quantized.onnx` | 8-bit weight-only (MatMulNBits, block 32) | 249 MB | WASM |

The fp16 pair is not a standard fp16 ONNX model. The graph is fp32 and reads its large weights as external data named `model_fp16w.onnx_data`. The `.f16` file holds those weights as fp16, at half the graph's offsets. Expand it to fp32 and pass it as external data. timesfm-web does this automatically.

## Graph

One prefill pass of the model for a fixed 512-point context, covering horizons up to 128 with no autoregressive steps.

- Inputs: `inputs` float32 `[batch, 512]`, `masks` float32 `[batch, 512]` with 1 marking padding.
- Outputs: `point` float32 `[batch, 128, 10]` (channel 0 is the mean head, 1 to 9 are q10 to q90) and `qspread` float32 `[batch, 128, 10]` (the continuous quantile head).

Forecast-level processing is not in the graph. Input normalization, flip invariance, the continuous quantile head, the quantile crossing fix and the positivity clamp run in the caller, as in `ForecastConfig(max_context=512, max_horizon=128, normalize_inputs=True, use_continuous_quantile_head=True, force_flip_invariance=True, infer_is_positive=True, fix_quantile_crossing=True)`.

## Accuracy

Agreement with the official `TimesFM_2p5_200M_torch.forecast()` across ten test series, as a fraction of each series' standard deviation:

| Format | Mean error | Worst error |
|---|---|---|
| fp32 | under 0.0005% | under 0.002% |
| fp16 weights | under 0.05% | under 0.2% |
| 8-bit | under 0.8% | under 3.5% |

## License

Apache-2.0, same as the original weights. TimesFM is by Google Research.
