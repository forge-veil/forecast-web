---
title: Deployment
group: Guides
---

# Deployment

## Bundlers

The package spawns its worker with `new Worker(new URL("./worker.js", import.meta.url), { type: "module" })`, which Vite and webpack 5 detect and bundle. A fresh Vite 8 app works with no configuration, in dev and in production builds.

Vite also copies onnxruntime-web's WASM binary (28 MB) into the build output. With the default `wasmPaths` that local copy is never requested, because the runtime loads from jsDelivr. To self-host it, point `wasmPaths` at the folder that serves those files.

onnxruntime-web only uses multiple WASM threads when the page is cross-origin isolated, which requires the `Cross-Origin-Opener-Policy` and `Cross-Origin-Embedder-Policy` headers. Without them, WASM runs on one thread. WebGPU is unaffected. The timings above were measured without isolation.

## Self-hosting the weights

Put these files in a folder on any static host and pass its URL as `weights`:

```
config.json
onnx/model_quantized.onnx     q8
onnx/model_fp16w.onnx         fp16 graph (under 1 MB)
onnx/model_fp16w.f16          fp16 weights
onnx/model.onnx               fp32
```

You only need the formats you plan to load. The fp32 and q8 files use Transformers.js file names. The fp16 pair is a compact format specific to this package: an fp32 graph whose weights ship as fp16 and are expanded in the browser. A standard fp16 conversion gives wrong results on WebGPU, which accumulates in half precision.

The `tools/` folder rebuilds every weights file from its upstream checkpoint. TimesFM:

```sh
cd tools
uv sync --python 3.12
uv run python models/timesfm/export_onnx.py timesfm      # fp32 graph
uv run python quantize.py timesfm                        # q8 and fp16 variants
uv run python models/timesfm/make_reference.py timesfm   # parity fixtures from the official library
uv run python package_hub.py timesfm                     # config.json with file sizes and hashes, model card
```

Chronos-Bolt sizes share one set of scripts; pass the size's model name:

```sh
uv run python models/chronos-bolt/export_onnx.py chronos-bolt-small      # fp32 graph
uv run python quantize.py chronos-bolt-small                             # q8 and fp16 variants
uv run python models/chronos-bolt/make_reference.py chronos-bolt-small   # parity fixtures from chronos-forecasting
uv run python package_hub.py chronos-bolt-small                          # config.json and model card
```

The loader always fetches `config.json` fresh and keys cached weights by the SHA-256 listed there, so uploading new weights with a regenerated config invalidates old cached copies.

Then `npm test` checks all three formats against the official forecasts.

## Browser support

| Browser | Status |
|---|---|
| Chrome and Edge, desktop | WebGPU and WASM. Measured in Chrome on macOS; WASM checked in CI. |
| Safari, desktop | WASM checked in CI on WebKit. WebGPU depends on the Safari version. |
| Firefox | Not tested. WASM is expected to work. |
| Phones | Not tested. Large models may exceed a phone's memory; prefer Chronos-Bolt Tiny or Mini. |

CI runs the built package in Chromium and WebKit on every change: it loads Chronos-Bolt Tiny from the Hub, forecasts and checks for anomalies. Node is not supported yet; support is planned for a minor release.
