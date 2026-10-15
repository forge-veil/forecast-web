---
title: Troubleshooting
group: Guides
---

# Troubleshooting

## Which backend am I on?

After loading, `model.backend` is `"webgpu"` or `"wasm"`, and `model.dtype` is the weight format in use.

```ts
const model = await loadForecaster();
console.log(model.backend, model.dtype); // "webgpu" "fp16"
```

`device: "auto"` (the default) uses WebGPU when the browser offers it and falls back to WASM otherwise. Pass `device: "wasm"` or `device: "webgpu"` to choose. Explicit `"webgpu"` fails with `LOAD_FAILED` instead of falling back.

## WebGPU is not being used

- The browser may not expose WebGPU, or may expose it only on some hardware. Check `"gpu" in navigator`.
- WebGPU needs a secure page: `https://` or `http://localhost`.
- `dtype: "q8"` runs on WASM only; asking for it with `device: "webgpu"` fails with a message saying so.

## The first forecast is slow

The first load downloads the weights; later visits read them from the Cache API. On WebGPU the first forecast also compiles shaders, which can take several seconds. Show progress with `onProgress`:

```ts
await loadForecaster({ onProgress: ({ loaded, total }) => setPercent(Math.round((100 * loaded) / total)) });
```

To start the download early, call `loadForecaster` (or pass `eager` to the React provider) before the user needs a forecast.

## The weights download on every visit

The Cache API is unavailable in some private browsing modes and on insecure pages (plain `http://` other than localhost), and browsers may evict it under storage pressure. Everything still works, but each visit downloads the weights again. `cache: false` turns caching off deliberately.

## WASM is slow

onnxruntime-web uses several WASM threads only when the page is cross-origin isolated, which needs these response headers:

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

Without them WASM runs on one thread. Smaller models help more than anything: Chronos-Bolt Tiny or Mini are much faster on WASM than TimesFM.

## The tab runs out of memory

Loading briefly needs a few times the file size in memory. Use the default formats rather than `fp32`, pick a smaller model on phones, and call `dispose()` on models you no longer need.

## Self-hosted weights fail to load (CORS)

When the weights live on another origin, that server must send `Access-Control-Allow-Origin` for the page's origin (or `*`). Without it the browser blocks the download and the load fails with `LOAD_FAILED`. See [Deployment](./deployment.md) for the expected folder layout.

## A download fails its SHA-256 check

`LOAD_FAILED` with "does not match its expected SHA-256" means the file that arrived is not the one the package was tested with: a truncated download, a proxy that rewrites responses, or changed files in a self-hosted repo. Reload to retry; if it persists, `clearCache()` and try again.

## INVALID_INPUT from detectAnomalies

The series is too short to check. Anomaly detection needs at least 20 observed points after the first value, and the first quarter of the series is used only as context. Pass more history.

## HORIZON_TOO_LONG

The horizon is above the model's maximum: 128 steps for TimesFM, 512 for Chronos-Bolt. See `MODELS[id].maxHorizon`.

## Still stuck?

[Open an issue](https://github.com/forge-veil/forecast-web/issues) with the browser, `model.backend`, `model.dtype` and the error's `code` and message.
