# timesfm-web

Run Google's [TimesFM 2.5](https://github.com/google-research/timesfm) time series forecasting model in the browser. Give it a history, get back a forecast up to 128 steps ahead with nine quantile bands. Everything runs on the visitor's machine, on WebGPU when available and WASM otherwise.

Forecasts match the official Python library. On the test series, the fp32 model agrees with `TimesFM_2p5_200M_torch.forecast()` to within 0.002% of each series' standard deviation. An automated parity test enforces agreement within 0.1%.

```sh
npm install timesfm-web
```

## Quick start

```ts
import { loadTimesFM } from "timesfm-web";

const model = await loadTimesFM({
  onProgress: ({ loaded, total }) => console.log(`${Math.round((100 * loaded) / total)}%`),
});

const [forecast] = await model.forecast([dailySignups], { horizon: 30 });

forecast.median; // the point forecast, 30 numbers
forecast.q10;    // lower band
forecast.q90;    // upper band
```

The first load downloads the weights (249 to 441 MB, see below) and stores them with the browser Cache API, so later visits skip the download.

Series are plain arrays, oldest first. `null`, `undefined` and `NaN` mark missing values. They are handled like the Python library: leading gaps are dropped, interior gaps are filled by linear interpolation, and trailing gaps hold the last value. Two edge cases differ on purpose. An empty or all-missing series gives a forecast of zeros, where Python raises an error. `Infinity` counts as missing, where Python keeps it and returns non-finite output. Pass several series to forecast them in one batch.

## React

```tsx
import { useEffect, useState } from "react";
import { TimesFMProvider, useTimesFM } from "timesfm-web/react";

function App() {
  return (
    <TimesFMProvider>
      <Forecast series={data} />
    </TimesFMProvider>
  );
}

function Forecast({ series }: { series: number[] }) {
  const { status, progress, forecast } = useTimesFM();
  const [result, setResult] = useState<number[] | null>(null);

  useEffect(() => {
    forecast([series], { horizon: 24 }).then(([f]) => setResult(f.median));
  }, [series, forecast]);

  if (status === "loading" && progress) return <p>Loading {Math.round((100 * progress.loaded) / progress.total)}%</p>;
  if (status === "error") return <p>Could not load the model.</p>;
  return <Chart history={series} forecast={result} />;
}
```

The provider loads lazily, on the first `forecast()` or `load()` call. Pass `eager` to start the download on mount. It accepts the same options as `loadTimesFM`, reads them when the load starts, and disposes the model on unmount. `useTimesFM()` returns `status` (`"idle" | "loading" | "ready" | "error"`), `progress`, `backend`, `error`, `load()` and `forecast()`.

## Options

| Option | Default | Meaning |
|---|---|---|
| `model` | `bakshinghampalace/timesfm-2.5-200m-onnx` | Hugging Face repo id, or a URL to a folder with the same layout for self-hosting. |
| `revision` | `"main"` | Hugging Face revision. |
| `dtype` | `"auto"` | Weight format: `"auto"`, `"q8"`, `"fp16"` or `"fp32"`. |
| `device` | `"auto"` | `"auto"`, `"webgpu"` or `"wasm"`. |
| `worker` | `true` | Run inference in a Web Worker so the page stays responsive. |
| `cache` | `true` | Keep downloads in the Cache API. |
| `wasmPaths` | jsDelivr | Where onnxruntime-web fetches its `.wasm` files. |
| `onProgress` | none | Called with `{ loaded, total, cached }` during the download. |
| `signal` | none | An `AbortSignal` that cancels the load. |

`forecast(series, { horizon })` accepts a horizon from 1 to 128 and returns, per series, `median`, `mean`, and `q10` through `q90`. The median is the official TimesFM point forecast. `mean` is the separate mean head.

Errors are `TimesFMError` instances with a `code`: `HORIZON_TOO_LONG`, `INVALID_HORIZON`, `EMPTY_INPUT`, `LOAD_FAILED`, `ABORTED` or `DISPOSED`.

## Choosing weights and backend

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

## Bundlers

The package spawns its worker with `new Worker(new URL("./worker.js", import.meta.url), { type: "module" })`, which Vite and webpack 5 detect and bundle. A fresh Vite 8 app works with no configuration, in dev and in production builds.

Vite also copies onnxruntime-web's WASM binary (28 MB) into the build output. With the default `wasmPaths` that local copy is never requested, because the runtime loads from jsDelivr. To self-host it, point `wasmPaths` at the folder that serves those files.

onnxruntime-web only uses multiple WASM threads when the page is cross-origin isolated, which requires the `Cross-Origin-Opener-Policy` and `Cross-Origin-Embedder-Policy` headers. Without them, WASM runs on one thread. WebGPU is unaffected. The timings above were measured without isolation.

## Self-hosting the weights

Put these files in a folder on any static host and pass its URL as `model`:

```
config.json
onnx/model_quantized.onnx     q8
onnx/model_fp16w.onnx         fp16 graph (under 1 MB)
onnx/model_fp16w.f16          fp16 weights
onnx/model.onnx               fp32
```

You only need the formats you plan to load. The fp32 and q8 files use Transformers.js file names. The fp16 pair is a compact format specific to this package: an fp32 graph whose weights ship as fp16 and are expanded in the browser. A standard fp16 conversion gives wrong results on WebGPU, which accumulates in half precision.

The `tools/` folder rebuilds every file from Google's checkpoint:

```sh
cd tools
uv sync --python 3.12
uv run python export_onnx.py      # fp32 graph
uv run python quantize.py         # q8 and fp16 variants
uv run python make_reference.py   # parity fixtures from the official library
uv run python package_hub.py      # config.json with file sizes and hashes, model card
```

The loader always fetches `config.json` fresh and keys cached weights by the SHA-256 listed there, so uploading new weights with a regenerated config invalidates old cached copies.

Then `npm test` checks all three formats against the official forecasts.

## Limits

- Context is the most recent 512 points. Longer series are truncated, as in the Python library with `max_context=512`.
- Horizon is at most 128 steps.
- TimesFM 3.0 is not supported. Its weights are licensed for non-commercial use only.
- No covariates. TimesFM 2.5 supports them in Python through a separate regression step, which this package does not port.
- No multivariate input. That is a TimesFM 3.0 feature.

## License

Apache-2.0. TimesFM is by Google Research, and its 2.5 weights are Apache-2.0. See `NOTICE`.
