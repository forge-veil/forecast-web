# Contributing

Thanks for helping improve timesfm-web. This guide covers setup, how the code is organized, and what a change needs before it can merge.

## Setup

You need Node 24 with npm 11.21 or newer. Older npm 11 releases skip rolldown's native binding on a clean install, which makes Vitest fail with "Cannot find native binding". Upgrade with `npm install -g npm@11`.

```sh
git clone <your fork>
cd timesfm-web
npm ci
npm test          # unit tests; the parity suite skips itself without weights
npm run typecheck
npm run build
```

Rebuilding the weights or running the parity suite also needs [uv](https://docs.astral.sh/uv/) and about 5 GB of free disk:

```sh
cd tools
uv sync --python 3.12
uv run python export_onnx.py      # fp32 graph, downloads Google's checkpoint
uv run python quantize.py         # q8 and fp16 variants
uv run python make_reference.py   # reference forecasts from the official library
uv run python package_hub.py      # config.json with sizes and hashes
cd ..
npm test                          # now includes parity for every format
```

## Layout

| Path | What lives there |
|---|---|
| `src/preprocess.ts`, `src/postprocess.ts` | The forecasting math around the graph. Pure functions, no onnxruntime. |
| `src/forecaster.ts` | Joins preprocessing, a graph runner, and postprocessing. |
| `src/assets.ts`, `src/half.ts` | File resolution, downloads, caching, fp16 weight expansion. |
| `src/loader.ts` | Picks weights and backend, downloads, creates the session. |
| `src/session.ts`, `src/gpu.ts` | onnxruntime-web session and WebGPU detection. |
| `src/protocol.ts`, `src/client.ts`, `src/worker.ts` | The Web Worker and its message protocol. |
| `src/index.ts`, `src/react.tsx` | The two public entry points. |
| `tools/` | Python scripts that export, compress, verify, and upload the weights. |
| `site/` | The public demo at vatsalbakshi.com/timesfm-web, a Vite and React app. |

## Conventions

- **Imports are root-relative.** Write `import { prepareBatch } from "src/preprocess"`, not `"./preprocess"`. tsconfig paths and the Vitest alias resolve them, and the build inlines them. The one exception is the worker URL in `src/index.ts`, which the browser resolves at runtime and must stay `./worker.js`.
- **Keep the pure modules pure.** Anything under `preprocess`, `postprocess` or `forecaster` must not import onnxruntime, so it stays testable in Node.
- **Write the test first.** Add a failing test, make it pass, then refactor.
- **No em dashes** in docs, comments, or UI text.
- **Commits** are one line with a type prefix, for example `fix: release the session after queued runs`.

## Changes to the forecasting math

The package promises parity with the official Python library, so changes to preprocessing, postprocessing, the export, or quantization need extra care:

1. Check the behavior against the official source. The reference is `TimesFM_2p5_200M_torch.forecast()` in [google-research/timesfm](https://github.com/google-research/timesfm), at the commit pinned in `tools/pyproject.toml`.
2. Rebuild the weights and run the full suite, including parity. Include the parity tables from the test output in your pull request.
3. If the change affects WebGPU or the graph, run the browser check below on both backends.

Do not loosen a parity tolerance to get a test passing without explaining the measured errors in the pull request.

## Checking in a browser

The demo site in `site/` installs `timesfm-web` from npm. To try your local changes, build the package and install it into the site:

```sh
npm run build
cd site
npm ci
npm install --no-save ..   # use the local package instead of the npm release
npm run data               # fetch the bundled datasets into public/data
npm run dev
```

Open http://localhost:5299/timesfm-web/. To test locally built weights, serve `build/hub` over HTTP with CORS and pass its URL:

```sh
npx http-server build/hub --cors -p 5199
# then open http://localhost:5299/timesfm-web/?model=http://localhost:5199/
```

Check both backends: Chrome with WebGPU, and the WASM path (for example with WebGPU disabled in `chrome://flags`). WebGPU bugs have shown up only in the browser, never in Node, so do not skip this for graph changes.

## Pull requests

- Keep each pull request to one change.
- CI must pass: typecheck, tests, build, and a package dry run.
- Describe what changed and how you verified it, including the browser and backend you tested.
- Releases are handled by the maintainer through GitHub Actions. See `RELEASING.md`.

## License

By contributing, you agree that your contributions are licensed under the Apache License 2.0, the same license as the project.
