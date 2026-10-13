# Contributing

Thanks for helping improve forecast-web. This guide covers setup, how the code is organized, and what a change needs before it can merge.

## Setup

You need Node 24 with npm 11.21 or newer. Older npm 11 releases skip rolldown's native binding on a clean install, which makes Vitest fail with "Cannot find native binding". Upgrade with `npm install -g npm@11`.

```sh
git clone <your fork>
cd forecast-web
npm ci
npm test          # unit tests; the parity suite skips itself without weights
npm run typecheck
npm run build
```

Rebuilding the weights or running the parity suite also needs [uv](https://docs.astral.sh/uv/) and about 5 GB of free disk:

```sh
cd tools
uv sync --python 3.12
uv run python models/timesfm/export_onnx.py timesfm      # fp32 graph, downloads Google's checkpoint
uv run python quantize.py timesfm                        # q8 and fp16 variants
uv run python models/timesfm/make_reference.py timesfm   # reference forecasts from the official library
uv run python package_hub.py timesfm                     # config.json with sizes and hashes
# Chronos-Bolt: the same steps with models/chronos-bolt/<script>.py and a size, e.g. chronos-bolt-small
uv run python models/chronos-bolt/export_onnx.py chronos-bolt-small
uv run python quantize.py chronos-bolt-small
uv run python models/chronos-bolt/make_reference.py chronos-bolt-small
uv run python package_hub.py chronos-bolt-small
cd ..
npm test                                                 # parity for every built model and format
```

## Layout

| Path | What lives there |
|---|---|
| `src/core/` | Model-agnostic download, cache, fp16 expansion, onnxruntime session, worker, protocol and loader. |
| `src/models/` | One folder per model: its info, files, config check, pre- and post-processing. `src/models/index.ts` is the registry and `adapter.ts` the interface. |
| `src/models/timesfm/` | TimesFM 2.5: `preprocess.ts` and `postprocess.ts` hold the forecasting math around the graph (pure functions, no onnxruntime). |
| `src/index.ts`, `src/react.tsx` | The two public entry points. |
| `tools/` | Python scripts. Shared packaging and upload scripts at the top; per-model export, compression and reference scripts in `tools/models/<model>/`. |
| `site/` | The public demo at vatsalbakshi.com/forecast-web, a Vite and React app. |

## Conventions

- **Imports are root-relative.** Write `import { prepareBatch } from "src/models/timesfm/preprocess"`, not `"./preprocess"`. tsconfig paths and the Vitest alias resolve them, and the build inlines them. The one exception is the worker URL in `src/index.ts`, which the browser resolves at runtime and must stay `./worker.js`.
- **Keep the pure modules pure.** Model code under `src/models/` must not import onnxruntime, so it stays testable in Node.
- **Write the test first.** Add a failing test, make it pass, then refactor.
- **No em dashes** in docs, comments, or UI text.
- **Commits** are one line with a type prefix, for example `fix: release the session after queued runs`.

## Changes to the forecasting math

The package promises parity with the official Python library, so changes to a model's preprocessing, postprocessing, the export, or quantization need extra care:

1. Check the behavior against the official source. The reference is `TimesFM_2p5_200M_torch.forecast()` in [google-research/timesfm](https://github.com/google-research/timesfm), at the commit pinned in `tools/pyproject.toml`.
2. Rebuild the weights and run the full suite, including parity. Include the parity tables from the test output in your pull request.
3. If the change affects WebGPU or the graph, run the browser check below on both backends.

Do not loosen a parity tolerance to get a test passing without explaining the measured errors in the pull request.

## Checking in a browser

The demo site in `site/` installs `forecast-web` from npm. To try your local changes, build the package and install it into the site:

```sh
npm run build
cd site
npm ci
npm install --no-save ..   # use the local package instead of the npm release
npm run data               # fetch the bundled datasets into public/data
npm run dev
```

Open http://localhost:5299/forecast-web/. To test locally built weights, serve `build/timesfm` over HTTP with CORS and pass its URL:

```sh
npx http-server build/timesfm --cors -p 5199
# then open http://localhost:5299/forecast-web/?weights=http://localhost:5199/
```

Check both backends: Chrome with WebGPU, and the WASM path (for example with WebGPU disabled in `chrome://flags`). WebGPU bugs have shown up only in the browser, never in Node, so do not skip this for graph changes.

## Pull requests

- Keep each pull request to one change.
- CI must pass: typecheck, tests, build, and a package dry run.
- Describe what changed and how you verified it, including the browser and backend you tested.
- Releases are handled by the maintainer through GitHub Actions. See `RELEASING.md`.

## License

By contributing, you agree that your contributions are licensed under the Apache License 2.0, the same license as the project.

## Adding a model

1. Pick a model whose weights allow commercial use (Apache-2.0, MIT, BSD).
2. `tools/models/<name>/`: `model.json` (repo, source, revision, files, required, and `scripts` when several sizes share one script folder), export and reference scripts, and `hub/README.md` plus `hub/config.json`. Compression is shared: `tools/quantize.py <name>`.
3. `src/models/<name>/`: an adapter implementing `ModelAdapter` from `src/models/adapter.ts`. Port the official forecast-level processing exactly.
4. Add the id to `ModelId` in `src/types.ts` and the adapter to `src/models/index.ts`.
5. `test/models/<name>/`: unit tests and a parity suite against the official Python library, with fixtures in `test/fixtures/<name>/`.
6. Add the model to the `model` choices in `.github/workflows/release-weights.yml` and `.github/workflows/release-model-card.yml`.
