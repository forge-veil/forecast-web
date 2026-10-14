---
license: bsd-3-clause
library_name: onnx
pipeline_tag: tabular-regression
base_model: jingang/TabICL
base_model_relation: quantized
tags:
  - tabicl
  - tabular
  - tabular-regression
  - in-context-learning
  - onnx
  - onnxruntime-web
  - webgpu
  - wasm
  - browser
  - javascript
---

# TabICL v2 regressor for the browser

[![npm](https://img.shields.io/npm/v/forecast-web?label=npm%20forecast-web)](https://www.npmjs.com/package/forecast-web)
[![Live demo](https://img.shields.io/badge/demo-live-0a84ff)](https://vatsalbakshi.com/forecast-web/)
[![GitHub](https://img.shields.io/badge/GitHub-forge--veil%2Fforecast--web-181717?logo=github)](https://github.com/forge-veil/forecast-web)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue)](#license)

ONNX exports of the [TabICLv2](https://github.com/soda-inria/tabicl) regression network (`tabicl-regressor-v2-20260212.ckpt` from [jingang/TabICL](https://huggingface.co/jingang/TabICL)), built to run tabular regression entirely in the browser on WebGPU or WebAssembly. They power `forecast-web/tabular` in the [`forecast-web`](https://www.npmjs.com/package/forecast-web) npm package.

TabICL predicts from a table with no training: it reads the training rows in context and predicts the test rows in one forward pass. No server, no API key, and the data never leaves the browser.

## Quick start

```sh
npm install forecast-web
```

```ts
import { loadTabular } from "forecast-web/tabular";

const reg = await loadTabular({ task: "regression" });
const fitted = reg.fit(trainRows, trainTargets);
const mean = await fitted.predict(testRows);                                 // point predictions
const bands = await fitted.predict(testRows, { output: "quantiles", alphas: [0.1, 0.9] });
```

## Files

| Format | Files | Size | Runs on | `dtype` |
|---|---|---|---|---|
| fp16 weights, fp32 compute | `onnx/model_fp16w.onnx` + `onnx/model_fp16w.f16` | 59 MB | WebGPU, WASM | `"fp16"` (default) |
| fp32 | `onnx/model.onnx` | 115 MB | WebGPU, WASM | `"fp32"` |

There is no 8-bit build: for this network it came out larger than the fp16-weight pair, because many small tensors stay fp32.

## The graph

The network only: column embedding (set transformer over rows), row interaction (with RoPE), in-context learning transformer, decoder.

| Name | Type | Shape | Meaning |
|---|---|---|---|
| `X` | float32 | `[batch, rows, features]` | preprocessed features, training rows first; batch = ensemble members |
| `y_train` | float32 | `[batch, train_rows]` | class index after the member's class shift, or standardized target |
| `quantiles` | float32 | `[batch, test_rows, 999]` | raw quantiles at levels 0.001 to 0.999 |

Everything around the graph runs in JavaScript and mirrors `tabicl` 2.2.0 exactly: ordinal encoding with categorical columns first, mean imputation, a constant-column filter, an ensemble of 8 members (Latin-square feature orders from Python's `random.Random(42)`, class shifts, and two normalizations), a standard scaler, Yeo-Johnson (scipy's bounded Brent search), and an outlier squash. Regression: sort each member's 999 quantiles; mean is their average; the median and other quantiles interpolate linearly inside [0.001, 0.999] with exponential tails outside; de-normalize, then average the members. The reference implementation is [`src/tabular`](https://github.com/forge-veil/forecast-web/tree/main/src/tabular).

Three export-only changes keep the graph shape-agnostic, with identical math: boolean-mask skips become `Where`, `log(n)` in the attention scaling uses `torch.log`, and classification returns all 10 logits instead of slicing to the class count.

## Accuracy

End-to-end agreement with `tabicl` 2.2.0 (`TabICLClassifier.predict_proba` / `TabICLRegressor.predict`, CPU, fp32) through the JavaScript pipeline, on a suite of tables: numeric, mixed categorical with booleans, missing values, constant columns, unseen categories, 2 to 10 classes, heavy-tailed targets, 15 to 900 training rows.

| Format | Worst error (fraction of the target's std; mean, median and quantiles) |
|---|---|
| fp32 | under 0.0004% (3.1e-6) |
| fp16 weights | under 0.08% (7.1e-4) |

The parity suite runs in CI before every upload; nothing is published if it fails.

## Limits

- Classification: at most 10 classes.
- Browser size limits: 5,000 training rows, 100 features and 20,000 cells (rows x features). Cost grows with rows x features: on a laptop CPU, 1,000 x 10 takes about 5 s for 100 test rows with 8 members; WebGPU is faster, and fewer members (`nEstimators`) cut time proportionally.
- Not included: more than 10 classes, the KV cache, TabICL's forecasting module and SHAP explanations.

## How these files were made

Built and verified by the [`release-weights`](https://github.com/forge-veil/forecast-web/blob/main/.github/workflows/release-weights.yml) GitHub Actions workflow: download the checkpoint at a pinned revision, export the graph (opset 17) and check it against PyTorch on unseen shapes and class counts, build the fp16-weight variant, generate reference predictions with `tabicl` 2.2.0, run the parity suite, and upload only if every check passes.

## Citation

```bibtex
@inproceedings{qu2025tabicl,
  title={Tab{ICL}: {A} Tabular Foundation Model for In-Context Learning on Large Data},
  author={Qu, Jingang and Holzm{\"u}ller, David and Varoquaux, Ga{\"e}l and Le Morvan, Marine},
  booktitle={International Conference on Machine Learning},
  year={2025}
}

@article{qu2026tabiclv2,
  title={{TabICLv2}: {A} better, faster, scalable, and open tabular foundation model},
  author={Qu, Jingang and Holzm{\"u}ller, David and Varoquaux, Ga{\"e}l and Le Morvan, Marine},
  journal={arXiv preprint arXiv:2602.11139},
  year={2026}
}
```

## License

These weights are converted from TabICL, which is licensed under the BSD 3-Clause License. The conversions and the `forecast-web` library are independent and not affiliated with or endorsed by Inria or the TabICL authors.

```
BSD 3-Clause License

Copyright (c) 2025, Soda team @ Inria

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
